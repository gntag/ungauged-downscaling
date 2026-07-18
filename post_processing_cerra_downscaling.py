"""
post_processing_cerra_downscaling.py
=====================================
Post-process the LightGBM quantile predictions from predict.py into
single best-estimate daily values and diagnostic plots.

MODES
-----
This script has two independent modes, controlled by the POSTPROCESS_MODE
setting in downscaling/config.py (or via command-line flags):

  daily  — Read the quantile Zarr produced by predict.py and write a new
            "daily values" Zarr (delos_downscaled_daily.zarr) with one
            best-estimate value per variable per day per grid point.

  plots  — Read the daily values Zarr and write diagnostic PNG plots:
            - Mean annual value maps (one per variable)
            - Average annual days-above-threshold maps (e.g. hot days, wet days)

  both   — Run both steps in sequence.

DAILY VALUE DERIVATION
-----------------------
For temperature, relative humidity, wind, and gust:
  The p50 (median) prediction is used directly as the daily best estimate.
  (p10 and p90 are retained in the source Zarr for uncertainty quantification
  but are not written to the daily values output.)

For wind and gust:
  The modelled mean (wind_mean, gust_mean) is used rather than p50.
  The mean model was specifically trained with a loss designed to minimise
  spatial artefacts at the coast; p50 from a quantile-regression model can
  exhibit step-changes near coastlines.

For precipitation:
  The two-part model produces:
    - precip_occurrence_prob  : P(wet day)    — unconditional probability
    - precip_p50, precip_p90  : conditional quantiles (wet days only)

  The expected daily precipitation is:
    E[P] = P(wet) × E[P | wet]

  E[P | wet] is estimated from the conditional p50 and p90 by inverting
  a gamma distribution:
    1. The ratio p90 / p50 constrains the shape parameter k of the gamma.
    2. k is recovered via a precomputed lookup table (logspace grid over k,
       ratio tabulated using scipy.stats.gamma.ppf).
    3. The scale parameter θ = p50 / Γ_ppf(0.5, k) ensures the fitted gamma
       median equals the predicted p50.
    4. The gamma mean is k × θ.

  This avoids the negative bias introduced by simply using p50 × P(wet),
  which underestimates daily totals on heavy-rain days (the conditional
  distribution is right-skewed, so the mean > median).

INPUT / OUTPUT
--------------
  Source Zarr  : predict.py output (quantile predictions, shape n_times × n_points)
  Daily Zarr   : post-processed best-estimate values (same shape)
  Plots dir    : PNG files, one per variable or threshold

CONFIGURATION
-------------
All paths and plot specifications are read from downscaling/config.py:
  POSTPROCESS_SOURCE_ZARR    : input Zarr from predict.py
  POSTPROCESS_DAILY_ZARR     : output daily-values Zarr
  POSTPROCESS_PLOT_DIR       : where PNG files are written
  POSTPROCESS_MODE           : default mode if no CLI flag given
  POSTPROCESS_PLOT_VARIABLES : which variables to plot (mean annual)
  POSTPROCESS_PLOT_STYLES    : colormap/vmin/vmax per variable
  POSTPROCESS_THRESHOLD_PLOTS: list of threshold-exceedance plot specs
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib
# Non-interactive backend — required on servers without a display.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc
from scipy.stats import gamma as _gamma_dist

from downscaling.config import (
    POSTPROCESS_DAILY_ZARR,
    POSTPROCESS_MODE,
    POSTPROCESS_PLOT_DIR,
    POSTPROCESS_PLOT_STYLES,
    POSTPROCESS_PLOT_VARIABLES,
    POSTPROCESS_SOURCE_ZARR,
    POSTPROCESS_THRESHOLD_PLOTS,
)

# Zarr output parameters — must match the chunking used in predict.py so
# downstream xarray/zarr reads can use the same chunk cache efficiently.
_ZARR_CHUNKS = (365, 50)
_COMPRESSOR  = Blosc(cname="zstd", clevel=5)

_VALID_MODES = {"daily", "plots", "both"}

# ---------------------------------------------------------------------------
# Daily-value derivation: source quantile variable → output variable metadata
# ---------------------------------------------------------------------------
# Each entry maps the desired output column name to:
#   source     : the Zarr variable to read from the quantile store
#   units      : CF-standard units string
#   long_name  : human-readable description for Zarr attribute
_DAILY_VALUE_DEFINITIONS = {
    "tmean_c": {
        "source":    "tmean_p50",
        "units":     "deg C",
        "long_name": "daily mean temperature",
    },
    "tmin_c": {
        "source":    "tmin_p50",
        "units":     "deg C",
        "long_name": "daily minimum temperature",
    },
    "tmax_c": {
        "source":    "tmax_p50",
        "units":     "deg C",
        "long_name": "daily maximum temperature",
    },
    "rh_pct": {
        "source":    "rh_p50",
        "units":     "%",
        "long_name": "daily relative humidity",
    },
    "wind_ms": {
        "source":    "wind_mean",
        "units":     "m/s",
        "long_name": "daily mean wind speed",
    },
    "gust_ms": {
        "source":    "gust_mean",
        "units":     "m/s",
        "long_name": "daily mean wind gust",
    },
}

_PRECIP_UNITS     = "mm"
_PRECIP_LONG_NAME = ("daily expected precipitation: "
                     "wet-day probability × gamma-fitted conditional mean")

# ---------------------------------------------------------------------------
# Gamma lookup table for _gamma_conditional_mean
# ---------------------------------------------------------------------------
# Building this at import time (rather than inside the function) avoids
# repeated scipy.stats.gamma.ppf calls for each grid point.
#
# For a Gamma(k, θ) distribution:
#   ratio = ppf(0.9, k) / ppf(0.5, k)
# is a strictly decreasing function of k (as k → 0, the distribution becomes
# more right-skewed and the ratio → ∞; as k → ∞ it approaches 1).
# We tabulate this mapping over 800 log-spaced k values so that we can later
# invert (ratio → k) via np.interp.
_K_TABLE     = np.logspace(-1, 2, 800)          # shape parameters to tabulate
_RATIO_TABLE = (_gamma_dist.ppf(0.9, _K_TABLE)  # p90/p50 ratio for each k
                / _gamma_dist.ppf(0.5, _K_TABLE))
_PPF50_TABLE = _gamma_dist.ppf(0.5, _K_TABLE)   # p50 as a fraction of scale θ


def _gamma_conditional_mean(p50: np.ndarray, p90: np.ndarray) -> np.ndarray:
    """
    Estimate E[P | wet] from the conditional p50 and p90 quantiles by
    fitting a gamma distribution to the two-quantile constraints.

    Algorithm
    ---------
    1. Compute the observed ratio r = p90 / p50.
    2. Invert r → k via the precomputed _RATIO_TABLE (linear interpolation).
       _RATIO_TABLE is decreasing, so we reverse both tables before passing
       to np.interp (which requires xp to be increasing).
    3. Recover the scale: θ = p50 / Γ.ppf(0.5, k)
    4. Return the gamma mean: k × θ.

    Fallback
    --------
    Where p50 ≤ 0 or p90 ≤ p50 (non-monotone quantiles), p50 itself is
    returned unchanged.  This avoids division-by-zero and preserves the
    physically-meaningful lower bound of zero.
    """
    valid    = (p50 > 0) & (p90 > p50)
    safe_p50 = np.where(valid, p50, 1.0)          # avoid div-by-zero before np.where
    ratio    = np.where(valid, p90 / safe_p50, np.nan)

    # Clip ratio to the tabulated range to avoid extrapolation.
    ratio_clipped = np.clip(ratio, _RATIO_TABLE[-1], _RATIO_TABLE[0])

    # Invert the decreasing ratio → k mapping (flip arrays so xp is increasing).
    k_est  = np.interp(ratio_clipped, _RATIO_TABLE[::-1], _K_TABLE[::-1])
    # Recover scale θ such that Gamma(k, θ).median() == p50.
    scale  = p50 / np.interp(k_est, _K_TABLE, _PPF50_TABLE)
    mean_est = (k_est * scale).astype("float32")

    return np.where(valid, mean_est, p50).astype("float32")


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """
    Parse command-line flags.  If no flag is given, the mode falls back to
    POSTPROCESS_MODE from config.py.
    """
    parser = argparse.ArgumentParser(
        description="Post-process Delos downscaled output into daily values, plots, or both."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-daily", "--daily",
        action="store_const", const="daily", dest="mode",
        help="Write only delos_downscaled_daily.zarr.",
    )
    group.add_argument(
        "-plots", "--plots",
        action="store_const", const="plots", dest="mode",
        help="Write only plots from the existing daily Zarr.",
    )
    group.add_argument(
        "-both", "--both",
        action="store_const", const="both", dest="mode",
        help="Write daily values and plots.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------

def run_post_processing(mode: str) -> None:
    """Orchestrate the chosen post-processing mode (daily / plots / both)."""
    mode  = normalize_mode(mode)
    daily = None

    if mode in {"daily", "both"}:
        source = zarr.open(str(POSTPROCESS_SOURCE_ZARR), mode="r")
        daily  = write_daily_values(source, POSTPROCESS_DAILY_ZARR)

    if mode in {"plots", "both"}:
        if daily is None:
            # Plots-only mode: open the pre-existing daily Zarr.
            if not POSTPROCESS_DAILY_ZARR.exists():
                raise FileNotFoundError(
                    f"Daily Zarr does not exist: {POSTPROCESS_DAILY_ZARR}. "
                    'Run with POSTPROCESS_MODE = "daily" or "both" first.'
                )
            daily = zarr.open(str(POSTPROCESS_DAILY_ZARR), mode="r")
        write_plots(daily, POSTPROCESS_PLOT_DIR)


def normalize_mode(mode: str) -> str:
    """
    Normalise the mode string.  Accepts 'daily_values' as an alias for 'daily'
    (matches the old config key name) and raises ValueError for invalid inputs.
    """
    value = str(mode).strip().lower()
    if value == "daily_values":
        value = "daily"
    if value not in _VALID_MODES:
        raise ValueError(
            f"POSTPROCESS_MODE must be one of {sorted(_VALID_MODES)}; got {mode!r}")
    return value


# ---------------------------------------------------------------------------
# Daily value writing
# ---------------------------------------------------------------------------

def write_daily_values(source: zarr.Group, output_path: Path) -> zarr.Group:
    """
    Convert the quantile Zarr produced by predict.py into a daily-values Zarr.

    Reads coordinate arrays (time, lat, lon, elevation_m) and each variable
    from _DAILY_VALUE_DEFINITIONS, then computes precipitation via the
    two-part gamma model.

    The output Zarr uses the same Zstandard-5 compression and (365, 50)
    chunking as the source, for compatibility with downstream analysis.

    Returns the opened daily Zarr group (mode="w", then implicitly readable).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    time      = source["time"][:]
    lat       = source["lat"][:]
    lon       = source["lon"][:]
    elevation = source["elevation_m"][:] if "elevation_m" in source else None
    n_times   = len(time)
    n_points  = len(lat)
    data_chunks = (min(n_times, _ZARR_CHUNKS[0]), min(n_points, _ZARR_CHUNKS[1]))

    daily = zarr.open(str(output_path), mode="w")
    daily.attrs["source_zarr"]           = str(POSTPROCESS_SOURCE_ZARR)
    daily.attrs["precip_mm_definition"]  = (
        "precip_best_eqm (frequency-anchored EQM) if present, "
        "else precip_occurrence_prob * E[P|wet]"
    )

    # --- Coordinate variables ------------------------------------------------
    create_1d_dataset(daily, "time", time, "int64", min(n_times, _ZARR_CHUNKS[0]))
    daily["time"].attrs.update(dict(source["time"].attrs))
    daily["time"].attrs["_ARRAY_DIMENSIONS"] = ["time"]

    create_1d_dataset(daily, "lat", lat, "float64", n_points)
    daily["lat"].attrs["_ARRAY_DIMENSIONS"] = ["point"]

    create_1d_dataset(daily, "lon", lon, "float64", n_points)
    daily["lon"].attrs["_ARRAY_DIMENSIONS"] = ["point"]

    if elevation is not None:
        create_1d_dataset(daily, "elevation_m", elevation, "float32", n_points)
        daily["elevation_m"].attrs.update({
            "units": "m", "long_name": "elevation",
            "_ARRAY_DIMENSIONS": ["point"],
        })

    # --- Standard variables (median or mean of quantile predictions) ---------
    for output_name, definition in _DAILY_VALUE_DEFINITIONS.items():
        source_name = definition["source"]
        require_array(source, source_name)
        data = source[source_name][:].astype("float32", copy=False)
        create_2d_dataset(daily, output_name, data, data_chunks)
        daily[output_name].attrs.update({
            "units":               definition["units"],
            "long_name":           definition["long_name"],
            "_ARRAY_DIMENSIONS":   ["time", "point"],
        })

    # --- Precipitation: final daily total ------------------------------------
    # Prefer the frequency-anchored EQM correction (precip_best_eqm) written by
    # predict.py — it restores annual totals and extreme frequency that the raw
    # two-part model compresses.  Fall back to the two-part expected value
    # E[P] = P(wet) × E[P | wet] when EQM output is not present in the source.
    if "precip_best_eqm" in source:
        precip = np.asarray(source["precip_best_eqm"][:], dtype="float32")
        precip = np.nan_to_num(precip, nan=0.0)
        precip_def = "precip_best_eqm (frequency-anchored EQM correction)"
    else:
        require_array(source, "precip_occurrence_prob")
        require_array(source, "precip_p50")
        require_array(source, "precip_p90")
        precip = (
            source["precip_occurrence_prob"][:]
            * _gamma_conditional_mean(source["precip_p50"][:], source["precip_p90"][:])
        ).astype("float32")
        precip_def = "precip_occurrence_prob * E[P | wet] (gamma-fitted from p50/p90)"

    create_2d_dataset(daily, "precip_mm", precip, data_chunks)
    daily["precip_mm"].attrs.update({
        "units":             _PRECIP_UNITS,
        "long_name":         _PRECIP_LONG_NAME,
        "definition":        precip_def,
        "_ARRAY_DIMENSIONS": ["time", "point"],
    })

    print(f"Wrote daily values to {output_path}")
    return daily


# ---------------------------------------------------------------------------
# Zarr dataset creation helpers
# ---------------------------------------------------------------------------

def create_1d_dataset(
    store: zarr.Group,
    name: str,
    data: np.ndarray,
    dtype: str,
    chunk_size: int,
) -> None:
    """Write a 1-D coordinate array to a Zarr store with standard compression."""
    store.create_dataset(
        name, data=data, dtype=dtype,
        chunks=(chunk_size,), compressor=_COMPRESSOR,
    )


def create_2d_dataset(
    store: zarr.Group,
    name: str,
    data: np.ndarray,
    chunks: tuple[int, int],
) -> None:
    """Write a 2-D (time × point) data array to a Zarr store as float32."""
    store.create_dataset(
        name, data=data, dtype="float32",
        chunks=chunks, compressor=_COMPRESSOR,
    )


def require_array(store: zarr.Group, name: str) -> None:
    """Raise KeyError with a descriptive message if 'name' is absent from store."""
    if name not in store:
        raise KeyError(f"Required Zarr array is missing: {name}")


# ---------------------------------------------------------------------------
# Plot writing
# ---------------------------------------------------------------------------

def write_plots(daily: zarr.Group, output_dir: Path) -> None:
    """
    Generate diagnostic maps from the daily-values Zarr.

    Two types of plots are produced:

    Mean annual value maps
      For each variable in POSTPROCESS_PLOT_VARIABLES, compute the mean of
      annual means (equal weight to each year regardless of leap years) and
      render a pcolormesh map with the colormap/vmin/vmax from
      POSTPROCESS_PLOT_STYLES.

    Threshold-exceedance maps
      For each entry in POSTPROCESS_THRESHOLD_PLOTS, count the average number
      of days per year that exceed the given threshold (e.g. tmax_c > 35 °C).

    All plots are saved as 180-dpi PNGs.  A CSV summary of min/mean/max
    values across all grid points is written alongside the images.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    time         = pd.to_datetime(daily["time"][:], unit="ns")
    years        = time.year.to_numpy()
    unique_years = np.unique(years)
    lon          = daily["lon"][:]
    lat          = daily["lat"][:]
    rows         = []

    # --- Mean annual value maps ----------------------------------------------
    for variable in POSTPROCESS_PLOT_VARIABLES:
        require_array(daily, variable)
        values       = daily[variable][:]
        annual_values = mean_of_annual_means(values, years, unique_years)
        attrs        = dict(daily[variable].attrs)
        style        = POSTPROCESS_PLOT_STYLES.get(variable, {})
        title        = (f"{attrs.get('long_name', variable)}: "
                        f"mean annual value ({unique_years[0]}-{unique_years[-1]})")
        output_path  = (output_dir
                        / f"mean_annual_{variable}_{unique_years[0]}_{unique_years[-1]}.png")
        plot_grid(lon, lat, annual_values, title, attrs.get("units", ""), style, output_path)
        rows.append(summary_row("mean annual value", variable,
                                attrs.get("units", ""), annual_values, output_path))

    # --- Threshold-exceedance maps -------------------------------------------
    for spec in POSTPROCESS_THRESHOLD_PLOTS:
        variable   = spec["variable"]
        require_array(daily, variable)
        values     = daily[variable][:]
        annual_days = average_annual_days_above(values, years, unique_years,
                                                float(spec["threshold"]))
        title       = f"{spec['title']} ({unique_years[0]}-{unique_years[-1]})"
        output_path = output_dir / spec["filename"]
        plot_grid(lon, lat, annual_days, title, spec.get("units", "days/year"),
                  spec, output_path)
        rows.append(summary_row("average annual threshold days", variable,
                                spec.get("units", ""), annual_days, output_path))

    # --- CSV summary ---------------------------------------------------------
    summary_path = output_dir / "post_processing_plot_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Wrote {len(rows)} plots to {output_dir}")
    print(f"Wrote plot summary to {summary_path}")


# ---------------------------------------------------------------------------
# Annual statistics helpers
# ---------------------------------------------------------------------------

def mean_of_annual_means(
    values: np.ndarray,
    years: np.ndarray,
    unique_years: np.ndarray,
) -> np.ndarray:
    """
    Compute the mean of annual means over all grid points.

    Takes the per-year daily mean first (so each year has equal weight
    regardless of the number of days), then averages across years.

    Returns a 1-D array of shape (n_points,).
    """
    annual_means = [np.nanmean(values[years == year], axis=0) for year in unique_years]
    return np.nanmean(np.stack(annual_means, axis=0), axis=0)


def average_annual_days_above(
    values: np.ndarray,
    years: np.ndarray,
    unique_years: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Compute the average number of days per year that exceed 'threshold'.

    Counts days exceeding the threshold within each year, then averages
    across years.  Returns a 1-D array of shape (n_points,).
    """
    annual_counts = [
        np.sum(values[years == year] > threshold, axis=0) for year in unique_years
    ]
    return np.mean(np.stack(annual_counts, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
    title: str,
    units: str,
    style: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Render a pcolormesh map of values on the Delos irregular grid.

    Converts the 1-D point arrays (lon, lat, values) to a regular 2-D grid
    via values_to_grid(), then constructs cell-edge coordinates for pcolormesh
    via edges_from_centers().

    The colormap is taken from style["cmap"] (default "viridis"); NaN cells
    are rendered as transparent so the background shows through.

    Parameters
    ----------
    style : dict with optional keys: cmap, vmin, vmax (passed to pcolormesh)
    """
    lon_centers, lat_centers, grid = values_to_grid(lon, lat, values)
    lon_edges = edges_from_centers(lon_centers)
    lat_edges = edges_from_centers(lat_centers)

    cmap = plt.get_cmap(style.get("cmap", "viridis")).copy()
    cmap.set_bad((1.0, 1.0, 1.0, 0.0))   # NaN → fully transparent

    fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
    mesh = ax.pcolormesh(
        lon_edges, lat_edges,
        np.ma.masked_invalid(grid),
        cmap=cmap, shading="auto",
        vmin=style.get("vmin"), vmax=style.get("vmax"),
    )
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(color="#d0d0d0", linewidth=0.5, alpha=0.5)

    colorbar = fig.colorbar(mesh, ax=ax, shrink=0.82)
    if units:
        colorbar.set_label(units)

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def values_to_grid(
    lon: np.ndarray,
    lat: np.ndarray,
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map 1-D (lon, lat, values) point arrays onto a 2-D regular grid.

    The Delos grid is assumed to lie on a regular lat/lon mesh; unique
    coordinate values are rounded to 9 decimal places before constructing
    the index to avoid floating-point equality mismatches.

    Returns (lon_centers, lat_centers, grid) where grid has shape
    (n_lats, n_lons) and NaN in cells with no data point.
    """
    lon_centers = np.unique(np.round(lon, 9))
    lat_centers = np.unique(np.round(lat, 9))
    grid = np.full((len(lat_centers), len(lon_centers)), np.nan, dtype=float)

    lon_index = {value: index for index, value in enumerate(lon_centers)}
    lat_index = {value: index for index, value in enumerate(lat_centers)}
    for x, y, value in zip(np.round(lon, 9), np.round(lat, 9), values):
        grid[lat_index[y], lon_index[x]] = value

    return lon_centers, lat_centers, grid


def edges_from_centers(centers: np.ndarray) -> np.ndarray:
    """
    Compute cell-edge coordinates from cell-centre coordinates.

    For n centres, returns n+1 edges by placing midpoints between consecutive
    centres and extrapolating half a cell-width at each end.  Required by
    pcolormesh (which expects edges, not centres) to avoid off-by-half-cell
    alignment.

    Edge: │    │    │    │
    Ctr:  · 0  · 1  · 2  ·  (3 centres → 4 edges)
    """
    centers = np.asarray(centers, dtype=float)
    if len(centers) == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5])

    edges        = np.empty(len(centers) + 1, dtype=float)
    edges[1:-1]  = (centers[:-1] + centers[1:]) / 2.0
    edges[0]     = centers[0] - (centers[1] - centers[0]) / 2.0
    edges[-1]    = centers[-1] + (centers[-1] - centers[-2]) / 2.0
    return edges


def summary_row(
    metric: str,
    variable: str,
    units: str,
    values: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    """Build one row of the plot summary CSV (finite-values only statistics)."""
    finite = values[np.isfinite(values)]
    return {
        "metric":   metric,
        "variable": variable,
        "units":    units,
        "min":      float(np.min(finite)),
        "mean":     float(np.mean(finite)),
        "max":      float(np.max(finite)),
        "file":     str(output_path),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    run_post_processing(args.mode or POSTPROCESS_MODE)


if __name__ == "__main__":
    main()
