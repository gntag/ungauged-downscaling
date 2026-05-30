"""
cerra_extract.py
================
Extract daily CERRA reanalysis values at arbitrary target locations using
cubic spatial interpolation.

CERRA GRID
----------
CERRA (Copernicus European Regional ReAnalysis) uses a ~5 km rotated
lat/lon grid covering Europe.  Variables are stored as 2-D (ny, nx) arrays
in NetCDF files.  The native grid is not regular in geographic coordinates,
so spatial interpolation is done via Delaunay triangulation + cubic
(CloughTocher2D) interpolation rather than a simple bilinear scheme.

INTERPOLATION APPROACH
----------------------
For each target location a 4×4 patch of surrounding CERRA grid cells is
assembled ("patch"), a Delaunay triangulation is built over those 16 points,
and the CloughTocher2DInterpolator (cubic, C1-continuous) is used to evaluate
the field at the target coordinates.  This approach:
  • Is more accurate than bilinear interpolation across complex terrain.
  • Handles the non-rectangular CERRA grid naturally.
  • Enables batch evaluation: the triangulation is precomputed once and
    reused across all time steps, months, and years.

LSM-AWARE "ISLAND PATCHING"
----------------------------
Small Aegean islands (e.g. Delos itself, 5 km²) often fall inside a CERRA
sea cell — the 5 km grid is too coarse to resolve them.  Interpolating from
sea cells would inject marine surface values (SST-driven temperatures,
ocean-surface humidity) into the predictors, corrupting the downscaling.

When a target's nearest CERRA cell is classified as sea (LSM < 0.5), the
patch is built from the nearest *land* cells instead of the geometrically
nearest cells.  Practically, this means the interpolated value at Delos
reflects the nearest land surface conditions rather than the surrounding sea.

ANALYSIS vs FORECAST VARIABLES
-------------------------------
CERRA distinguishes two stream types, each with a different file layout:

  Analysis stream  — 3-hourly, stored in monthly files per year
    t2m  : 2-m temperature (K)
    r2   : 2-m relative humidity (%)
    si10 : 10-m wind speed (m/s)
    Each is aggregated to daily mean / min / max before merging.

  Forecast stream  — sub-daily accumulations, stored in yearly files
    tp   : total precipitation (m) — accumulated to midnight; one value
           per day taken at the 00:00 Z step.
    fg10 : 10-m wind gust (m/s) — daily maximum across all lead times
           within each forecast initialisation day.

CACHING
-------
The CERRA land-sea mask (LSM) is read once from `cerra_lsm.nc` and cached
in `_lsm_cache`.  The same cache persists for the lifetime of the process,
so calling `extract_all()` multiple times (e.g. for stations then for the
Delos grid) only reads the LSM file once.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.interpolate import CloughTocher2DInterpolator
from scipy.spatial import Delaunay, KDTree

# Offset for converting Kelvin to Celsius (CERRA t2m is stored in K)
_KELVIN_OFFSET = 273.15

# Dimensions of the square patch assembled around each target location.
# A 4×4 patch gives 16 points, which is the minimum for C1-cubic interpolation
# with CloughTocher2DInterpolator (needs at least 4 non-collinear points).
_PATCH_SIZE = 4
_N_PATCH    = _PATCH_SIZE * _PATCH_SIZE   # = 16 grid cells per target

# ---------------------------------------------------------------------------
# CERRA analysis variables
# ---------------------------------------------------------------------------
# Mapping: short_key → (subfolder_name, nc_variable_name, unit_conversion_fn, daily_aggregations)
# - subfolder_name: the directory under cerra_dir that holds monthly NetCDF files
# - nc_variable_name: the variable name inside each NetCDF
# - unit_conversion_fn: applied element-wise to the raw array before aggregation
# - daily_aggregations: which daily statistics to compute ("mean", "min", "max")
_ANALYSIS_VARS = {
    "t2m":  ("2m_temperature",       "t2m",  lambda x: x - _KELVIN_OFFSET, ["mean", "min", "max"]),
    "r2":   ("2m_relative_humidity", "r2",   lambda x: x,                   ["mean"]),
    "si10": ("10m_wind_speed",       "si10", lambda x: x,                   ["mean"]),
}

# Maps (short_key, aggregation) → output column name in the returned DataFrame
_ANALYSIS_OUT = {
    "t2m":  {"mean": "cerra_tmean", "min": "cerra_tmin", "max": "cerra_tmax"},
    "r2":   {"mean": "cerra_rh"},
    "si10": {"mean": "cerra_wind"},
}

# ---------------------------------------------------------------------------
# CERRA forecast variables
# ---------------------------------------------------------------------------
# Mapping: short_key → (subfolder_name, nc_variable_name)
# tp   : total precipitation accumulated over the forecast period (m)
# fg10 : maximum 10-m wind gust since the previous post-processing step (m/s)
_FORECAST_VARS = {
    "tp":   ("total_precipitation",                                   "tp"),
    "fg10": ("10m_wind_gust_since_previous_post_processing", "fg10"),
}

# Output column names for forecast variables in the returned DataFrame
_FORECAST_OUT = {"tp": "cerra_precip", "fg10": "cerra_gust"}

# Module-level LSM cache: populated on the first call to _load_lsm()
_lsm_cache: np.ndarray | None = None


def _load_lsm(lsm_path: Path) -> np.ndarray:
    """
    Load the CERRA land-sea mask and return a flat boolean array.

    Reads the first time slice of the 'lsm' variable (shape ny × nx) and
    flattens it to (ny*nx,).  Values >= 0.5 are considered land (True).

    Results are cached in _lsm_cache so the file is only read once per
    process, regardless of how many times extract_all() is called.
    """
    global _lsm_cache
    if _lsm_cache is not None:
        return _lsm_cache
    with xr.open_dataset(lsm_path, engine="netcdf4") as ds:
        lsm2d = ds["lsm"].values[0]  # drop the single time dimension → (ny, nx)
    _lsm_cache = (lsm2d >= 0.5).ravel()   # True = land
    return _lsm_cache


def extract_all(
    cerra_dir: Path,
    locations: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Extract daily CERRA values at every location for all years in [year_start, year_end].

    Parameters
    ----------
    cerra_dir  : Root CERRA data directory.  Expected layout:
                   <cerra_dir>/<subfolder>/<year>/<month>.nc  (analysis)
                   <cerra_dir>/<subfolder>/<var>_<year>.nc    (forecast)
    locations  : DataFrame with columns: id, lat, lon.
    year_start : First year to extract (inclusive).
    year_end   : Last year to extract (inclusive).

    Returns
    -------
    Long-format DataFrame with columns:
      id, date, cerra_tmean, cerra_tmin, cerra_tmax, cerra_rh, cerra_wind,
      cerra_precip, cerra_gust
    One row per (location id, calendar day).

    IMPLEMENTATION NOTES
    --------------------
    Interpolation is performed in four stages:
      1. Load the CERRA grid lat/lon arrays from one sample NetCDF.
      2. Precompute per-target patches and Delaunay triangulations once.
      3. Compute the global sub-region bounding box to minimise array reads.
      4. For each year, extract analysis and forecast streams and merge them.
    """
    # --- Stage 1: read CERRA grid coordinates from a representative file ----
    # We only need one file to get the lat/lon grid — all CERRA files share
    # the same grid topology.
    sample = next((Path(cerra_dir) / "2m_temperature" / str(year_start)).glob("*.nc"))
    lat2d, lon2d = _load_grid(sample)
    ny, nx = lat2d.shape

    # --- Stage 2: precompute per-target patch and Delaunay triangulation ----
    target_ll = locations[["lat", "lon"]].values   # (N_locs, 2)

    # Load the CERRA LSM if available; if the file is missing, all targets
    # are treated as land (no LSM-aware island patching).
    from downscaling.config import LSM_PATH
    is_land_flat = _load_lsm(LSM_PATH) if LSM_PATH.exists() else None

    # patch_info: list of (row_indices, col_indices, Delaunay, target) per location
    patch_info = _precompute_patches(lat2d, lon2d, target_ll, is_land_flat)

    # --- Stage 3: compute the minimal bounding sub-region -------------------
    # Reading only the rows y_min:y_max and columns x_min:x_max from each
    # NetCDF file (instead of the entire ~5k×5k grid) reduces I/O by ~99 %.
    y_min, y_max, x_min, x_max = _patch_bbox(patch_info, ny, nx)

    # Convert absolute grid indices to relative indices within the sub-region
    # so _interpolate_subregion() can index directly into the sub-array.
    rel_info = [(rows - y_min, cols - x_min, tri, tgt) for rows, cols, tri, tgt in patch_info]
    loc_ids  = locations["id"].values

    # --- Stage 4: extract year by year and concatenate ----------------------
    yearly = []
    for year in range(year_start, year_end + 1):
        ana = _extract_analysis_year(cerra_dir, year, rel_info, y_min, y_max, x_min, x_max, loc_ids)
        fct = _extract_forecast_year(cerra_dir, year, rel_info, y_min, y_max, x_min, x_max, loc_ids)
        # Outer join: keeps all dates even if forecast or analysis data is missing
        merged = ana.merge(fct, on=["id", "date"], how="outer") if not fct.empty else ana
        yearly.append(merged)

    return (
        pd.concat(yearly, ignore_index=True)
        .drop_duplicates(subset=["id", "date"])   # defensive: remove stale cache duplicates
        .sort_values(["id", "date"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Grid and patch helpers
# ---------------------------------------------------------------------------

def _load_grid(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Read the 2-D latitude and longitude arrays from a CERRA NetCDF file.

    CERRA uses a rotated lat/lon projection stored as 2-D 'latitude' and
    'longitude' variables of shape (ny, nx).  These are geographic coordinates
    (not rotated), so they can be used directly for distance calculations.
    """
    with xr.open_dataset(sample_path, engine="netcdf4") as ds:
        lat2d = ds["latitude"].values    # shape (ny, nx), degrees north
        lon2d = ds["longitude"].values   # shape (ny, nx), degrees east
    return lat2d, lon2d


def _precompute_patches(
    lat2d: np.ndarray,
    lon2d: np.ndarray,
    target_latlons: np.ndarray,
    is_land_flat: np.ndarray | None = None,
) -> list[tuple[np.ndarray, np.ndarray, Delaunay, np.ndarray]]:
    """
    For each target location, select the interpolation patch cells and build
    a Delaunay triangulation over them.

    Parameters
    ----------
    lat2d, lon2d   : 2-D geographic coordinates of the CERRA grid (degrees).
    target_latlons : (N_targets, 2) array of (lat, lon) pairs.
    is_land_flat   : Optional flat boolean array (ny*nx,) where True = land
                     (LSM >= 0.5).  When provided, sea-falling targets use
                     the nearest land cells instead of geometrically nearest.

    Returns
    -------
    List of (rows, cols, tri, target) per target location:
      rows  : 1-D array of grid row indices for the patch cells
      cols  : 1-D array of grid column indices for the patch cells
      tri   : Delaunay triangulation over patch (lat, lon) coordinates
      target: (lat, lon) of the target point

    LONGITUDE SCALING
    -----------------
    The KDTree is built in a [lat, lon*cos(mean_lat)] space so that one
    unit in lat and lon correspond to approximately equal arc-length at the
    centre of the domain.  Without this scaling, longitude differences are
    underweighted at high latitudes, leading to incorrect nearest-neighbour
    selection.
    """
    ny, nx        = lat2d.shape
    flat_lats     = lat2d.ravel()
    flat_lons     = lon2d.ravel()
    # Scale factor converts longitude degrees to approximately the same
    # angular distance as latitude degrees at the domain's mean latitude.
    cos_lat = np.cos(np.radians(float(np.mean(flat_lats))))
    tree    = KDTree(np.column_stack([flat_lats, flat_lons * cos_lat]))
    half    = _PATCH_SIZE // 2    # = 2 cells on each side of the nearest cell
    results = []

    for lat, lon in target_latlons:
        q                = [lat, lon * cos_lat]   # query point in scaled space
        use_land_filter  = False

        # Check whether the nearest grid cell is a sea cell.
        # If so, switch to LSM-aware land-cell patching.
        if is_land_flat is not None:
            _, nearest_idx  = tree.query(q)
            use_land_filter = not bool(is_land_flat[nearest_idx])

        if use_land_filter:
            # --- LSM-aware patch: nearest _N_PATCH land cells ----------------
            # Search through up to ~100 km radius worth of candidates.
            # At 5 km CERRA spacing that is about 500 grid cells, but we scan
            # _N_PATCH * 500 candidates to be safe near coastlines.
            k_cand    = min(_N_PATCH * 500, ny * nx)
            _, flat_idxs = tree.query(q, k=k_cand)
            # Keep only land cells, take the closest _N_PATCH of them.
            chosen = flat_idxs[is_land_flat[flat_idxs]][:_N_PATCH]
            if len(chosen) < 4:
                # CloughTocher2D requires at least 4 non-collinear points.
                raise RuntimeError(
                    'Too few land cells ({}) found near target ({:.4f}, {:.4f}) '
                    'within {} candidates. Check that the land mask covers this '
                    'location.'.format(len(chosen), lat, lon, k_cand)
                )
        else:
            # --- Standard patch: 4×4 block around the nearest grid cell ------
            _, flat_idx = tree.query(q)
            y0, x0      = np.unravel_index(flat_idx, (ny, nx))
            # Clamp the patch to stay inside the grid boundaries.
            y_start = max(0, y0 - half)
            x_start = max(0, x0 - half)
            y_end   = min(ny, y_start + _PATCH_SIZE)
            x_end   = min(nx, x_start + _PATCH_SIZE)
            ys      = np.arange(y_start, y_end)
            xs      = np.arange(x_start, x_end)
            rows_2d, cols_2d = np.meshgrid(ys, xs, indexing='ij')
            chosen   = np.ravel_multi_index((rows_2d.ravel(), cols_2d.ravel()), (ny, nx))

        rows, cols = np.unravel_index(chosen, (ny, nx))
        patch_lat  = flat_lats[chosen]
        patch_lon  = flat_lons[chosen]
        # Build Delaunay triangulation once; re-used for every time step.
        tri = Delaunay(np.column_stack([patch_lat, patch_lon]))
        results.append((rows, cols, tri, np.array([lat, lon])))

    return results


def _patch_bbox(
    patch_info: list[tuple[np.ndarray, np.ndarray, Delaunay, np.ndarray]],
    ny: int,
    nx: int,
) -> tuple[int, int, int, int]:
    """
    Compute the minimal grid sub-region that contains all patches.

    Returns (y_min, y_max, x_min, x_max) as integer slice bounds such that
    data[:, y_min:y_max, x_min:x_max] contains every patch cell from
    every target location.  Clamped to valid grid bounds [0, ny) × [0, nx).
    """
    y_min = min(int(rows.min()) for rows, _, _, _ in patch_info)
    y_max = max(int(rows.max()) + 1 for rows, _, _, _ in patch_info)
    x_min = min(int(cols.min()) for _, cols, _, _ in patch_info)
    x_max = max(int(cols.max()) + 1 for _, cols, _, _ in patch_info)
    return max(0, y_min), min(ny, y_max), max(0, x_min), min(nx, x_max)


def _interpolate_subregion(
    data: np.ndarray,
    rel_rows: np.ndarray,
    rel_cols: np.ndarray,
    tri: Delaunay,
    target: np.ndarray,
) -> np.ndarray:
    """
    Evaluate a spatially varying field at one target location using cubic
    interpolation over a precomputed Delaunay triangulation.

    Parameters
    ----------
    data     : Sub-region array of shape (T, H, W), where T is the number
               of time steps and H×W is the spatial extent of the sub-region.
    rel_rows : 1-D index array into the H dimension of data.
    rel_cols : 1-D index array into the W dimension of data.
    tri      : Precomputed Delaunay triangulation over the patch's (lat, lon)
               coordinates (geographic, not grid indices).
    target   : (lat, lon) of the target location.

    Returns
    -------
    1-D array of shape (T,) — interpolated value at target for each time step.

    FALLBACK FOR NaN
    ----------------
    CloughTocher2DInterpolator returns NaN for points that lie outside the
    convex hull of the triangulation (i.e. the target is outside the patch).
    In this edge case we fall back to the nearest patch cell (constant
    extrapolation).  The cosine scaling applied to the longitude axis
    ensures the distance metric is approximately equidistant.
    """
    # Extract patch values for all T time steps: (T, N_patch_cells)
    patch_flat = data[:, rel_rows, rel_cols]

    # CloughTocher2D expects (N_points, N_values) — hence the transpose.
    # The interpolator takes the Delaunay as its first argument (node locations)
    # and the value matrix as its second argument (values at nodes).
    interp = CloughTocher2DInterpolator(tri, patch_flat.T)

    # Evaluate at the single target point; result is (T,) after squeezing.
    result = interp(target.reshape(1, -1))[0]

    # Fill any NaN (outside convex hull) with nearest-neighbour fallback.
    nan_mask = np.isnan(result)
    if np.any(nan_mask):
        # Use cosine-scaled distance so latitude and longitude have equal weight.
        scale   = np.array([1.0, np.cos(np.radians(target[0]))])
        nearest = np.argmin(np.linalg.norm((tri.points - target) * scale, axis=1))
        result[nan_mask] = patch_flat[nan_mask, nearest]

    return result


# ---------------------------------------------------------------------------
# Analysis variable extraction (monthly NetCDF files, 3-hourly)
# ---------------------------------------------------------------------------

def _extract_analysis_year(
    cerra_dir: Path,
    year: int,
    rel_info: list,
    y_min: int, y_max: int, x_min: int, x_max: int,
    loc_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Extract and aggregate all analysis-stream variables for one calendar year.

    For each variable in _ANALYSIS_VARS:
      1. Read all monthly NetCDF files for the given year.
      2. Apply the unit conversion function.
      3. Interpolate to each target location.
      4. Aggregate 3-hourly values to daily (mean / min / max as configured).

    Returns a wide-then-stacked DataFrame with columns: id, date, <cerra_vars>.
    Multiple variable chunks are combined by concatenating on axis=1 over the
    shared (id, date) multi-index.
    """
    chunks: list[pd.DataFrame] = []

    for nc_key, (folder, var_name, conv_fn, agg_methods) in _ANALYSIS_VARS.items():
        times, values = _read_analysis_variable(
            Path(cerra_dir) / folder, year, var_name, conv_fn,
            rel_info, y_min, y_max, x_min, x_max,
        )
        if times is None:
            continue

        # Normalise timestamps to midnight so groupby-date aggregation is clean.
        dates  = pd.DatetimeIndex(times).normalize()
        # df_wide: rows = sub-daily times, columns = location ids
        df_wide = pd.DataFrame(values, index=dates, columns=loc_ids)

        for agg in agg_methods:
            col     = _ANALYSIS_OUT[nc_key][agg]
            grouped = df_wide.groupby(df_wide.index)
            if agg == "mean":
                daily = grouped.mean()
            elif agg == "min":
                daily = grouped.min()
            else:
                daily = grouped.max()

            daily.index.name = "date"
            # Stack wide → long: (date, id) → value
            long = daily.stack().reset_index()
            long.columns = ["date", "id", col]
            chunks.append(long.set_index(["id", "date"]))

    if not chunks:
        return pd.DataFrame(columns=["id", "date"])

    # Combine all variables on the shared (id, date) index.
    # axis=1 join ensures no row duplication — each chunk contributes one column.
    return pd.concat(chunks, axis=1).reset_index()


def _read_analysis_variable(
    var_dir: Path,
    year: int,
    var_name: str,
    conv_fn,
    rel_info: list,
    y_min: int, y_max: int, x_min: int, x_max: int,
) -> tuple[pd.DatetimeIndex | None, np.ndarray | None]:
    """
    Read all monthly NetCDF files for one analysis variable in one year.

    Files are expected under var_dir/<year>/*.nc (one file per month).
    The spatial sub-region slice y_min:y_max, x_min:x_max is extracted from
    each file to avoid loading the full grid.

    Returns (all_times, values) where:
      all_times : DatetimeIndex of length T (all sub-daily timestamps for the year)
      values    : (T, N_locations) float array after unit conversion and interpolation
    """
    folder = var_dir / str(year)
    if not folder.exists():
        return None, None

    times_chunks, val_chunks = [], []
    for fpath in sorted(folder.glob("*.nc")):
        with xr.open_dataset(fpath, engine="netcdf4") as ds:
            # Read only the spatial sub-region to minimise memory usage.
            sub = (
                ds[var_name]
                .isel(y=slice(y_min, y_max), x=slice(x_min, x_max))
                .values.astype(float)   # (T_chunk, H_sub, W_sub)
            )
            chunk_times = pd.DatetimeIndex(ds.valid_time.values)

        # Apply unit conversion (e.g. K → °C for temperature).
        sub = conv_fn(sub)

        # Interpolate each location; stack results into (T_chunk, N_locs).
        loc_vals = np.column_stack(
            [_interpolate_subregion(sub, rows, cols, tri, tgt) for rows, cols, tri, tgt in rel_info]
        )
        times_chunks.append(chunk_times)
        val_chunks.append(loc_vals)

    if not times_chunks:
        return None, None

    all_times = pd.DatetimeIndex(
        np.concatenate([np.asarray(t, dtype="datetime64[ns]") for t in times_chunks])
    )
    return all_times, np.concatenate(val_chunks, axis=0)


# ---------------------------------------------------------------------------
# Forecast variable extraction (yearly NetCDF files, multiple lead times)
# ---------------------------------------------------------------------------

def _extract_forecast_year(
    cerra_dir: Path,
    year: int,
    rel_info: list,
    y_min: int, y_max: int, x_min: int, x_max: int,
    loc_ids: np.ndarray,
) -> pd.DataFrame:
    """
    Extract forecast-stream variables (precipitation, wind gust) for one year.

    Forecast files are stored as single yearly NetCDF files:
      <cerra_dir>/<subfolder>/<subfolder>_<year>.nc

    Precipitation (tp):
      CERRA total precipitation is an accumulation over the forecast period.
      The 00:00 Z step contains the complete 24-hour daily total.
      Other steps are skipped.  The valid_time at 00:00 Z is shifted back
      by one day to align with the calendar day the accumulation covers.

    Wind gust (fg10):
      The maximum gust since the previous post-processing step.
      Daily maximum is taken across all forecast lead times within each
      forecast initialisation day (see _daily_max_from_forecast).

    Returns a DataFrame with columns: id, date, cerra_precip, cerra_gust.
    """
    chunks: list[pd.DataFrame] = []

    for nc_key, (folder, var_name) in _FORECAST_VARS.items():
        fpath = Path(cerra_dir) / folder / f"{folder}_{year}.nc"
        if not fpath.exists():
            continue

        with xr.open_dataset(fpath, engine="netcdf4") as ds:
            # Slice to the minimal sub-region to reduce memory.
            sub = (
                ds[var_name]
                .isel(y=slice(y_min, y_max), x=slice(x_min, x_max))
                .values.astype(float)   # (T_all, H_sub, W_sub)
            )
            valid_times = pd.DatetimeIndex(ds.valid_time.values)

        # Interpolate all locations: (T_all, N_locs)
        values = np.column_stack(
            [_interpolate_subregion(sub, rows, cols, tri, tgt) for rows, cols, tri, tgt in rel_info]
        )

        col = _FORECAST_OUT[nc_key]
        if nc_key == "tp":
            # Keep only the 00:00 Z step; shift date back by one day.
            mask    = valid_times.hour == 0
            dates   = (valid_times[mask] - pd.Timedelta(days=1)).normalize()
            df_wide = pd.DataFrame(values[mask], index=dates, columns=loc_ids)
        else:
            # Daily maximum: group by forecast init day, take the max over lead times.
            dates, vals2d = _daily_max_from_forecast(valid_times, values)
            df_wide = pd.DataFrame(vals2d, index=dates, columns=loc_ids)

        df_wide.index.name = "date"
        long = df_wide.stack().reset_index()
        long.columns = ["date", "id", col]
        chunks.append(long.set_index(["id", "date"]))

    if not chunks:
        return pd.DataFrame(columns=["id", "date"])

    # Join on shared (id, date) multi-index (not stack) to avoid row duplication.
    # axis=0 would concatenate rows, creating duplicate (id, date) entries for
    # dates that have both tp and fg10 data.
    return pd.concat(chunks, axis=1).reset_index()


def _daily_max_from_forecast(
    valid_times: pd.DatetimeIndex,
    values: np.ndarray,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """
    Aggregate sub-daily forecast values to daily maximum.

    The CERRA forecast is initialised twice daily (00 Z, 12 Z), producing
    multiple valid_times per calendar day.  We assign each time step to its
    initialisation day (floor to midnight, or previous midnight for the
    00:00 Z step which belongs to the day that ended), then take the maximum
    over all steps assigned to the same init day.

    Parameters
    ----------
    valid_times : DatetimeIndex of all forecast valid times.
    values      : Array of shape (T,) or (T, N_locations).

    Returns
    -------
    (dates, daily_max):
      dates     : DatetimeIndex of unique initialisation days
      daily_max : Array of shape (N_days,) or (N_days, N_locations)
    """
    # Time steps at 00:00 Z are the "end" of the previous day's forecast;
    # assign them to the previous calendar day.
    init_dates = np.where(
        valid_times.hour != 0,
        valid_times.floor("D").values,
        (valid_times - pd.Timedelta(days=1)).floor("D").values,
    )
    idx = pd.DatetimeIndex(init_dates)

    if values.ndim == 1:
        daily = pd.Series(values, index=idx).groupby(level=0).max()
        return pd.DatetimeIndex(daily.index), daily.values

    # 2-D case: (T, N_locs) → (N_days, N_locs)
    daily = pd.DataFrame(values, index=idx).groupby(level=0).max()
    return pd.DatetimeIndex(daily.index), daily.values
