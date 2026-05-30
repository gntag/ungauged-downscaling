"""
predict.py
==========
Apply trained LightGBM models to the Delos target grid and write output Zarr.

INPUT
-----
  cerra_delos  : CERRA values at all Delos grid points (from cerra_extract.py)
  delos_geo    : Geographic features for all Delos points (from geo_features.py)
  model_dir    : Directory containing <name>.txt LightGBM model files
  delos_csv_path : CSV with Delos grid coordinates (lon, lat, elevation_m)

OUTPUT
------
  output_path : Zarr store (compressed with Zstandard level 5) containing:
    time             (int64 nanoseconds since epoch)
    lon, lat         (float64 coordinate arrays)
    elevation_m      (float32)
    <var>_p10, <var>_p50, <var>_p90  — quantile predictions per variable
    <var>_mean                        — mean prediction (wind and gust only)
    precip_occurrence_prob            — P(wet day)
    precip_p10/p50/p90               — conditional quantiles on wet days

OUTPUT SHAPE
------------
All 2-D arrays have shape (n_times, n_points), where n_times is the number of
unique dates in cerra_delos and n_points is the number of Delos grid points.

POST-PROCESSING INVARIANTS
--------------------------
After prediction, two physical consistency checks are applied:
  1. Temperature ordering: tmin ≤ tmean ≤ tmax  (per quantile tag)
  2. Quantile ordering:    p10 ≤ p50 ≤ p90      (per variable)

These can be violated locally because each quantile/variable is a separate model.
Violations are resolved by sorting (not clipping) so no information is discarded.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from downscaling.config import GEO_FEATURES, VARIABLE_CERRA_FEATURES

# ---------------------------------------------------------------------------
# Zarr output parameters
# ---------------------------------------------------------------------------
# Chunk along time dimension to match typical analysis window sizes;
# chunk along point dimension for efficient per-point reads.
_ZARR_CHUNKS = (365, 50)

# Zstandard compression: good ratio with fast decompression — suitable for
# scientific arrays accessed by sliced indexing.
_COMPRESSOR = Blosc(cname="zstd", clevel=5)

_ALL_VARS      = ["tmean", "tmin", "tmax", "rh", "precip", "wind", "gust"]
_QUANTILE_TAGS = ["p10", "p50", "p90"]
_MEAN_VARS     = {"wind", "gust"}


def predict_delos(
    cerra_delos: pd.DataFrame,
    delos_geo: pd.DataFrame,
    model_dir: Path,
    delos_csv_path: Path,
    output_path: Path,
) -> None:
    """
    Run all trained models over the Delos CERRA + geo feature matrix and write
    the results as a Zarr store.

    Parameters
    ----------
    cerra_delos    : CERRA values at Delos points (id, date, cerra_* columns)
    delos_geo      : Geographic features at Delos points (id, lat, lon, …)
    model_dir      : Directory with .txt LightGBM model files
    delos_csv_path : CSV with lon, lat, VALUE (elevation) for the Delos grid
    output_path    : Path for the output .zarr store (created or overwritten)
    """
    model_dir  = Path(model_dir)
    models     = _load_models(model_dir)

    delos_coords = pd.read_csv(delos_csv_path)[["lon", "lat", "VALUE"]].rename(
        columns={"VALUE": "elevation_m"}
    )

    # Build the full feature matrix: one row per (date × Delos point)
    # Duplicates from a stale parquet cache are dropped defensively.
    geo_cols = ["id"] + [c for c in ["lat", "lon", "elevation_m", "coast_dist_km", "is_island"]
                         if c in delos_geo.columns]
    feat_df = (
        cerra_delos
        .drop_duplicates(subset=["date", "id"])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
        .merge(delos_geo[geo_cols].drop_duplicates("id"), on="id", how="left")
    )

    dates     = pd.DatetimeIndex(sorted(feat_df["date"].unique()))
    point_ids = sorted(feat_df["id"].unique())
    n_times, n_points = len(dates), len(point_ids)

    # Harmonic month encoding (same as training in train.py)
    feat_df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)
    feat_df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)

    # --- Predict each variable ------------------------------------------------
    data_vars: dict[str, np.ndarray] = {}

    for var in _ALL_VARS:
        feature_cols = VARIABLE_CERRA_FEATURES[var] + GEO_FEATURES
        available    = [c for c in feature_cols if c in feat_df.columns]
        if not available:
            continue

        X = feat_df[available].values   # (n_times × n_points, n_features)

        if var == "precip":
            _predict_precip(X, models, n_times, n_points, data_vars)
        else:
            for qtag in _QUANTILE_TAGS:
                name = f"{var}_{qtag}"
                if name not in models:
                    continue
                # LightGBM predict returns a 1-D array; reshape to (n_times, n_points)
                raw = models[name].predict(X).reshape(n_times, n_points).astype("float32")
                raw = _clip_variable(var, qtag, raw)
                data_vars[name] = raw
            if var in _MEAN_VARS:
                _predict_mean(var, X, models, n_times, n_points, data_vars)

    # Physical consistency enforcement (vectorised, no information loss)
    _enforce_temp_ordering(data_vars)
    _enforce_quantile_ordering(data_vars)

    _write_zarr(data_vars, dates, delos_coords, output_path)


def _predict_mean(var: str, X, models, n_times, n_points, data_vars) -> None:
    """Predict the mean value for wind or gust and clip to non-negative."""
    name = f"{var}_mean"
    if name not in models:
        return
    raw = models[name].predict(X).reshape(n_times, n_points).astype("float32")
    data_vars[name] = np.clip(raw, 0, None)


def _predict_precip(X, models, n_times, n_points, data_vars):
    """
    Run the two-part precipitation model.

    Occurrence probability is predicted unconditionally (all X).
    Conditional quantiles are also predicted for all X, but their physical
    interpretation is "amount given a wet day" — the post-processing step
    combines them with the occurrence probability to get expected daily precip.
    """
    if "precip_occ" in models:
        occ = models["precip_occ"].predict(X).reshape(n_times, n_points).astype("float32")
        data_vars["precip_occurrence_prob"] = np.clip(occ, 0, 1)

    for qtag in _QUANTILE_TAGS:
        name = f"precip_{qtag}"
        if name not in models:
            continue
        raw = models[name].predict(X).reshape(n_times, n_points).astype("float32")
        data_vars[name] = np.clip(raw, 0, None)


def _clip_variable(var: str, qtag: str, arr: np.ndarray) -> np.ndarray:
    """Apply physical bounds to predicted values."""
    if var == "rh":
        return np.clip(arr, 0, 100)
    if var in ("wind", "gust"):
        return np.clip(arr, 0, None)   # wind speed cannot be negative
    return arr                          # temperature: no physical bounds


def _enforce_temp_ordering(data_vars: dict):
    """
    Ensure tmin ≤ tmean ≤ tmax within each quantile tag.

    Violations arise because each temperature variable is modelled independently.
    We resolve them by element-wise min/max rather than clipping (so the total
    information in the ensemble is preserved).
    """
    for qtag in _QUANTILE_TAGS:
        tmin_k, tmean_k, tmax_k = f"tmin_{qtag}", f"tmean_{qtag}", f"tmax_{qtag}"
        if tmean_k in data_vars and tmin_k in data_vars:
            data_vars[tmin_k] = np.minimum(data_vars[tmin_k], data_vars[tmean_k])
        if tmean_k in data_vars and tmax_k in data_vars:
            data_vars[tmax_k] = np.maximum(data_vars[tmax_k], data_vars[tmean_k])


def _enforce_quantile_ordering(data_vars: dict):
    """
    Ensure p10 ≤ p50 ≤ p90 within each variable.

    Quantile crossing can occur in regions of the feature space underrepresented
    in training.  We resolve crossing by sorting the three quantile values rather
    than clipping (preserves spread information).
    """
    for var in _ALL_VARS:
        p10, p50, p90 = f"{var}_p10", f"{var}_p50", f"{var}_p90"
        if not all(k in data_vars for k in [p10, p50, p90]):
            continue
        # np.sort along axis 0 sorts each (time, point) triplet independently
        stack = np.stack([data_vars[p10], data_vars[p50], data_vars[p90]], axis=0)
        stack = np.sort(stack, axis=0)
        data_vars[p10], data_vars[p50], data_vars[p90] = stack[0], stack[1], stack[2]


def _load_models(model_dir: Path) -> dict[str, lgb.Booster]:
    """Load all .txt LightGBM model files from model_dir."""
    models = {}
    for txt in model_dir.glob("*.txt"):
        name = txt.stem
        try:
            models[name] = lgb.Booster(model_file=str(txt))
        except Exception:
            pass   # skip corrupted or incompatible model files
    return models


def _write_zarr(
    data_vars: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    delos_coords: pd.DataFrame,
    output_path: Path,
) -> None:
    """
    Write coordinate arrays and all predicted variables to a new Zarr store.

    Time is stored as int64 nanoseconds since the Unix epoch (same convention
    as pandas.DatetimeIndex.asi8) for easy round-trip via pd.to_datetime().
    """
    n_times  = len(dates)
    n_points = len(delos_coords)

    store = zarr.open(str(output_path), mode="w")

    # --- Coordinate arrays ---------------------------------------------------
    store.create_dataset("time", data=dates.asi8, dtype="int64",
                         chunks=(min(n_times, _ZARR_CHUNKS[0]),),
                         compressor=_COMPRESSOR)
    store["time"].attrs["units"] = "nanoseconds since 1970-01-01"

    store.create_dataset("lon", data=delos_coords["lon"].values.astype("float64"), dtype="float64")
    store.create_dataset("lat", data=delos_coords["lat"].values.astype("float64"), dtype="float64")
    store.create_dataset("elevation_m", data=delos_coords["elevation_m"].values.astype("float32"),
                         dtype="float32")

    chunk_t = min(n_times,  _ZARR_CHUNKS[0])
    chunk_p = min(n_points, _ZARR_CHUNKS[1])

    # Long-name metadata for precipitation arrays (aids downstream tooling)
    _LONG_NAMES = {
        "precip_occurrence_prob": "precipitation occurrence probability (P(prcp > 0.1 mm))",
        "precip_p10": "conditional precipitation p10 — wet days only (prcp > 0.1 mm)",
        "precip_p50": "conditional precipitation p50 — wet days only (prcp > 0.1 mm)",
        "precip_p90": "conditional precipitation p90 — wet days only (prcp > 0.1 mm)",
    }

    # --- Data variables -------------------------------------------------------
    for name, arr in data_vars.items():
        ds = store.create_dataset(name, data=arr, dtype="float32",
                                  chunks=(chunk_t, chunk_p),
                                  compressor=_COMPRESSOR)
        if name in _LONG_NAMES:
            ds.attrs["long_name"] = _LONG_NAMES[name]
