"""
run_pipeline_cerra_downscaling.py
==================================
End-to-end CERRA statistical downscaling pipeline for the Delos grid.

OVERVIEW
--------
This script orchestrates the full seven-step pipeline that trains LightGBM
quantile-regression models on NOAA GSOD station observations + CERRA reanalysis
predictors, then applies those models to every point in the Delos target grid.

  Step 1  Load NOAA GSOD station data (station_ingest.py)
  Step 2  Extract CERRA predictors at station locations (cerra_extract.py)
  Step 3  Extract CERRA predictors at Delos grid points (cerra_extract.py)
  Step 4  Compute geographic features — coast distance, is_island (geo_features.py)
  Step 5  Compute donor-similarity weights for training (similarity.py)
  Step 6  Build training DataFrame and train LightGBM models (train.py)
  Step 7  Apply models to Delos grid, write output Zarr (predict.py)

CACHING
-------
CERRA extraction (steps 2 and 3) is slow (~hours on first run).  The results
are cached as Parquet files under downscaling/cache/.  Delete those files to
force a full re-extraction (e.g. after adding new CERRA years or changing PATCH_SIZE).

USAGE
-----
  python run_pipeline_cerra_downscaling.py             # all variables
  python run_pipeline_cerra_downscaling.py --vars wind gust   # retrain wind + gust only
  python run_pipeline_cerra_downscaling.py --vars precip      # retrain precip only

  --vars filters model training in step 6.  Prediction in step 7 always runs
  all variables using whatever model files exist on disk, so the output Zarr
  is always complete even when only some models were retrained.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from downscaling.cerra_extract import extract_all
from downscaling.config import (
    CERRA_DIR,
    DELOS_CSV,
    MODEL_DIR,
    OUT_DIR,
    ROOT,
    STATION_BBOX,
    STATION_DIR,
    YEAR_END,
    YEAR_START,
)
from downscaling.geo_features import compute_geo_features
from downscaling.predict import predict_delos
from downscaling.similarity import compute_similarity
from downscaling.station_ingest import load_stations
from downscaling.train import build_training_df, train_all

# ---------------------------------------------------------------------------
# Cache directory for intermediate Parquet files
# ---------------------------------------------------------------------------
# Derived from ROOT (config.py) so no machine-specific path is hardcoded here.
_CACHE = ROOT / "downscaling" / "cache"

_ALL_VARS = ["tmean", "tmin", "tmax", "rh", "precip", "wind", "gust"]


def _assert_full_month_coverage(cerra_df: pd.DataFrame, label: str) -> None:
    """
    Raise ValueError if any of the 12 calendar months is missing from cerra_df.

    A missing month typically means a CERRA NetCDF file for that month/year was
    not found.  The check runs before training to prevent silently fitting models
    on an incomplete seasonal cycle.
    """
    months = pd.to_datetime(cerra_df["date"]).dt.month.unique()
    missing = sorted(set(range(1, 13)) - set(months))
    if missing:
        raise ValueError(
            f"{label}: CERRA extraction is missing months {missing}. "
            "Delete the parquet cache and verify all CERRA NetCDF files are present."
        )


def _monthly_clim(cerra_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """
    Compute monthly CERRA tmean climatology per location.

    Returns a DataFrame with columns id_col, cerra_tmean_m01 … cerra_tmean_m12.
    These 12 columns are used by compute_similarity() to assess climatic
    similarity between each donor station and the Delos target grid.
    """
    df = cerra_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    pivot = df.groupby([id_col, "month"])["cerra_tmean"].mean().unstack("month")
    pivot.columns = [f"cerra_tmean_m{int(m):02d}" for m in pivot.columns]
    return pivot.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Delos statistical downscaling pipeline")
    parser.add_argument(
        "--vars",
        nargs="+",
        choices=_ALL_VARS,
        default=None,
        metavar="VAR",
        help="Variables to (re)train LightGBM models for. Default: all. "
             "Prediction step 7 always runs all vars regardless of this flag.",
    )
    args = parser.parse_args()

    # Ensure output directories exist before anything is written
    _CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Station observations ───────────────────────────────────────────────
    # Load daily GSOD CSV files for stations within STATION_BBOX.
    # Stations with no measurable precipitation are filtered out (see station_ingest).
    print("━━ 1/7  Ingesting GSOD station data")
    t0 = time.time()
    stations = load_stations(STATION_DIR, YEAR_START, YEAR_END, **STATION_BBOX)
    print(f"   {len(stations)} station-days from {stations['station_id'].nunique()} stations "
          f"({time.time()-t0:.0f}s)")

    # Unique station locations for CERRA extraction (one row per station)
    station_locs = (
        stations[["station_id", "lat", "lon", "elevation_m"]]
        .drop_duplicates("station_id")
        .rename(columns={"station_id": "id"})
    )

    # ── 2. CERRA at station locations ─────────────────────────────────────────
    # Extract daily CERRA values at each station using cubic interpolation over a
    # 4×4 patch of CERRA cells (with LSM-aware fallback for island stations).
    print("━━ 2/7  Extracting CERRA values at station locations")
    t0 = time.time()
    cache_path = _CACHE / "cerra_stations.parquet"
    if cache_path.exists():
        cerra_stations = pd.read_parquet(cache_path)
        print(f"   loaded from cache ({time.time()-t0:.0f}s)")
    else:
        cerra_stations = extract_all(CERRA_DIR, station_locs, YEAR_START, YEAR_END)
        cerra_stations.to_parquet(cache_path)
        print(f"   extracted and cached ({time.time()-t0:.0f}s)")
    _assert_full_month_coverage(cerra_stations, "cerra_stations")
    print(f"   {len(cerra_stations)} rows")

    # ── 3. CERRA at Delos grid ────────────────────────────────────────────────
    # Same extraction as step 2 but for the target Delos points.
    # These become the predictor features for inference in step 7.
    print("━━ 3/7  Extracting CERRA values at Delos grid points")
    delos_locs = (
        pd.read_csv(DELOS_CSV)[["lon", "lat", "VALUE"]]
        .rename(columns={"VALUE": "elevation_m"})
        .assign(id=lambda df: range(len(df)))[["id", "lat", "lon", "elevation_m"]]
    )
    t0 = time.time()
    cache_path = _CACHE / "cerra_delos.parquet"
    if cache_path.exists():
        cerra_delos = pd.read_parquet(cache_path)
        print(f"   loaded from cache ({time.time()-t0:.0f}s)")
    else:
        cerra_delos = extract_all(CERRA_DIR, delos_locs, YEAR_START, YEAR_END)
        cerra_delos.to_parquet(cache_path)
        print(f"   extracted and cached ({time.time()-t0:.0f}s)")
    _assert_full_month_coverage(cerra_delos, "cerra_delos")
    print(f"   {len(cerra_delos)} rows")

    # ── 4. Geographic features ────────────────────────────────────────────────
    # Compute coast_dist_km and is_island for both station and Delos locations.
    # These require the S2Coast-2023 shapefile (see geo_features.py).
    print("━━ 4/7  Computing geographic features")
    t0 = time.time()
    geo_stations = compute_geo_features(station_locs)
    geo_delos    = compute_geo_features(delos_locs)
    print(f"   done ({time.time()-t0:.0f}s)")

    # ── 5. Donor-similarity weights ───────────────────────────────────────────
    # Weight each donor station by its climatic and physiographic similarity to
    # the Delos grid.  Islands get an extra bonus (see similarity.py).
    print("━━ 5/7  Computing donor-similarity weights")
    clim_stations = _monthly_clim(cerra_stations, "id").rename(columns={"id": "station_id"})
    clim_delos    = _monthly_clim(cerra_delos,    "id").rename(columns={"id": "station_id"})

    station_sim = geo_stations.rename(columns={"id": "station_id"}).merge(clim_stations, on="station_id")
    delos_sim   = geo_delos.rename(  columns={"id": "station_id"}).merge(clim_delos,    on="station_id")

    weights = compute_similarity(station_sim, delos_sim)
    print(f"   weights: min={weights.min():.2f}  mean={weights.mean():.2f}  max={weights.max():.2f}")

    # ── 6. Train ──────────────────────────────────────────────────────────────
    # Merge stations + CERRA predictors + geo features into one training table,
    # then train separate LightGBM models for each variable (see train.py).
    print("━━ 6/7  Building training data and training LightGBM models")
    training_df = build_training_df(stations, cerra_stations, geo_stations, weights)
    print(f"   {len(training_df)} training rows")
    t0 = time.time()
    models = train_all(training_df, MODEL_DIR, vars=args.vars)
    print(f"   models trained: {sorted(models.keys())} ({time.time()-t0:.0f}s)")

    # ── 7. Predict at Delos ───────────────────────────────────────────────────
    # Apply all trained models to the Delos CERRA + geo feature matrix and
    # write a compressed Zarr store with p10/p50/p90 quantiles per variable.
    print("━━ 7/7  Predicting at Delos grid")
    t0 = time.time()
    output_path = OUT_DIR / "delos_downscaled.zarr"
    predict_delos(cerra_delos, geo_delos, MODEL_DIR, DELOS_CSV, output_path)
    print(f"   output written to {output_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
