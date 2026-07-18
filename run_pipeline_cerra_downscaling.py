"""
run_pipeline_cerra_downscaling.py
==================================
End-to-end CERRA statistical downscaling pipeline for the Delos grid.

OVERVIEW
--------
This script orchestrates the full nine-step pipeline that trains per-zone
LightGBM quantile-regression models on NOAA GSOD station observations +
CERRA reanalysis predictors, then applies those models to every point in
the Delos target grid.

  Step 1  Load NOAA GSOD station data (station_ingest.py)
  Step 2  Extract CERRA predictors at station locations (cerra_extract.py)
  Step 3  Extract CERRA predictors at Delos grid points (cerra_extract.py)
  Step 4  Extract static CERRA terrain features at both sets of locations
          (cerra_extract.extract_cerra_static) — orography, northness, eastness
  Step 5  Compute geographic features — coast distance, is_island, directional
          fetch, DEM-based northness/eastness/TPI/SVF (geo_features.py)
  Step 6  Compute soft zone memberships for Delos and stations (similarity.py)
  Step 7  Compute per-zone donor-similarity weights for training (similarity.py)
  Step 8  Build training DataFrame and train per-zone LightGBM models (train.py)
  Step 9  Apply zone-ensemble models to Delos grid, write output Zarr (predict.py)

NEW FEATURES ADDED (vs. original pipeline)
-------------------------------------------
  delta_elev_m      : target elevation − CERRA grid elevation (Prop. 1)
  lapse_correction_c: delta_elev_m × monthly ELR / 1000 (Prop. 2)
  fetch_N … fetch_NW: 8-direction ocean fetch (Prop. 3)
  cerra_u10/v10     : CERRA wind components derived from wdir10 + si10 (Prop. 4)
  dem_northness/east: aspect from high-res DEM (Prop. 5)
  tpi_500m/2000m    : topographic position index from DEM (Prop. 6)
  svf               : sky view factor from DEM (Prop. 7)
  Per-zone training : soft zone memberships replace single mean-Delos target (Prop. 8)

CACHING
-------
CERRA extraction (steps 2 and 3) is slow (~hours on first run).  Results are
cached as Parquet files under downscaling/cache/.  Delete these files to force
a full re-extraction (e.g. after adding new CERRA years or downloading wind direction).

USAGE
-----
  python run_pipeline_cerra_downscaling.py             # all variables
  python run_pipeline_cerra_downscaling.py --vars wind gust
  python run_pipeline_cerra_downscaling.py --vars precip

  --vars filters model training in step 8.  Prediction in step 9 always runs
  all variables using whatever model files exist on disk.
"""
from __future__ import annotations

import argparse
import time

import pandas as pd

from downscaling.cerra_extract import extract_all, extract_cerra_static
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
from downscaling.similarity import (
    N_ZONES,
    compute_zone_memberships,
    compute_zone_similarity,
)
from downscaling.station_ingest import load_stations
from downscaling.train import build_training_df, train_all

_CACHE = ROOT / "downscaling" / "cache"
_ALL_VARS = ["tmean", "tmin", "tmax", "rh", "precip", "wind", "gust"]


# ---------------------------------------------------------------------------
# Helpers (unchanged from original)
# ---------------------------------------------------------------------------

def _assert_full_month_coverage(cerra_df: pd.DataFrame, label: str) -> None:
    months  = pd.to_datetime(cerra_df["date"]).dt.month.unique()
    missing = sorted(set(range(1, 13)) - set(months))
    if missing:
        raise ValueError(
            f"{label}: CERRA extraction is missing months {missing}. "
            "Delete the parquet cache and verify all CERRA NetCDF files are present."
        )


def _monthly_clim(cerra_df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    df    = cerra_df.copy()
    df["month"] = pd.to_datetime(df["date"]).dt.month
    pivot = df.groupby([id_col, "month"])["cerra_tmean"].mean().unstack("month")
    pivot.columns = [f"cerra_tmean_m{int(m):02d}" for m in pivot.columns]
    return pivot.reset_index()


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Delos statistical downscaling pipeline")
    parser.add_argument(
        "--vars", nargs="+", choices=_ALL_VARS, default=None, metavar="VAR",
        help="Variables to (re)train LightGBM models for. Default: all.",
    )
    args = parser.parse_args()

    _CACHE.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ── 1. Station observations ──────────────────────────────────────────────
    print("━━ 1/9  Ingesting GSOD station data")
    t0 = time.time()
    stations = load_stations(STATION_DIR, YEAR_START, YEAR_END, **STATION_BBOX)
    print(f"   {len(stations)} station-days from {stations['station_id'].nunique()} stations "
          f"({time.time()-t0:.0f}s)")

    station_locs = (
        stations[["station_id", "lat", "lon", "elevation_m"]]
        .drop_duplicates("station_id")
        .rename(columns={"station_id": "id"})
    )

    # ── 2. CERRA at station locations ────────────────────────────────────────
    print("━━ 2/9  Extracting CERRA values at station locations")
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

    # ── 3. CERRA at Delos grid ───────────────────────────────────────────────
    print("━━ 3/9  Extracting CERRA values at Delos grid points")
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

    # ── 4. Static CERRA terrain features ────────────────────────────────────
    # Requires cerra_data/orography/cerra_orography.nc (download via cerra_data.py).
    # Returns cerra_elev_m, cerra_northness, cerra_eastness for each location.
    print("━━ 4/9  Extracting static CERRA terrain features (orography)")
    t0 = time.time()
    cerra_static_stations = extract_cerra_static(CERRA_DIR, station_locs)
    cerra_static_delos    = extract_cerra_static(CERRA_DIR, delos_locs)
    print(f"   done ({time.time()-t0:.0f}s)")

    # ── 5. Geographic features ───────────────────────────────────────────────
    # Computes: coast_dist_km, is_island, fetch_N…NW, dem_northness/eastness,
    # dem_slope_deg, tpi_500m, tpi_2000m, svf
    print("━━ 5/9  Computing geographic features")
    t0 = time.time()
    geo_stations_base = compute_geo_features(station_locs)
    geo_delos_base    = compute_geo_features(delos_locs)

    # Merge static CERRA terrain features into the geo DataFrames
    geo_stations = geo_stations_base.merge(
        cerra_static_stations, on="id", how="left"
    )
    geo_delos = geo_delos_base.merge(
        cerra_static_delos, on="id", how="left"
    )
    print(f"   done ({time.time()-t0:.0f}s)")

    # ── 6. Soft zone memberships ─────────────────────────────────────────────
    # Each Delos point gets a (N_ZONES,) vector of soft memberships based on
    # its coast distance, elevation, and dem_northness.  Station memberships
    # use cerra_northness as a fallback (coarser resolution but consistent).
    print("━━ 6/9  Computing soft zone memberships")
    t0 = time.time()

    # Rename id → station_id for the membership functions
    delos_feats_for_zones = geo_delos.rename(columns={"id": "station_id"}).merge(
        _monthly_clim(cerra_delos, "id").rename(columns={"id": "station_id"}),
        on="station_id", how="left",
    )
    station_feats_for_zones = geo_stations.rename(columns={"id": "station_id"}).merge(
        _monthly_clim(cerra_stations, "id").rename(columns={"id": "station_id"}),
        on="station_id", how="left",
    )

    # Delos: prefer fine-scale DEM northness for zone assignments
    delos_memberships   = compute_zone_memberships(
        delos_feats_for_zones, northness_col="dem_northness"
    )
    # Stations: use CERRA-scale northness (fine DEM not available everywhere)
    station_memberships = compute_zone_memberships(
        station_feats_for_zones, northness_col="cerra_northness"
    )

    for k in range(N_ZONES):
        col = f"zone_{k}"
        m_min = delos_memberships[col].min()
        m_max = delos_memberships[col].max()
        print(f"   zone {k} Delos membership: min={m_min:.3f}  max={m_max:.3f}")
    print(f"   done ({time.time()-t0:.0f}s)")

    # ── 7. Per-zone donor-similarity weights ─────────────────────────────────
    print("━━ 7/9  Computing per-zone donor-similarity weights")
    t0 = time.time()

    zone_weights = compute_zone_similarity(
        station_feats_for_zones, delos_feats_for_zones, delos_memberships
    )
    for k, w in zone_weights.items():
        print(f"   zone {k}: min={w.min():.2f}  mean={w.mean():.2f}  max={w.max():.2f}")
    print(f"   done ({time.time()-t0:.0f}s)")

    # ── 8. Train ─────────────────────────────────────────────────────────────
    print("━━ 8/9  Building training data and training per-zone LightGBM models")
    training_df = build_training_df(
        stations, cerra_stations, geo_stations,
        zone_weights, station_memberships,
    )
    print(f"   {len(training_df)} training rows, {N_ZONES} zones")
    t0 = time.time()
    models = train_all(training_df, MODEL_DIR, vars=args.vars)
    print(f"   models trained: {len(models)} total ({time.time()-t0:.0f}s)")

    # ── 9. Predict at Delos ──────────────────────────────────────────────────
    print("━━ 9/9  Predicting at Delos grid (zone-ensemble)")
    t0 = time.time()
    output_path = OUT_DIR / "delos_downscaled.zarr"
    # Donor pool for the precipitation EQM: paired (obs, CERRA) per station-day,
    # so predict.py can fit the QM(CERRA→obs) transfer and blend with the model.
    _donor = stations[["station_id", "date", "prcp_mm"]].merge(
        cerra_stations[["id", "date", "cerra_precip"]].rename(columns={"id": "station_id"}),
        on=["station_id", "date"], how="inner",
    )
    _donor_obs   = _donor["prcp_mm"].values
    _donor_cerra = _donor["cerra_precip"].values
    _donor_doy   = pd.to_datetime(_donor["date"]).dt.dayofyear.values
    predict_delos(
        cerra_delos, geo_delos, MODEL_DIR, DELOS_CSV, output_path,
        delos_memberships=delos_memberships,
        donor_obs=_donor_obs, donor_cerra=_donor_cerra, donor_doy=_donor_doy,
    )
    print(f"   output written to {output_path} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
