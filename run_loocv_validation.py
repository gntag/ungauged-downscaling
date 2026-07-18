"""
run_loocv_validation.py
=======================
Leave-one-out cross-validation (LOOCV) for the Delos CERRA downscaling pipeline.

Three sequential phases (each is resumable — already-computed outputs are skipped):

  Phase 1  Station dataset preparation
           • Load all station observations for VAL_YEAR_START..VAL_YEAR_END
           • Extract CERRA values at station locations via nearest-grid-cell lookup
           • Compute geographic + CERRA-static features at station locations
           • Cache everything in validation/cache/  (skipped if files already exist)

  Phase 2  LOOCV training and prediction
           • Reuse the cubic-interpolated CERRA station cache from
             downscaling/cache/cerra_stations.parquet (same as the main pipeline)
           • For each station s: retrain the full pipeline on all other stations,
             then predict at s for the training period.
           • Save per-station daily predictions + observations to validation/loocv/
             (individual files are skipped unless LOOCV_OVERWRITE=True)

  Phase 3  Skill metric computation
           • Annual and monthly R², RMSE, Bias for 13 climate indices
           • Daily POD, FAR, CSI for 7 threshold events
           • Per-station and pooled, for both LOOCV predictions and raw CERRA baseline
           • Exports 4 CSVs to validation/metrics/

Edit downscaling/validation_config.py to change paths, thresholds, and flags.
"""
from __future__ import annotations

import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from downscaling.cerra_extract import extract_all, extract_cerra_static
from downscaling.config import (
    CERRA_DIR,
    MONTHLY_LAPSE_RATES,
    GEO_FEATURES,
    STATION_BBOX,
    STATION_DIR,
    VARIABLE_CERRA_FEATURES,
    VARIABLE_GEO_FEATURES,
    DELOS_CSV,
    YEAR_START,
    YEAR_END,
)
from downscaling.geo_features import compute_geo_features
from downscaling.similarity import (
    N_ZONES,
    compute_zone_memberships,
    compute_zone_similarity,
)
from downscaling.station_ingest import load_stations
from downscaling.bias_correct import blended_precip_eqm
from downscaling.config import PRECIP_EQM_BLEND
from downscaling.train import build_training_df, train_all
from downscaling.validation_config import (
    CACHE_OVERWRITE,
    CERRA_PRECIP_MM_FACTOR,
    ENV_LOOCV_OVERWRITE,
    ENVIRONMENT_GROUPS,
    LOOCV_FAST_MODE,
    LOOCV_OVERWRITE,
    OCC_PROB_THRESHOLD,
    PR20_MM,
    PR50_MM,
    PRECIP_EQM_ENABLE,
    PRECIP_EQM_MIN_WET,
    PRECIP_EQM_STEP_DAYS,
    PRECIP_EQM_WINDOW_DAYS,
    R99P_PERCENTILE,
    TMAX35_C,
    TMAX37_C,
    TMIN0_C,
    VALIDATION_DIR,
    VAL_YEAR_END,
    VAL_YEAR_START,
    WET_DAY_MM,
)

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------
_CACHE_DIR   = VALIDATION_DIR / "cache"
_LOOCV_DIR   = VALIDATION_DIR / "loocv"
_METRICS_DIR = VALIDATION_DIR / "metrics"

# Cache file paths (Phase 1 outputs)
_F_STATION_OBS     = _CACHE_DIR / "station_obs.parquet"
_F_CERRA_NEAREST   = _CACHE_DIR / "cerra_stations_nearest.parquet"
_F_STATION_GEO     = _CACHE_DIR / "station_geo.parquet"
_F_CERRA_STATIC    = _CACHE_DIR / "station_cerra_static.parquet"

# Environment-specific LOOCV directories (Phase 2/3 env outputs)
_ENV_LOOCV_BASE  = VALIDATION_DIR / "env_loocv"
_ENV_METRICS_DIR = VALIDATION_DIR / "env_metrics"

# Production interpolated CERRA cache (built by run_pipeline_cerra_downscaling.py)
_CERRA_INTERP_CACHE = _HERE / "downscaling" / "cache" / "cerra_stations.parquet"


# ===========================================================================
# Shared helpers
# ===========================================================================

def _monthly_clim(df: pd.DataFrame, id_col: str) -> pd.DataFrame:
    """Compute per-location monthly mean CERRA temperature."""
    tmp = df.copy()
    tmp["month"] = pd.to_datetime(tmp["date"]).dt.month
    pivot = tmp.groupby([id_col, "month"])["cerra_tmean"].mean().unstack("month")
    pivot.columns = [f"cerra_tmean_m{int(m):02d}" for m in pivot.columns]
    return pivot.reset_index()


def _precip_point_estimate(df: pd.DataFrame) -> np.ndarray:
    """
    Best daily precipitation point estimate used by all metric computations.

    Preference order (best available first):
      1. precip_best_eqm  — EQM-corrected best estimate (PR-next-EQM)
      2. precip_best      — Hurdle model occ_prob × precip_mean
      3. occ_prob × precip_p50 gated at OCC_PROB_THRESHOLD  (legacy fallback)
    """
    if "precip_best_eqm" in df.columns:
        return df["precip_best_eqm"].values
    if "precip_best" in df.columns:
        return df["precip_best"].values
    occ = df["precip_occ_prob"].values if "precip_occ_prob" in df.columns else np.zeros(len(df))
    p50 = df["precip_p50"].values      if "precip_p50"       in df.columns else np.zeros(len(df))
    return np.where(occ > OCC_PROB_THRESHOLD, p50, 0.0)


# ===========================================================================
# Phase 1 — Station dataset preparation
# ===========================================================================

def _extract_cerra_nearest(
    cerra_dir: Path,
    locations: pd.DataFrame,
    year_start: int,
    year_end: int,
) -> pd.DataFrame:
    """
    Extract CERRA values at station locations using nearest grid cell (no interpolation).

    Delegates to extract_all(nearest_cell=True) to share the LSM cache,
    circular wind-direction mean, and precipitation date-shift logic with the
    cubic-interpolation path — eliminating silent divergence risk.
    Precipitation is returned in the native CERRA unit (metres); convert to mm
    by multiplying by CERRA_PRECIP_MM_FACTOR in the calling code.
    """
    n_years = year_end - year_start + 1
    print(f"   CERRA nearest-index: {len(locations)} station locations, "
          f"{n_years} years ({year_start}–{year_end})")
    result = extract_all(cerra_dir, locations, year_start, year_end, nearest_cell=True)
    print(f"   CERRA nearest-index done: {len(result)} station-days extracted")
    return result


def phase1_build_dataset() -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
    """
    Load/compute and cache all station-level datasets.

    Returns
    -------
    stations         : GSOD observations (station_id, date, tmean_c, …)
    station_locs     : unique station locations  (id=station_id, lat, lon, elevation_m)
    cerra_nearest    : nearest-index CERRA values (id, date, cerra_tmean, …)
    geo_stations     : geographic + CERRA-static features (id, lat, lon, …)
    cerra_interp     : cubic-interpolated CERRA features for LOOCV training
    """
    for d in (_CACHE_DIR, _LOOCV_DIR, _METRICS_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # ── Observations ─────────────────────────────────────────────────────────
    if _F_STATION_OBS.exists() and not CACHE_OVERWRITE:
        print(f"   Phase 1 [1/4]: loading station observations from cache  ({_F_STATION_OBS.name})")
        stations = pd.read_parquet(_F_STATION_OBS)
        print(f"   -> {len(stations):,} station-days, {stations['station_id'].nunique()} stations, "
              f"{pd.to_datetime(stations['date']).dt.year.min()}–"
              f"{pd.to_datetime(stations['date']).dt.year.max()}")
    else:
        print(f"   Phase 1 [1/4]: ingesting station observations from {STATION_DIR}")
        stations = load_stations(STATION_DIR, VAL_YEAR_START, VAL_YEAR_END, **STATION_BBOX)
        stations.to_parquet(_F_STATION_OBS)
        print(f"   -> {len(stations):,} station-days from {stations['station_id'].nunique()} stations  "
              f"(saved to {_F_STATION_OBS.name})")

    station_locs = (
        stations[["station_id", "lat", "lon", "elevation_m"]]
        .drop_duplicates("station_id")
        .rename(columns={"station_id": "id"})
        .reset_index(drop=True)
    )

    # ── CERRA nearest-index (baseline) ───────────────────────────────────────
    if _F_CERRA_NEAREST.exists() and not CACHE_OVERWRITE:
        print(f"   Phase 1 [2/4]: loading CERRA nearest-index from cache  ({_F_CERRA_NEAREST.name})")
        cerra_nearest = pd.read_parquet(_F_CERRA_NEAREST)
        print(f"   -> {len(cerra_nearest):,} rows, {cerra_nearest['id'].nunique()} stations")
    else:
        print(f"   Phase 1 [2/4]: extracting CERRA nearest-index at {len(station_locs)} station locations")
        t0 = time.time()
        cerra_nearest = _extract_cerra_nearest(
            CERRA_DIR, station_locs, VAL_YEAR_START, VAL_YEAR_END
        )
        cerra_nearest.to_parquet(_F_CERRA_NEAREST)
        print(f"   -> {len(cerra_nearest):,} rows saved to {_F_CERRA_NEAREST.name}  "
              f"({time.time()-t0:.0f}s)")

    # ── Geographic features ──────────────────────────────────────────────────
    if _F_STATION_GEO.exists() and _F_CERRA_STATIC.exists() and not CACHE_OVERWRITE:
        print(f"   Phase 1 [3/4]: loading geographic + CERRA-static features from cache")
        geo_base    = pd.read_parquet(_F_STATION_GEO)
        cerra_static = pd.read_parquet(_F_CERRA_STATIC)
        print(f"   -> {len(geo_base)} stations ({geo_base.shape[1]-1} geo cols, "
              f"{cerra_static.shape[1]-1} CERRA-static cols)")
    else:
        print(f"   Phase 1 [3/4]: computing geographic and CERRA-static features "
              f"at {len(station_locs)} station locations")
        t0 = time.time()
        print("   ... running compute_geo_features (DEM, coast distance, northness, …)")
        geo_base    = compute_geo_features(station_locs)
        print("   ... running extract_cerra_static (orography, land-sea mask, …)")
        cerra_static = extract_cerra_static(CERRA_DIR, station_locs)
        geo_base.to_parquet(_F_STATION_GEO)
        cerra_static.to_parquet(_F_CERRA_STATIC)
        print(f"   -> {geo_base.shape[1]-1} geo cols + {cerra_static.shape[1]-1} CERRA-static cols  "
              f"({time.time()-t0:.0f}s)")

    geo_stations = geo_base.merge(cerra_static, on="id", how="left")

    # ── Cubic-interpolated CERRA (training features) ─────────────────────────
    # Reuse the main pipeline cache if it exists; otherwise create it here
    # using the same YEAR_START/YEAR_END and station locations so LOOCV models
    # see exactly the same feature values as the production models.
    _MAIN_CACHE = _HERE / "downscaling" / "cache"
    _MAIN_CACHE.mkdir(parents=True, exist_ok=True)
    if _CERRA_INTERP_CACHE.exists():
        print(f"   Phase 1 [4/4]: loading cubic-interpolated CERRA from main pipeline cache  "
              f"({_CERRA_INTERP_CACHE.name})")
        cerra_interp = pd.read_parquet(_CERRA_INTERP_CACHE)
        print(f"   -> {len(cerra_interp):,} rows, {cerra_interp['id'].nunique()} stations")
    else:
        print(f"   Phase 1 [4/4]: extracting cubic-interpolated CERRA at station locations "
              f"({YEAR_START}–{YEAR_END})  — this may take several minutes")
        t0 = time.time()
        cerra_interp = extract_all(CERRA_DIR, station_locs, YEAR_START, YEAR_END)
        cerra_interp.to_parquet(_CERRA_INTERP_CACHE)
        print(f"   -> {len(cerra_interp):,} rows saved to {_CERRA_INTERP_CACHE.name}  "
              f"({time.time()-t0:.0f}s)")

    print(f"\n   Phase 1 complete — datasets ready for {len(station_locs)} stations")
    return stations, station_locs, cerra_nearest, geo_stations, cerra_interp


# ===========================================================================
# Phase 2 — LOOCV
# ===========================================================================

def _build_feat_matrix(
    cerra_df: pd.DataFrame,
    geo_df: pd.DataFrame,
) -> tuple[pd.DataFrame, list, list]:
    """
    Merge CERRA and geo features, add harmonic month encoding and lapse features.
    Returns (feat_df, point_ids, dates).
    """
    geo_cols = ["id"] + [c for c in geo_df.columns if c != "id"]
    feat_df = (
        cerra_df.drop_duplicates(subset=["date", "id"])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
        .merge(geo_df[geo_cols].drop_duplicates("id"), on="id", how="left")
    )
    feat_df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)
    feat_df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)
    if "cerra_elev_m" in feat_df.columns:
        feat_df["delta_elev_m"] = feat_df["elevation_m"] - feat_df["cerra_elev_m"]
    else:
        feat_df["delta_elev_m"] = 0.0
    month_lapse = pd.to_datetime(feat_df["date"]).dt.month.map(MONTHLY_LAPSE_RATES)
    feat_df["lapse_correction_c"] = feat_df["delta_elev_m"] * month_lapse / 1000.0
    # Gust factor and wind-terrain alignment (must match train.build_training_df)
    if "cerra_gust" in feat_df.columns and "cerra_wind" in feat_df.columns:
        feat_df["gust_factor_cerra"] = (
            feat_df["cerra_gust"] / feat_df["cerra_wind"].replace(0.0, np.nan)
        ).clip(1.0, 10.0).fillna(1.0)
    else:
        feat_df["gust_factor_cerra"] = 1.0
    if "cerra_u10" in feat_df.columns and "dem_eastness" in feat_df.columns:
        feat_df["wind_terrain_align"] = (
            feat_df["cerra_u10"] * feat_df["dem_eastness"]
            + feat_df["cerra_v10"] * feat_df["dem_northness"]
        )
    else:
        feat_df["wind_terrain_align"] = 0.0
    point_ids = sorted(feat_df["id"].unique())
    dates     = sorted(feat_df["date"].unique())
    return feat_df, point_ids, dates


def _zone_ensemble_predict(
    X: np.ndarray,
    models: dict,
    n_times: int,
    n_points: int,
    base_name: str,
    zone_weights_arr: np.ndarray,
) -> np.ndarray | None:
    accumulated = None
    weight_sum  = np.zeros(n_points, dtype=np.float32)
    for k in range(N_ZONES):
        name = f"{base_name}_zone{k}"
        if name not in models:
            continue
        raw = models[name].predict(X).reshape(n_times, n_points).astype("float32")
        w   = zone_weights_arr[k].astype("float32")
        accumulated = raw * w[np.newaxis, :] if accumulated is None else accumulated + raw * w[np.newaxis, :]
        weight_sum += w
    if accumulated is None:
        return None
    safe = np.where(weight_sum == 0, 1.0, weight_sum).astype("float32")
    return accumulated / safe[np.newaxis, :]


def _single_predict(
    X: np.ndarray,
    models: dict,
    n_times: int,
    n_points: int,
    name: str,
) -> np.ndarray | None:
    if name not in models:
        return None
    return models[name].predict(X).reshape(n_times, n_points).astype("float32")


def predict_at_stations(
    cerra_df: pd.DataFrame,
    geo_df: pd.DataFrame,
    memberships: pd.DataFrame,
    models: dict,
) -> pd.DataFrame:
    """
    Apply trained zone-ensemble models to one or more station locations.
    Returns a DataFrame with columns: id, date, plus one column per
    predicted quantity (e.g. tmean_p50, precip_p50, precip_occ_prob, …).
    """
    feat_df, point_ids, dates = _build_feat_matrix(cerra_df, geo_df)
    n_times, n_points = len(dates), len(point_ids)

    # Zone membership weights — shape (N_ZONES, n_points)
    mem_indexed = memberships.reindex(point_ids)
    zone_w = np.array(
        [mem_indexed[f"zone_{k}"].fillna(1.0 / N_ZONES).values for k in range(N_ZONES)],
        dtype=np.float32,
    )
    row_sums = zone_w.sum(axis=0, keepdims=True)
    zone_w  /= np.where(row_sums == 0, 1.0, row_sums)

    _ZONE_VARS  = {"tmean", "tmin", "tmax", "rh"}
    _MEAN_VARS  = {"wind", "gust"}
    _ALL_VARS   = ["tmean", "tmin", "tmax", "rh", "precip", "wind", "gust"]
    _Q_TAGS     = ["p10", "p50", "p90"]

    pred_cols: dict[str, np.ndarray] = {}

    for var in _ALL_VARS:
        fcols     = VARIABLE_CERRA_FEATURES[var] + VARIABLE_GEO_FEATURES.get(var, GEO_FEATURES)
        available = [c for c in fcols if c in feat_df.columns]
        if not available:
            continue
        X = feat_df[available].values.astype(np.float32)

        if var in _ZONE_VARS:
            for qtag in _Q_TAGS:
                arr = _zone_ensemble_predict(X, models, n_times, n_points,
                                             f"{var}_{qtag}", zone_w)
                if arr is not None:
                    pred_cols[f"{var}_{qtag}"] = arr.ravel()
        elif var == "precip":
            occ = _single_predict(X, models, n_times, n_points, "precip_occ")
            if occ is not None:
                pred_cols["precip_occ_prob"] = np.clip(occ, 0, 1).ravel()
            # Quantiles p10/p50/p90 plus p95/p99 upper-tail bands
            for qtag in _Q_TAGS + ["p95", "p99"]:
                arr = _single_predict(X, models, n_times, n_points, f"precip_{qtag}")
                if arr is not None:
                    pred_cols[f"precip_{qtag}"] = np.clip(arr, 0, None).ravel()
            arr = _single_predict(X, models, n_times, n_points, "precip_mean")
            if arr is not None:
                pred_cols["precip_mean"] = np.clip(arr, 0, None).ravel()
            # Amount source occ_prob × precip_mean (blended with QM(CERRA) in phase 3)
            if "precip_occ_prob" in pred_cols and "precip_mean" in pred_cols:
                pred_cols["precip_best"] = np.clip(
                    pred_cols["precip_occ_prob"] * pred_cols["precip_mean"], 0, None
                )
        else:
            for qtag in _Q_TAGS:
                arr = _single_predict(X, models, n_times, n_points, f"{var}_{qtag}")
                if arr is not None:
                    pred_cols[f"{var}_{qtag}"] = (
                        np.clip(arr, 0, None) if var in _MEAN_VARS else arr
                    ).ravel()
            if var in _MEAN_VARS:
                arr = _single_predict(X, models, n_times, n_points, f"{var}_mean")
                if arr is not None:
                    pred_cols[f"{var}_mean"] = np.clip(arr, 0, None).ravel()

    result = feat_df[["id", "date"]].copy()
    for col, vals in pred_cols.items():
        result[col] = vals
    return result


def phase2_loocv(
    stations: pd.DataFrame,
    station_locs: pd.DataFrame,
    cerra_interp: pd.DataFrame,
    geo_stations: pd.DataFrame,
    cerra_nearest: pd.DataFrame,
) -> None:
    """
    Run the full LOOCV loop, writing one parquet per station to validation/loocv/.
    Each parquet contains: date, station_id, obs_*, pred_*, cerra_* columns.
    """
    # Precompute Delos memberships (fixed, independent of which station is left out)
    print("   Phase 2: loading Delos grid coordinates …")
    delos_locs = (
        pd.read_csv(DELOS_CSV)[["lon", "lat", "VALUE"]]
        .rename(columns={"VALUE": "elevation_m"})
        .assign(id=lambda df: range(len(df)))[["id", "lat", "lon", "elevation_m"]]
    )
    print(f"   -> {len(delos_locs)} Delos grid points")

    # Reuse cached Delos CERRA if available; create it if not
    _delos_cache = _HERE / "downscaling" / "cache" / "cerra_delos.parquet"
    (_HERE / "downscaling" / "cache").mkdir(parents=True, exist_ok=True)
    if _delos_cache.exists():
        print(f"   Phase 2: loading Delos CERRA from cache  ({_delos_cache.name})")
        cerra_delos = pd.read_parquet(_delos_cache)
        print(f"   -> {len(cerra_delos):,} rows")
    else:
        print(f"   Phase 2: extracting cubic-interpolated CERRA at Delos grid "
              f"({YEAR_START}–{YEAR_END})  — this may take several minutes")
        t0 = time.time()
        cerra_delos = extract_all(CERRA_DIR, delos_locs, YEAR_START, YEAR_END)
        cerra_delos.to_parquet(_delos_cache)
        print(f"   -> {len(cerra_delos):,} rows saved  ({time.time()-t0:.0f}s)")

    print("   Phase 2: computing Delos geo features and zone memberships …")
    cerra_static_delos = extract_cerra_static(CERRA_DIR, delos_locs)
    geo_delos_base     = compute_geo_features(delos_locs)
    geo_delos          = geo_delos_base.merge(cerra_static_delos, on="id", how="left")

    delos_feats_for_zones = geo_delos.rename(columns={"id": "station_id"}).merge(
        _monthly_clim(cerra_delos, "id").rename(columns={"id": "station_id"}),
        on="station_id", how="left",
    )
    delos_memberships = compute_zone_memberships(
        delos_feats_for_zones, northness_col="dem_northness"
    )
    print(f"   -> Delos zone memberships computed for {len(delos_memberships)} grid points")

    station_ids = station_locs["id"].tolist()
    print(f"\n   Phase 2: starting LOOCV over {len(station_ids)} stations")

    for i, s in enumerate(station_ids):
        out_path = _LOOCV_DIR / f"loocv_{s}.parquet"
        if out_path.exists() and not LOOCV_OVERWRITE:
            print(f"   [{i+1}/{len(station_ids)}] {s}: skipping (file exists)")
            continue

        t0 = time.time()
        print(f"\n   ── Fold [{i+1}/{len(station_ids)}]  held-out: {s}")

        # ── Training data (all stations except s) ───────────────────────────
        train_stations = stations[stations["station_id"] != s].copy()
        train_cerra    = cerra_interp[cerra_interp["id"] != s].copy()
        train_geo      = geo_stations[geo_stations["id"] != s].copy()

        n_train = train_stations["station_id"].nunique()
        if n_train < 5:
            print(f"   [{i+1}/{len(station_ids)}] {s}: SKIP — only {n_train} training stations (min 5)")
            continue
        print(f"   Training set: {n_train} stations, "
              f"{len(train_stations):,} station-days")

        # ── Zone features for training stations ─────────────────────────────
        print("   Computing zone memberships and similarity weights for training set …")
        train_feats = train_geo.rename(columns={"id": "station_id"}).merge(
            _monthly_clim(train_cerra, "id").rename(columns={"id": "station_id"}),
            on="station_id", how="left",
        )
        train_memberships = compute_zone_memberships(
            train_feats, northness_col="cerra_northness"
        )
        zone_weights = compute_zone_similarity(
            train_feats, delos_feats_for_zones, delos_memberships
        )

        # ── Train ────────────────────────────────────────────────────────────
        print("   Building training DataFrame …")
        training_df = build_training_df(
            train_stations, train_cerra, train_geo,
            zone_weights, train_memberships,
        )
        print(f"   Training DataFrame: {len(training_df):,} rows × {training_df.shape[1]} cols  "
              f"— fitting LightGBM models (fast={LOOCV_FAST_MODE}) …")
        with tempfile.TemporaryDirectory() as tmp_dir:
            models = train_all(training_df, Path(tmp_dir), fast=LOOCV_FAST_MODE)
        print(f"   Models trained ({len(models)} model objects)")

        # ── Zone membership for held-out station (for prediction) ────────────
        s_geo  = geo_stations[geo_stations["id"] == s]
        s_cerra_interp = cerra_interp[cerra_interp["id"] == s]
        if s_cerra_interp.empty or s_geo.empty:
            print(f"   [{i+1}/{len(station_ids)}] {s}: SKIP — no CERRA/geo data for held-out station")
            continue

        print("   Computing zone membership for held-out station …")
        s_feats = s_geo.rename(columns={"id": "station_id"}).merge(
            _monthly_clim(s_cerra_interp, "id").rename(columns={"id": "station_id"}),
            on="station_id", how="left",
        )
        s_membership = compute_zone_memberships(s_feats, northness_col="cerra_northness")

        # ── Predict at held-out station ──────────────────────────────────────
        print("   Predicting at held-out station …")
        preds = predict_at_stations(s_cerra_interp, s_geo, s_membership, models)

        # ── Merge: obs + predictions + CERRA nearest ─────────────────────────
        s_obs = stations[stations["station_id"] == s][
            ["station_id", "date", "tmean_c", "tmin_c", "tmax_c",
             "rh_pct", "prcp_mm", "wdsp_ms", "gust_ms"]
        ].rename(columns={
            "tmean_c": "obs_tmean_c", "tmin_c": "obs_tmin_c", "tmax_c": "obs_tmax_c",
            "rh_pct": "obs_rh_pct", "prcp_mm": "obs_prcp_mm",
            "wdsp_ms": "obs_wdsp_ms", "gust_ms": "obs_gust_ms",
        })

        preds_renamed = preds.rename(columns={"id": "station_id"})
        s_nearest = cerra_nearest[cerra_nearest["id"] == s].rename(
            columns={"id": "station_id"}
        )

        merged = (
            s_obs
            .merge(preds_renamed, on=["station_id", "date"], how="inner")
            .merge(s_nearest, on=["station_id", "date"], how="left")
        )

        # Convert CERRA precipitation to mm for direct comparison
        if "cerra_precip" in merged.columns:
            merged["cerra_precip_mm"] = merged["cerra_precip"] * CERRA_PRECIP_MM_FACTOR

        merged.to_parquet(out_path, index=False)
        elapsed = time.time() - t0
        date_range = (f"{pd.to_datetime(merged['date']).dt.year.min()}–"
                      f"{pd.to_datetime(merged['date']).dt.year.max()}")
        print(f"   Fold [{i+1}/{len(station_ids)}] DONE  {s}: "
              f"{len(merged):,} rows ({date_range}), "
              f"{elapsed:.0f}s  -> {out_path.name}")

    completed = sorted(_LOOCV_DIR.glob("loocv_*.parquet"))
    print(f"\n   Phase 2 complete — {len(completed)} station files in {_LOOCV_DIR}")


# ===========================================================================
# Phase 2 (env) — Environment-specific LOOCV
# ===========================================================================

def phase2_env_loocv(
    stations: pd.DataFrame,
    station_locs: pd.DataFrame,
    cerra_interp: pd.DataFrame,
    geo_stations: pd.DataFrame,
    cerra_nearest: pd.DataFrame,
) -> None:
    """
    For each group in ENVIRONMENT_GROUPS, run LOOCV over that group's stations.

    For each held-out station s:
      - Training uses all other stations in the full dataset (n-1 of all stations).
      - Zone similarity weights are computed relative to s itself, so donors most
        similar to s are upweighted.
      - Prediction is made at s only.

    The groups define which stations are evaluated; they do not restrict training.

    Outputs: env_loocv/{group_name}/loocv_{station_id}.parquet
    """
    _ENV_LOOCV_BASE.mkdir(parents=True, exist_ok=True)
    _ENV_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    station_ids_all = set(station_locs["id"].tolist())

    for env_name, env_cfg in ENVIRONMENT_GROUPS.items():
        env_label = env_cfg["label"]
        env_station_ids = [s for s in env_cfg["stations"] if s in station_ids_all]
        missing = [s for s in env_cfg["stations"] if s not in station_ids_all]
        if missing:
            print(f"   Warning [{env_name}]: {len(missing)} station(s) not in dataset: {missing}")
        if not env_station_ids:
            print(f"   Group '{env_name}': no stations found in dataset, skipping")
            continue

        env_dir = _ENV_LOOCV_BASE / env_name
        env_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n   ━━ Environment: {env_label}  ({len(env_station_ids)} stations)")

        # --- LOOCV folds for this environment ---
        for i, s in enumerate(env_station_ids):
            out_path = env_dir / f"loocv_{s}.parquet"
            if out_path.exists() and not ENV_LOOCV_OVERWRITE:
                print(f"   [{i+1}/{len(env_station_ids)}] {s}: skipping (file exists)")
                continue

            t0 = time.time()
            print(f"\n   ── Fold [{i+1}/{len(env_station_ids)}]  held-out: {s}  (env: {env_name})")

            # Held-out station geo/CERRA — needed for zone target and prediction
            s_geo          = geo_stations[geo_stations["id"] == s]
            s_cerra_interp = cerra_interp[cerra_interp["id"] == s]
            if s_cerra_interp.empty or s_geo.empty:
                print(f"   [{i+1}] {s}: SKIP — no CERRA/geo data for held-out station")
                continue

            # Training data: all stations except s
            train_stations = stations[stations["station_id"] != s].copy()
            train_cerra    = cerra_interp[cerra_interp["id"] != s].copy()
            train_geo      = geo_stations[geo_stations["id"] != s].copy()

            n_train = train_stations["station_id"].nunique()
            if n_train < 5:
                print(f"   [{i+1}] {s}: SKIP — only {n_train} training stations (min 5)")
                continue
            print(f"   Training set: {n_train} stations, {len(train_stations):,} station-days")

            # Zone target: s itself — donors most similar to s get highest weights
            s_feats_for_zones = s_geo.rename(columns={"id": "station_id"}).merge(
                _monthly_clim(s_cerra_interp, "id").rename(columns={"id": "station_id"}),
                on="station_id", how="left",
            )
            s_membership = compute_zone_memberships(
                s_feats_for_zones, northness_col="cerra_northness"
            )

            # Zone features for training stations
            print("   Computing zone memberships and similarity weights …")
            train_feats = train_geo.rename(columns={"id": "station_id"}).merge(
                _monthly_clim(train_cerra, "id").rename(columns={"id": "station_id"}),
                on="station_id", how="left",
            )
            train_memberships = compute_zone_memberships(
                train_feats, northness_col="cerra_northness"
            )
            zone_weights = compute_zone_similarity(
                train_feats, s_feats_for_zones, s_membership
            )

            # Train
            print("   Building training DataFrame …")
            training_df = build_training_df(
                train_stations, train_cerra, train_geo,
                zone_weights, train_memberships,
            )
            print(f"   Training DataFrame: {len(training_df):,} rows × {training_df.shape[1]} cols"
                  f"  — fitting LightGBM models (fast={LOOCV_FAST_MODE}) …")
            with tempfile.TemporaryDirectory() as tmp_dir:
                models = train_all(training_df, Path(tmp_dir), fast=LOOCV_FAST_MODE)
            print(f"   Models trained ({len(models)} model objects)")

            # Predict
            print("   Predicting at held-out station …")
            preds = predict_at_stations(s_cerra_interp, s_geo, s_membership, models)

            # Merge: obs + predictions + CERRA nearest
            s_obs = stations[stations["station_id"] == s][[
                "station_id", "date", "tmean_c", "tmin_c", "tmax_c",
                "rh_pct", "prcp_mm", "wdsp_ms", "gust_ms",
            ]].rename(columns={
                "tmean_c":  "obs_tmean_c",  "tmin_c":   "obs_tmin_c",
                "tmax_c":   "obs_tmax_c",   "rh_pct":   "obs_rh_pct",
                "prcp_mm":  "obs_prcp_mm",  "wdsp_ms":  "obs_wdsp_ms",
                "gust_ms":  "obs_gust_ms",
            })

            preds_renamed = preds.rename(columns={"id": "station_id"})
            s_nearest = cerra_nearest[cerra_nearest["id"] == s].rename(
                columns={"id": "station_id"}
            )

            merged = (
                s_obs
                .merge(preds_renamed, on=["station_id", "date"], how="inner")
                .merge(s_nearest,     on=["station_id", "date"], how="left")
            )
            if "cerra_precip" in merged.columns:
                merged["cerra_precip_mm"] = merged["cerra_precip"] * CERRA_PRECIP_MM_FACTOR

            merged.to_parquet(out_path, index=False)
            elapsed    = time.time() - t0
            date_range = (f"{pd.to_datetime(merged['date']).dt.year.min()}–"
                          f"{pd.to_datetime(merged['date']).dt.year.max()}")
            print(f"   Fold [{i+1}/{len(env_station_ids)}] DONE  {s}: "
                  f"{len(merged):,} rows ({date_range}), {elapsed:.0f}s  -> {out_path.name}")

        env_done = sorted(env_dir.glob("loocv_*.parquet"))
        print(f"\n   Group '{env_name}' complete — {len(env_done)} files in {env_dir}")

    total = sorted(_ENV_LOOCV_BASE.glob("*/loocv_*.parquet"))
    print(f"\n   Phase 2 (env) complete — {len(total)} total files in {_ENV_LOOCV_BASE}")


# ===========================================================================
# Phase 3 — Metrics
# ===========================================================================

def _thornthwaite_pet(T_monthly: np.ndarray) -> float:
    """
    Annual PET (mm/yr) from 12-element monthly mean temperature array (°C).
    Thornthwaite (1948) method.
    """
    i_m = np.where(T_monthly > 0, (T_monthly / 5.0) ** 1.514, 0.0)
    I   = i_m.sum()
    if I <= 0:
        return 0.0
    a    = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239
    PET_m = np.where(T_monthly > 0, 16.0 * (10.0 * T_monthly / I) ** a, 0.0)
    return float(PET_m.sum())


def _annual_indices(df: pd.DataFrame, r99p_thresh: float) -> pd.DataFrame:
    """
    Compute annual climate indices for obs, pred, and CERRA columns.
    Returns one row per calendar year.
    """
    df = df.copy()
    df["year"] = pd.to_datetime(df["date"]).dt.year

    # Derived precipitation series
    df["pred_prcp"] = _precip_point_estimate(df)
    if "cerra_precip_mm" in df.columns:
        df["cerra_prcp"] = df["cerra_precip_mm"]
    elif "cerra_precip" in df.columns:
        df["cerra_prcp"] = df["cerra_precip"] * CERRA_PRECIP_MM_FACTOR
    else:
        df["cerra_prcp"] = 0.0

    rows = []
    for yr, g in df.groupby("year"):
        row = {"year": yr}
        # Continuous variables — annual mean
        for src, obs_col, pred_col, cerra_col in [
            ("tmean",  "obs_tmean_c",  "tmean_p50",  "cerra_tmean"),
            ("tmin",   "obs_tmin_c",   "tmin_p50",   "cerra_tmin"),
            ("tmax",   "obs_tmax_c",   "tmax_p50",   "cerra_tmax"),
            ("wind",   "obs_wdsp_ms",  "wind_p50",   "cerra_wind"),
            ("gust",   "obs_gust_ms",  "gust_p50",   "cerra_gust"),
        ]:
            row[f"obs_{src}"]   = g[obs_col].mean()   if obs_col   in g else np.nan
            row[f"pred_{src}"]  = g[pred_col].mean()  if pred_col  in g else np.nan
            row[f"cerra_{src}"] = g[cerra_col].mean() if cerra_col in g else np.nan

        # Precipitation — annual total
        row["obs_precip_annual"]   = g["obs_prcp_mm"].sum()   if "obs_prcp_mm"   in g else np.nan
        row["pred_precip_annual"]  = g["pred_prcp"].sum()
        row["cerra_precip_annual"] = g["cerra_prcp"].sum()

        # Threshold-exceedance counts
        for tag, obs_c, pred_c, cerra_c, op, thr in [
            ("pr20",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", PR20_MM),
            ("pr50",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", PR50_MM),
            ("r99p",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", r99p_thresh),
            ("tmax35", "obs_tmax_c",  "tmax_p50",   "cerra_tmax", ">", TMAX35_C),
            ("tmax37", "obs_tmax_c",  "tmax_p50",   "cerra_tmax", ">", TMAX37_C),
            ("tmin0",  "obs_tmin_c",  "tmin_p50",   "cerra_tmin", "<", TMIN0_C),
        ]:
            fn = (lambda a, t: a > t) if op == ">" else (lambda a, t: a < t)
            row[f"obs_{tag}"]   = fn(g[obs_c],   thr).sum()   if obs_c   in g.columns else np.nan
            row[f"pred_{tag}"]  = fn(g[pred_c],  thr).sum()   if pred_c  in g.columns else np.nan
            row[f"cerra_{tag}"] = fn(g[cerra_c], thr).sum()   if cerra_c in g.columns else np.nan

        # Aridity index (annual P / Thornthwaite PET)
        for prefix, t_col, p_col in [
            ("obs",   "obs_tmean_c",  "obs_prcp_mm"),
            ("pred",  "tmean_p50",    "pred_prcp"),
            ("cerra", "cerra_tmean",  "cerra_prcp"),
        ]:
            if t_col not in g.columns:
                row[f"{prefix}_aridity"] = np.nan
                continue
            month_means = (
                g.assign(_mo=pd.to_datetime(g["date"]).dt.month)
                .groupby("_mo")[t_col].mean()
                .reindex(range(1, 13), fill_value=0.0)
            )
            T_m  = month_means.values.astype(float)
            pet  = _thornthwaite_pet(T_m)
            P    = g[p_col].sum() if p_col in g.columns else np.nan
            row[f"{prefix}_aridity"] = P / pet if pet > 0 else np.nan

        rows.append(row)

    return pd.DataFrame(rows)


def _monthly_indices(df: pd.DataFrame, r99p_thresh: float) -> pd.DataFrame:
    """
    Compute monthly climate indices for obs, pred, and CERRA columns.
    Returns one row per (year, month).
    """
    df = df.copy()
    df["year"]  = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["pred_prcp"] = _precip_point_estimate(df)
    if "cerra_precip_mm" in df.columns:
        df["cerra_prcp"] = df["cerra_precip_mm"]
    elif "cerra_precip" in df.columns:
        df["cerra_prcp"] = df["cerra_precip"] * CERRA_PRECIP_MM_FACTOR
    else:
        df["cerra_prcp"] = 0.0

    rows = []
    for (yr, mo), g in df.groupby(["year", "month"]):
        row = {"year": yr, "month": mo}
        for src, obs_col, pred_col, cerra_col in [
            ("tmean",  "obs_tmean_c", "tmean_p50",  "cerra_tmean"),
            ("tmin",   "obs_tmin_c",  "tmin_p50",   "cerra_tmin"),
            ("tmax",   "obs_tmax_c",  "tmax_p50",   "cerra_tmax"),
            ("wind",   "obs_wdsp_ms", "wind_p50",   "cerra_wind"),
            ("gust",   "obs_gust_ms", "gust_p50",   "cerra_gust"),
        ]:
            row[f"obs_{src}"]   = g[obs_col].mean()   if obs_col   in g.columns else np.nan
            row[f"pred_{src}"]  = g[pred_col].mean()  if pred_col  in g.columns else np.nan
            row[f"cerra_{src}"] = g[cerra_col].mean() if cerra_col in g.columns else np.nan

        row["obs_precip"]   = g["obs_prcp_mm"].sum() if "obs_prcp_mm" in g.columns else np.nan
        row["pred_precip"]  = g["pred_prcp"].sum()
        row["cerra_precip"] = g["cerra_prcp"].sum()

        for tag, obs_c, pred_c, cerra_c, op, thr in [
            ("pr20",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", PR20_MM),
            ("pr50",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", PR50_MM),
            ("r99p",   "obs_prcp_mm", "pred_prcp",  "cerra_prcp", ">", r99p_thresh),
            ("tmax35", "obs_tmax_c",  "tmax_p50",   "cerra_tmax", ">", TMAX35_C),
            ("tmax37", "obs_tmax_c",  "tmax_p50",   "cerra_tmax", ">", TMAX37_C),
            ("tmin0",  "obs_tmin_c",  "tmin_p50",   "cerra_tmin", "<", TMIN0_C),
        ]:
            fn = (lambda a, t: a > t) if op == ">" else (lambda a, t: a < t)
            row[f"obs_{tag}"]   = fn(g[obs_c],   thr).sum() if obs_c   in g.columns else np.nan
            row[f"pred_{tag}"]  = fn(g[pred_c],  thr).sum()  if pred_c  in g.columns else np.nan
            row[f"cerra_{tag}"] = fn(g[cerra_c], thr).sum()  if cerra_c in g.columns else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _r2_rmse_bias(obs: np.ndarray, pred: np.ndarray) -> dict:
    mask = np.isfinite(obs) & np.isfinite(pred)
    if mask.sum() < 3:
        return {"R2": np.nan, "RMSE": np.nan, "Bias": np.nan, "n": int(mask.sum())}
    o, p = obs[mask], pred[mask]
    ss_res = ((o - p) ** 2).sum()
    ss_tot = ((o - o.mean()) ** 2).sum()
    r2   = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = float(np.sqrt(ss_res / mask.sum()))
    bias = float((p - o).mean())
    return {"R2": float(r2), "RMSE": rmse, "Bias": bias, "n": int(mask.sum())}


def _contingency_metrics(obs_event: np.ndarray, pred_event: np.ndarray) -> dict:
    H = int(( obs_event &  pred_event).sum())
    M = int(( obs_event & ~pred_event).sum())
    F = int((~obs_event &  pred_event).sum())
    pod = H / (H + M) if (H + M) > 0 else np.nan
    far = F / (F + H) if (F + H) > 0 else np.nan
    csi = H / (H + M + F) if (H + M + F) > 0 else np.nan
    return {"POD": pod, "FAR": far, "CSI": csi,
            "H": H, "M": M, "F": F, "n_obs_events": H + M}


_INDEX_ANNUAL_COLS = [
    # (index_name, obs_col, pred_col, cerra_col)
    ("tmean",         "obs_tmean",         "pred_tmean",         "cerra_tmean"),
    ("tmin",          "obs_tmin",          "pred_tmin",          "cerra_tmin"),
    ("tmax",          "obs_tmax",          "pred_tmax",          "cerra_tmax"),
    ("wind",          "obs_wind",          "pred_wind",          "cerra_wind"),
    ("gust",          "obs_gust",          "pred_gust",          "cerra_gust"),
    ("precip_annual", "obs_precip_annual", "pred_precip_annual", "cerra_precip_annual"),
    ("pr20",          "obs_pr20",          "pred_pr20",          "cerra_pr20"),
    ("pr50",          "obs_pr50",          "pred_pr50",          "cerra_pr50"),
    ("r99p",          "obs_r99p",          "pred_r99p",          "cerra_r99p"),
    ("tmax35",        "obs_tmax35",        "pred_tmax35",        "cerra_tmax35"),
    ("tmax37",        "obs_tmax37",        "pred_tmax37",        "cerra_tmax37"),
    ("tmin0",         "obs_tmin0",         "pred_tmin0",         "cerra_tmin0"),
    ("aridity_index", "obs_aridity",       "pred_aridity",       "cerra_aridity"),
]

_INDEX_MONTHLY_COLS = [
    # aridity_index excluded (annual-only)
    ("tmean",   "obs_tmean",   "pred_tmean",   "cerra_tmean"),
    ("tmin",    "obs_tmin",    "pred_tmin",    "cerra_tmin"),
    ("tmax",    "obs_tmax",    "pred_tmax",    "cerra_tmax"),
    ("wind",    "obs_wind",    "pred_wind",    "cerra_wind"),
    ("gust",    "obs_gust",    "pred_gust",    "cerra_gust"),
    ("precip",  "obs_precip",  "pred_precip",  "cerra_precip"),
    ("pr20",    "obs_pr20",    "pred_pr20",    "cerra_pr20"),
    ("pr50",    "obs_pr50",    "pred_pr50",    "cerra_pr50"),
    ("r99p",    "obs_r99p",    "pred_r99p",    "cerra_r99p"),
    ("tmax35",  "obs_tmax35",  "pred_tmax35",  "cerra_tmax35"),
    ("tmax37",  "obs_tmax37",  "pred_tmax37",  "cerra_tmax37"),
    ("tmin0",   "obs_tmin0",   "pred_tmin0",   "cerra_tmin0"),
]

_CONTINGENCY_EVENTS = [
    # (event_name, obs_col, pred_col, cerra_col, op, threshold)
    ("wet_1mm",  "obs_prcp_mm", "pred_prcp", "cerra_prcp", ">", WET_DAY_MM),
    ("pr20",     "obs_prcp_mm", "pred_prcp", "cerra_prcp", ">", PR20_MM),
    ("pr50",     "obs_prcp_mm", "pred_prcp", "cerra_prcp", ">", PR50_MM),
    ("tmax35",   "obs_tmax_c",  "tmax_p50",  "cerra_tmax", ">", TMAX35_C),
    ("tmax37",   "obs_tmax_c",  "tmax_p50",  "cerra_tmax", ">", TMAX37_C),
    ("r99p",     None,          None,        None,          ">", None),  # threshold set per station
    ("tmin0",    "obs_tmin_c",  "tmin_p50",  "cerra_tmin", "<", TMIN0_C),
]


def _model_best_source(df: pd.DataFrame) -> np.ndarray:
    """
    Pre-EQM daily precipitation best-estimate (the EQM *source* variable).

    This is the model's Hurdle estimate occ_prob × precip_mean (the user-approved
    EQM source).  Falls back to occ_prob × precip_p50 if precip_mean is absent.
    Distinct from _precip_point_estimate, which prefers the already-EQM'd column.
    """
    if "precip_mean" in df.columns and "precip_occ_prob" in df.columns:
        return (df["precip_occ_prob"] * df["precip_mean"]).values
    if "precip_best" in df.columns:
        return df["precip_best"].values
    occ = df["precip_occ_prob"].values if "precip_occ_prob" in df.columns else np.zeros(len(df))
    p50 = df["precip_p50"].values      if "precip_p50"       in df.columns else np.zeros(len(df))
    return np.where(occ > OCC_PROB_THRESHOLD, p50, 0.0)


def _add_precip_eqm_to_dir(loocv_dir: Path) -> None:
    """
    Compute the EQM-corrected precipitation best-estimate for every station file
    in ``loocv_dir`` and write it back as the ``precip_best_eqm`` column.

    For each held-out station, the EQM transfer is fitted from ALL OTHER stations
    in the directory (the donor pool = the environment group in env-LOOCV), using
    smooth DOY windows (seasonal_precip_eqm).  Idempotent: re-running overwrites
    the column.  No-op if PRECIP_EQM_ENABLE is False or fewer than 2 files exist.
    """
    if not PRECIP_EQM_ENABLE:
        return
    files = sorted(loocv_dir.glob("loocv_*.parquet"))
    if len(files) < 2:
        print(f"   EQM: <2 station files in {loocv_dir.name}, skipping")
        return

    # Load once; cache model best-estimate, target CERRA, obs, and DOY per file
    cache = {}
    for fp in files:
        d = pd.read_parquet(fp)
        doy = pd.to_datetime(d["date"]).dt.dayofyear.values
        cache[fp] = {
            "src":   _model_best_source(d),
            "cerra": d["cerra_precip"].values,     # already mm
            "obs":   d["obs_prcp_mm"].values,
            "doy":   doy,
        }

    print(f"   EQM: correcting {len(files)} stations in {loocv_dir.name} "
          f"(QM(CERRA)+model blend={PRECIP_EQM_BLEND}, donor = other stations, "
          f"DOY window ±{PRECIP_EQM_WINDOW_DAYS}d)")
    for fp in files:
        # Donor pool = every OTHER station's paired (obs, CERRA, DOY)
        d_obs, d_cerra, d_doy = [], [], []
        for fp2 in files:
            if fp2 == fp:
                continue
            d_obs.append(cache[fp2]["obs"])
            d_cerra.append(cache[fp2]["cerra"])
            d_doy.append(cache[fp2]["doy"])
        donor_obs   = np.concatenate(d_obs)
        donor_cerra = np.concatenate(d_cerra)
        donor_doy   = np.concatenate(d_doy)
        keep = np.isfinite(donor_obs) & np.isfinite(donor_cerra)

        corrected = blended_precip_eqm(
            cache[fp]["cerra"], cache[fp]["src"], cache[fp]["doy"],
            donor_obs[keep], donor_cerra[keep], donor_doy[keep],
            alpha=PRECIP_EQM_BLEND,
            window_days=PRECIP_EQM_WINDOW_DAYS,
            step_days=PRECIP_EQM_STEP_DAYS,
            min_wet=PRECIP_EQM_MIN_WET,
        )
        d_out = pd.read_parquet(fp)
        d_out["precip_best_eqm"] = corrected
        d_out.to_parquet(fp, index=False)


def phase3_metrics() -> None:
    """
    Load all LOOCV parquet files and compute R²/RMSE/Bias and POD/FAR/CSI metrics.
    Exports 4 CSVs to validation/metrics/.
    """
    loocv_files = sorted(_LOOCV_DIR.glob("loocv_*.parquet"))
    if not loocv_files:
        print(f"   Phase 3: no LOOCV parquet files found in {_LOOCV_DIR}")
        return
    # Distribution-correct the precipitation best-estimate before scoring
    _add_precip_eqm_to_dir(_LOOCV_DIR)
    n_files = len(loocv_files)
    print(f"   Phase 3: computing metrics from {n_files} station files in {_LOOCV_DIR}")

    cont_rows_per_station: list[dict] = []
    cont_rows_overall_loocv: list[dict] = []
    cont_rows_overall_cerra: list[dict] = []
    ctg_rows_per_station:   list[dict] = []

    # Accumulators for pooled (all-station) contingency
    pooled_loocv: dict[str, dict] = {}
    pooled_cerra: dict[str, dict] = {}

    # Accumulators for pooled continuous annual / monthly
    ann_all_loocv: dict[str, list] = {}
    ann_all_cerra: dict[str, list] = {}
    mon_all_loocv: dict[str, list] = {}
    mon_all_cerra: dict[str, list] = {}

    for fi, fpath in enumerate(loocv_files, 1):
        sid = fpath.stem.replace("loocv_", "")
        print(f"   [{fi}/{n_files}] {sid} …", flush=True)
        df  = pd.read_parquet(fpath)
        df["date"] = pd.to_datetime(df["date"])

        # Derived series needed by both metric types
        df["pred_prcp"] = _precip_point_estimate(df)
        if "cerra_precip_mm" in df.columns:
            df["cerra_prcp"] = df["cerra_precip_mm"].values
        elif "cerra_precip" in df.columns:
            df["cerra_prcp"] = df["cerra_precip"].values * CERRA_PRECIP_MM_FACTOR
        else:
            df["cerra_prcp"] = 0.0

        # r99p threshold from this station's observed wet days
        wet_obs = df.loc[df["obs_prcp_mm"] > WET_DAY_MM, "obs_prcp_mm"]
        r99p_thresh = float(np.nanpercentile(wet_obs, R99P_PERCENTILE)) if len(wet_obs) > 10 else 999.0

        # ── Continuous metrics ────────────────────────────────────────────────
        ann = _annual_indices(df, r99p_thresh)
        mon = _monthly_indices(df, r99p_thresh)

        for idx_name, obs_c, pred_c, cerra_c in _INDEX_ANNUAL_COLS:
            if obs_c not in ann.columns:
                continue
            m_loocv = _r2_rmse_bias(ann[obs_c].values, ann[pred_c].values  if pred_c  in ann else np.full(len(ann), np.nan))
            m_cerra = _r2_rmse_bias(ann[obs_c].values, ann[cerra_c].values if cerra_c in ann else np.full(len(ann), np.nan))
            cont_rows_per_station.append({
                "station_id": sid, "index": idx_name, "frequency": "annual",
                "source": "loocv", **m_loocv,
            })
            cont_rows_per_station.append({
                "station_id": sid, "index": idx_name, "frequency": "annual",
                "source": "cerra", **m_cerra,
            })
            # Accumulate for pooled metrics (keep only rows where both are finite)
            for key, pool in [("loocv", ann_all_loocv), ("cerra", ann_all_cerra)]:
                col = pred_c if key == "loocv" else cerra_c
                if col in ann.columns and obs_c in ann.columns:
                    mask = ann[obs_c].notna() & ann[col].notna()
                    pool.setdefault(idx_name, [[], []])
                    pool[idx_name][0].extend(ann.loc[mask, obs_c].tolist())
                    pool[idx_name][1].extend(ann.loc[mask, col].tolist())

        for idx_name, obs_c, pred_c, cerra_c in _INDEX_MONTHLY_COLS:
            if obs_c not in mon.columns:
                continue
            m_loocv = _r2_rmse_bias(mon[obs_c].values, mon[pred_c].values  if pred_c  in mon else np.full(len(mon), np.nan))
            m_cerra = _r2_rmse_bias(mon[obs_c].values, mon[cerra_c].values if cerra_c in mon else np.full(len(mon), np.nan))
            cont_rows_per_station.append({
                "station_id": sid, "index": idx_name, "frequency": "monthly",
                "source": "loocv", **m_loocv,
            })
            cont_rows_per_station.append({
                "station_id": sid, "index": idx_name, "frequency": "monthly",
                "source": "cerra", **m_cerra,
            })
            for key, pool in [("loocv", mon_all_loocv), ("cerra", mon_all_cerra)]:
                col = pred_c if key == "loocv" else cerra_c
                if col in mon.columns and obs_c in mon.columns:
                    mask = mon[obs_c].notna() & mon[col].notna()
                    pool.setdefault(idx_name, [[], []])
                    pool[idx_name][0].extend(mon.loc[mask, obs_c].tolist())
                    pool[idx_name][1].extend(mon.loc[mask, col].tolist())

        # ── Contingency metrics ───────────────────────────────────────────────
        for event, obs_c, pred_c, cerra_c, op, thr in _CONTINGENCY_EVENTS:
            # r99p: use station-specific threshold
            if event == "r99p":
                obs_c   = "obs_prcp_mm"
                pred_c  = "pred_prcp"
                cerra_c = "cerra_prcp"
                thr     = r99p_thresh

            if obs_c not in df.columns:
                continue
            fn = (lambda a, t: a > t) if op == ">" else (lambda a, t: a < t)

            pred_present  = pred_c  in df.columns
            cerra_present = cerra_c in df.columns

            # Matched sample: only rows where obs, pred, and cerra are all non-NaN.
            valid = df[obs_c].notna()
            if pred_present:
                valid = valid & df[pred_c].notna()
            if cerra_present:
                valid = valid & df[cerra_c].notna()

            df_v  = df[valid]
            n_v   = len(df_v)
            obs_ev   = fn(df_v[obs_c],   thr).values
            cerra_ev = fn(df_v[cerra_c], thr).values if cerra_present else np.zeros(n_v, bool)
            # pred_prcp = precip_best_eqm (blended); thresholding it gives the
            # event prediction for all precip events including pr20/pr50.
            pred_ev  = fn(df_v[pred_c], thr).values if pred_present else np.zeros(n_v, bool)

            m_l = _contingency_metrics(obs_ev, pred_ev)
            m_c = _contingency_metrics(obs_ev, cerra_ev)
            ctg_rows_per_station.append({
                "station_id": sid, "event": event, "source": "loocv", **m_l,
            })
            ctg_rows_per_station.append({
                "station_id": sid, "event": event, "source": "cerra", **m_c,
            })

            # Accumulate for pooled contingency
            for key, pool in [("loocv", pooled_loocv), ("cerra", pooled_cerra)]:
                ev = pred_ev if key == "loocv" else cerra_ev
                pool.setdefault(event, {"H": 0, "M": 0, "F": 0})
                pool[event]["H"] += int(( obs_ev &  ev).sum())
                pool[event]["M"] += int(( obs_ev & ~ev).sum())
                pool[event]["F"] += int((~obs_ev &  ev).sum())

    # ── Build pooled continuous rows ─────────────────────────────────────────
    for freq, pool_l, pool_c in [
        ("annual",  ann_all_loocv, ann_all_cerra),
        ("monthly", mon_all_loocv, mon_all_cerra),
    ]:
        for idx_name in pool_l:
            obs_l  = np.array(pool_l[idx_name][0])
            pred_l = np.array(pool_l[idx_name][1])
            m = _r2_rmse_bias(obs_l, pred_l)
            cont_rows_overall_loocv.append({
                "station_id": "ALL", "index": idx_name,
                "frequency": freq, "source": "loocv", **m,
            })
        for idx_name in pool_c:
            obs_c2  = np.array(pool_c[idx_name][0])
            pred_c2 = np.array(pool_c[idx_name][1])
            m = _r2_rmse_bias(obs_c2, pred_c2)
            cont_rows_overall_cerra.append({
                "station_id": "ALL", "index": idx_name,
                "frequency": freq, "source": "cerra", **m,
            })

    # ── Build pooled contingency rows ────────────────────────────────────────
    ctg_rows_overall: list[dict] = []
    for key, pool in [("loocv", pooled_loocv), ("cerra", pooled_cerra)]:
        for event, counts in pool.items():
            H, M, F = counts["H"], counts["M"], counts["F"]
            pod = H / (H + M) if (H + M) > 0 else np.nan
            far = F / (F + H) if (F + H) > 0 else np.nan
            csi = H / (H + M + F) if (H + M + F) > 0 else np.nan
            ctg_rows_overall.append({
                "station_id": "ALL", "event": event, "source": key,
                "POD": pod, "FAR": far, "CSI": csi,
                "H": H, "M": M, "F": F, "n_obs_events": H + M,
            })

    # ── Export ───────────────────────────────────────────────────────────────
    all_cont = cont_rows_per_station + cont_rows_overall_loocv + cont_rows_overall_cerra
    all_ctg  = ctg_rows_per_station + ctg_rows_overall

    cont_df = pd.DataFrame(all_cont)
    ctg_df  = pd.DataFrame(all_ctg)

    cont_loocv = cont_df[cont_df["source"] == "loocv"].drop(columns="source")
    cont_cerra = cont_df[cont_df["source"] == "cerra"].drop(columns="source")
    ctg_loocv  = ctg_df[ctg_df["source"]  == "loocv"].drop(columns="source")
    ctg_cerra  = ctg_df[ctg_df["source"]  == "cerra"].drop(columns="source")

    f_cl = _METRICS_DIR / "metrics_loocv_continuous.csv"
    f_cc = _METRICS_DIR / "metrics_cerra_continuous.csv"
    f_tl = _METRICS_DIR / "metrics_loocv_contingency.csv"
    f_tc = _METRICS_DIR / "metrics_cerra_contingency.csv"

    cont_loocv.to_csv(f_cl, index=False)
    cont_cerra.to_csv(f_cc, index=False)
    ctg_loocv.to_csv( f_tl, index=False)
    ctg_cerra.to_csv( f_tc, index=False)

    print(f"\n   Phase 3 complete — metrics written to {_METRICS_DIR}")
    print(f"   {f_cl.name}:  {len(cont_loocv):,} rows")
    print(f"   {f_cc.name}:  {len(cont_cerra):,} rows")
    print(f"   {f_tl.name}: {len(ctg_loocv):,} rows")
    print(f"   {f_tc.name}: {len(ctg_cerra):,} rows")


# ===========================================================================
# Phase 3 (env) — Metrics for environment-specific LOOCV
# ===========================================================================

def phase3_env_metrics() -> None:
    """
    Compute R²/RMSE/Bias and POD/FAR/CSI for all environment LOOCV outputs.

    Reads from env_loocv/{group_name}/loocv_*.parquet and writes two CSVs to
    env_metrics/:
      metrics_env_continuous.csv   — columns: group, station_id, index, frequency, R2, RMSE, Bias, n
      metrics_env_contingency.csv  — columns: group, station_id, event, POD, FAR, CSI, H, M, F, n_obs_events

    A per-group summary table (median monthly R² for key indices) is printed
    at the end so differences across environments are immediately visible.
    """
    _ENV_METRICS_DIR.mkdir(parents=True, exist_ok=True)

    cont_rows: list[dict] = []
    ctg_rows:  list[dict] = []

    for env_name, env_cfg in ENVIRONMENT_GROUPS.items():
        env_dir = _ENV_LOOCV_BASE / env_name
        if not env_dir.exists():
            print(f"   Group '{env_name}': directory not found, skipping")
            continue

        loocv_files = sorted(env_dir.glob("loocv_*.parquet"))
        if not loocv_files:
            print(f"   Group '{env_name}': no parquet files found, skipping")
            continue

        # Distribution-correct precip within this group (donor = other group stations)
        _add_precip_eqm_to_dir(env_dir)

        print(f"\n   ── Group: {env_cfg['label']}  ({len(loocv_files)} files)")

        for fi, fpath in enumerate(loocv_files, 1):
            sid = fpath.stem.replace("loocv_", "")
            print(f"   [{fi}/{len(loocv_files)}] {sid} …", flush=True)
            df = pd.read_parquet(fpath)
            df["date"] = pd.to_datetime(df["date"])

            # Derived precipitation columns (same logic as phase3_metrics)
            df["pred_prcp"] = _precip_point_estimate(df)
            if "cerra_precip_mm" in df.columns:
                df["cerra_prcp"] = df["cerra_precip_mm"].values
            elif "cerra_precip" in df.columns:
                df["cerra_prcp"] = df["cerra_precip"].values * CERRA_PRECIP_MM_FACTOR
            else:
                df["cerra_prcp"] = 0.0

            wet_obs      = df.loc[df["obs_prcp_mm"] > WET_DAY_MM, "obs_prcp_mm"]
            r99p_thresh  = float(np.nanpercentile(wet_obs, R99P_PERCENTILE)) if len(wet_obs) > 10 else 999.0

            ann = _annual_indices(df, r99p_thresh)
            mon = _monthly_indices(df, r99p_thresh)

            # Continuous metrics (LOOCV only — no CERRA baseline split needed here)
            for idx_name, obs_c, pred_c, _ in _INDEX_ANNUAL_COLS:
                if obs_c not in ann.columns:
                    continue
                pred_vals = ann[pred_c].values if pred_c in ann.columns else np.full(len(ann), np.nan)
                m = _r2_rmse_bias(ann[obs_c].values, pred_vals)
                cont_rows.append({"group": env_name, "station_id": sid,
                                   "index": idx_name, "frequency": "annual", **m})

            for idx_name, obs_c, pred_c, _ in _INDEX_MONTHLY_COLS:
                if obs_c not in mon.columns:
                    continue
                pred_vals = mon[pred_c].values if pred_c in mon.columns else np.full(len(mon), np.nan)
                m = _r2_rmse_bias(mon[obs_c].values, pred_vals)
                cont_rows.append({"group": env_name, "station_id": sid,
                                   "index": idx_name, "frequency": "monthly", **m})

            # Contingency metrics
            for event, obs_c, pred_c, _, op, thr in _CONTINGENCY_EVENTS:
                if event == "r99p":
                    obs_c, pred_c, thr = "obs_prcp_mm", "pred_prcp", r99p_thresh
                if obs_c not in df.columns:
                    continue
                fn           = (lambda a, t: a > t) if op == ">" else (lambda a, t: a < t)
                pred_present = pred_c in df.columns
                valid        = df[obs_c].notna() & (df[pred_c].notna() if pred_present else True)
                df_v         = df[valid]
                n_v          = len(df_v)
                obs_ev  = fn(df_v[obs_c], thr).values
                # pred_prcp = precip_best_eqm (blended) — threshold for all events
                pred_ev = (fn(df_v[pred_c], thr).values if pred_present
                           else np.zeros(n_v, bool))
                m = _contingency_metrics(obs_ev, pred_ev)
                ctg_rows.append({"group": env_name, "station_id": sid, "event": event, **m})

    if not cont_rows:
        print("   Phase 3 (env): no data found — run phase 2 (env) first")
        return

    cont_df = pd.DataFrame(cont_rows)
    ctg_df  = pd.DataFrame(ctg_rows)

    f_cont = _ENV_METRICS_DIR / "metrics_env_continuous.csv"
    f_ctg  = _ENV_METRICS_DIR / "metrics_env_contingency.csv"
    cont_df.to_csv(f_cont, index=False, float_format="%.4f")
    ctg_df.to_csv( f_ctg,  index=False, float_format="%.4f")
    print(f"\n   Phase 3 (env) complete — metrics written to {_ENV_METRICS_DIR}")
    print(f"   {f_cont.name}:  {len(cont_df):,} rows")
    print(f"   {f_ctg.name}:   {len(ctg_df):,} rows")

    # Quick summary: median monthly R² per group for key indices
    key_indices = ["tmean", "tmin", "tmax", "precip", "wind"]
    monthly = cont_df[(cont_df["frequency"] == "monthly") & (cont_df["index"].isin(key_indices))]
    if not monthly.empty:
        pivot = (monthly.groupby(["group", "index"])["R2"]
                 .median()
                 .unstack("index")
                 .reindex(columns=key_indices))
        print("\n   Median monthly R² by environment group:")
        print(pivot.to_string())


# ===========================================================================
# Entry point
# ===========================================================================

def inspect_zone_weights() -> None:
    """
    Compute and export per-station similarity weights for each Delos zone using
    the full station network (no fold held out).  Requires Phase 1 cache to exist.

    Output: validation/zone_weights.csv
      Columns: station_id, lat, lon, elevation_m, is_island,
               zone_0_weight, …, zone_{N-1}_weight,
               dominant_zone (zone with highest weight)
    """
    print("Loading Phase 1 cache …")
    stations     = pd.read_parquet(_F_STATION_OBS)
    geo_base     = pd.read_parquet(_F_STATION_GEO)
    cerra_static = pd.read_parquet(_F_CERRA_STATIC)
    geo_stations = geo_base.merge(cerra_static, on="id", how="left")
    cerra_interp = pd.read_parquet(_CERRA_INTERP_CACHE)

    delos_locs = (
        pd.read_csv(DELOS_CSV)[["lon", "lat", "VALUE"]]
        .rename(columns={"VALUE": "elevation_m"})
        .assign(id=lambda df: range(len(df)))[["id", "lat", "lon", "elevation_m"]]
    )

    _delos_cache = _HERE / "downscaling" / "cache" / "cerra_delos.parquet"
    if not _delos_cache.exists():
        print(f"   {_delos_cache.name} not found — extracting cubic-interpolated CERRA "
              f"at Delos grid ({YEAR_START}–{YEAR_END})  … this may take several minutes")
        (_HERE / "downscaling" / "cache").mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        cerra_delos = extract_all(CERRA_DIR, delos_locs, YEAR_START, YEAR_END)
        cerra_delos.to_parquet(_delos_cache)
        print(f"   -> {len(cerra_delos):,} rows saved  ({time.time()-t0:.0f}s)")
    else:
        cerra_delos = pd.read_parquet(_delos_cache)

    cerra_static_delos    = extract_cerra_static(CERRA_DIR, delos_locs)
    geo_delos_base        = compute_geo_features(delos_locs)
    geo_delos             = geo_delos_base.merge(cerra_static_delos, on="id", how="left")
    delos_feats_for_zones = geo_delos.rename(columns={"id": "station_id"}).merge(
        _monthly_clim(cerra_delos, "id").rename(columns={"id": "station_id"}),
        on="station_id", how="left",
    )
    delos_memberships = compute_zone_memberships(
        delos_feats_for_zones, northness_col="dem_northness"
    )

    # Full station feature matrix (all stations, no fold exclusion)
    train_feats = geo_stations.rename(columns={"id": "station_id"}).merge(
        _monthly_clim(cerra_interp, "id").rename(columns={"id": "station_id"}),
        on="station_id", how="left",
    )
    zone_weights = compute_zone_similarity(
        train_feats, delos_feats_for_zones, delos_memberships
    )

    # Assemble into a wide DataFrame
    weight_df = pd.DataFrame(
        {f"zone_{k}_weight": zone_weights[k] for k in range(N_ZONES)}
    )
    weight_df.index.name = "station_id"
    weight_df = weight_df.reset_index()

    # Join station metadata for interpretability
    station_locs = (
        stations[["station_id", "lat", "lon", "elevation_m"]]
        .drop_duplicates("station_id")
    )
    meta_cols = ["station_id", "lat", "lon", "elevation_m"]
    if "is_island" in train_feats.columns:
        is_island = (
            train_feats[["station_id", "is_island"]]
            .drop_duplicates("station_id")
        )
        station_locs = station_locs.merge(is_island, on="station_id", how="left")
        meta_cols.append("is_island")

    weight_df = station_locs[meta_cols].merge(weight_df, on="station_id", how="right")

    zone_cols = [f"zone_{k}_weight" for k in range(N_ZONES)]
    weight_df["dominant_zone"] = weight_df[zone_cols].idxmax(axis=1).str.replace("_weight", "")

    out = VALIDATION_DIR / "zone_weights.csv"
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    weight_df.to_csv(out, index=False, float_format="%.4f")

    print(f"\nZone weights written to {out}")
    print(f"{len(weight_df)} stations  ×  {N_ZONES} zones\n")
    print(weight_df[["station_id"] + zone_cols + ["dominant_zone"]]
          .sort_values(zone_cols[0], ascending=False)
          .to_string(index=False))


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Delos LOOCV validation pipeline")
    parser.add_argument(
        "--env-phase", type=int, nargs="+", choices=[1, 2, 3], default=[1, 2, 3],
        help="Environment-group LOOCV phases to run (default: 1 2 3).  "
             "1 = build/load station dataset, 2 = train/predict per group station, "
             "3 = compute group metrics.  Example: --env-phase 3 to rerun metrics only.",
    )
    parser.add_argument(
        "--delos-phase", type=int, nargs="+", choices=[1, 2, 3], default=[],
        help="Run full Delos LOOCV phases (all 126 stations, zone target = Delos grid).  "
             "2 = train/predict, 3 = compute metrics.  Example: --delos-phase 2 3",
    )
    parser.add_argument(
        "--inspect-weights", action="store_true",
        help="Export per-station Delos zone similarity weights to validation/zone_weights.csv "
             "and print a summary table.  Requires Phase 1 cache to exist.",
    )
    args = parser.parse_args()

    if args.inspect_weights:
        inspect_zone_weights()
        return

    env_phases   = set(args.env_phase)
    delos_phases = set(args.delos_phase)

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print("    Delos LOOCV Validation Pipeline")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"   Output directory : {VALIDATION_DIR}")
    print(f"   Year range       : {VAL_YEAR_START}–{VAL_YEAR_END}")
    if env_phases:
        grp_names = ", ".join(ENVIRONMENT_GROUPS)
        print(f"   Env phases       : {sorted(env_phases)}  ({grp_names})")
    if delos_phases:
        print(f"   Delos phases     : {sorted(delos_phases)}  (all stations, Delos zone target)")
    print(f"   LOOCV overwrite  : {LOOCV_OVERWRITE}  |  Env overwrite: {ENV_LOOCV_OVERWRITE}")
    print(f"   Cache overwrite  : {CACHE_OVERWRITE}   |  Fast mode: {LOOCV_FAST_MODE}")

    # Phase 1 (data load/build) is needed by any downstream phase
    needs_data = (env_phases - {3}) or (delos_phases - {3})
    if 1 in env_phases or 1 in delos_phases:
        print("\n━━ Phase 1  Station dataset preparation")
        (stations, station_locs,
         cerra_nearest, geo_stations, cerra_interp) = phase1_build_dataset()
    elif needs_data:
        print("\n━━ Phase 1  Skipped — loading cached data")
        stations      = pd.read_parquet(_F_STATION_OBS)
        cerra_nearest = pd.read_parquet(_F_CERRA_NEAREST)
        geo_base      = pd.read_parquet(_F_STATION_GEO)
        cerra_static  = pd.read_parquet(_F_CERRA_STATIC)
        geo_stations  = geo_base.merge(cerra_static, on="id", how="left")
        cerra_interp  = pd.read_parquet(_CERRA_INTERP_CACHE)
        station_locs  = (
            stations[["station_id", "lat", "lon", "elevation_m"]]
            .drop_duplicates("station_id")
            .rename(columns={"station_id": "id"})
            .reset_index(drop=True)
        )

    if 2 in env_phases:
        print("\n━━ Env Phase 2  Environment-group LOOCV training and prediction")
        print(f"   Groups: {', '.join(ENVIRONMENT_GROUPS)}")
        phase2_env_loocv(stations, station_locs, cerra_interp, geo_stations, cerra_nearest)

    if 3 in env_phases:
        print("\n━━ Env Phase 3  Environment-group skill metrics")
        phase3_env_metrics()

    if 2 in delos_phases:
        print("\n━━ Delos Phase 2  Full LOOCV training and prediction (all stations)")
        phase2_loocv(stations, station_locs, cerra_interp, geo_stations, cerra_nearest)

    if 3 in delos_phases:
        print("\n━━ Delos Phase 3  Full LOOCV skill metrics")
        phase3_metrics()

    print("\n━━ Done ━━")


if __name__ == "__main__":
    main()
