"""
train.py
========
Build the training dataset and train per-zone LightGBM quantile-regression
models.

MODEL ARCHITECTURE
------------------
Seven climate variables are modelled independently:
  tmean, tmin, tmax  — temperature (°C)
  rh                 — relative humidity (%)
  precip             — precipitation (mm/day)
  wind               — mean wind speed (m/s)
  gust               — maximum wind gust (m/s)

For each variable (except precipitation) three LightGBM models are trained per
zone, one per quantile level: p10 (q=0.1), p50 (q=0.5), p90 (q=0.9).
Wind and gust additionally get a separate mean-regression model per zone.

ZONE ARCHITECTURE (Proposition 8)
----------------------------------
N_ZONES independent model families are trained.  For zone k, each donor
station's sample weight is:

    weight = zone_station_similarity_k × zone_membership_k[station]

where:
  zone_station_similarity_k  : how climatically / physiographically similar
                                the station is to the zone-k Delos centroid
                                (from similarity.compute_zone_similarity)
  zone_membership_k[station] : how strongly the station itself belongs to
                                micro-climate zone k (from
                                similarity.compute_zone_memberships, using
                                the station's own coast_dist/elevation/northness)

This gives each zone model a smoothly weighted training set that peaks on
donors most representative of that zone, rather than using a hard station subset.

Prediction time: each Delos point gets predictions from all N_ZONES models,
then takes the membership-weighted average (see predict.py).

Model file naming convention:
  <var>_<tag>_zone<k>.txt   e.g. tmean_p50_zone0.txt, wind_mean_zone2.txt
  precip_occ_zone<k>.txt    e.g. precip_occ_zone1.txt

PRECIPITATION MODEL
-------------------
Two sub-models per zone (same two-part architecture as before):
  1. Occurrence classifier: P(daily precip > 0.1 mm)
  2. Conditional quantile regressors: p10, p50, p90 on wet days

FEATURE MATRIX
--------------
Each row represents one station × one day.
Features = CERRA predictors + geographic features (from config).

New features computed here before training:
  delta_elev_m      : elevation_m − cerra_elev_m  (sub-grid elevation residual)
  lapse_correction_c: delta_elev_m × MONTHLY_LAPSE_RATES[month] / 1000
                      (expected temperature offset in °C from terrain mismatch)
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from downscaling.config import (
    EXTRA_QUANTILES,
    EXTREME_SAMPLE_WEIGHT,
    GEO_FEATURES,
    MONTHLY_LAPSE_RATES,
    QUANTILES,
    VARIABLE_CERRA_FEATURES,
    VARIABLE_GEO_FEATURES,
)
from downscaling.similarity import N_ZONES

# ---------------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------------
_LGB_BASE = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    verbosity=-1,
    num_threads=-1,
)

# Wind/gust mean model: more leaves and more estimators to resolve fine-scale
# spatial structure (coast_dist_km, directional fetch, cerra_u10/v10).
_LGB_MEAN = dict(
    learning_rate=0.05,
    num_leaves=127,
    min_data_in_leaf=20,
    n_estimators=800,
    colsample_bytree=0.8,
    subsample=0.8,
    verbosity=-1,
    num_threads=-1,
)

_MIN_ROWS = 1000   # minimum training rows to fit a model (per zone)
_MEAN_VARS = {"wind", "gust"}

# Variables trained with the zone ensemble (N_ZONES independent models).
# precip / wind / gust use a single model trained on all stations uniformly —
# the spatial signal on a small island is too weak and noisy to benefit from
# zone splitting, which only reduces effective sample size and adds variance.
_ZONE_VARS = {"tmean", "tmin", "tmax", "rh"}

_TARGET_COL = {
    "tmean":  "tmean_c",
    "tmin":   "tmin_c",
    "tmax":   "tmax_c",
    "rh":     "rh_pct",
    "precip": "prcp_mm",
    "wind":   "wdsp_ms",
    "gust":   "gust_ms",
}


# ---------------------------------------------------------------------------
# Training DataFrame construction
# ---------------------------------------------------------------------------

def build_training_df(
    stations: pd.DataFrame,
    cerra: pd.DataFrame,
    geo: pd.DataFrame,
    zone_weights: dict[int, pd.Series],
    station_memberships: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge station observations, CERRA predictors, geographic features, and
    per-zone similarity weights into one flat training DataFrame.

    Parameters
    ----------
    stations            : station obs from station_ingest.load_stations()
    cerra               : CERRA-at-stations from cerra_extract.extract_all()
    geo                 : geographic features from geo_features.compute_geo_features(),
                          enriched with cerra_elev_m / cerra_northness / cerra_eastness
                          from cerra_extract.extract_cerra_static()
    zone_weights        : dict zone_k → pd.Series(station_id → weight)
                          from similarity.compute_zone_similarity()
    station_memberships : DataFrame (station_id × zone_k columns) from
                          similarity.compute_zone_memberships(station_features)

    Returns
    -------
    DataFrame with one row per station × day.  Contains all feature columns,
    target variable columns, and per-zone combined weight columns:
      zone_weight_0, zone_weight_1, …, zone_weight_{N_ZONES-1}
    """
    cerra_renamed = cerra.rename(columns={"id": "station_id"})
    geo_renamed   = geo.rename(  columns={"id": "station_id"})

    # Select all geo columns to carry forward (drop id after rename)
    geo_cols = [c for c in geo_renamed.columns if c != "station_id"] + ["station_id"]

    df = (
        stations
        .merge(cerra_renamed, on=["station_id", "date"], how="inner",
               suffixes=("", "_cerra"))
        .merge(geo_renamed[geo_cols], on="station_id", how="left",
               suffixes=("", "_geo"))
    )

    # --- Harmonic month encoding -----------------------------------------
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    # --- Elevation residual (Proposition 1) ------------------------------
    # delta_elev_m: how much higher/lower the station is than CERRA's terrain.
    # Positive → station above CERRA mean patch elevation → should be cooler.
    if "cerra_elev_m" in df.columns:
        df["delta_elev_m"] = df["elevation_m"] - df["cerra_elev_m"]
    else:
        df["delta_elev_m"] = 0.0

    # --- Lapse rate correction (Proposition 2) ---------------------------
    # Expected temperature offset in °C for the elevation mismatch.
    month_lapse = df["date"].dt.month.map(MONTHLY_LAPSE_RATES)   # K/km
    df["lapse_correction_c"] = df["delta_elev_m"] * month_lapse / 1000.0

    # --- Gust factor (INTRIGUE 2023) -------------------------------------
    # cerra_gust / cerra_wind separates synoptic flow magnitude from
    # boundary-layer amplification.  Used as a predictor for the gust model.
    if "cerra_gust" in df.columns and "cerra_wind" in df.columns:
        df["gust_factor_cerra"] = (
            df["cerra_gust"] / df["cerra_wind"].replace(0.0, np.nan)
        ).clip(1.0, 10.0).fillna(1.0)
    else:
        df["gust_factor_cerra"] = 1.0

    # --- Wind-terrain alignment (Dujardin 2022) --------------------------
    # Projects synoptic flow onto local upslope direction.
    # dem_eastness = sin(aspect), dem_northness = cos(aspect) — both in
    # standard meteorological convention compatible with cerra_u10/v10.
    if "cerra_u10" in df.columns and "dem_eastness" in df.columns:
        df["wind_terrain_align"] = (
            df["cerra_u10"] * df["dem_eastness"]
            + df["cerra_v10"] * df["dem_northness"]
        )
    else:
        df["wind_terrain_align"] = 0.0

    # --- Per-zone combined training weights (Proposition 8) --------------
    # weight_k = zone_station_similarity_k × station_membership_k
    # Station memberships give extra weight to stations whose own physiology
    # resembles zone k (e.g. island coastal stations for the N-Coastal zone).
    station_memberships = station_memberships.reset_index()
    mem_id_col = "station_id" if "station_id" in station_memberships.columns else "id"
    station_memberships = station_memberships.rename(columns={mem_id_col: "station_id"})

    for k in range(N_ZONES):
        sim_k = zone_weights.get(k, pd.Series(dtype=float))
        mem_k = (
            station_memberships.set_index("station_id")[f"zone_{k}"]
            if f"zone_{k}" in station_memberships.columns
            else pd.Series(1.0 / N_ZONES, index=station_memberships["station_id"])
        )
        # Map each row's station_id to its zone-k combined weight
        sim_mapped = df["station_id"].map(sim_k).fillna(1.0)
        mem_mapped = df["station_id"].map(mem_k).fillna(1.0 / N_ZONES)
        raw_weight = sim_mapped * mem_mapped
        # Normalise so the mean equals 1 (preserves LightGBM calibration)
        mean_w = raw_weight.mean()
        df[f"zone_weight_{k}"] = raw_weight / mean_w if mean_w > 0 else 1.0

    return df


# ---------------------------------------------------------------------------
# Per-zone model training
# ---------------------------------------------------------------------------

def train_all(
    training_df: pd.DataFrame,
    model_dir: Path,
    vars: list[str] | None = None,
    fast: bool = False,
) -> dict[str, lgb.Booster]:
    """
    Train per-zone LightGBM models for all (or a subset of) climate variables.

    For each combination of variable × quantile_tag × zone, one LightGBM model
    is trained using the corresponding zone_weight_k column as sample weights.

    Models are saved as:
      <var>_<tag>_zone<k>.txt     e.g. tmean_p50_zone0.txt
      <var>_mean_zone<k>.txt      e.g. wind_mean_zone2.txt
      precip_occ_zone<k>.txt      e.g. precip_occ_zone1.txt

    Parameters
    ----------
    training_df : training DataFrame from build_training_df()
    model_dir   : directory where .txt files are written
    vars        : if not None, only train models for these variable names

    Returns
    -------
    dict mapping model name → lgb.Booster
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    models: dict[str, lgb.Booster] = {}
    n_boost = 20 if fast else None

    for var, target_col in _TARGET_COL.items():
        if vars is not None and var not in vars:
            continue
        if target_col not in training_df.columns:
            continue

        feature_cols = VARIABLE_CERRA_FEATURES[var] + VARIABLE_GEO_FEATURES.get(var, GEO_FEATURES)
        available    = [c for c in feature_cols if c in training_df.columns]
        sub          = training_df.dropna(subset=[target_col])

        if len(sub) < _MIN_ROWS:
            continue

        X = sub[available].values
        y = sub[target_col].values

        if var in _ZONE_VARS:
            for k in range(N_ZONES):
                weight_col = f"zone_weight_{k}"
                w = sub[weight_col].values if weight_col in sub.columns else np.ones(len(sub))
                for q in QUANTILES:
                    name    = f"{var}_p{int(q * 100):02d}_zone{k}"
                    booster = _fit_quantile(X, y, w, q, n_boost=n_boost)
                    booster.save_model(str(model_dir / f"{name}.txt"))
                    models[name] = booster
        else:
            w = np.ones(len(sub))
            if var == "precip":
                _train_precip(X, y, w, model_dir, models, zone=None, n_boost=n_boost)
            else:
                for q in QUANTILES:
                    name    = f"{var}_p{int(q * 100):02d}"
                    booster = _fit_quantile(X, y, w, q, n_boost=n_boost)
                    booster.save_model(str(model_dir / f"{name}.txt"))
                    models[name] = booster
                if var in _MEAN_VARS:
                    _train_mean(var, X, y, w, model_dir, models, zone=None, n_boost=n_boost)

    return models


# ---------------------------------------------------------------------------
# Internal training helpers
# ---------------------------------------------------------------------------

def _train_mean(var: str, X, y, w, model_dir, models, zone: int | None,
                n_boost: int | None = None) -> None:
    """Train a mean-regression LightGBM model for wind or gust."""
    params  = {**_LGB_MEAN, "objective": "regression"}
    n       = n_boost if n_boost is not None else params.pop("n_estimators", 800)
    params.pop("n_estimators", None)
    ds      = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=n)
    name    = f"{var}_mean" if zone is None else f"{var}_mean_zone{zone}"
    booster.save_model(str(model_dir / f"{name}.txt"))
    models[name] = booster


def _train_precip(X, y, w, model_dir, models, zone: int | None,
                  n_boost: int | None = None) -> None:
    """
    Train the precipitation model:
      1. Binary occurrence classifier  — P(prcp > 0.1 mm), all days
      2. Conditional quantile models   — p10, p50, p90, p95, p99, wet days only
         p90/p95/p99 upweight obs > 95th-pct wet day by EXTREME_SAMPLE_WEIGHT
      3. Conditional mean (Gamma GLM)  — E[prcp | wet], wet days only

    precip_mean × occurrence is the amount source blended with QM(CERRA) to form
    the final precip_best_eqm (see predict.py / bias_correct.blended_precip_eqm).
    """
    sfx = "" if zone is None else f"_zone{zone}"
    n   = n_boost if n_boost is not None else _LGB_BASE["n_estimators"]

    # ── 1. Occurrence classifier (wet / dry) ────────────────────────────
    occ_label = (y > 0.1).astype(float)
    params = {**_LGB_BASE, "objective": "binary"}
    params.pop("n_estimators", None)
    booster = lgb.train(params, lgb.Dataset(X, label=occ_label, weight=w,
                                            free_raw_data=False), num_boost_round=n)
    occ_name = f"precip_occ{sfx}"
    booster.save_model(str(model_dir / f"{occ_name}.txt"))
    models[occ_name] = booster

    wet = y > 0.1
    if wet.sum() < 50:
        return
    X_wet, y_wet, w_wet = X[wet], y[wet], w[wet]

    # ── 2. Conditional quantile models ──────────────────────────────────
    # p90/p95/p99 get extreme-event upweighting (Q-SRDRN, arXiv 2605.12762).
    p95_wet   = np.percentile(y_wet, 95) if len(y_wet) >= 20 else np.inf
    w_extreme = np.where(y_wet > p95_wet, EXTREME_SAMPLE_WEIGHT, 1.0)

    for q in QUANTILES:
        w_q  = w_wet * w_extreme if q == 0.9 else w_wet
        name = f"precip_p{int(q * 100):02d}{sfx}"
        booster = _fit_quantile(X_wet, y_wet, w_q, q, n_boost=n_boost)
        booster.save_model(str(model_dir / f"{name}.txt"))
        models[name] = booster

    for q in EXTRA_QUANTILES:
        name    = f"precip_p{int(q * 100):02d}{sfx}"
        booster = _fit_quantile(X_wet, y_wet, w_wet * w_extreme, q, n_boost=n_boost)
        booster.save_model(str(model_dir / f"{name}.txt"))
        models[name] = booster

    # ── 3. Conditional mean — Gamma GLM ─────────────────────────────────
    mean_params = {**_LGB_MEAN, "objective": "gamma"}
    n_mean = n_boost if n_boost is not None else mean_params.pop("n_estimators", 800)
    mean_params.pop("n_estimators", None)
    booster = lgb.train(mean_params,
                        lgb.Dataset(X_wet, label=y_wet, weight=w_wet,
                                    free_raw_data=False),
                        num_boost_round=n_mean)
    mean_name = f"precip_mean{sfx}"
    booster.save_model(str(model_dir / f"{mean_name}.txt"))
    models[mean_name] = booster


def _fit_quantile(X, y, w, q: float, n_boost: int | None = None) -> lgb.Booster:
    """Fit one LightGBM quantile-regression model at probability level q."""
    params = {**_LGB_BASE, "objective": "quantile", "alpha": q}
    n      = n_boost if n_boost is not None else params.pop("n_estimators", 500)
    params.pop("n_estimators", None)
    ds     = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=n)
