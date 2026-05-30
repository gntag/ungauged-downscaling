"""
train.py
========
Build the training dataset and train LightGBM quantile-regression models.

MODEL ARCHITECTURE
------------------
Seven climate variables are modelled independently:
  tmean, tmin, tmax  — temperature (°C)
  rh                 — relative humidity (%)
  precip             — precipitation (mm/day)
  wind               — mean wind speed (m/s)
  gust               — maximum wind gust (m/s)

For each variable (except precipitation) three LightGBM models are trained,
one per quantile level: p10 (q=0.1), p50 (q=0.5), p90 (q=0.9).
Wind and gust additionally get a separate mean-regression model.

PRECIPITATION MODEL
-------------------
Precipitation is split into two sub-models:
  1. Occurrence model (binary classification):
     Predicts P(daily precip > 0.1 mm) — the wet-day probability.
  2. Conditional quantile models (regression on wet days only):
     p10, p50, p90 of the conditional distribution given a wet day.

This "two-part" approach avoids modelling a mixed discrete-continuous
distribution with a single regression tree, which would smear the wet/dry
boundary.  At prediction time the occurrence probability and the conditional
quantiles are combined (see post_processing_cerra_downscaling.py).

FEATURE MATRIX
--------------
Each row in the training DataFrame represents one station × one day.
Features = CERRA predictors for that station's location + geographic features.
  CERRA predictors: listed in config.VARIABLE_CERRA_FEATURES (variable-specific)
  Geographic:       month_sin, month_cos, lat, lon, elevation_m,
                    coast_dist_km, is_island  (config.GEO_FEATURES)

SIMILARITY WEIGHTS
------------------
Stations closer (climatically and geographically) to Delos receive higher
sample weights in LightGBM training.  The weights come from similarity.py
and are stored in the 'similarity_weight' column of the training DataFrame.

HYPERPARAMETERS
---------------
See _LGB_BASE and _LGB_MEAN.  The wind/gust mean model uses finer-grained
trees to capture the steep spatial gradient introduced by coast_dist_km
without creating a hard boundary artefact in the output maps.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from downscaling.config import GEO_FEATURES, QUANTILES, VARIABLE_CERRA_FEATURES

# ---------------------------------------------------------------------------
# LightGBM hyperparameters
# ---------------------------------------------------------------------------
# Base params shared by most quantile models.
# verbosity=-1 suppresses LightGBM's per-iteration console output.
_LGB_BASE = dict(
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    n_estimators=500,
    colsample_bytree=0.8,
    subsample=0.8,
    verbosity=-1,
    num_threads=-1,   # use all available CPU cores
)

# Wind/gust mean model: more leaves and more estimators to resolve the
# fine-scale spatial structure driven by coast_dist_km.
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

# Minimum training rows per variable; variables with fewer rows are skipped
_MIN_ROWS = 1000

# Variables that get a mean-regression model in addition to the three quantiles.
# The mean models reduce spatial artefacts in the wind/gust output maps.
_MEAN_VARS = {"wind", "gust"}

# Map pipeline variable name → observation column in the training DataFrame
_TARGET_COL = {
    "tmean":  "tmean_c",
    "tmin":   "tmin_c",
    "tmax":   "tmax_c",
    "rh":     "rh_pct",
    "precip": "prcp_mm",
    "wind":   "wdsp_ms",
    "gust":   "gust_ms",
}


def build_training_df(
    stations: pd.DataFrame,
    cerra: pd.DataFrame,
    geo: pd.DataFrame,
    similarity: pd.Series,
) -> pd.DataFrame:
    """
    Merge station observations, CERRA predictors, geographic features, and
    similarity weights into one flat training DataFrame.

    Parameters
    ----------
    stations   : station obs DataFrame from station_ingest.load_stations()
    cerra      : CERRA-at-stations DataFrame from cerra_extract.extract_all()
    geo        : geographic features from geo_features.compute_geo_features()
    similarity : per-station weights from similarity.compute_similarity()

    Returns
    -------
    DataFrame with one row per station × day, containing all feature columns
    and the target variable columns.
    """
    # CERRA and geo DataFrames use 'id' as the station key; align with 'station_id'
    cerra_renamed = cerra.rename(columns={"id": "station_id"})
    geo_renamed   = geo.rename(  columns={"id": "station_id"})

    df = (
        stations
        .merge(cerra_renamed, on=["station_id", "date"], how="inner",
               suffixes=("", "_cerra"))
        .merge(
            geo_renamed[["station_id", "coast_dist_km", "is_island"]],
            on="station_id", how="left", suffixes=("", "_geo"),
        )
    )

    # Add similarity weight; fill missing stations with uniform weight = 1
    df["similarity_weight"] = df["station_id"].map(similarity).fillna(1.0)

    # Harmonic month encoding: encodes month 1..12 as a smooth circular feature
    # so December–January is continuous (no discontinuity at year boundary).
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)

    return df


def train_all(
    training_df: pd.DataFrame,
    model_dir: Path,
    vars: list[str] | None = None,
) -> dict[str, lgb.Booster]:
    """
    Train LightGBM models for all (or a subset of) climate variables.

    Models are saved to model_dir as <varname>_<tag>.txt files:
      <var>_p10.txt, <var>_p50.txt, <var>_p90.txt  — quantile models
      <var>_mean.txt                                 — mean model (wind/gust only)
      precip_occ.txt                                 — precipitation occurrence

    Parameters
    ----------
    training_df : training DataFrame from build_training_df()
    model_dir   : directory where model .txt files are written
    vars        : if not None, only train models for these variable names

    Returns
    -------
    dict mapping model name → lgb.Booster
    """
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    models: dict[str, lgb.Booster] = {}

    for var, target_col in _TARGET_COL.items():
        if vars is not None and var not in vars:
            continue
        if target_col not in training_df.columns:
            continue

        # Build feature list: CERRA predictors + geo features (drop missing cols)
        feature_cols = VARIABLE_CERRA_FEATURES[var] + GEO_FEATURES
        available    = [c for c in feature_cols if c in training_df.columns]

        sub = training_df.dropna(subset=[target_col])
        if len(sub) < _MIN_ROWS:
            continue

        X = sub[available].values
        y = sub[target_col].values
        w = sub["similarity_weight"].values

        if var == "precip":
            _train_precip(X, y, w, model_dir, models)
        else:
            # Three quantile models per variable
            for q in QUANTILES:
                name    = f"{var}_p{int(q * 100):02d}"
                booster = _fit_quantile(X, y, w, q)
                booster.save_model(str(model_dir / f"{name}.txt"))
                models[name] = booster
            # Extra mean model for wind/gust to smooth spatial artefacts
            if var in _MEAN_VARS:
                _train_mean(var, X, y, w, model_dir, models)

    return models


def _train_mean(var: str, X, y, w, model_dir, models) -> None:
    """Train a standard mean-regression LightGBM model (for wind/gust)."""
    params  = {**_LGB_MEAN, "objective": "regression"}
    n       = params.pop("n_estimators", 800)
    ds      = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=n)
    name    = f"{var}_mean"
    booster.save_model(str(model_dir / f"{name}.txt"))
    models[name] = booster


def _train_precip(X, y, w, model_dir, models):
    """
    Train the two-part precipitation model:
      1. Binary occurrence classifier (all days)
      2. Quantile regressors (wet days only, precip > 0.1 mm)
    """
    # --- Occurrence model (P(wet) = P(precip > 0.1 mm)) ----------------------
    occ    = (y > 0.1).astype(float)
    params = {**_LGB_BASE, "objective": "binary"}
    n_boost = params.pop("n_estimators", 500)
    ds     = lgb.Dataset(X, label=occ, weight=w, free_raw_data=False)
    booster = lgb.train(params, ds, num_boost_round=n_boost)
    booster.save_model(str(model_dir / "precip_occ.txt"))
    models["precip_occ"] = booster

    # --- Conditional quantile models (wet-day subsample) ----------------------
    wet = y > 0.1
    if wet.sum() < 50:
        return   # too few wet days to fit a reliable quantile model
    X_wet, y_wet, w_wet = X[wet], y[wet], w[wet]
    for q in QUANTILES:
        name    = f"precip_p{int(q * 100):02d}"
        booster = _fit_quantile(X_wet, y_wet, w_wet, q)
        booster.save_model(str(model_dir / f"{name}.txt"))
        models[name] = booster


def _fit_quantile(X, y, w, q: float) -> lgb.Booster:
    """
    Fit one LightGBM quantile-regression model at probability level q.

    LightGBM's 'quantile' objective minimises the pinball (quantile) loss:
        L = (y − ŷ) × (q − 1_{y < ŷ})
    which produces a calibrated conditional quantile estimator.
    """
    params = {**_LGB_BASE, "objective": "quantile", "alpha": q}
    ds     = lgb.Dataset(X, label=y, weight=w, free_raw_data=False)
    return lgb.train(params, ds, num_boost_round=params.pop("n_estimators", 500))
