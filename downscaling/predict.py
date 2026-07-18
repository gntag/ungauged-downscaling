"""
predict.py
==========
Apply trained per-zone LightGBM models to the Delos target grid and write
the zone-ensemble output as a Zarr store.

ZONE ENSEMBLE PREDICTION
------------------------
For each Delos point p and each climate variable, N_ZONES independent models
are evaluated.  The final prediction is the membership-weighted average:

    prediction[p] = Σ_k (delos_membership_k[p] × zone_k_prediction[p])

where delos_membership_k[p] is the soft zone membership computed in
similarity.compute_zone_memberships().  This avoids hard zone boundaries and
produces spatially smooth output even where zones overlap.

The zone memberships are computed once from the Delos geographic feature
matrix and stored as weights in a (n_times × n_points) weighted average.

LAPSE CORRECTION IN PREDICTION
--------------------------------
The same delta_elev_m and lapse_correction_c features used at training time
are recomputed here from the Delos CERRA and geo feature DataFrames, ensuring
consistency between training and prediction.

INPUT
-----
  cerra_delos       : CERRA values at all Delos grid points (from cerra_extract.py)
  delos_geo         : Geographic + CERRA-static features for Delos points
  delos_memberships : Zone memberships (n_points × n_zones) from similarity.py
  model_dir         : Directory containing <name>.txt LightGBM model files
  delos_csv_path    : CSV with Delos grid coordinates (lon, lat, elevation_m)
  output_path       : Zarr store to write

OUTPUT
------
  Zarr store identical in structure to the original predict.py, with all
  predictions being the zone-weighted ensemble values.
"""
from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import zarr
from numcodecs import Blosc

from downscaling.bias_correct import blended_precip_eqm
from downscaling.config import (
    GEO_FEATURES,
    MONTHLY_LAPSE_RATES,
    PRECIP_EQM_BLEND,
    PRECIP_EQM_ENABLE,
    PRECIP_EQM_MIN_WET,
    PRECIP_EQM_STEP_DAYS,
    PRECIP_EQM_WINDOW_DAYS,
    VARIABLE_CERRA_FEATURES,
    VARIABLE_GEO_FEATURES,
)
from downscaling.similarity import N_ZONES

# ---------------------------------------------------------------------------
# Zarr output parameters
# ---------------------------------------------------------------------------
_ZARR_CHUNKS = (365, 50)
_COMPRESSOR  = Blosc(cname="zstd", clevel=5)

_ALL_VARS      = ["tmean", "tmin", "tmax", "rh", "precip", "wind", "gust"]
_QUANTILE_TAGS = ["p10", "p50", "p90"]
_MEAN_VARS     = {"wind", "gust"}

# Variables predicted with zone ensemble (must match train._ZONE_VARS).
# precip / wind / gust use single non-zone models for spatial coherence.
_ZONE_VARS     = {"tmean", "tmin", "tmax", "rh"}


def predict_delos(
    cerra_delos: pd.DataFrame,
    delos_geo: pd.DataFrame,
    model_dir: Path,
    delos_csv_path: Path,
    output_path: Path,
    delos_memberships: pd.DataFrame | None = None,
    donor_obs: np.ndarray | None = None,
    donor_cerra: np.ndarray | None = None,
    donor_doy: np.ndarray | None = None,
) -> None:
    """
    Run all trained zone models over the Delos feature matrix and write the
    zone-weighted ensemble predictions as a Zarr store.

    Parameters
    ----------
    cerra_delos        : CERRA values at Delos points (id, date, cerra_* columns)
    delos_geo          : Geographic + CERRA-static features (id, lat, lon, …)
    model_dir          : Directory with .txt LightGBM model files
    delos_csv_path     : CSV with lon, lat, VALUE (elevation) for the Delos grid
    output_path        : Path for the output .zarr store
    delos_memberships  : Optional (n_points × n_zones) DataFrame of zone weights;
                         if None, equal weights (1/N_ZONES) are used for all zones.
    donor_obs          : Optional (m,) array of station observed daily precip (mm)
                         used as the donor pool for the precipitation EQM
                         correction (precip_best_eqm).  If None, EQM is skipped.
    donor_cerra        : Optional (m,) CERRA precip paired with donor_obs (same
                         rows) — the QM(CERRA→obs) transfer input.
    donor_doy          : Optional (m,) day-of-year for each donor_obs sample.
    """
    model_dir  = Path(model_dir)
    models     = _load_models(model_dir)

    delos_coords = pd.read_csv(delos_csv_path)[["lon", "lat", "VALUE"]].rename(
        columns={"VALUE": "elevation_m"}
    )

    # --- Build the full feature matrix ------------------------------------
    geo_cols = ["id"] + [c for c in delos_geo.columns if c != "id"]
    feat_df  = (
        cerra_delos
        .drop_duplicates(subset=["date", "id"])
        .sort_values(["date", "id"])
        .reset_index(drop=True)
        .merge(delos_geo[geo_cols].drop_duplicates("id"), on="id", how="left")
    )

    dates     = pd.DatetimeIndex(sorted(feat_df["date"].unique()))
    point_ids = sorted(feat_df["id"].unique())
    n_times, n_points = len(dates), len(point_ids)

    # Harmonic month encoding
    feat_df["month_sin"] = np.sin(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)
    feat_df["month_cos"] = np.cos(2 * np.pi * pd.to_datetime(feat_df["date"]).dt.month / 12)

    # Lapse-rate derived features (Propositions 1 & 2)
    if "cerra_elev_m" in feat_df.columns:
        feat_df["delta_elev_m"] = feat_df["elevation_m"] - feat_df["cerra_elev_m"]
    else:
        feat_df["delta_elev_m"] = 0.0

    month_lapse = pd.to_datetime(feat_df["date"]).dt.month.map(MONTHLY_LAPSE_RATES)
    feat_df["lapse_correction_c"] = feat_df["delta_elev_m"] * month_lapse / 1000.0

    # Gust factor — separates synoptic flow magnitude from boundary-layer amplification
    if "cerra_gust" in feat_df.columns and "cerra_wind" in feat_df.columns:
        feat_df["gust_factor_cerra"] = (
            feat_df["cerra_gust"] / feat_df["cerra_wind"].replace(0.0, np.nan)
        ).clip(1.0, 10.0).fillna(1.0)
    else:
        feat_df["gust_factor_cerra"] = 1.0

    # Wind-terrain alignment — projects synoptic flow onto local upslope direction
    if "cerra_u10" in feat_df.columns and "dem_eastness" in feat_df.columns:
        feat_df["wind_terrain_align"] = (
            feat_df["cerra_u10"] * feat_df["dem_eastness"]
            + feat_df["cerra_v10"] * feat_df["dem_northness"]
        )
    else:
        feat_df["wind_terrain_align"] = 0.0

    # --- Zone membership weights (n_points,) per zone --------------------
    if delos_memberships is not None:
        # Align to point_ids ordering
        mem_indexed = delos_memberships.reindex(point_ids)
        zone_weights_arr = np.array([
            mem_indexed[f"zone_{k}"].fillna(1.0 / N_ZONES).values
            for k in range(N_ZONES)
        ])   # shape (N_ZONES, n_points)
    else:
        zone_weights_arr = np.full((N_ZONES, n_points), 1.0 / N_ZONES)

    # Normalise so zone weights sum to 1 per point
    zone_weight_sums = zone_weights_arr.sum(axis=0, keepdims=True)
    zone_weight_sums = np.where(zone_weight_sums == 0, 1.0, zone_weight_sums)
    zone_weights_arr = zone_weights_arr / zone_weight_sums  # (N_ZONES, n_points)

    # --- Predict each variable -------------------------------------------
    data_vars: dict[str, np.ndarray] = {}

    for var in _ALL_VARS:
        feature_cols = VARIABLE_CERRA_FEATURES[var] + VARIABLE_GEO_FEATURES.get(var, GEO_FEATURES)
        available    = [c for c in feature_cols if c in feat_df.columns]
        if not available:
            continue

        X = feat_df[available].values   # (n_times * n_points, n_features)

        if var in _ZONE_VARS:
            for qtag in _QUANTILE_TAGS:
                arr = _predict_ensemble(
                    X, models, n_times, n_points,
                    f"{var}_{qtag}", zone_weights_arr,
                )
                if arr is not None:
                    arr = _clip_variable(var, qtag, arr)
                    data_vars[f"{var}_{qtag}"] = arr
        elif var == "precip":
            has_zone_models = any(f"precip_occ_zone{k}" in models for k in range(N_ZONES))
            if has_zone_models:
                _predict_precip_zones(X, models, n_times, n_points,
                                      zone_weights_arr, data_vars)
            else:
                _predict_precip_single(X, models, n_times, n_points, data_vars)
        else:
            for qtag in _QUANTILE_TAGS:
                arr = _predict_single(X, models, n_times, n_points, f"{var}_{qtag}")
                if arr is not None:
                    arr = _clip_variable(var, qtag, arr)
                    data_vars[f"{var}_{qtag}"] = arr
            if var in _MEAN_VARS:
                arr = _predict_single(X, models, n_times, n_points, f"{var}_mean")
                if arr is not None:
                    data_vars[f"{var}_mean"] = np.clip(arr, 0, None)

    _enforce_temp_ordering(data_vars)
    _enforce_quantile_ordering(data_vars)
    target_cerra = None
    if "cerra_precip" in feat_df.columns:
        target_cerra = feat_df["cerra_precip"].values.reshape(n_times, n_points)
    _apply_precip_eqm_delos(data_vars, dates, target_cerra,
                            donor_obs, donor_cerra, donor_doy)
    _write_zarr(data_vars, dates, delos_coords, output_path)


def _apply_precip_eqm_delos(
    data_vars: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    target_cerra: np.ndarray | None,
    donor_obs: np.ndarray | None,
    donor_cerra: np.ndarray | None,
    donor_doy: np.ndarray | None,
) -> None:
    """
    Add ``precip_best_eqm`` — the final blended precipitation — to ``data_vars``
    (in place), one column per Delos point:

        precip_best_eqm = blend · QM_seasonal(CERRA → obs) + (1 − blend) · precip_best

    QM(CERRA) supplies day-matched timing + distribution correction; precip_best
    (occ × mean) supplies sub-grid spatial structure.  No-op if EQM is disabled,
    donors / target CERRA are missing, or precip_best is absent.
    See PR-next-EQM in ai_context/ai_context.md.
    """
    if not PRECIP_EQM_ENABLE:
        return
    if (donor_obs is None or donor_cerra is None or donor_doy is None
            or len(donor_obs) == 0 or target_cerra is None):
        return
    if "precip_best" not in data_vars:
        return

    src  = data_vars["precip_best"]                # (n_times, n_points)
    doys = pd.DatetimeIndex(dates).dayofyear.values

    d_obs   = np.asarray(donor_obs, dtype=float)
    d_cerra = np.asarray(donor_cerra, dtype=float)
    d_doy   = np.asarray(donor_doy)
    keep = np.isfinite(d_obs) & np.isfinite(d_cerra)
    d_obs, d_cerra, d_doy = d_obs[keep], d_cerra[keep], d_doy[keep]

    n_points = src.shape[1]
    out = np.empty_like(src, dtype="float32")
    for p in range(n_points):
        out[:, p] = blended_precip_eqm(
            target_cerra[:, p], src[:, p], doys,
            d_obs, d_cerra, d_doy,
            alpha=PRECIP_EQM_BLEND,
            window_days=PRECIP_EQM_WINDOW_DAYS,
            step_days=PRECIP_EQM_STEP_DAYS,
            min_wet=PRECIP_EQM_MIN_WET,
        )
    data_vars["precip_best_eqm"] = out


# ---------------------------------------------------------------------------
# Zone-ensemble prediction helpers
# ---------------------------------------------------------------------------

def _predict_ensemble(
    X: np.ndarray,
    models: dict[str, lgb.Booster],
    n_times: int,
    n_points: int,
    base_name: str,
    zone_weights_arr: np.ndarray,
) -> np.ndarray | None:
    """
    Compute the zone-weighted ensemble prediction for one output variable.

    Looks for models named <base_name>_zone0 … <base_name>_zone{N-1}.
    Returns None if no zone models exist.  Falls back to any zone model
    that is available (others contribute zero weight).
    """
    accumulated = None
    weight_sum  = np.zeros(n_points, dtype=np.float32)

    for k in range(N_ZONES):
        name = f"{base_name}_zone{k}"
        if name not in models:
            continue
        raw = models[name].predict(X).reshape(n_times, n_points).astype("float32")
        w   = zone_weights_arr[k].astype("float32")  # (n_points,)

        if accumulated is None:
            accumulated = raw * w[np.newaxis, :]
        else:
            accumulated = accumulated + raw * w[np.newaxis, :]
        weight_sum += w

    if accumulated is None:
        return None

    # Divide by total weight (handles zones where all models are missing)
    safe_ws = np.where(weight_sum == 0, 1.0, weight_sum).astype("float32")
    return accumulated / safe_ws[np.newaxis, :]


def _predict_single(
    X: np.ndarray,
    models: dict[str, lgb.Booster],
    n_times: int,
    n_points: int,
    name: str,
) -> np.ndarray | None:
    """Predict using a single non-zone model."""
    if name not in models:
        return None
    return models[name].predict(X).reshape(n_times, n_points).astype("float32")


def _predict_precip_single(
    X: np.ndarray,
    models: dict[str, lgb.Booster],
    n_times: int,
    n_points: int,
    data_vars: dict,
) -> None:
    """Run the precipitation model suite using single non-zone models."""
    occ = _predict_single(X, models, n_times, n_points, "precip_occ")
    if occ is not None:
        data_vars["precip_occurrence_prob"] = np.clip(occ, 0, 1)

    # Quantiles p10/p50/p90 plus p95/p99 upper-tail uncertainty bands
    for qtag in _QUANTILE_TAGS + ["p95", "p99"]:
        arr = _predict_single(X, models, n_times, n_points, f"precip_{qtag}")
        if arr is not None:
            data_vars[f"precip_{qtag}"] = np.clip(arr, 0, None)

    arr = _predict_single(X, models, n_times, n_points, "precip_mean")
    if arr is not None:
        data_vars["precip_mean"] = np.clip(arr, 0, None)

    _compute_precip_best(data_vars)


def _predict_precip_zones(
    X: np.ndarray,
    models: dict[str, lgb.Booster],
    n_times: int,
    n_points: int,
    zone_weights_arr: np.ndarray,
    data_vars: dict,
) -> None:
    """Run the precipitation model suite as a zone-weighted ensemble."""
    occ = _predict_ensemble(X, models, n_times, n_points,
                            "precip_occ", zone_weights_arr)
    if occ is not None:
        data_vars["precip_occurrence_prob"] = np.clip(occ, 0, 1)

    for qtag in _QUANTILE_TAGS + ["p95", "p99"]:
        arr = _predict_ensemble(X, models, n_times, n_points,
                                f"precip_{qtag}", zone_weights_arr)
        if arr is not None:
            data_vars[f"precip_{qtag}"] = np.clip(arr, 0, None)

    arr = _predict_ensemble(X, models, n_times, n_points,
                            "precip_mean", zone_weights_arr)
    if arr is not None:
        data_vars["precip_mean"] = np.clip(arr, 0, None)

    _compute_precip_best(data_vars)


def _compute_precip_best(data_vars: dict) -> None:
    """
    Compute precip_best: the Hurdle-model amount estimate occ_prob × precip_mean
    (E[Y] = P(Y>0) × E[Y|Y>0]).

    This is the *source* the final precip_best_eqm blends with QM(CERRA); on its
    own it is variance-compressed and underestimates totals — do not use it as the
    final field. See _apply_precip_eqm_delos / bias_correct.blended_precip_eqm.
    """
    occ  = data_vars.get("precip_occurrence_prob")
    mean = data_vars.get("precip_mean")

    if occ is None or mean is None:
        return

    data_vars["precip_best"] = np.clip(occ * mean, 0, None)


# ---------------------------------------------------------------------------
# Physical consistency checks
# ---------------------------------------------------------------------------

def _clip_variable(var: str, qtag: str, arr: np.ndarray) -> np.ndarray:
    """Apply physical bounds to predicted values."""
    if var == "rh":
        return np.clip(arr, 0, 100)
    if var in ("wind", "gust"):
        return np.clip(arr, 0, None)
    return arr


def _enforce_temp_ordering(data_vars: dict) -> None:
    """Ensure tmin ≤ tmean ≤ tmax within each quantile tag."""
    for qtag in _QUANTILE_TAGS:
        tmin_k, tmean_k, tmax_k = f"tmin_{qtag}", f"tmean_{qtag}", f"tmax_{qtag}"
        if tmean_k in data_vars and tmin_k in data_vars:
            data_vars[tmin_k] = np.minimum(data_vars[tmin_k], data_vars[tmean_k])
        if tmean_k in data_vars and tmax_k in data_vars:
            data_vars[tmax_k] = np.maximum(data_vars[tmax_k], data_vars[tmean_k])


def _enforce_quantile_ordering(data_vars: dict) -> None:
    """Ensure p10 ≤ p50 ≤ p90 within each variable."""
    for var in _ALL_VARS:
        p10, p50, p90 = f"{var}_p10", f"{var}_p50", f"{var}_p90"
        if not all(k in data_vars for k in [p10, p50, p90]):
            continue
        stack = np.stack([data_vars[p10], data_vars[p50], data_vars[p90]], axis=0)
        stack = np.sort(stack, axis=0)
        data_vars[p10], data_vars[p50], data_vars[p90] = stack[0], stack[1], stack[2]


# ---------------------------------------------------------------------------
# Model loading and Zarr writing
# ---------------------------------------------------------------------------

def _load_models(model_dir: Path) -> dict[str, lgb.Booster]:
    """Load all .txt LightGBM model files from model_dir."""
    models = {}
    for txt in model_dir.glob("*.txt"):
        name = txt.stem
        try:
            models[name] = lgb.Booster(model_file=str(txt))
        except Exception:
            pass
    return models


def _write_zarr(
    data_vars: dict[str, np.ndarray],
    dates: pd.DatetimeIndex,
    delos_coords: pd.DataFrame,
    output_path: Path,
) -> None:
    """Write coordinate arrays and all predicted variables to a new Zarr store."""
    n_times  = len(dates)
    n_points = len(delos_coords)

    store = zarr.open(str(output_path), mode="w")

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

    _LONG_NAMES = {
        "precip_occurrence_prob": "precipitation occurrence probability (P(prcp > 0.1 mm))",
        "precip_p10":        "conditional precipitation p10 — wet days only",
        "precip_p50":        "conditional precipitation p50 — wet days only",
        "precip_p90":        "conditional precipitation p90 — wet days only",
        "precip_mean":        "conditional precipitation mean — Gamma GLM, wet days only",
        "precip_p95":         "conditional precipitation p95 — wet days only (upper-tail band)",
        "precip_p99":         "conditional precipitation p99 — wet days only (upper-tail band)",
        "precip_best":        "amount source occ_prob x precip_mean (Hurdle E[Y]); blended into precip_best_eqm",
        "precip_best_eqm":    "final daily precipitation: blend of QM(CERRA->obs) and precip_best",
    }

    for name, arr in data_vars.items():
        ds = store.create_dataset(name, data=arr, dtype="float32",
                                  chunks=(chunk_t, chunk_p),
                                  compressor=_COMPRESSOR)
        if name in _LONG_NAMES:
            ds.attrs["long_name"] = _LONG_NAMES[name]
