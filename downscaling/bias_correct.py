"""
bias_correct.py
===============
Empirical Quantile Mapping (EQM) bias correction — core library.

SCIENTIFIC BASIS
----------------
EQM (also called EDCDF or quantile delta mapping in the literature) maps each
quantile of a model's distribution to the corresponding quantile of the
observed distribution, estimated from the historical calibration period.

The transfer function T(x) is:
    T(x) = Q_obs(F_mdl(x))

where F_mdl is the model CDF and Q_obs is the observed quantile function,
both estimated empirically from the calibration sample.

For values outside the training range of the model (extrapolation), a "delta"
correction is applied:
    T(x) = x + (obs_mean − mdl_mean)

This adds the mean bias as a fallback rather than clipping or extrapolating the
empirical CDF, which can produce artefacts.

For precipitation, a wet-day correction is applied before QM (Gudmundsson et al.
2012 §2.2):
  1. Compute the wet-day fraction f_wet in observations (days > wet_threshold).
  2. Find the model quantile Q_mdl(1 − f_wet) — the model's "dry-day threshold".
  3. Map model values ≤ dry-day threshold to 0 (dry).
  4. Apply QM only to the remaining "wet" values.

The vectorised grid variants (fit_eqm_grid / apply_eqm_grid) process all Delos
grid points simultaneously using (n_days × n_points) NumPy arrays for efficiency.

REFERENCES
----------
  Gudmundsson et al. (2012). HESS 16(9) — the qmap::fitQmapQUANT method.
  Themessl et al. (2011). Int. J. Climatology 31(10) — per-DOY window approach.
  Wood et al. (2004). Climatic Change 62 — QM applied to climate projections.
"""
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Transfer function container
# ---------------------------------------------------------------------------

class EQMTransfer:
    """
    Fitted EQM transfer function for one variable / one DOY window / one point.

    Attributes
    ----------
    mdl_quantiles : (n_q,) array — model quantiles (x-axis of the mapping)
    obs_quantiles : (n_q,) array — corresponding observed quantiles (y-axis)
    obs_mean      : float — mean of observed values in the calibration window
    mdl_mean      : float — mean of model values in the calibration window
    wet_mdl_threshold : float — model precipitation threshold below which days
                       are treated as dry (0 for non-precipitation variables)
    """

    def __init__(self, mdl_quantiles, obs_quantiles, obs_mean, mdl_mean, wet_mdl_threshold=0.0):
        self.mdl_quantiles = mdl_quantiles
        self.obs_quantiles = obs_quantiles
        self.obs_mean = obs_mean
        self.mdl_mean = mdl_mean
        self.wet_mdl_threshold = wet_mdl_threshold


# ---------------------------------------------------------------------------
# Calendar utilities
# ---------------------------------------------------------------------------

def doy_window_mask(
    doys: np.ndarray,
    target: int,
    window: int = 15,
    cal_len: int = 365,
) -> np.ndarray:
    """
    Boolean mask for days within ±window of target DOY, wrapping at year boundary.

    Uses circular distance so the window wraps correctly at day 1 / day 365:
        dist = min(|d − target|, cal_len − |d − target|) ≤ window

    This avoids a discontinuity in the QM transfer function around New Year's Day.

    Parameters
    ----------
    doys    : array of integer DOY values (1-based)
    target  : centre DOY of the window
    window  : half-width in days (default 15 → ±15, 31-day window)
    cal_len : calendar length (365 for Gregorian, 360 for HadGEM2)
    """
    diff = np.abs(doys.astype(int) - target)
    return np.minimum(diff, cal_len - diff) <= window


def map_360_to_365(doy360: np.ndarray) -> np.ndarray:
    """
    Map HadGEM2 360-day DOY values to the nearest Gregorian equivalent.

    HadGEM2 uses a uniform 30-day month calendar; its DOY 1..360 does not align
    with Gregorian DOY 1..365.  We convert via fractional year position:
        frac   = (d360 − 0.5) / 360
        d365   = round(frac × 365 + 0.5),  clipped to [1, 365]

    The −0.5/+0.5 shifts put the DOY at the midpoint of each day.

    This mapping is never 1:1 — multiple HadGEM2 DOYs can map to the same
    Gregorian DOY, which is correct: HadGEM2's 360 days spread uniformly over
    the Gregorian 365-day year.
    """
    frac = (np.asarray(doy360, dtype=float) - 0.5) / 360.0
    d365 = np.round(frac * 365.0 + 0.5).astype(int)
    return np.clip(d365, 1, 365)


# ---------------------------------------------------------------------------
# Single-point EQM fit and apply
# ---------------------------------------------------------------------------

def fit_eqm(
    obs: np.ndarray,
    mdl: np.ndarray,
    n_quantiles: int = 250,
    is_precip: bool = False,
    wet_threshold: float = 0.1,
) -> EQMTransfer:
    """
    Fit an EQM transfer function from model distribution to observed distribution.

    Parameters
    ----------
    obs           : 1-D array of observed values in the calibration DOY window
    mdl           : 1-D array of model  values in the calibration DOY window
    n_quantiles   : number of equally spaced probability levels to estimate
    is_precip     : if True, apply wet-day correction before fitting
    wet_threshold : daily precipitation threshold for "wet day" (mm; WMO default 0.1)

    Returns
    -------
    EQMTransfer fitted to this (obs, mdl) pair
    """
    obs_clean = obs[np.isfinite(obs)]
    mdl_clean = mdl[np.isfinite(mdl)]

    obs_mean = float(np.mean(obs_clean)) if len(obs_clean) > 0 else 0.0
    mdl_mean = float(np.mean(mdl_clean)) if len(mdl_clean) > 0 else 0.0

    wet_mdl_threshold = 0.0

    if is_precip:
        # Wet-day fraction in observations drives the model's dry-day threshold
        f_wet = float(np.mean(obs_clean > wet_threshold)) if len(obs_clean) > 0 else 0.0
        if f_wet > 0.0 and len(mdl_clean) > 0:
            # The (1 − f_wet) quantile of the model separates its dry and wet days
            wet_mdl_threshold = float(np.nanquantile(mdl_clean, max(0.0, 1.0 - f_wet)))
        # Restrict QM fitting to wet days only in both datasets
        obs_clean = obs_clean[obs_clean > wet_threshold]
        mdl_clean = mdl_clean[mdl_clean > wet_mdl_threshold]

    q_probs = np.linspace(0.0, 1.0, n_quantiles)

    if len(obs_clean) >= 2 and len(mdl_clean) >= 2:
        obs_qs = np.quantile(obs_clean, q_probs)
        mdl_qs = np.quantile(mdl_clean, q_probs)
    else:
        # Degenerate case (too few wet days): fall back to identity mapping
        combined = (
            np.concatenate([obs_clean, mdl_clean])
            if len(obs_clean) + len(mdl_clean) > 0
            else np.array([0.0, 1.0])
        )
        lo, hi = float(np.min(combined)), float(np.max(combined))
        mdl_qs = np.linspace(lo, hi, n_quantiles)
        obs_qs = mdl_qs.copy()

    return EQMTransfer(
        mdl_quantiles=mdl_qs,
        obs_quantiles=obs_qs,
        obs_mean=obs_mean,
        mdl_mean=mdl_mean,
        wet_mdl_threshold=wet_mdl_threshold,
    )


def apply_eqm(
    values: np.ndarray,
    transfer: EQMTransfer,
    is_precip: bool = False,
) -> np.ndarray:
    """
    Apply an EQM transfer function to a 1-D array (single grid point).

    For in-range values: linear interpolation between the stored quantile pairs.
    For out-of-range values: delta correction (obs_mean − mdl_mean).
    For precipitation: values ≤ wet_mdl_threshold are set to 0 before QM, and
    the result is clipped to ≥ 0.

    Returns an array of the same shape as 'values', with NaN where input is NaN.
    """
    result = np.full_like(values, np.nan, dtype=float)
    valid = np.isfinite(values)
    if not np.any(valid):
        return result

    v = values[valid].astype(float)

    if is_precip:
        dry = v <= transfer.wet_mdl_threshold
        corrected = np.where(
            dry,
            0.0,
            _interp_delta(v, transfer.mdl_quantiles, transfer.obs_quantiles,
                          transfer.obs_mean - transfer.mdl_mean),
        )
        corrected = np.maximum(corrected, 0.0)
    else:
        corrected = _interp_delta(
            v, transfer.mdl_quantiles, transfer.obs_quantiles,
            transfer.obs_mean - transfer.mdl_mean,
        )

    result[valid] = corrected
    return result


def _interp_delta(
    values: np.ndarray,
    mdl_q: np.ndarray,
    obs_q: np.ndarray,
    delta: float,
) -> np.ndarray:
    """
    Interpolate values within the training range; delta-correct outside it.

    np.interp handles in-range values with linear interpolation.  Values below
    mdl_q[0] or above mdl_q[-1] get the constant delta offset instead of
    extrapolating the empirical CDF, which can diverge badly.
    """
    mapped = np.interp(values, mdl_q, obs_q)
    mapped[values < mdl_q[0]]  = values[values < mdl_q[0]]  + delta
    mapped[values > mdl_q[-1]] = values[values > mdl_q[-1]] + delta
    return mapped


# ---------------------------------------------------------------------------
# Vectorised multi-point variants (used by cordex_bias_correct.py)
# ---------------------------------------------------------------------------


def fit_eqm_grid(
    obs_window: np.ndarray,
    mdl_window: np.ndarray,
    n_quantiles: int = 250,
    is_precip: bool = False,
    wet_threshold: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit EQM for all grid points simultaneously for one DOY window.

    Rather than calling fit_eqm() in a Python loop over points, this function
    uses np.nanquantile with axis=0 to compute all point quantiles at once,
    dramatically reducing overhead for large grids.

    Precipitation wet-day correction still loops over points because each point
    may have a different wet-day fraction and dry-day threshold.

    Parameters
    ----------
    obs_window : (n_obs_days, n_pts) — observed values in the DOY window
    mdl_window : (n_mdl_days, n_pts) — model    values in the DOY window

    Returns
    -------
    mdl_qs   : (n_q, n_pts) model quantile matrix
    obs_qs   : (n_q, n_pts) observed quantile matrix
    obs_mean : (n_pts,) observed mean per point
    mdl_mean : (n_pts,) model    mean per point
    wet_thr  : (n_pts,) per-point dry-day threshold (0 for non-precipitation)
    """
    n_pts   = obs_window.shape[1]
    q_probs = np.linspace(0.0, 1.0, n_quantiles)

    obs_mean = np.nanmean(obs_window, axis=0)
    mdl_mean = np.nanmean(mdl_window, axis=0)
    wet_thr  = np.zeros(n_pts)

    if not is_precip:
        # Non-precipitation variables: vectorised quantile computation for all points
        obs_qs = np.nanquantile(obs_window, q_probs, axis=0)
        mdl_qs = np.nanquantile(mdl_window, q_probs, axis=0)
        return mdl_qs, obs_qs, obs_mean, mdl_mean, wet_thr

    # Precipitation: compute per-point wet-day threshold (must loop)
    f_wet = np.nanmean(obs_window > wet_threshold, axis=0)
    for p in range(n_pts):
        col = mdl_window[:, p]
        col = col[np.isfinite(col)]
        if f_wet[p] > 0.0 and len(col) > 0:
            wet_thr[p] = float(np.nanquantile(col, max(0.0, 1.0 - f_wet[p])))

    obs_qs = np.zeros((n_quantiles, n_pts))
    mdl_qs = np.zeros((n_quantiles, n_pts))
    for p in range(n_pts):
        obs_col = obs_window[:, p]
        obs_wet = obs_col[np.isfinite(obs_col) & (obs_col > wet_threshold)]
        mdl_col = mdl_window[:, p]
        mdl_wet = mdl_col[np.isfinite(mdl_col) & (mdl_col > wet_thr[p])]

        if len(obs_wet) >= 2 and len(mdl_wet) >= 2:
            obs_qs[:, p] = np.quantile(obs_wet, q_probs)
            mdl_qs[:, p] = np.quantile(mdl_wet, q_probs)
        else:
            # Degenerate point: identity mapping
            combined = (
                np.concatenate([obs_wet, mdl_wet])
                if len(obs_wet) + len(mdl_wet) > 0
                else np.array([0.0, 1.0])
            )
            lo, hi = float(np.min(combined)), float(np.max(combined))
            mdl_qs[:, p] = np.linspace(lo, hi, n_quantiles)
            obs_qs[:, p] = mdl_qs[:, p].copy()

    return mdl_qs, obs_qs, obs_mean, mdl_mean, wet_thr


def apply_eqm_grid(
    values: np.ndarray,
    mdl_qs: np.ndarray,
    obs_qs: np.ndarray,
    obs_mean: np.ndarray,
    mdl_mean: np.ndarray,
    wet_thr: np.ndarray,
    is_precip: bool = False,
) -> np.ndarray:
    """
    Apply EQM to a (n_days, n_pts) block using per-point transfer functions.

    Each grid point uses its own column slice of mdl_qs / obs_qs and its own
    scalar values of obs_mean, mdl_mean, wet_thr.  A Python loop over points
    is unavoidable here because np.interp cannot vectorise across different xp
    arrays.  The number of points for Delos is small (~few thousand), so the
    loop is acceptably fast.

    Returns a (n_days, n_pts) float32 array.
    """
    result = np.full(values.shape, np.nan, dtype=float)
    delta  = obs_mean - mdl_mean   # (n_pts,) mean bias per point

    for p in range(values.shape[1]):
        v     = values[:, p].astype(float)
        valid = np.isfinite(v)
        if not np.any(valid):
            continue

        v_v       = v[valid]
        corrected = _interp_delta(v_v, mdl_qs[:, p], obs_qs[:, p], delta[p])

        if is_precip:
            dry       = v_v <= wet_thr[p]
            corrected = np.where(dry, 0.0, corrected)
            corrected = np.maximum(corrected, 0.0)

        r        = np.full(len(v), np.nan)
        r[valid] = corrected
        result[:, p] = r

    return result
