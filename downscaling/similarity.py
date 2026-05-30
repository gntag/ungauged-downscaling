"""
similarity.py
=============
Donor-station similarity weighting for the Delos statistical downscaling.

PURPOSE
-------
Not all weather stations in the region are equally representative of Delos.
A station at 200 m elevation on a continental mainland is a poor donor for a
low-lying Aegean island.  These weights down-weight such poor donors so that
stations that closely resemble the Delos environment contribute more to
LightGBM training.

SIMILARITY FACTORS
------------------
The weight of each station is the product of four Gaussian kernel terms:

  w_elev   : elevation similarity     — penalises large elevation differences
  w_coast  : coastal proximity         — rewards stations close to the coast
  w_clim   : monthly CERRA tmean RMSE — rewards stations with similar seasonal cycle
  w_island : is_island bonus           — multiplies island stations by _ISLAND_BONUS

The Gaussian bandwidth σ for each factor is its IQR across the donor pool
(Silverman's rule of thumb adapted to one predictor at a time).  Using the IQR
rather than a fixed value makes the weights robust to the range of the donor pool.

NORMALISATION
-------------
Weights are normalised so their mean equals 1.  This preserves the total
effective sample size and keeps LightGBM's learning rate calibration stable.

USAGE
-----
  from downscaling.similarity import compute_similarity
  weights = compute_similarity(station_features, delos_features)
  # weights is a pd.Series indexed by station_id
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Islands get this multiplicative bonus on top of the three Gaussian terms.
# Rationale: an island station (e.g. Mykonos, Syros) has marine exposure and
# summer dryness much more similar to Delos than a mainland coastal station.
_ISLAND_BONUS = 1.5

# Monthly CERRA tmean column names expected in both feature DataFrames
_CLIM_COLS = [f"cerra_tmean_m{m:02d}" for m in range(1, 13)]


def compute_similarity(
    station_features: pd.DataFrame,
    delos_features: pd.DataFrame,
) -> pd.Series:
    """
    Compute donor-similarity weights for each station relative to the Delos mean.

    Both DataFrames must contain: station_id, elevation_m, coast_dist_km,
    is_island, cerra_tmean_m01 … cerra_tmean_m12.

    The Delos target is represented as the point-wise mean of delos_features
    (all Delos grid points are averaged to a single "target" vector).

    Returns
    -------
    pd.Series indexed by station_id, normalised so mean == 1.
    """
    # Represent Delos as a single averaged target point
    delos = delos_features.mean(numeric_only=True)

    # --- Elevation: penalise large elevation differences ----------------------
    elev_diff = station_features["elevation_m"] - delos["elevation_m"]

    # --- Coastal proximity: penalise stations far from the coast --------------
    coast_diff = station_features["coast_dist_km"] - delos["coast_dist_km"]

    # --- Climatic similarity: RMSE of monthly tmean vs. Delos climatology ----
    clim_cols = [c for c in _CLIM_COLS if c in station_features.columns]
    if clim_cols:
        clim_diff = np.sqrt(
            ((station_features[clim_cols].values - delos[clim_cols].values) ** 2).sum(axis=1)
        )
    else:
        clim_diff = pd.Series(0.0, index=station_features.index)

    # --- Gaussian kernels (bandwidth = IQR of each predictor) -----------------
    w_elev   = _gaussian(elev_diff,                              _iqr(station_features["elevation_m"]))
    w_coast  = _gaussian(coast_diff,                             _iqr(station_features["coast_dist_km"]))
    w_clim   = _gaussian(pd.Series(clim_diff, index=station_features.index), _iqr(pd.Series(clim_diff)))
    w_island = station_features["is_island"].map({True: _ISLAND_BONUS, False: 1.0}).fillna(1.0)

    # --- Combined weight (product of all four terms) --------------------------
    raw  = w_elev * w_coast * w_clim * w_island
    mean = raw.mean()

    # Guard against degenerate case where all weights collapse to 0
    if mean == 0:
        return pd.Series(1.0, index=station_features.index)

    # Normalise so the mean weight is 1 (preserves total effective sample size)
    result = raw / mean
    result.index = station_features["station_id"].values
    return result


def _gaussian(diff: pd.Series, sigma: float) -> pd.Series:
    """
    Evaluate a zero-mean Gaussian kernel: exp(−diff² / (2σ²)).

    When σ = 0 (all stations have identical values), returns 1 everywhere
    (uniform weights — no information to discriminate on this axis).
    """
    if sigma == 0:
        return pd.Series(1.0, index=diff.index)
    return np.exp(-(diff ** 2) / (2 * sigma ** 2))


def _iqr(series: pd.Series) -> float:
    """
    Interquartile range (Q75 − Q25) as the Gaussian bandwidth.

    The IQR is more robust than the standard deviation for skewed distributions
    (e.g. elevation, coast distance), which prevents a few extreme stations from
    dominating the bandwidth estimate.
    """
    q75, q25 = np.nanpercentile(series, [75, 25])
    return float(q75 - q25)
