"""
similarity.py
=============
Donor-station similarity weighting and soft zone membership for the Delos
statistical downscaling pipeline.

OVERVIEW
--------
The module provides two related but distinct functions:

  compute_zone_memberships(features, delos_context)
      For each location (station or Delos point), compute a soft membership
      weight for each of the N_ZONES micro-climate zones defined in config.py.
      Memberships are Gaussian kernels in a normalised (coast, elevation,
      northness) space; they sum to 1 across zones for each point, so they
      function as fuzzy partition-of-unity weights.

  compute_zone_similarity(station_features, delos_features, delos_memberships)
      For each zone k, compute a per-station similarity weight that measures
      how representative each donor station is of zone k on Delos.
      The "target" for zone k is the membership-weighted centroid of all Delos
      grid points in that zone, not a simple mean.

      Similarity factors per zone are the same Gaussian kernels as before
      (elevation, coast distance, climatic RMSE, island bonus), but computed
      relative to the zone centroid rather than the mean of all Delos points.

ZONE ARCHITECTURE
-----------------
Zone definitions live in config.ZONE_DEFINITIONS.  Each zone is characterised
by a center and per-dimension sigma in the normalised space:
  coast_norm = 1 − coast_dist_km / ZONE_COAST_MAX_KM   (clipped to [0, 1])
  elev_norm  = elevation_m / ZONE_ELEV_MAX_M            (clipped to [0, 1])
  northness  = dem_northness or cerra_northness          (−1 to +1)

Using the geometric mean of per-dimension Gaussian kernels gives a smooth,
direction-aware zone membership that avoids hard boundaries.  A Delos shoreline
point at the north coast gets high membership in "N-Coastal" and low membership
in "S-Coastal" and "Ridge", with no discontinuous step at any boundary.

NORMALISATION
-------------
Within each zone k, station similarity weights are normalised so their mean
equals 1.  This preserves the total effective sample size and keeps LightGBM's
learning rate calibration stable.

USAGE
-----
  from downscaling.similarity import compute_zone_memberships, compute_zone_similarity

  # Delos zone memberships (n_delos_points × n_zones DataFrame)
  delos_memberships = compute_zone_memberships(delos_features, context="delos")

  # Station similarity per zone (dict: zone_idx → pd.Series indexed by station_id)
  zone_weights = compute_zone_similarity(station_features, delos_features,
                                         delos_memberships)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from downscaling.config import (
    ZONE_COAST_MAX_KM,
    ZONE_DEFINITIONS,
    ZONE_ELEV_MAX_M,
)

# Island stations receive this multiplicative similarity bonus because their
# marine exposure and summer dryness more closely resemble Delos than mainland
# coastal stations.
_ISLAND_BONUS = 1.5

# Monthly CERRA tmean column names expected in both feature DataFrames
_CLIM_COLS = [f"cerra_tmean_m{m:02d}" for m in range(1, 13)]

# Number of zones defined in config
N_ZONES = len(ZONE_DEFINITIONS)


# ---------------------------------------------------------------------------
# Soft zone membership
# ---------------------------------------------------------------------------

def compute_zone_memberships(
    features: pd.DataFrame,
    northness_col: str = "dem_northness",
) -> pd.DataFrame:
    """
    Compute soft Gaussian zone memberships for each location.

    Parameters
    ----------
    features      : DataFrame containing at least coast_dist_km, elevation_m,
                    and a northness column (dem_northness or cerra_northness).
                    Also requires a 'station_id' or 'id' column — whichever
                    is present is used as the index of the returned DataFrame.
    northness_col : Column to use as the northness axis. Falls back to
                    'cerra_northness' if the preferred column is absent.

    Returns
    -------
    DataFrame indexed by the location id with columns
    zone_0, zone_1, …, zone_{N_ZONES-1}.
    Each row sums to 1.0 (partition of unity).
    """
    if northness_col not in features.columns:
        northness_col = "cerra_northness" if "cerra_northness" in features.columns else None

    # Normalise input dimensions to [0, 1] / [−1, +1]
    # coast_dist_km is stored as raw km; log1p applied here keeps zone centres
    # in log1p-km units (ZONE_COAST_MAX_KM = 1.0 is in log1p-km).
    coast_dist_log1p = np.log1p(features["coast_dist_km"].clip(0))
    coast_norm = (1.0 - coast_dist_log1p.clip(0, ZONE_COAST_MAX_KM)
                  / ZONE_COAST_MAX_KM).values
    elev_norm  = (features["elevation_m"].clip(0, ZONE_ELEV_MAX_M)
                  / ZONE_ELEV_MAX_M).values

    if northness_col is not None:
        northness = features[northness_col].fillna(0.0).values
    else:
        northness = np.zeros(len(features), dtype=np.float64)

    # Compute raw (unnormalised) Gaussian membership for each zone
    raw = np.zeros((len(features), N_ZONES), dtype=np.float64)
    for k, zone in enumerate(ZONE_DEFINITIONS):
        c = zone["center"]
        s = zone["sigma"]
        raw[:, k] = (
            np.exp(-0.5 * ((coast_norm - c["coast_norm"]) / s["coast_norm"]) ** 2)
            * np.exp(-0.5 * ((elev_norm  - c["elev_norm"])  / s["elev_norm"])  ** 2)
            * np.exp(-0.5 * ((northness  - c["northness"])  / s["northness"])  ** 2)
        )

    # Normalise rows to sum to 1 (partition of unity)
    row_sums = raw.sum(axis=1, keepdims=True)
    # Guard against the degenerate case where all zones have zero membership
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    memberships = raw / row_sums

    id_col = "station_id" if "station_id" in features.columns else "id"
    zone_cols = {f"zone_{k}": memberships[:, k] for k in range(N_ZONES)}
    df = pd.DataFrame(zone_cols, index=features[id_col].values)
    df.index.name = id_col
    return df


# ---------------------------------------------------------------------------
# Per-zone similarity
# ---------------------------------------------------------------------------

def compute_zone_similarity(
    station_features: pd.DataFrame,
    delos_features: pd.DataFrame,
    delos_memberships: pd.DataFrame,
) -> dict[int, pd.Series]:
    """
    Compute per-zone donor-station similarity weights.

    For each zone k, the Delos "target" is defined as the membership-weighted
    centroid of all Delos grid points (more weight to points strongly belonging
    to zone k).  Station similarity is then computed relative to that centroid
    using Gaussian kernels on elevation, coast distance, and climatic RMSE.

    Parameters
    ----------
    station_features   : DataFrame with station_id, elevation_m, coast_dist_km,
                         is_island, cerra_tmean_m01 … m12.
    delos_features     : Same columns but for Delos grid points (id column).
    delos_memberships  : DataFrame from compute_zone_memberships(delos_features),
                         indexed by Delos id, columns zone_0 … zone_{N-1}.

    Returns
    -------
    dict mapping zone_index (int) → pd.Series of similarity weights indexed
    by station_id.  Each Series is normalised so mean == 1.
    """
    # Align Delos features with their zone memberships on id.
    # coast_dist_km is transformed to log1p space here so that zone centroids
    # are computed in the same space as compute_zone_memberships, which uses
    # log1p(coast_dist_km) for zone placement.
    delos_num = delos_features.rename(columns={"id": "station_id"}) \
                               .select_dtypes(include="number").copy()
    if "coast_dist_km" in delos_num.columns:
        delos_num["coast_dist_km"] = np.log1p(delos_num["coast_dist_km"].clip(0))

    zone_weights: dict[int, pd.Series] = {}

    for k in range(N_ZONES):
        zone_col    = f"zone_{k}"
        zone_w      = delos_memberships[zone_col].values  # (n_delos_points,)
        total_w     = zone_w.sum()

        if total_w == 0:
            # Degenerate zone: no Delos points belong to it — fall back to uniform
            zone_weights[k] = pd.Series(
                1.0, index=station_features["station_id"].values
            )
            continue

        # Weighted centroid of Delos numeric features for zone k (coast in log1p space)
        delos_centroid = (delos_num.multiply(zone_w / total_w, axis=0)).sum()

        # --- Elevation similarity -----------------------------------------
        elev_diff  = station_features["elevation_m"] - delos_centroid.get("elevation_m", 0)
        sigma_elev = _iqr(station_features["elevation_m"])
        w_elev     = _gaussian(elev_diff, sigma_elev)

        # --- Coastal proximity similarity (log1p space, consistent with zone membership) ---
        station_coast_log1p = np.log1p(station_features["coast_dist_km"].clip(0))
        coast_diff  = station_coast_log1p.values - delos_centroid.get("coast_dist_km", 0)
        sigma_coast = _iqr(station_coast_log1p)
        w_coast     = _gaussian(pd.Series(coast_diff, index=station_features.index), sigma_coast)

        # --- Climatic similarity (monthly tmean RMSE) ---------------------
        clim_cols = [c for c in _CLIM_COLS if c in station_features.columns]
        if clim_cols:
            target_clim = delos_centroid[clim_cols].values
            clim_diff   = np.sqrt(
                ((station_features[clim_cols].values - target_clim) ** 2).sum(axis=1)
            )
        else:
            clim_diff = np.zeros(len(station_features))
        sigma_clim = _iqr(pd.Series(clim_diff))
        w_clim     = _gaussian(pd.Series(clim_diff, index=station_features.index), sigma_clim)

        # --- Island bonus -------------------------------------------------
        w_island = (
            station_features["is_island"]
            .map({True: _ISLAND_BONUS, False: 1.0})
            .fillna(1.0)
        )

        # --- Combined weight (product) ------------------------------------
        raw  = w_elev * w_coast * w_clim * w_island
        mean = raw.mean()
        if mean == 0:
            raw = pd.Series(1.0, index=station_features.index)
            mean = 1.0

        result = raw / mean
        result.index = station_features["station_id"].values
        zone_weights[k] = result

    return zone_weights


# ---------------------------------------------------------------------------
# Backward-compatible single-target similarity (used for sanity checks)
# ---------------------------------------------------------------------------

def compute_similarity(
    station_features: pd.DataFrame,
    delos_features: pd.DataFrame,
) -> pd.Series:
    """
    Original single-target similarity (mean of all Delos points as target).

    Kept for backward compatibility and unit tests.  In the main pipeline,
    compute_zone_similarity() is used instead.
    """
    delos = delos_features.mean(numeric_only=True)

    elev_diff  = station_features["elevation_m"] - delos["elevation_m"]
    coast_diff = station_features["coast_dist_km"] - delos["coast_dist_km"]

    clim_cols = [c for c in _CLIM_COLS if c in station_features.columns]
    if clim_cols:
        clim_diff = np.sqrt(
            ((station_features[clim_cols].values - delos[clim_cols].values) ** 2).sum(axis=1)
        )
    else:
        clim_diff = pd.Series(0.0, index=station_features.index)

    w_elev   = _gaussian(elev_diff,                              _iqr(station_features["elevation_m"]))
    w_coast  = _gaussian(coast_diff,                             _iqr(station_features["coast_dist_km"]))
    w_clim   = _gaussian(pd.Series(clim_diff, index=station_features.index), _iqr(pd.Series(clim_diff)))
    w_island = station_features["is_island"].map({True: _ISLAND_BONUS, False: 1.0}).fillna(1.0)

    raw  = w_elev * w_coast * w_clim * w_island
    mean = raw.mean()
    if mean == 0:
        return pd.Series(1.0, index=station_features.index)

    result = raw / mean
    result.index = station_features["station_id"].values
    return result


# ---------------------------------------------------------------------------
# Kernel helpers
# ---------------------------------------------------------------------------

def _gaussian(diff: pd.Series, sigma: float) -> pd.Series:
    """Zero-mean Gaussian kernel: exp(−diff² / (2σ²))."""
    if sigma == 0:
        return pd.Series(1.0, index=diff.index)
    return np.exp(-(diff ** 2) / (2 * sigma ** 2))


def _iqr(series: pd.Series) -> float:
    """Interquartile range as a robust bandwidth estimate."""
    q75, q25 = np.nanpercentile(series, [75, 25])
    return float(q75 - q25)
