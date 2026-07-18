"""
geo_features.py
===============
Compute geographic predictor features for the Delos downscaling pipeline.

FEATURES COMPUTED
-----------------
Existing features:
  coast_dist_km : Euclidean distance to the nearest coastline (km), raw (not transformed).
                  Zone membership in similarity.py applies log1p internally so the
                  zone partition is not affected by the raw-km representation here.
  is_island     : True if the point lies on a land mass < _ISLAND_AREA_KM2 km².

New features added for improved terrain and sea-breeze representation:

  fetch_N_km … fetch_NW_km
      log(1 + open-ocean fetch in km) for 8 compass directions (N, NE, E, SE,
      S, SW, W, NW).  The raw fetch (km to nearest coastline in that bearing,
      excluding the point's own island coast) is log1p-transformed before
      storage.  This compresses the 5–400 km training range to ~1.8–6.0,
      preventing models from assigning near-zero occurrence to the sheltered
      faces of small islands (where raw fetch ≈ 5 km because a nearby island
      interrupts the ray, but the climate exposure is still maritime).
      Raw fetch is capped at _FETCH_MAX_KM = 400 km before transformation.
      Reference: PRISM coastal-trajectory model (Daly et al. 2003);
                 Effective fetch (Rao et al. 2018).

  dem_northness : cos(aspect) from the high-resolution DEM (delos_mapzen_s2coast.tif).
      +1 = north-facing, −1 = south-facing.  Encodes differential solar radiation
      and exposure to the Etesian / Meltemi northerly winds.
      Only computed where the DEM covers the point; falls back to 0.0 elsewhere.

  dem_eastness  : sin(aspect) from the high-resolution DEM.
      +1 = east-facing, −1 = west-facing.

  dem_slope_deg : terrain slope in degrees from the high-resolution DEM.
      Steeper slopes have stronger orographic wind channelling and more
      pronounced radiation asymmetries between aspects.

  tpi_500m      : Topographic Position Index at 500 m radius.
      tpi = pixel_elevation − mean(elevations within 500 m).
      Negative = valley / depression (cold-air pooling, warmer Tmin).
      Positive = ridge / hilltop (exposed, lower Tmin from radiative loss).
      Reference: Holden et al. (2011); Daly et al. (2009 PRISM).

  tpi_2000m     : Same as tpi_500m but at 2 km radius.  Captures the
                  mesoscale topographic position of a point (valley floor
                  vs. ridge relative to the whole island).

  svf           : Sky View Factor ∈ [0, 1].  Proportion of the sky hemisphere
                  visible from the point, unobstructed by surrounding terrain.
                  High SVF (ridges) → greater nocturnal radiative cooling → lower Tmin.
                  Low SVF (narrow valleys) → less radiative loss but cold-air pooling.
                  Computed using a 16-direction horizon-angle approximation.
                  Reference: TopoSCALE (Fiddes & Gruber 2014);
                             Longwave radiation over rugged terrain (Yang et al. 2024).

DATA REQUIREMENTS
-----------------
  S2Coast-2023 shapefile : for coast_dist_km, is_island, and directional fetch.
  delos_mapzen_s2coast.tif : for dem_northness, dem_eastness, dem_slope_deg,
                              tpi_500m, tpi_2000m, svf.
    All DEM features fall back to 0.0 when the TIF is not present or when the
    point lies outside the raster extent (e.g. mainland/large-island stations).

CACHING
-------
The S2Coast layers are loaded once per process and cached in _cache.
The DEM raster array is loaded once and cached in _dem_cache.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from shapely.geometry import LineString, Point, box as shapely_box

from downscaling.config import DELOS_DEM_TIF, S2COAST_DIR, S2COAST_STUDY_BBOX

# ---------------------------------------------------------------------------
# S2Coast shapefile paths
# ---------------------------------------------------------------------------
_POLY_PATH = S2COAST_DIR / "S2Coast-2023_Polygon_fishnet.shp"
_LINE_PATH = S2COAST_DIR / "S2Coast-2023_Polyline_diss.shp"

# Island area threshold (km²) — same as original
_ISLAND_AREA_KM2 = 25_000.0

# Study-area mask for S2Coast clipping
_STUDY_MASK = shapely_box(*S2COAST_STUDY_BBOX)

# ---------------------------------------------------------------------------
# Directional fetch parameters
# ---------------------------------------------------------------------------
# 8 compass directions: N=0°, NE=45°, E=90°, SE=135°, S=180°, SW=225°, W=270°, NW=315°
_FETCH_BEARINGS   = [0, 45, 90, 135, 180, 225, 270, 315]
_FETCH_NAMES      = ["fetch_N_km", "fetch_NE_km", "fetch_E_km", "fetch_SE_km",
                     "fetch_S_km", "fetch_SW_km", "fetch_W_km", "fetch_NW_km"]
# Coastline intersections closer than this are assumed to be the point's own
# island coast and are ignored; only farther coastlines count as fetch limit.
_OWN_COAST_BUFFER_M = 6_500.0   # 6.5 km — excludes Rhenia (~5.06 km NW) from fetch
# Maximum fetch cap (km); points with no far coast receive this value.
_FETCH_MAX_KM = 400.0

# ---------------------------------------------------------------------------
# DEM parameters
# ---------------------------------------------------------------------------
# Sky View Factor: number of horizontal sectors used for horizon-angle sampling.
_SVF_N_SECTORS = 16
# Maximum radius (pixels) used for horizon search in the SVF computation.
_SVF_MAX_RADIUS_PX = 150   # ~600 m at 4 m/px DEM resolution

# Module-level caches
_cache: dict | None = None
_dem_cache: dict | None = None


# ---------------------------------------------------------------------------
# S2Coast loader
# ---------------------------------------------------------------------------

def _load_s2coast() -> dict:
    """Load S2Coast layers (first call) or return cached result."""
    global _cache
    if _cache is not None:
        return _cache

    coast_gdf = gpd.read_file(_LINE_PATH, mask=_STUDY_MASK).to_crs("EPSG:3035")
    land_gdf  = gpd.read_file(_POLY_PATH, mask=_STUDY_MASK).to_crs("EPSG:3035")

    land_union = land_gdf.explode(index_parts=False).geometry.union_all()
    land_parts = (
        list(land_union.geoms)
        if land_union.geom_type == "MultiPolygon"
        else [land_union]
    )

    # Single unified coastline geometry for fast intersection tests
    coast_union = coast_gdf.geometry.union_all()

    _cache = {
        "coast":      coast_gdf,
        "coast_union": coast_union,
        "land_parts": land_parts,
    }
    return _cache


# ---------------------------------------------------------------------------
# DEM loader
# ---------------------------------------------------------------------------

def _load_dem() -> dict | None:
    """
    Load the Delos high-res DEM raster.  Returns None if the TIF is absent.

    Cached globally so the file is read only once per process.
    """
    global _dem_cache
    if _dem_cache is not None:
        return _dem_cache

    if not DELOS_DEM_TIF.exists():
        return None

    with rasterio.open(DELOS_DEM_TIF) as src:
        elev   = src.read(1).astype(np.float32)        # (rows, cols)
        nodata = src.nodata
        tf     = src.transform
        crs    = src.crs
        res    = src.res                               # (row_size_deg, col_size_deg)

    # Mask nodata → NaN
    if nodata is not None:
        elev = np.where(np.isnan(elev) | (elev == nodata), np.nan, elev)

    n_rows, n_cols = elev.shape

    # Approximate pixel size in metres at the raster's mean latitude
    # res = (pixel_height_deg, pixel_width_deg) in rasterio
    lat_c   = (tf.f + tf.e * n_rows / 2)              # centre latitude (approx)
    cos_lat = np.cos(np.radians(abs(lat_c)))
    py_m    = abs(tf.e) * 111_320.0                    # north-south pixel in metres
    px_m    = abs(tf.a) * 111_320.0 * cos_lat          # east-west pixel in metres

    _dem_cache = {
        "elev":   elev,
        "tf":     tf,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "py_m":   py_m,
        "px_m":   px_m,
    }
    return _dem_cache


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_geo_features(locations: pd.DataFrame) -> pd.DataFrame:
    """
    Add all geographic predictor features to a locations DataFrame.

    Parameters
    ----------
    locations : DataFrame with columns id, lat, lon, elevation_m.

    Returns
    -------
    Copy of locations with the following new columns:

    From S2Coast (all locations):
      coast_dist_km, is_island,
      fetch_N_km, fetch_NE_km, fetch_E_km, fetch_SE_km,
      fetch_S_km, fetch_SW_km, fetch_W_km, fetch_NW_km

    From DEM (Delos points where TIF covers; 0 elsewhere):
      dem_northness, dem_eastness, dem_slope_deg,
      tpi_500m, tpi_2000m, svf
    """
    geo = _load_s2coast()

    # Project to EPSG:3035 (metric) for all distance computations
    gdf = gpd.GeoDataFrame(
        locations.copy(),
        geometry=gpd.points_from_xy(locations["lon"], locations["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")

    result = locations.copy()

    # --- Existing features: coast distance and is_island ----------------------
    result["coast_dist_km"] = [
        _coast_dist_km(pt, geo["coast"]) for pt in gdf.geometry
    ]
    result["is_island"] = [
        _is_island(pt, geo["land_parts"]) for pt in gdf.geometry
    ]

    # --- Directional fetch (8 directions) ------------------------------------
    fetch_rows = [
        _directional_fetch(pt, geo["coast_union"]) for pt in gdf.geometry
    ]
    fetch_df = pd.DataFrame(fetch_rows, index=result.index)
    for col in _FETCH_NAMES:
        result[col] = fetch_df[col]

    # --- DEM-derived features (per-point, falls back to 0 off-DEM) -----------
    dem = _load_dem()
    if dem is not None:
        dem_rows = [
            _dem_features_at(lat, lon, dem)
            for lat, lon in zip(locations["lat"], locations["lon"])
        ]
        dem_df = pd.DataFrame(dem_rows, index=result.index)
    else:
        n = len(result)
        dem_df = pd.DataFrame({
            "dem_northness": np.zeros(n, dtype=np.float32),
            "dem_eastness":  np.zeros(n, dtype=np.float32),
            "dem_slope_deg": np.zeros(n, dtype=np.float32),
            "tpi_500m":      np.zeros(n, dtype=np.float32),
            "tpi_2000m":     np.zeros(n, dtype=np.float32),
            "svf":           np.ones(n,  dtype=np.float32),
        }, index=result.index)

    for col in dem_df.columns:
        result[col] = dem_df[col]

    return result


# ---------------------------------------------------------------------------
# S2Coast helpers
# ---------------------------------------------------------------------------

def _coast_dist_km(pt_proj: Point, coast_gdf: gpd.GeoDataFrame) -> float:
    """Distance to nearest coastline in km (raw, not transformed)."""
    _, (pos,) = coast_gdf.sindex.nearest(pt_proj, return_all=False)
    return float(coast_gdf.geometry.iloc[int(pos)].distance(pt_proj) / 1000.0)


def _is_island(pt_proj: Point, land_parts: list) -> bool:
    """True if the point lies on a land mass smaller than _ISLAND_AREA_KM2."""
    for geom in land_parts:
        if geom.contains(pt_proj):
            return (geom.area / 1e6) < _ISLAND_AREA_KM2
    return False


def _directional_fetch(pt_proj: Point, coast_union) -> dict:
    """
    Compute log(1 + open-ocean fetch in km) in 8 compass directions.

    For each bearing, a ray is cast from the point to _FETCH_MAX_KM distance.
    All intersections with the S2Coast dissolved polyline are collected;
    intersections closer than _OWN_COAST_BUFFER_M are discarded (own island
    coast).  The nearest remaining intersection gives the raw fetch (km);
    log1p is applied before returning.

    log1p transform: compresses the 5–400 km range to ~1.8–6.0, preventing
    LightGBM from assigning near-zero precipitation occurrence to the sheltered
    face of small islands where the raw fetch is small but the climate exposure
    is still maritime.

    Works in EPSG:3035 (metric CRS) so distance arithmetic is straightforward.
    """
    max_m  = _FETCH_MAX_KM * 1000.0
    result = {}

    for bearing, name in zip(_FETCH_BEARINGS, _FETCH_NAMES):
        rad = np.radians(bearing)
        dx  = np.sin(rad)   # eastward component
        dy  = np.cos(rad)   # northward component

        endpoint = Point(pt_proj.x + dx * max_m, pt_proj.y + dy * max_m)
        ray      = LineString([pt_proj, endpoint])

        try:
            intersection = coast_union.intersection(ray)
        except Exception:
            result[name] = round(np.log1p(_FETCH_MAX_KM), 6)
            continue

        if intersection.is_empty:
            result[name] = round(np.log1p(_FETCH_MAX_KM), 6)
            continue

        # Collect all intersection point distances
        dists = _collect_intersection_distances(pt_proj, intersection)

        # Filter out own-island coast (too close)
        far_dists = [d for d in dists if d > _OWN_COAST_BUFFER_M]

        if far_dists:
            raw_km = min(far_dists) / 1000.0
            result[name] = round(np.log1p(raw_km), 6)
        else:
            result[name] = round(np.log1p(_FETCH_MAX_KM), 6)

    return result


def _collect_intersection_distances(origin: Point, geom) -> list[float]:
    """
    Recursively collect distances from origin to all Point geometries in geom.
    """
    distances = []
    if geom.geom_type == "Point":
        distances.append(origin.distance(geom))
    elif geom.geom_type in ("MultiPoint", "GeometryCollection", "MultiLineString"):
        for sub in geom.geoms:
            distances.extend(_collect_intersection_distances(origin, sub))
    elif geom.geom_type == "LineString":
        # Intersection is a shared segment; use midpoint distance
        mid = geom.interpolate(0.5, normalized=True)
        distances.append(origin.distance(mid))
    elif geom.geom_type == "Polygon":
        # Degenerate intersection (ray clipped through a land polygon feature).
        # Use the nearest point on the exterior boundary as a conservative estimate.
        nearest_pt = geom.exterior.interpolate(geom.exterior.project(origin))
        distances.append(origin.distance(nearest_pt))
    elif geom.geom_type == "MultiPolygon":
        for sub in geom.geoms:
            nearest_pt = sub.exterior.interpolate(sub.exterior.project(origin))
            distances.append(origin.distance(nearest_pt))
    return distances


# ---------------------------------------------------------------------------
# DEM-derived features
# ---------------------------------------------------------------------------

def _dem_features_at(lat: float, lon: float, dem: dict) -> dict:
    """
    Compute DEM-derived features for one (lat, lon) point.

    Returns a dict with keys: dem_northness, dem_eastness, dem_slope_deg,
    tpi_500m, tpi_2000m, svf.  Falls back to 0 / 1 (neutral values) when the
    point lies outside the DEM extent.
    """
    elev   = dem["elev"]
    tf     = dem["tf"]
    n_rows = dem["n_rows"]
    n_cols = dem["n_cols"]
    py_m   = dem["py_m"]
    px_m   = dem["px_m"]

    _zero = {
        "dem_northness": 0.0, "dem_eastness": 0.0, "dem_slope_deg": 0.0,
        "tpi_500m": 0.0, "tpi_2000m": 0.0, "svf": 1.0,
    }

    try:
        row_f, col_f = rowcol(tf, lon, lat, op=float)
    except Exception:
        return _zero

    row_c = int(round(row_f))
    col_c = int(round(col_f))

    if not (0 <= row_c < n_rows and 0 <= col_c < n_cols):
        return _zero

    if np.isnan(elev[row_c, col_c]):
        return _zero

    # --- Slope and aspect (from local gradient) ----------------------------
    # Use a 3×3 neighbourhood for gradient estimation (Horn's method)
    r0, r1 = max(0, row_c - 1), min(n_rows, row_c + 2)
    c0, c1 = max(0, col_c - 1), min(n_cols, col_c + 2)
    patch = elev[r0:r1, c0:c1]

    if patch.shape == (3, 3) and not np.any(np.isnan(patch)):
        # Central-difference in row (south) and col (east) directions
        # Note: rows increase southward, so d/drow = southward derivative
        dz_drow = (patch[2, :].mean() - patch[0, :].mean()) / (2 * py_m)
        dz_dcol = (patch[:, 2].mean() - patch[:, 0].mean()) / (2 * px_m)
        # northward derivative = -dz_drow (south is positive row direction)
        dz_dy_north =  -dz_drow
        dz_dx_east  =   dz_dcol

        slope_rad   = np.arctan(np.hypot(dz_dx_east, dz_dy_north))
        aspect_rad  = np.arctan2(dz_dx_east, dz_dy_north)   # from N, clockwise = E
        northness   = float(np.cos(aspect_rad))
        eastness    = float(np.sin(aspect_rad))
        slope_deg   = float(np.degrees(slope_rad))
    else:
        northness = 0.0
        eastness  = 0.0
        slope_deg = 0.0

    # --- Topographic Position Index at two radii --------------------------
    tpi_500m  = _tpi(elev, row_c, col_c, radius_m=500.0,  py_m=py_m, px_m=px_m)
    tpi_2000m = _tpi(elev, row_c, col_c, radius_m=2000.0, py_m=py_m, px_m=px_m)

    # --- Sky View Factor --------------------------------------------------
    svf = _sky_view_factor(elev, row_c, col_c, py_m=py_m, px_m=px_m)

    return {
        "dem_northness": northness,
        "dem_eastness":  eastness,
        "dem_slope_deg": slope_deg,
        "tpi_500m":      tpi_500m,
        "tpi_2000m":     tpi_2000m,
        "svf":           svf,
    }


def _tpi(
    elev: np.ndarray,
    row_c: int,
    col_c: int,
    radius_m: float,
    py_m: float,
    px_m: float,
) -> float:
    """
    Topographic Position Index = centre_elevation − mean(neighbourhood elevation).

    Neighbourhood is a circular window of radius radius_m metres.
    Negative → depression (valley/basin); positive → ridge/hilltop.
    Returns 0.0 when fewer than 4 valid neighbours exist.
    """
    r_rows = int(np.ceil(radius_m / py_m))
    r_cols = int(np.ceil(radius_m / px_m))
    n_rows, n_cols = elev.shape

    r0 = max(0, row_c - r_rows);  r1 = min(n_rows, row_c + r_rows + 1)
    c0 = max(0, col_c - r_cols);  c1 = min(n_cols, col_c + r_cols + 1)

    sub = elev[r0:r1, c0:c1]
    # Build a distance mask to restrict to the circular radius
    rows_idx = np.arange(r0, r1) - row_c
    cols_idx = np.arange(c0, c1) - col_c
    rr, cc   = np.meshgrid(rows_idx, cols_idx, indexing="ij")
    dist_m   = np.hypot(rr * py_m, cc * px_m)

    mask  = (dist_m <= radius_m) & ~np.isnan(sub)
    # Exclude the centre pixel from the neighbourhood mean
    mask[row_c - r0, col_c - c0] = False

    if mask.sum() < 4:
        return 0.0

    centre_elev = elev[row_c, col_c]
    if np.isnan(centre_elev):
        return 0.0

    return float(centre_elev - np.nanmean(sub[mask]))


def _sky_view_factor(
    elev: np.ndarray,
    row_c: int,
    col_c: int,
    py_m: float,
    px_m: float,
    n_sectors: int = _SVF_N_SECTORS,
    max_radius_px: int = _SVF_MAX_RADIUS_PX,
) -> float:
    """
    Approximate Sky View Factor using a horizon-angle method.

    For each of n_sectors equally spaced compass directions, the maximum
    elevation angle to any surrounding pixel within max_radius_px pixels
    is computed.  SVF is approximated as:

        SVF ≈ 1 − mean(sin²(max_horizon_angle))

    This is the simplified formula from Oke (1987) adapted for a planar DEM.
    Returns 1.0 (fully open sky) when terrain is flat or point is on sea.
    """
    n_rows, n_cols = elev.shape
    centre_elev    = elev[row_c, col_c]

    if np.isnan(centre_elev):
        return 1.0

    sin2_horizons = []
    steps = np.arange(1, max_radius_px + 1, dtype=float)

    for k in range(n_sectors):
        bearing = 2 * np.pi * k / n_sectors
        dx      = np.sin(bearing)   # east
        dy      = -np.cos(bearing)  # south (positive row direction for north=0)

        nr_arr = (row_c + np.round(dy * steps)).astype(int)
        nc_arr = (col_c + np.round(dx * steps)).astype(int)

        valid = (nr_arr >= 0) & (nr_arr < n_rows) & (nc_arr >= 0) & (nc_arr < n_cols)
        if not valid.any():
            sin2_horizons.append(0.0)
            continue

        nr_safe = nr_arr.clip(0, n_rows - 1)
        nc_safe = nc_arr.clip(0, n_cols - 1)
        neighbour_elev = np.where(valid, elev[nr_safe, nc_safe], np.nan)

        dist_m = np.hypot((nr_arr - row_c) * py_m, (nc_arr - col_c) * px_m)
        with np.errstate(invalid="ignore", divide="ignore"):
            tan_angle = np.where(
                valid & (dist_m > 1e-6),
                (neighbour_elev - centre_elev) / dist_m,
                np.nan,
            )

        max_tan = float(np.nanmax(tan_angle)) if not np.all(np.isnan(tan_angle)) else 0.0
        max_tan = max(max_tan, 0.0)

        horizon_angle = np.arctan(max_tan)
        sin2_horizons.append(np.sin(horizon_angle) ** 2)

    if not sin2_horizons:
        return 1.0

    return float(1.0 - np.mean(sin2_horizons))
