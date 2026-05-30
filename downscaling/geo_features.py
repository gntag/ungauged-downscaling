"""
geo_features.py
===============
Compute geographic predictor features for the Delos downscaling pipeline.

FEATURES COMPUTED
-----------------
coast_dist_km : great-circle distance to the nearest coastline (km).
    Derived from the S2Coast-2023 dissolved polyline in EPSG:3035 (metric CRS).
    This is the primary orographic proxy in the model — stations and grid points
    close to the coast experience stronger sea-breeze cooling and higher humidity
    than interior points.

is_island : True if the point lies on a land mass < _ISLAND_AREA_KM2 km².
    Classifies small Aegean islands separately from the mainland.  Delos itself
    (5 km²) is flagged as an island; large islands like Crete are not.

DATA REQUIREMENTS
-----------------
The S2Coast-2023 shapefile must be available at the path configured in
downscaling/config.py (S2COAST_DIR).  The two required files are:
  S2Coast-2023_Polygon_fishnet.shp  — land area polygons (for is_island)
  S2Coast-2023_Polyline_diss.shp   — dissolved coastline (for coast_dist_km)

Both are clipped to S2COAST_STUDY_BBOX before loading to avoid reading the
global 40 GB dataset.  GeoPandas re-projects to EPSG:3035 (Laea Europe) so
distance calculations are in metres with minimal distortion.

CACHING
-------
The S2Coast layers are loaded once per process and cached in _cache.  Subsequent
calls to compute_geo_features() skip the expensive file-read.
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely.geometry import Point, box as shapely_box

from downscaling.config import S2COAST_DIR, S2COAST_STUDY_BBOX

# ---------------------------------------------------------------------------
# S2Coast shapefile paths (derived from config)
# ---------------------------------------------------------------------------
_POLY_PATH = S2COAST_DIR / "S2Coast-2023_Polygon_fishnet.shp"
_LINE_PATH = S2COAST_DIR / "S2Coast-2023_Polyline_diss.shp"

# Threshold for is_island classification (km²).
# Crete is ~8 300 km², so 25 000 km² safely separates small islands from large ones.
_ISLAND_AREA_KM2 = 25_000.0

# Pre-build the study-area bounding polygon for GeoPandas masking
_STUDY_MASK = shapely_box(*S2COAST_STUDY_BBOX)

# Module-level cache so the shapefile read happens only once per process
_cache: dict | None = None


def _load_s2coast() -> dict:
    """
    Load S2Coast layers from disk (first call) or return cached result.

    Returns a dict with keys:
      'coast'      — GeoDataFrame of dissolved coastline in EPSG:3035
      'land_parts' — list of Polygon geometries (individual land masses)
    """
    global _cache
    if _cache is not None:
        return _cache

    # Read only the bounding-box subset to avoid loading the global dataset
    coast_gdf = gpd.read_file(_LINE_PATH, mask=_STUDY_MASK).to_crs("EPSG:3035")
    land_gdf  = gpd.read_file(_POLY_PATH, mask=_STUDY_MASK).to_crs("EPSG:3035")

    # Dissolve the fishnet grid cells into contiguous land masses.
    # explode() decomposes MultiPolygons so union_all() can merge cross-cell
    # fragments of the same land mass (e.g. Crete spans many fishnet tiles).
    land_union = land_gdf.explode(index_parts=False).geometry.union_all()
    land_parts = (
        list(land_union.geoms)
        if land_union.geom_type == "MultiPolygon"
        else [land_union]
    )

    _cache = {"coast": coast_gdf, "land_parts": land_parts}
    return _cache


def compute_geo_features(locations: pd.DataFrame) -> pd.DataFrame:
    """
    Add coast_dist_km and is_island columns to a locations DataFrame.

    Parameters
    ----------
    locations : DataFrame with columns id, lat, lon, elevation_m.

    Returns
    -------
    Copy of locations with two new columns appended:
      coast_dist_km : float (km to nearest S2Coast polyline)
      is_island     : bool (True if the point lies on a land mass < 25 000 km²)
    """
    geo = _load_s2coast()

    # Project points to EPSG:3035 for metric distance calculations
    gdf = gpd.GeoDataFrame(
        locations.copy(),
        geometry=gpd.points_from_xy(locations["lon"], locations["lat"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:3035")

    result = locations.copy()
    result["coast_dist_km"] = [
        _coast_dist_km(pt, geo["coast"]) for pt in gdf.geometry
    ]
    result["is_island"] = [
        _is_island(pt, geo["land_parts"]) for pt in gdf.geometry
    ]
    return result


def _coast_dist_km(pt_proj: Point, coast_gdf: gpd.GeoDataFrame) -> float:
    """
    Distance from a projected point to the nearest coastline segment (km).

    Uses GeoDataFrame.geometry.distance() which operates on the already-
    projected EPSG:3035 coordinates (distance in metres).
    """
    return float(coast_gdf.geometry.distance(pt_proj).min() / 1000.0)


def _is_island(pt_proj: Point, land_parts: list) -> bool:
    """
    Return True if the projected point lies on a land mass < _ISLAND_AREA_KM2.

    Iterates over individual contiguous land polygons and checks containment.
    Area is converted from m² (EPSG:3035) to km² by dividing by 1e6.
    """
    for geom in land_parts:
        if geom.contains(pt_proj):
            return (geom.area / 1e6) < _ISLAND_AREA_KM2
    return False
