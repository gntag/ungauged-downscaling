"""
config_template.py
==================
Template for downscaling/config.py — copy this file to config.py and fill in
the MACHINE-SPECIFIC sections before running the pipeline.

HOW TO USE
----------
1. Copy this file:
       cp config_template.py config.py          # Linux / macOS
       copy config_template.py config.py        # Windows

2. Fill in every value marked "EDIT" below.

3. config.py is excluded from version control (.gitignore).
   This template file (config_template.py) IS version-controlled and contains
   no personal paths or credentials.

SECTIONS
--------
  Root              — auto-derived; no edit needed
  S2Coast           — path to the S2Coast 2023 shapefile on your machine
  CORDEX (server)   — paths to the CORDEX archive (needed only on the server)
  Everything else   — safe defaults that work without editing
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directory — auto-derived, no edit needed
# ---------------------------------------------------------------------------
# This file is at <root>/downscaling/config_template.py (or config.py);
# two parents up gives the project root on any machine.
ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# CERRA reanalysis data
# ---------------------------------------------------------------------------
# Expected layout (produced by cerra_data/cerra_data.py):
#   <CERRA_DIR>/2m_temperature/<year>/<var>_<year>_<month>.nc
#   <CERRA_DIR>/total_precipitation/<var>_<year>.nc
CERRA_DIR = ROOT / "cerra_data"

# CERRA native land-sea mask (downloaded by land_mask/download_land_masks.py).
LSM_PATH = ROOT / "land_mask" / "cerra_lsm.nc"

# ---------------------------------------------------------------------------
# CORDEX land masks
# ---------------------------------------------------------------------------
CORDEX_LAND_MASK_DIR = ROOT / "land_mask"

# ---------------------------------------------------------------------------
# S2Coast 2023 high-resolution shoreline  ← EDIT FOR YOUR MACHINE
# ---------------------------------------------------------------------------
# Download from https://coastline.eu/ (free academic registration required).
# The pipeline needs two files inside this directory:
#   S2Coast-2023_Polygon_fishnet.shp   — land polygons   (is_island feature)
#   S2Coast-2023_Polyline_diss.shp    — dissolved coast  (coast_dist_km feature)
#
S2COAST_DIR = Path("/path/to/S2Coast2023_ShapeFile_vector")  # <-- EDIT

# Bounding box used to clip S2Coast to the study area before loading.
# (lon_min, lat_min, lon_max, lat_max)
S2COAST_STUDY_BBOX = (6.0, 30.0, 37.0, 47.0)

# ---------------------------------------------------------------------------
# Delos grid CSV
# ---------------------------------------------------------------------------
# CSV with columns: lat, lon, VALUE (elevation in metres).
# Place the file in the project root before running the pipeline.
DELOS_CSV = ROOT / "aurehly_PCs_dilos.csv"

# ---------------------------------------------------------------------------
# Station observations (NOAA GSOD)
# ---------------------------------------------------------------------------
STATION_DIR  = ROOT / "Station_Data" / "downloaded_station_data"
STATION_BBOX = dict(lat_min=33.0, lat_max=43.0, lon_min=11.0, lon_max=32.0)

# ---------------------------------------------------------------------------
# Pipeline outputs (CERRA-based downscaling)
# ---------------------------------------------------------------------------
OUT_DIR   = ROOT / "downscaling" / "output"
MODEL_DIR = ROOT / "downscaling" / "models"

POSTPROCESS_SOURCE_ZARR = OUT_DIR / "delos_downscaled.zarr"
POSTPROCESS_DAILY_ZARR  = OUT_DIR / "delos_downscaled_daily.zarr"
POSTPROCESS_PLOT_DIR    = OUT_DIR / "delos_downscaled_plots"
POSTPROCESS_MODE        = "both"    # "daily" | "plots" | "both"

POSTPROCESS_PLOT_VARIABLES = [
    "tmean_c", "tmin_c", "tmax_c", "rh_pct", "precip_mm", "wind_ms", "gust_ms",
]

POSTPROCESS_PLOT_STYLES = {
    "tmean_c":   {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "tmin_c":    {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "tmax_c":    {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "rh_pct":    {"cmap": "YlGnBu",  "vmin": 0.0,   "vmax": 100.0},
    "precip_mm": {"cmap": "Blues",   "vmin": 0.0,   "vmax": None},
    "wind_ms":   {"cmap": "viridis", "vmin": 0.0,   "vmax": None},
    "gust_ms":   {"cmap": "viridis", "vmin": 0.0,   "vmax": None},
}

POSTPROCESS_THRESHOLD_PLOTS = [
    {
        "variable":  "tmax_c",
        "threshold": 35.0,
        "title":     "Average annual days with tmax > 35 deg C",
        "units":     "days/year",
        "cmap":      "inferno",
        "vmin":      0.0,
        "vmax":      None,
        "filename":  "avg_annual_days_tmax_gt_35c.png",
    },
    {
        "variable":  "precip_mm",
        "threshold": 20.0,
        "title":     "Average annual days with precipitation > 20 mm",
        "units":     "days/year",
        "cmap":      "magma",
        "vmin":      0.0,
        "vmax":      None,
        "filename":  "avg_annual_days_precip_gt_20mm.png",
    },
    {
        "variable":  "precip_mm",
        "threshold": 50.0,
        "title":     "Average annual days with precipitation > 50 mm",
        "units":     "days/year",
        "cmap":      "magma",
        "vmin":      0.0,
        "vmax":      None,
        "filename":  "avg_annual_days_precip_gt_50mm.png",
    },
]

# ---------------------------------------------------------------------------
# Training period
# ---------------------------------------------------------------------------
YEAR_START = 1985
YEAR_END   = 2005

# ---------------------------------------------------------------------------
# Model hyperparameters (LightGBM)
# ---------------------------------------------------------------------------
PATCH_SIZE = 4
QUANTILES  = [0.1, 0.5, 0.9]

VARIABLE_CERRA_FEATURES = {
    "tmean":  ["cerra_tmean", "cerra_wind"],
    "tmin":   ["cerra_tmin",  "cerra_tmean"],
    "tmax":   ["cerra_tmax",  "cerra_tmean"],
    "rh":     ["cerra_rh",    "cerra_tmean"],
    "precip": ["cerra_precip", "cerra_rh", "cerra_wind"],
    "wind":   ["cerra_wind",  "cerra_tmean"],
    "gust":   ["cerra_gust",  "cerra_wind"],
}

GEO_FEATURES = [
    "month_sin", "month_cos",
    "lat", "lon",
    "elevation_m", "coast_dist_km",
    "is_island",
]

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
KELVIN_OFFSET = 273.15

# ---------------------------------------------------------------------------
# CORDEX EUR-11 extraction  ← EDIT FOR YOUR SERVER
# ---------------------------------------------------------------------------
# Only needed when running cordex_extract.py on the server hosting the CORDEX
# archive.  These paths are not used by the local CERRA-only pipeline.
#
CORDEX_DIR        = Path("/path/to/CORDEX/archive")        # <-- EDIT FOR SERVER
CORDEX_OUTPUT_DIR = Path("/path/to/cordex_downscaled/output")  # <-- EDIT FOR SERVER
MIN_FREE_GB       = 5.0

CORDEX_YEAR_RANGES = {
    "historical": (1985, 2005),
    "rcp45":      (2041, 2100),
    "rcp85":      (2041, 2100),
}

CORDEX_VAR_MAP = {
    "tmean":  ("tas",     lambda x: x - KELVIN_OFFSET),
    "tmin":   ("tasmin",  lambda x: x - KELVIN_OFFSET),
    "tmax":   ("tasmax",  lambda x: x - KELVIN_OFFSET),
    "wind":   ("sfcWind", lambda x: x),
    "precip": ("pr",      lambda x: x * 86400.0),
}

CORDEX_HURS_MODELS = frozenset({
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
    "CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E",
    "MPI-M-MPI-ESM-LR_SMHI-RCA4",
    "ICHEC-EC-EARTH_DMI-HIRHAM5",
})

CORDEX_HADGEM2_MODELS = frozenset({
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
})

CORDEX_ALL_MODELS = [
    "MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009",
    "ICHEC-EC-EARTH_CLMcom-CCLM4-8-17",
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
    "CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E",
    "MPI-M-MPI-ESM-LR_SMHI-RCA4",
    "ICHEC-EC-EARTH_DMI-HIRHAM5",
]
