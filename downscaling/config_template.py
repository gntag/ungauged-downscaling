"""
config_template.py
===================
Template for downscaling/config.py — copy this file to config.py and fill in
the MACHINE-SPECIFIC sections before running the pipeline.

HOW TO USE
----------
1. Copy this file:
       cp config_template.py config.py          # Linux / macOS
       copy config_template.py config.py        # Windows

2. Fill in every value marked "EDIT" below.  Only three paths are
   machine-specific; everything else is auto-derived from the project root or
   is a safe default you can leave untouched.

3. config.py is excluded from version control (.gitignore).
   This template file (config_template.py) IS version-controlled and contains
   no personal paths or credentials.

MACHINE-SPECIFIC paths
----------------------
  ROOT               — auto-derived from this file's location (no edit needed)
  S2COAST_DIR        — path to the S2Coast 2023 shapefile on your machine
  CORDEX_DIR         — (server only) path to the raw CORDEX EUR-11 archive
  CORDEX_OUTPUT_DIR  — (server only) where cordex_extract.py writes .nc files
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Root directory
# ---------------------------------------------------------------------------
# Derived from the location of this file.
# This file sits at <root>/downscaling/config.py, so two parents up = project root.
# DO NOT hard-code a machine-specific path here — the auto-derivation makes
# the project portable across Windows and Linux development environments.
ROOT = Path(__file__).resolve().parent.parent  # <project_root>/

# ---------------------------------------------------------------------------
# CERRA reanalysis data (local copy, ~5 km EUR resolution)
# ---------------------------------------------------------------------------
# Directory layout expected by cerra_extract.py:
#   <CERRA_DIR>/2m_temperature/<year>/*.nc        (analysis, monthly files)
#   <CERRA_DIR>/2m_relative_humidity/<year>/*.nc
#   <CERRA_DIR>/10m_wind_speed/<year>/*.nc
#   <CERRA_DIR>/10m_wind_gust_since_.../<var>_<year>.nc   (forecast, yearly)
#   <CERRA_DIR>/total_precipitation/<var>_<year>.nc
CERRA_DIR = ROOT / "cerra_data"

# CERRA native land-sea mask (static, time-invariant).
# Cells with LSM >= 0.5 are treated as land; the rest as sea.
# Used to redirect island-station patches to the nearest land cells.
LSM_PATH = ROOT / "land_mask" / "cerra_lsm.nc"

# ---------------------------------------------------------------------------
# CORDEX land masks
# ---------------------------------------------------------------------------
# One sftlf_<model>.nc per CORDEX RCM.
# Downloaded by land_mask/download_land_masks.py.
# sftlf > 50 % → land;  used identically to LSM_PATH for CORDEX patches.
CORDEX_LAND_MASK_DIR = ROOT / "land_mask"

# ---------------------------------------------------------------------------
# S2Coast 2023 high-resolution shoreline (MACHINE-SPECIFIC)
# ---------------------------------------------------------------------------
# S2Coast-2023 is a global 10 m Sentinel-2 derived coastline product.
# Obtain it from: https://coastline.eu/  (free academic download)
# The pipeline uses two layers:
#   S2Coast-2023_Polygon_fishnet.shp  — land polygons   (for is_island feature)
#   S2Coast-2023_Polyline_diss.shp   — dissolved coast line (for distance feature)
#
# Point this to the directory that contains both shapefiles on your machine.
S2COAST_DIR = Path("/path/to/S2Coast2023_ShapeFile_vector")  # <-- EDIT FOR YOUR MACHINE

# Bounding box used to clip S2Coast to the Mediterranean study area before
# loading into GeoPandas (avoids reading the global 40 GB file entirely).
# Format: (lon_min, lat_min, lon_max, lat_max)
S2COAST_STUDY_BBOX = (6.0, 30.0, 37.0, 47.0)

# ---------------------------------------------------------------------------
# Delos grid (target locations)
# ---------------------------------------------------------------------------
# CSV produced by the aurehly/aurehly_PCs package containing one row per
# Delos grid point.  Required columns: lat, lon, VALUE (elevation in metres).
# VALUE is sampled from the Mapzen DEM by raster/fix_elevation.py.
DELOS_CSV = ROOT / "aurehly_PCs_dilos.csv"

# High-resolution Mapzen DEM clipped to S2Coast island boundary (~4 m resolution).
# Created by raster/delos_mapzen_s2coast.py; used by geo_features.py for
# dem_northness, dem_eastness, dem_slope_deg, tpi_500m, tpi_2000m, svf.
DELOS_DEM_TIF = ROOT / "raster" / "delos_mapzen_s2coast.tif"

# ---------------------------------------------------------------------------
# Station observations (NOAA GSOD, via Station_Data/Station_Data.py)
# ---------------------------------------------------------------------------
# Layout: <STATION_DIR>/<year>/<station_id>.csv
STATION_DIR = ROOT / "Station_Data" / "downloaded_station_data"

# Geographic bounding box that filters which stations are loaded.
# Covers the eastern Mediterranean / Aegean region generously.
STATION_BBOX = dict(lat_min=33.0, lat_max=43.0, lon_min=11.0, lon_max=32.0)

# ---------------------------------------------------------------------------
# Pipeline outputs (local, CERRA-based downscaling)
# ---------------------------------------------------------------------------
OUT_DIR   = ROOT / "downscaling" / "output"
MODEL_DIR = ROOT / "downscaling" / "models"

# Zarr stores produced by predict.py and consumed by post_processing_cerra_downscaling.py
POSTPROCESS_SOURCE_ZARR  = OUT_DIR / "delos_downscaled.zarr"          # raw quantile output
POSTPROCESS_DAILY_ZARR   = OUT_DIR / "delos_downscaled_daily.zarr"    # daily best-estimate
POSTPROCESS_PLOT_DIR     = OUT_DIR / "delos_downscaled_plots"

# Which post-processing steps to execute: "daily", "plots", or "both"
POSTPROCESS_MODE = "both"

# Variables written to the daily Zarr and rendered as spatial maps
POSTPROCESS_PLOT_VARIABLES = [
    "tmean_c",
    "tmin_c",
    "tmax_c",
    "rh_pct",
    "precip_mm",
    "wind_ms",
    "gust_ms",
]

# Colormap / range settings for each plotted variable.
# vmin/vmax = None lets matplotlib choose the data range automatically.
POSTPROCESS_PLOT_STYLES = {
    "tmean_c":   {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "tmin_c":    {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "tmax_c":    {"cmap": "coolwarm", "vmin": None,  "vmax": None},
    "rh_pct":    {"cmap": "YlGnBu",  "vmin": 0.0,   "vmax": 100.0},
    "precip_mm": {"cmap": "Blues",   "vmin": 0.0,   "vmax": None},
    "wind_ms":   {"cmap": "viridis", "vmin": 0.0,   "vmax": None},
    "gust_ms":   {"cmap": "viridis", "vmin": 0.0,   "vmax": None},
}

# Threshold-exceedance maps (average annual days above a threshold).
# Each entry produces one additional PNG alongside the standard variable plots.
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
# Calibration window shared by CERRA extraction, model training, and EQM fitting.
# Chosen to match the CORDEX historical experiment (typically 1950-2005).
YEAR_START = 1985
YEAR_END   = 2005

# ---------------------------------------------------------------------------
# Model hyper-parameters (LightGBM)
# ---------------------------------------------------------------------------
# Patch size: each station or Delos point borrows a PATCH_SIZE × PATCH_SIZE
# sub-grid of CERRA cells for spatial interpolation via CloughTocher.
PATCH_SIZE = 4

# Quantiles predicted by LightGBM for each continuous variable.
# p10 / p50 / p90 span a calibrated uncertainty range around the median.
QUANTILES = [0.1, 0.5, 0.9]

# Mapping: pipeline variable name → CERRA predictor columns expected in the
# feature matrix.  Entries are ordered by variable importance (primary feature
# first).  Changing this alters which CERRA fields each LightGBM model sees.
#
# cerra_u10 / cerra_v10 capture wind direction — whether Meltemi (northerly) or
# southerly flow dominates on a given day, the primary driver of sea-breeze
# induced coastal cooling/drying.
VARIABLE_CERRA_FEATURES = {
    "tmean":  ["cerra_tmean", "cerra_wind", "cerra_u10", "cerra_v10"],
    "tmin":   ["cerra_tmin",  "cerra_tmean", "cerra_u10", "cerra_v10"],
    "tmax":   ["cerra_tmax",  "cerra_tmean", "cerra_u10", "cerra_v10"],
    "rh":     ["cerra_rh",    "cerra_tmean", "cerra_u10", "cerra_v10"],
    # u10/v10 added: southerly flow (positive v10) strongly predicts extreme precip
    "precip": ["cerra_precip", "cerra_rh", "cerra_wind", "cerra_u10", "cerra_v10"],
    # u10/v10 added: Meltemi (negative v10) is the dominant Aegean wind regime
    "wind":   ["cerra_wind",  "cerra_tmean", "cerra_u10", "cerra_v10"],
    # u10/v10 + gust_factor_cerra (derived in train.py) for boundary-layer state
    "gust":   ["cerra_gust",  "cerra_wind",  "cerra_u10", "cerra_v10",
               "gust_factor_cerra"],
}

# Geographic / temporal features appended to each variable's feature matrix.
# The base list (GEO_FEATURES) is used as-is for temperature and RH.
# VARIABLE_GEO_FEATURES overrides this per variable where physics differs.
#
#   Temporal    : month_sin, month_cos  — smooth circular seasonal cycle
#   Location    : lat, lon              — broad regional climate gradient
#
#   Elevation   : elevation_m           — absolute target elevation
#                 delta_elev_m          — target elevation − CERRA grid elevation
#                                         (encodes sub-grid orographic correction)
#                 lapse_correction_c    — delta_elev_m × monthly_ELR/1000
#                                         (expected temperature offset in °C)
#                                         TEMPERATURE/RH ONLY — excluded from
#                                         precip/wind/gust to avoid spurious
#                                         "cold = dry" artifacts on island terrain
#
#   Coastal     : coast_dist_km         — Euclidean distance to nearest coast
#                 is_island             — island vs. mainland classification
#                 fetch_N_km … fetch_NW_km — log(1 + directional ocean fetch km)
#                                            captures sea-breeze / Meltemi exposure
#
#   CERRA slope : cerra_northness       — cos(terrain aspect) at CERRA scale
#                 cerra_eastness        — sin(terrain aspect) at CERRA scale
#
#   Fine terrain: dem_northness         — cos(aspect) from high-res DEM
#                 dem_eastness          — sin(aspect) from high-res DEM
#                 dem_slope_deg         — slope angle from high-res DEM
#                 tpi_500m              — TPI at 500 m radius (cold-air pooling)
#                 tpi_2000m             — TPI at 2 km radius (ridge vs. valley)
#                 svf                   — sky view factor (nocturnal cooling)
GEO_FEATURES = [
    # Temporal
    "month_sin", "month_cos",
    # Location
    "lat", "lon",
    # Elevation
    "elevation_m", "delta_elev_m", "lapse_correction_c",
    # Coastal
    "coast_dist_km", "is_island",
    "fetch_N_km", "fetch_NE_km", "fetch_E_km", "fetch_SE_km",
    "fetch_S_km", "fetch_SW_km", "fetch_W_km", "fetch_NW_km",
    # CERRA-scale slope
    "cerra_northness", "cerra_eastness",
    # Fine-terrain (DEM-derived; 0 for off-DEM stations — graceful degradation)
    "dem_northness", "dem_eastness", "dem_slope_deg",
    "tpi_500m", "tpi_2000m", "svf",
]

# Per-variable overrides of GEO_FEATURES.
#
#   tmean / tmin / tmax / rh  — full GEO_FEATURES (including lapse correction and
#     fine-terrain features, which are physically valid for temperature and humidity).
#
#   precip / wind / gust — fetch columns added so LightGBM can discover
#     fetch × wind-direction interactions (u10/v10 × fetch_X_km).
#     wind_terrain_align (derived in train.py) is also added to wind/gust.
_GEO_PRECIP: list[str] = [
    "month_sin", "month_cos",
    "lat", "lon",
    "elevation_m", "coast_dist_km",
    "is_island",
    "fetch_N_km", "fetch_NE_km", "fetch_E_km", "fetch_SE_km",
    "fetch_S_km", "fetch_SW_km", "fetch_W_km", "fetch_NW_km",
]
_GEO_WIND: list[str] = [
    "month_sin", "month_cos",
    "lat", "lon",
    "elevation_m", "coast_dist_km",
    "is_island",
    "fetch_N_km", "fetch_NE_km", "fetch_E_km", "fetch_SE_km",
    "fetch_S_km", "fetch_SW_km", "fetch_W_km", "fetch_NW_km",
    "wind_terrain_align",
]
VARIABLE_GEO_FEATURES: dict[str, list[str]] = {
    "tmean":  GEO_FEATURES,
    "tmin":   GEO_FEATURES,
    "tmax":   GEO_FEATURES,
    "rh":     GEO_FEATURES,
    "precip": _GEO_PRECIP,
    "wind":   _GEO_WIND,
    "gust":   _GEO_WIND,
}

# ---------------------------------------------------------------------------
# Precipitation quantile training constant
# ---------------------------------------------------------------------------
# Sample weight multiplier applied to obs > 95th-percentile wet days when
# training the p90/p95/p99 quantile models (Q-SRDRN upweighting, arXiv 2605.12762).
EXTREME_SAMPLE_WEIGHT: float = 6.0

# ---------------------------------------------------------------------------
# Precipitation EQM (empirical quantile mapping) distribution correction
# ---------------------------------------------------------------------------
# The Gamma amount model is variance-compressed (predicts ~5-12 mm for every wet
# day), which collapses annual totals and extreme frequency.  The final precip
# estimate blends an interpolation-QM correction of CERRA (which has day-matched
# timing) with the model's occ×mean (which carries sub-grid spatial structure).
# Used by predict.py (production Delos) and run_loocv_validation.py (validation).
# See the Precipitation subsection under "Scientific Methods" in README.md.
PRECIP_EQM_ENABLE      = True   # produce precip_best_eqm as the corrected estimate
PRECIP_EQM_WINDOW_DAYS = 45     # half-width of the circular DOY window for donors
PRECIP_EQM_STEP_DAYS   = 30     # spacing between DOY transfer centres (30 -> 12 windows)
PRECIP_EQM_MIN_WET     = 30     # min wet donor days in a window before all-season fallback
PRECIP_EQM_WET_MM      = 1.0    # donor wet-day threshold (mm) for the intensity pool

# Final precip = PRECIP_EQM_BLEND · QM_seasonal(CERRA→obs) + (1−blend) · (occ×mean).
# Interpolation QM of CERRA preserves CERRA's day-matched timing (positive monthly
# R²); the model term adds sub-grid spatial structure.  Blending partially
# independent errors also lifts monthly R² above either component alone.
# Validated sweep (env LOOCV): 0.6 → R² 0.17, totals 94%, CSI20 0.159, CSI50 0.052
# (beats the model-only baseline on every metric).  0.4 favours R², 0.7 favours totals.
PRECIP_EQM_BLEND       = 0.6

# Additional quantile levels trained for precipitation (beyond p10/p50/p90).
# p95 and p99 provide higher-tail uncertainty bands in the output Zarr.
EXTRA_QUANTILES: list[float] = [0.95, 0.99]

# ---------------------------------------------------------------------------
# Monthly environmental lapse rates for the Eastern Mediterranean (K/km)
# ---------------------------------------------------------------------------
# Values derived from published ELR studies (Dutra et al. 2020; Blandford 2008)
# adapted for Mediterranean insular climate: drier/more unstable in summer
# (steeper lapse), wetter/more stable in winter (shallower lapse).
# Used to compute lapse_correction_c = delta_elev_m × MONTHLY_LAPSE_RATES[m] / 1000
# A positive delta_elev_m (target higher than CERRA) → negative correction (cooler).
MONTHLY_LAPSE_RATES: dict[int, float] = {
    1:  -4.5,   # January   — shallow (stable, moist)
    2:  -4.8,
    3:  -5.2,
    4:  -5.5,
    5:  -5.8,
    6:  -6.2,
    7:  -6.5,   # July      — steep (dry, unstable)
    8:  -6.5,   # August    — steep (dry, unstable)
    9:  -6.0,
    10: -5.5,
    11: -5.0,
    12: -4.7,   # December  — shallow (stable, moist)
}

# ---------------------------------------------------------------------------
# Soft zone definitions for per-zone training
# ---------------------------------------------------------------------------
# Zones partition the Delos micro-climate space using a Gaussian soft-membership
# function over three normalised dimensions:
#   - coast_norm     : 1 − coast_dist_km / coast_dist_max_km  (1 = shoreline)
#   - elev_norm      : elevation_m / elev_max_m              (1 = highest point)
#   - northness      : dem_northness or cerra_northness       (−1 to +1)
#
# Each zone is defined by (center, sigma) in this 3-D space.
# Membership: w_k = exp(−0.5 × sum[(feature − center_k)² / sigma_k²])
# Memberships are normalised to sum to 1 across all zones for each point.
#
# Zone semantics:
#   N-Coastal  : north-facing shoreline — maximum Meltemi and sea-breeze exposure
#   S-Coastal  : south/harbour shoreline — sheltered, calmer, warmer
#   Ridge      : elevated terrain — lapse-rate cooling, more wind
#   Transitional: intermediate / interior — blended characteristics
#
# Adjust zone centers and sigmas if the Delos DEM reveals a different
# micro-climate structure in the diagnostic plots.
ZONE_DEFINITIONS: list[dict] = [
    {
        "name":          "N-Coastal",
        "center":        {"coast_norm": 0.9, "elev_norm": 0.05, "northness": 0.7},
        "sigma":         {"coast_norm": 0.15, "elev_norm": 0.15, "northness": 0.35},
    },
    {
        "name":          "S-Coastal",
        "center":        {"coast_norm": 0.9, "elev_norm": 0.05, "northness": -0.7},
        "sigma":         {"coast_norm": 0.15, "elev_norm": 0.15, "northness": 0.35},
    },
    {
        "name":          "Ridge",
        "center":        {"coast_norm": 0.4, "elev_norm": 0.60, "northness": 0.0},
        "sigma":         {"coast_norm": 0.30, "elev_norm": 0.20, "northness": 0.50},
    },
    {
        "name":          "Transitional",
        "center":        {"coast_norm": 0.6, "elev_norm": 0.20, "northness": 0.0},
        "sigma":         {"coast_norm": 0.35, "elev_norm": 0.25, "northness": 0.55},
    },
]

# Maximum normalisation values for coast distance and elevation in zone membership.
# coast_dist_max_km : distances beyond this are clamped to 1 in coast_norm.
# elev_max_m        : elevations above this are clamped to 1 in elev_norm.
ZONE_COAST_MAX_KM = 1.0    # in log1p-km units — similarity.py applies log1p(coast_dist_km)
                           # before normalising, so this value is independent of whether
                           # geo_features stores raw or log1p km.
                           # log1p(0.8 km) = 0.588 → Kynthos hill coast_norm ≈ 0.41,
                           # matching the Ridge zone center (coast_norm = 0.40).
ZONE_ELEV_MAX_M   = 110.0  # slightly above Delos's highest peak (~102 m)

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
# 0 °C in Kelvin — used when converting CERRA/CORDEX temperature from K → °C.
KELVIN_OFFSET = 273.15

# ---------------------------------------------------------------------------
# CORDEX EUR-11 extraction (server-side paths — MACHINE-SPECIFIC)
# ---------------------------------------------------------------------------
# These paths are relevant only when running cordex_extract.py on the server
# that stores the raw CORDEX NetCDF archive.  They are NOT needed for the
# local CERRA-based downscaling pipeline.
#
# CORDEX_DIR        — root of the CORDEX EUR-11 archive; one sub-directory
#                     per GCM–RCM pair, each containing daily .nc files.
# CORDEX_OUTPUT_DIR — where cordex_extract.py writes the interpolated .nc
#                     files (one per model × scenario combination).
# MIN_FREE_GB       — disk-space safety check; aborts if less than this
#                     number of GiB is free at the output path.
CORDEX_DIR        = Path("/path/to/CORDEX/archive")           # <-- EDIT FOR SERVER
CORDEX_OUTPUT_DIR = Path("/path/to/cordex_downscaled/output") # <-- EDIT FOR SERVER
MIN_FREE_GB       = 5.0

# Year ranges for each CORDEX scenario.
# "historical" calibrates the bias-correction transfer functions.
# "rcp45" and "rcp85" are the future projection windows.
CORDEX_YEAR_RANGES = {
    "historical": (1985, 2005),
    "rcp45":      (2041, 2100),
    "rcp85":      (2041, 2100),
}

# Mapping: pipeline variable name → (CORDEX NetCDF variable name, unit conversion).
# Temperatures arrive in Kelvin; precipitation in kg m⁻² s⁻¹ (= mm/s).
# Wind speed (sfcWind) is already in m/s — identity conversion.
CORDEX_VAR_MAP = {
    "tmean":  ("tas",     lambda x: x - KELVIN_OFFSET),
    "tmin":   ("tasmin",  lambda x: x - KELVIN_OFFSET),
    "tmax":   ("tasmax",  lambda x: x - KELVIN_OFFSET),
    "wind":   ("sfcWind", lambda x: x),
    "precip": ("pr",      lambda x: x * 86400.0),   # kg m⁻² s⁻¹ → mm day⁻¹
}

# Models that carry relative humidity (hurs) directly.
# All other models carry specific humidity (huss) + surface pressure (psl),
# from which RH is derived via the Magnus formula (see cordex_extract._rh_from_huss).
CORDEX_HURS_MODELS = frozenset({
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
    "CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E",
    "MPI-M-MPI-ESM-LR_SMHI-RCA4",
    "ICHEC-EC-EARTH_DMI-HIRHAM5",
})

# HadGEM2-ES uses a 360-day calendar (12 × 30-day months).
# These models require special DOY mapping when computing EQM transfer functions;
# see bias_correct.map_360_to_365 for the conversion formula.
CORDEX_HADGEM2_MODELS = frozenset({
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
})

# All 7 CORDEX EUR-11 models used in the ensemble.
# Order matters only for logging; processing is independent per model.
CORDEX_ALL_MODELS = [
    "MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009",
    "ICHEC-EC-EARTH_CLMcom-CCLM4-8-17",
    "MOHC-HadGEM2-ES_SMHI-RCA4",
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
    "CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E",
    "MPI-M-MPI-ESM-LR_SMHI-RCA4",
    "ICHEC-EC-EARTH_DMI-HIRHAM5",
]
