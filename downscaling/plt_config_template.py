"""
plt_config_template.py
======================
Template for downscaling/plt_config.py — copy to plt_config.py and fill in
the MACHINE-SPECIFIC sections before running the plotting scripts.

HOW TO USE
----------
1. Copy this file:
       cp plt_config_template.py plt_config.py          # Linux / macOS
       copy plt_config_template.py plt_config.py        # Windows

2. Fill in every value marked "EDIT" below.

3. plt_config.py is excluded from version control (.gitignore).
   This template (plt_config_template.py) IS version-controlled and contains
   no personal paths.
"""
from pathlib import Path

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------------------
# Root — auto-derived, no edit needed
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Paths  ← EDIT THESE FOR YOUR MACHINE
# ---------------------------------------------------------------------------

# Directory containing bias-corrected CORDEX NetCDF files produced by
# cordex_bias_correct.py.  ba_postprocessing.py reads {model}_bc_*.nc here.
INPUT_DIR = str(_ROOT / "downscaling" / "ba_downscaled")   # <-- EDIT IF NEEDED

# Directory where ba_postprocessing.py and plot_ba_data.py write output files.
OUTPUT_DIR = str(_ROOT / "downscaling" / "output" / "plots")  # <-- EDIT IF NEEDED

# Path to the S2Coast-2023 dissolved shoreline shapefile used in map figures.
# Download from: https://coastline.eu/
S2COAST_SHP = "/path/to/S2Coast2023_ShapeFile_vector/S2Coast-2023_Polyline_diss.shp"  # <-- EDIT

# Mapzen elevation raster clipped to Delos.  Used by plot_delos_sealevel.py.
TIF_PATH = str(_ROOT / "raster" / "delos_mapzen_s2coast.tif")  # <-- EDIT IF NEEDED

# Output path for the sea-level-risk two-panel figure.
SEALEVEL_OUTPUT_PATH = str(_ROOT / "downscaling" / "output" / "plots" / "delos_sealevel_risk.png")

# ---------------------------------------------------------------------------
# Custom colour maps
# ---------------------------------------------------------------------------
PRECIP_CMAP = LinearSegmentedColormap.from_list(
    "blue_to_brown",
    ["#1d6ea5", "#5fa2ce", "#aad3e8", "#f7f2ea", "#b07030", "#6b2010"],
)

# ---------------------------------------------------------------------------
# Figure aesthetics
# ---------------------------------------------------------------------------
PLOT_DPI    = 300
FIG_W       = 9.0
FIG_H       = 36.0
MAP_PADDING = 0.005

# ---------------------------------------------------------------------------
# Font sizes (points)
# ---------------------------------------------------------------------------
FONT_BASE           = 14
FONT_AXES_TITLE     = 15
FONT_AXES_LABEL     = 13
FONT_TICK           = 11
FONT_SUPTITLE       = 28
FONT_PANEL_SUBTITLE = 20
FONT_PANEL_LETTER   = 18
FONT_CBAR_LABEL     = 24
FONT_CBAR_TICK      = 20
FONT_FOOTER         = 9
FONT_NODATA         = 14

# ---------------------------------------------------------------------------
# Colorbar geometry
# ---------------------------------------------------------------------------
CBAR_SHRINK = 0.35
CBAR_ASPECT = 40

# ---------------------------------------------------------------------------
# Variable plot specifications (see plt_config.py for full documentation)
# ---------------------------------------------------------------------------
VARIABLES = {
    "tmean": {
        "varname":   "tmean",
        "stat_type": "annual_mean",
        "title":     "Annual Mean Temperature",
        "unit":      "°C",
        "cmap":      "RdYlGn_r",
        "levels":    np.arange(14.0, 27.0, 0.5),
        "extend":    "both",
    },
    "tmin": {
        "varname":   "tmin",
        "stat_type": "annual_mean",
        "title":     r"Annual Mean $T_{\mathrm{min}}$",
        "unit":      "°C",
        "cmap":      "RdYlGn_r",
        "levels":    np.arange(9.0, 22.0, 0.5),
        "extend":    "both",
    },
    "tmax": {
        "varname":   "tmax",
        "stat_type": "annual_mean",
        "title":     r"Annual Mean $T_{\mathrm{max}}$",
        "unit":      "°C",
        "cmap":      "RdYlGn_r",
        "levels":    np.arange(19.0, 33.0, 0.5),
        "extend":    "both",
    },
    "wind": {
        "varname":   "wind",
        "stat_type": "annual_mean",
        "title":     "Annual Mean Wind Speed",
        "unit":      r"m s$^{-1}$",
        "cmap":      "plasma",
        "levels":    np.arange(2.5, 8.75, 0.25),
        "extend":    "both",
    },
    "precip_annual": {
        "varname":           "precip",
        "stat_type":         "annual_sum",
        "title":             "Mean Annual Total Precipitation",
        "unit":              r"mm yr$^{-1}$",
        "cmap":              PRECIP_CMAP,
        "levels":            np.arange(330, 431, 25),
        "extend":            "both",
        "use_config_levels": True,
    },
    "days_tmax35": {
        "varname":   "tmax",
        "stat_type": "count_gt:35",
        "title":     r"Mean Annual Days $T_{\mathrm{max}}$ > 35 °C",
        "unit":      r"days yr$^{-1}$",
        "cmap":      "RdYlGn_r",
        "levels":    np.arange(0, 100, 5),
        "extend":    "max",
    },
    "days_tmax37": {
        "varname":           "tmax",
        "stat_type":         "count_gt:37",
        "title":             r"Mean Annual Days $T_{\mathrm{max}}$ > 37 °C",
        "unit":              r"days yr$^{-1}$",
        "cmap":              "RdYlGn_r",
        "levels":            np.arange(0, 7, 1),
        "extend":            "max",
        "use_config_levels": True,
    },
    "days_tmin0": {
        "varname":           "tmin",
        "stat_type":         "count_lt:0",
        "title":             r"Mean Annual Frost Days ($T_{\mathrm{min}}$ < 0 °C)",
        "unit":              r"days yr$^{-1}$",
        "cmap":              "Blues",
        "levels":            np.arange(0, 0.45, 0.05),
        "extend":            "max",
        "use_config_levels": True,
    },
    "days_r99p": {
        "varname":   "precip",
        "stat_type": "precip_r99p",
        "title":     "Mean Decadal Extreme Precipitation Days\n(daily precip > 99th percentile of wet days)",
        "unit":      "days\ndecade$^{-1}$",
        "cmap":      PRECIP_CMAP,
        "levels":    np.arange(4, 16, 1),
        "extend":    "both",
        "scale":     10.0,
    },
    "days_pr20": {
        "varname":   "precip",
        "stat_type": "count_gt:20",
        "title":     "Mean Decadal Days Precipitation > 20 mm",
        "unit":      "days\ndecade$^{-1}$",
        "cmap":      PRECIP_CMAP,
        "levels":    np.arange(0, 630, 30),
        "extend":    "max",
        "scale":     10.0,
    },
    "days_pr50": {
        "varname":   "precip",
        "stat_type": "count_gt:50",
        "title":     "Mean Decadal Days Precipitation > 50 mm",
        "unit":      "days\ndecade$^{-1}$",
        "cmap":      PRECIP_CMAP,
        "levels":    np.arange(0, 260, 10),
        "extend":    "max",
        "scale":     10.0,
    },
    "aridity_index": {
        "varname":   None,
        "stat_type": "aridity_index",
        "title":     r"Aridity Index  ($P\,/\,\mathrm{PET}_{\mathrm{TW}}$)",
        "unit":      "—",
        "cmap":      "RdYlGn",
        "levels":    np.array([
                        0.03, 0.05, 0.10, 0.15,
                        0.20, 0.225, 0.25, 0.275, 0.30, 0.325, 0.35, 0.375, 0.40,
                        0.425, 0.45, 0.475, 0.50,
                        0.55, 0.60, 0.65, 0.70, 0.80,
                     ]),
        "extend":    "both",
    },
}
