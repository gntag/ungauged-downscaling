"""
validation_config.py
====================
Configuration for the LOOCV validation pipeline (run_loocv_validation.py).

All paths and tunable constants live here.  Edit this file; do not hardcode
values in run_loocv_validation.py.
"""
from pathlib import Path

from downscaling.config import ROOT, YEAR_START, YEAR_END

# ---------------------------------------------------------------------------
# Output directory for all validation artefacts
# ---------------------------------------------------------------------------
# VALIDATION_DIR = ROOT / "validation"
VALIDATION_DIR = Path(r"E:\Dilos_Validation")

# ---------------------------------------------------------------------------
# Year range for CERRA extraction and LOOCV evaluation
# ---------------------------------------------------------------------------
# Defaults to the main pipeline calibration window (1985-2005).
# Extend if you have CERRA data for more years.
VAL_YEAR_START = YEAR_START
VAL_YEAR_END   = YEAR_END

# ---------------------------------------------------------------------------
# Climate-event thresholds
# ---------------------------------------------------------------------------
WET_DAY_MM      = 1.0     # mm/day  — wet-day definition
PR20_MM         = 20.0    # mm/day  — heavy precipitation
PR50_MM         = 50.0    # mm/day  — very heavy precipitation
TMAX35_C        = 35.0    # °C      — hot day
TMAX37_C        = 37.0    # °C      — extreme heat
TMIN0_C         = 0.0     # °C      — frost day  (tmin < 0)
R99P_PERCENTILE = 99      # percentile for r99p  (from observed wet days per station)

# ---------------------------------------------------------------------------
# Precipitation best-estimate for LOOCV
# ---------------------------------------------------------------------------
# Days where predicted occurrence probability exceeds this value are treated
# as wet; all others receive 0 mm.
OCC_PROB_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# CERRA precipitation unit conversion
# ---------------------------------------------------------------------------
# cerra_data.py downloads tp as "precipitation accumulation (tp, m)" — metres.
# cerra_extract.py applies no unit conversion, so cerra_precip in the parquet
# cache is in metres.  Multiply by 1000 to compare with station prcp_mm (mm).
CERRA_PRECIP_MM_FACTOR = 1000.0

# ---------------------------------------------------------------------------
# LOOCV behaviour flags
# ---------------------------------------------------------------------------
LOOCV_FAST_MODE = False   # True: fewer LightGBM trees — for development only
LOOCV_OVERWRITE = False   # True: recompute even if loocv_{station}.parquet exists
CACHE_OVERWRITE = False   # True: recompute Phase-1 cache files even if present
ENV_LOOCV_OVERWRITE = False  # True: recompute env-LOOCV folds even if files exist

# ---------------------------------------------------------------------------
# Precipitation EQM (empirical quantile mapping) distribution correction
# ---------------------------------------------------------------------------
# Single source of truth lives in downscaling/config.py (shared with predict.py).
from downscaling.config import (           # noqa: E402
    PRECIP_EQM_ENABLE,
    PRECIP_EQM_MIN_WET,
    PRECIP_EQM_STEP_DAYS,
    PRECIP_EQM_WET_MM,
    PRECIP_EQM_WINDOW_DAYS,
)

# ---------------------------------------------------------------------------
# Environment groups for cross-environment validation
# ---------------------------------------------------------------------------
# Each group defines a set of evaluation stations and a label.
# Zone similarity weights are recomputed relative to each group's own station
# features (not Delos), so the model is optimised for that environment type.
# Training always uses all available stations (n-1 per fold).
#
# ~6 stations per group, 24 total — keeps per-run cost ~20% of full LOOCV.
ENVIRONMENT_GROUPS: dict[str, dict] = {
    "aegean_islands": {
        "label": "Aegean Islands (Delos-like)",
        "stations": [
            "16732099999",  # Naxos         37.1N 25.4E   10m  (adjacent to Delos)
            "16723099999",  # Samos          37.7N 26.9E    6m
            "16742099999",  # Kos            36.8N 27.1E  126m
            "16749099999",  # Rhodes         36.4N 28.1E    6m
            "16754099999",  # Sitia, Crete   35.3N 25.2E   35m
            "16746099999",  # Heraklion      35.5N 24.1E  149m
        ],
    },
    "cities": {
        "label": "Urban / City environments",
        "stations": [
            "16622099999",  # Thessaloniki   40.5N 23.0E    7m  (Greece)
            "16716099999",  # Athens         37.9N 23.7E   21m  (Greece)
            "16253099999",  # Naples         41.1N 14.1E    9m  (S Italy)
            "16289099999",  # Nola/Naples    40.9N 14.3E   72m  (S Italy)
            "16405099999",  # Palermo        38.2N 13.1E   20m  (Sicily)
            "16420099999",  # Reggio Calabria 38.2N 15.6E  51m  (S Italy)
        ],
    },
    "high_elevation": {
        "label": "High elevation mountainous",
        "stations": [
            "16219099999",  # L'Aquila        42.5N 13.0E 1875m (central Apennines)
            "16344099999",  # Calabrian mts   39.3N 16.4E 1677m (S Apennines)
            "16262099999",  # Campobasso      41.1N 15.2E 1093m (Apennines)
            "16300099999",  # Campania mts    40.6N 15.8E  843m (Apennines)
            "16710099999",  # Tripolis, Arcadia 37.5N 22.4E 644m (Greece)
            "16614099999",  # Ptolemaida/Kozani 40.4N 21.3E 660m (Greece)
        ],
    },
    "continental_greece": {
        "label": "Continental / Ionian Greece",
        "stations": [
            "16641099999",  # Kerkira/Corfu   39.6N 19.9E    2m
            "16643099999",  # Preveza         38.9N 20.8E    3m
            "16648099999",  # Larissa         39.7N 22.5E   73m  (Thessaly)
            "16675099999",  # Lamia           38.9N 22.4E   14m
            "16687099999",  # Agrinio         38.2N 21.4E   14m
            "16627099999",  # Alexandroupolis  40.9N 26.0E    7m  (Thrace)
        ],
    },
}
