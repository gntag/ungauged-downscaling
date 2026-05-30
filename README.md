# Delos Climate Downscaling Pipeline (v1)

Statistical downscaling of daily climate variables to the Delos island grid
using LightGBM quantile regression trained on CERRA reanalysis and NOAA GSOD
station observations, with Empirical Quantile Mapping (EQM) bias correction
of CORDEX EUR-11 RCM projections.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Quick Start](#quick-start)
4. [Directory Structure](#directory-structure)
5. [Data Requirements](#data-requirements)
6. [Configuration](#configuration)
7. [Running the Pipeline](#running-the-pipeline)
8. [Output Format](#output-format)
9. [CORDEX Bias Correction (Server)](#cordex-bias-correction-server)
10. [Visualisation](#visualisation)
11. [Scientific Methods](#scientific-methods)
12. [Dependencies](#dependencies)
13. [References](#references)

---

## Overview

This project produces high-resolution (~250 m) daily climate projections for
the island of Delos (Aegean Sea, Greece) for the period 1985–2100 under
historical, RCP 4.5, and RCP 8.5 scenarios.

**Variables produced:**

| Variable | Description | Units |
|---|---|---|
| tmean | Daily mean temperature | °C |
| tmin | Daily minimum temperature | °C |
| tmax | Daily maximum temperature | °C |
| rh | Relative humidity | % |
| precip | Precipitation (expected daily total) | mm |
| wind | Mean 10-m wind speed | m/s |
| gust | Maximum wind gust | m/s |

For each variable (except precipitation), three quantile predictions are
produced: p10, p50, p90.  Wind and gust additionally receive a mean model.
Precipitation uses a two-part model (occurrence probability + conditional
quantile distribution).

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│  DATA ACQUISITION                                                   │
│                                                                     │
│  cerra_data/cerra_data.py         ← CERRA NetCDF via CDS API       │
│  Station_Data/Station_Data.py     ← NOAA GSOD station CSVs         │
│  land_mask/download_land_masks.py ← CERRA + CORDEX land masks      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  CERRA-BASED STATISTICAL DOWNSCALING  (run_pipeline_cerra_...)      │
│                                                                     │
│  1. cerra_extract.py       Extract CERRA values at station locs     │
│  2. station_ingest.py      Load + clean NOAA GSOD observations      │
│  3. geo_features.py        Compute coast_dist_km, is_island         │
│  4. similarity.py          Gaussian similarity weights per station  │
│  5. train.py               Train LightGBM p10/p50/p90 models        │
│  6. cerra_extract.py       Extract CERRA values at Delos grid       │
│  7. predict.py             Apply models → quantile Zarr             │
│  8. post_processing...py   p50 / gamma mean → daily-values Zarr     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────────┐
│  CORDEX EQM BIAS CORRECTION  (server-side)                          │
│                                                                     │
│  cordex_extract.py         Extract CORDEX EUR-11 to Delos grid      │
│  cordex_bias_correct.py    Fit + apply EQM transfer functions       │
│  ba_postprocessing.py      Compute climate statistics per model     │
│  plot_ba_data.py           7-panel ensemble projection maps         │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Clone and install dependencies

```bash
git clone https://github.com/<your-org>/delos-downscaling.git
cd delos-downscaling
pip install -r requirements.txt
```

### 2. Configure paths

```bash
cp downscaling/config_template.py downscaling/config.py
# Edit downscaling/config.py:
#   S2COAST_DIR  ← path to S2Coast 2023 shapefile directory
#   (all other paths are auto-derived from the project root)
```

For plotting scripts:
```bash
cp downscaling/plt_config_template.py downscaling/plt_config.py
# Edit downscaling/plt_config.py:
#   S2COAST_SHP  ← path to S2Coast-2023_Polyline_diss.shp
```

### 3. Download input data

```bash
# CERRA reanalysis (requires CDS API key in ~/.cdsapirc)
python cerra_data/cerra_data.py --start-year 1985 --end-year 2005

# NOAA GSOD station observations
python Station_Data/Station_Data.py --start-year 1985 --end-year 2005

# Land-sea masks
python land_mask/download_land_masks.py --skip-cordex
```

### 4. Run the downscaling pipeline

```bash
python run_pipeline_cerra_downscaling.py
```

---

## Directory Structure

```
delos-downscaling/
├── downscaling/                  # Core Python package
│   ├── config_template.py        # Template — copy to config.py and edit
│   ├── plt_config_template.py    # Template — copy to plt_config.py and edit
│   ├── bias_correct.py           # Empirical Quantile Mapping (EQM)
│   ├── cerra_extract.py          # CERRA spatial interpolation
│   ├── cordex_extract.py         # CORDEX spatial interpolation (server)
│   ├── geo_features.py           # coast_dist_km, is_island computation
│   ├── predict.py                # Apply LightGBM models → Zarr output
│   ├── similarity.py             # Gaussian donor-station weighting
│   ├── station_ingest.py         # Load and clean NOAA GSOD station data
│   └── train.py                  # LightGBM model training
│
├── cerra_data/
│   └── cerra_data.py             # Download CERRA NetCDF via CDS API
│
├── Station_Data/
│   └── Station_Data.py           # Download NOAA GSOD station CSVs
│
├── land_mask/
│   └── download_land_masks.py    # Download CERRA + CORDEX land masks
│
├── run_pipeline_cerra_downscaling.py  # Main pipeline entry point
├── post_processing_cerra_downscaling.py  # Convert quantiles → daily values
├── cordex_bias_correct.py        # EQM bias correction for CORDEX (server)
├── ba_postprocessing.py          # CORDEX ensemble statistics (server)
├── plot_ba_data.py               # 7-panel ensemble projection plots
├── plot_delos_sealevel.py        # Sea-level risk visualisation
└── README.md
```

---

## Data Requirements

### Required (local machine)

| Data | Source | Script | Notes |
|---|---|---|---|
| CERRA reanalysis | Copernicus CDS | `cerra_data/cerra_data.py` | Requires CDS API key |
| NOAA GSOD stations | NOAA CDO | `Station_Data/Station_Data.py` | Free, no key needed |
| CERRA land mask | Copernicus CDS | `land_mask/download_land_masks.py` | |
| Delos grid CSV | Private | — | `aurehly_PCs_dilos.csv` in project root |
| S2Coast 2023 | coastline.eu | Manual download | Free academic registration |

### Required (server only, CORDEX pipeline)

| Data | Source | Notes |
|---|---|---|
| CORDEX EUR-11 NetCDF | ESGF / local archive | 7 GCM-RCM pairs × 3 scenarios |
| CORDEX land masks | Copernicus CDS | `land_mask/download_land_masks.py --skip-cerra` |

### CDS API Setup

Create `~/.cdsapirc` with your Copernicus credentials:
```
url: https://cds.climate.copernicus.eu/api
key: <your-personal-api-key>
```
Register at [https://cds.climate.copernicus.eu](https://cds.climate.copernicus.eu).

---

## Configuration

All machine-specific settings live in two files that are **not** committed to
version control.  Copy the templates and edit:

### `downscaling/config.py`

Key settings (most paths are auto-derived from the project root):

```python
# Path to the S2Coast 2023 shapefile directory  ← REQUIRED EDIT
S2COAST_DIR = Path("/path/to/S2Coast2023_ShapeFile_vector")

# Server-only: CORDEX archive paths
CORDEX_DIR        = Path("/mnt/data/CORDEX")
CORDEX_OUTPUT_DIR = Path("/mnt/data/Dilos/cordex_downscaled_v2/output")
```

### `downscaling/plt_config.py`

```python
# Path to the S2Coast dissolved polyline shapefile  ← REQUIRED EDIT
S2COAST_SHP = "/path/to/S2Coast2023_ShapeFile_vector/S2Coast-2023_Polyline_diss.shp"
```

---

## Running the Pipeline

### Full CERRA downscaling pipeline

```bash
python run_pipeline_cerra_downscaling.py
```

This runs seven sequential steps:
1. Extract CERRA values at station locations
2. Load and clean NOAA GSOD station observations
3. Compute geographic features (coast distance, is_island)
4. Compute donor-station similarity weights
5. Train LightGBM p10/p50/p90 models (+ mean for wind/gust)
6. Extract CERRA values at the Delos grid
7. Generate quantile predictions → `downscaling/output/delos_downscaled.zarr`

### Post-processing (quantiles → daily values)

```bash
python post_processing_cerra_downscaling.py --both
# or
python post_processing_cerra_downscaling.py --daily   # Zarr only
python post_processing_cerra_downscaling.py --plots   # plots from existing Zarr
```

### Individual training (one variable at a time)

```python
from downscaling import config, station_ingest, cerra_extract, geo_features, similarity, train

stations = station_ingest.load_stations(config.STATION_DIR, config.YEAR_START, config.YEAR_END)
cerra    = cerra_extract.extract_all(config.CERRA_DIR, stations[["station_id","lat","lon"]]
                                     .rename(columns={"station_id":"id"}),
                                     config.YEAR_START, config.YEAR_END)
geo      = geo_features.compute_geo_features(...)
weights  = similarity.compute_similarity(geo, delos_geo)
df       = train.build_training_df(stations, cerra, geo, weights)
models   = train.train_all(df, config.MODEL_DIR, vars=["tmean"])
```

---

## Output Format

### `delos_downscaled.zarr` (from `predict.py`)

Zarr store compressed with Zstandard level 5, chunked (365, 50):

| Variable | Shape | Description |
|---|---|---|
| `time` | (n_times,) int64 | Nanoseconds since Unix epoch |
| `lon`, `lat` | (n_points,) float64 | Grid coordinates |
| `elevation_m` | (n_points,) float32 | Elevation |
| `<var>_p10/p50/p90` | (n_times, n_points) float32 | Quantile predictions |
| `<var>_mean` | (n_times, n_points) float32 | Wind/gust mean models |
| `precip_occurrence_prob` | (n_times, n_points) float32 | P(wet day) |
| `precip_p10/p50/p90` | (n_times, n_points) float32 | Conditional precip quantiles |

### `delos_downscaled_daily.zarr` (from `post_processing_cerra_downscaling.py`)

| Variable | Description |
|---|---|
| `tmean_c`, `tmin_c`, `tmax_c` | Temperature best estimates (p50), °C |
| `rh_pct` | Relative humidity (p50), % |
| `wind_ms`, `gust_ms` | Wind speed (modelled mean), m/s |
| `precip_mm` | Expected daily precip = P(wet) × gamma-mean, mm |

Read example:
```python
import zarr, pandas as pd, numpy as np

store = zarr.open("downscaling/output/delos_downscaled_daily.zarr", mode="r")
time  = pd.to_datetime(store["time"][:], unit="ns")
tmean = store["tmean_c"][:]   # shape (n_times, n_points)
```

---

## CORDEX Bias Correction (Server)

The CORDEX branch of the pipeline runs on a server hosting the EUR-11 archive.
It is not required for the CERRA-based local downscaling.

```bash
# 1. Extract CORDEX to Delos grid (all 7 models × 3 scenarios)
python downscaling/cordex_extract.py

# 2. Bias correct using EQM
python cordex_bias_correct.py

# 3. Compute ensemble statistics per variable
python ba_postprocessing.py

# 4. Plot 7-panel ensemble projections
python plot_ba_data.py
```

**CORDEX models included:**

| GCM | RCM | Calendar |
|---|---|---|
| MPI-M-MPI-ESM-LR | MPI-CSC-REMO2009 | Gregorian |
| ICHEC-EC-EARTH | CLMcom-CCLM4-8-17 | Gregorian |
| MOHC-HadGEM2-ES | SMHI-RCA4 | 360-day |
| MOHC-HadGEM2-ES | KNMI-RACMO22E | 360-day |
| CNRM-CERFACS-CNRM-CM5 | KNMI-RACMO22E | Gregorian |
| MPI-M-MPI-ESM-LR | SMHI-RCA4 | Gregorian |
| ICHEC-EC-EARTH | DMI-HIRHAM5 | Gregorian |

---

## Visualisation

```bash
# Mean annual / threshold-exceedance maps (CERRA downscaling)
python post_processing_cerra_downscaling.py --plots

# 7-panel ensemble projection maps (CORDEX pipeline)
python plot_ba_data.py

# Delos sea-level risk figure
python plot_delos_sealevel.py
```

---

## Scientific Methods

### Statistical downscaling

LightGBM quantile regression maps daily CERRA reanalysis predictors (at each
Delos grid point) to observed climate distributions from nearby NOAA GSOD
stations.  Key design choices:

- **Two-part precipitation model**: separate binary occurrence model and
  conditional quantile models on wet days (daily precip > 0.1 mm), avoiding
  the mixed discrete-continuous distribution problem.
- **Donor-station weighting**: Gaussian similarity kernels on elevation, coast
  distance, monthly CERRA temperature climatology, and island-status ensure
  that stations resembling Delos (low elevation, coastal, Aegean island)
  contribute more to training.
- **LSM-aware interpolation**: targets falling on sea cells in CERRA are
  interpolated from the nearest land cells, preventing contamination by
  SST-driven marine surface values.
- **Physical consistency**: tmin ≤ tmean ≤ tmax enforced by sorting (not
  clipping), and quantile ordering p10 ≤ p50 ≤ p90 enforced element-wise
  after prediction.

### Empirical Quantile Mapping (EQM)

EQM corrects systematic biases in CORDEX RCM output by matching empirical
quantile distributions between the model and the downscaled truth:

- Calibration period: 1985–2005 (historical overlap)
- 250 quantile levels; ±15-day DOY window (≈ 650 training samples per point)
- Wet-day threshold: 0.1 mm/day (WMO standard)
- Extrapolation: mean-delta correction for values outside the training range
- 360-day calendar models (HadGEM2): DOY mapped to Gregorian via linear
  interpolation before truth lookup

Reference: Gudmundsson et al. (2012), Hydrol. Earth Syst. Sci., 16(9).

### Expected daily precipitation

Daily precipitation is derived from the two-part model as:
```
E[P] = P(wet day) × E[P | wet day]
```
where E[P | wet day] is estimated from the conditional p50 and p90 by fitting
a gamma distribution: the p90/p50 ratio uniquely constrains the shape parameter
k via a precomputed lookup table, and the scale is recovered from p50 and the
gamma median.

---

## Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Core packages:

| Package | Purpose |
|---|---|
| `lightgbm` | Quantile regression models |
| `numpy`, `pandas` | Numerical computing |
| `xarray`, `zarr`, `numcodecs` | NetCDF / Zarr I/O |
| `scipy` | Delaunay triangulation, CloughTocher interpolation |
| `geopandas`, `shapely` | Geographic feature computation |
| `netCDF4` | Low-level NetCDF4 reading/writing (CORDEX pipeline) |
| `matplotlib` | Plotting |
| `cdsapi` | CDS API client (data download) |

---

## References

- Gudmundsson, L., Bremnes, J. B., Haugen, J. E., and Engen‐Skaugen, T. (2012).
  Technical note: Downscaling RCM precipitation to the station scale using
  statistical transformations — a comparison of methods.
  *Hydrology and Earth System Sciences*, 16(9), 3383–3390.
  https://doi.org/10.5194/hess-16-3383-2012

- Déqué, M. (2007). Frequency of precipitation and temperature extremes over
  France in an anthropogenic scenario: Model results and statistical correction
  according to observed values. *Global and Planetary Change*, 57(1–2), 16–26.

- Skamarock, W. C., et al. (2021). CERRA: The Copernicus European Regional
  ReAnalysis. *Copernicus Climate Change Service (C3S)*.

- Thornthwaite, C. W. (1948). An approach toward a rational classification of
  climate. *Geographical Review*, 38(1), 55–94.

- UNEP (1992). World Atlas of Desertification. United Nations Environment
  Programme, Nairobi.
