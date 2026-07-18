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
9. [LOOCV Validation](#loocv-validation)
10. [CORDEX Bias Correction (Server)](#cordex-bias-correction-server)
11. [Visualisation](#visualisation)
12. [Scientific Methods](#scientific-methods)
13. [Dependencies](#dependencies)
14. [References](#references)

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
Precipitation combines an occurrence classifier, conditional quantiles
(p10/p50/p90/p95/p99) and a Gamma conditional-mean; the final daily value is a
distribution-corrected blend of quantile-mapped CERRA and the model estimate
(see the Precipitation section under Scientific Methods).

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
│  LOOCV VALIDATION  (run_loocv_validation.py)                        │
│                                                                     │
│  Phase 1  Build station dataset cache (CERRA nearest + geo feats)  │
│  Phase 2  Leave-one-out loop: retrain on N-1 stations, predict     │
│           at held-out station for each station in the network       │
│  Phase 3  Compute R²/RMSE/Bias and POD/FAR/CSI metrics             │
│           → validation/metrics/*.csv                                │
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
├── downscaling/
│   └── validation_config.py      # Thresholds and flags for LOOCV pipeline
│
├── run_pipeline_cerra_downscaling.py     # Main downscaling pipeline entry point
├── post_processing_cerra_downscaling.py  # Convert quantiles → daily values
├── run_loocv_validation.py       # LOOCV validation pipeline (3-phase)
├── cordex_bias_correct.py        # EQM bias correction for CORDEX (server)
├── ba_postprocessing.py          # CORDEX ensemble statistics (server)
├── plot_ba_data.py               # 7-panel ensemble projection plots
├── plot_delos_sealevel.py        # Sea-level risk visualisation
│
├── validation/                   # Created by run_loocv_validation.py
│   ├── cache/                    # Phase 1 outputs (station obs, CERRA nearest, geo)
│   ├── loocv/                    # Phase 2 outputs — one parquet per station
│   ├── metrics/                  # Phase 3 outputs — 4 CSV skill-score tables
│   └── zone_weights.csv          # Station × zone weight table (--inspect-weights)
│
├── requirements.txt              # Python dependencies
├── CITATIONS.md                  # Research sources consulted
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
| `precip_p10/p50/p90/p95/p99` | (n_times, n_points) float32 | Conditional precip quantiles (p95/p99 = upper-tail bands) |
| `precip_mean` | (n_times, n_points) float32 | Gamma conditional-mean amount |
| `precip_best` | (n_times, n_points) float32 | Hurdle amount `occ_prob × precip_mean` (EQM source) |
| `precip_best_eqm` | (n_times, n_points) float32 | **Final** daily precip: QM(CERRA)+model blend |

### `delos_downscaled_daily.zarr` (from `post_processing_cerra_downscaling.py`)

| Variable | Description |
|---|---|
| `tmean_c`, `tmin_c`, `tmax_c` | Temperature best estimates (p50), °C |
| `rh_pct` | Relative humidity (p50), % |
| `wind_ms`, `gust_ms` | Wind speed (modelled mean), m/s |
| `precip_mm` | Final daily precip = `precip_best_eqm` (QM(CERRA)+model blend), mm |

Read example:
```python
import zarr, pandas as pd, numpy as np

store = zarr.open("downscaling/output/delos_downscaled_daily.zarr", mode="r")
time  = pd.to_datetime(store["time"][:], unit="ns")
tmean = store["tmean_c"][:]   # shape (n_times, n_points)
```

---

## LOOCV Validation

`run_loocv_validation.py` evaluates the statistical downscaling pipeline using
**leave-one-out cross-validation (LOOCV)** across the full training station
network.  It also computes a **raw CERRA baseline** (nearest-grid-cell
reanalysis, no downscaling) for comparison.  All configuration is in
`downscaling/validation_config.py`.

### Running

```bash
# Full 126-station LOOCV (all three phases, Delos zone weights)
python run_loocv_validation.py --delos-phase 1 2 3

# Recompute metrics only (e.g. after changing thresholds or PRECIP_EQM_*)
python run_loocv_validation.py --delos-phase 3

# Environment-specific LOOCV (4 groups × 6 stations; needs the Phase-1 cache)
python run_loocv_validation.py --env-phase 2 3

# Inspect per-station Delos zone similarity weights (see below)
python run_loocv_validation.py --inspect-weights
```

Phase 1 is resumable: if `validation/cache/` files already exist and
`CACHE_OVERWRITE = False` (the default), they are loaded directly.
Phase 2 is also resumable: individual `loocv_{station_id}.parquet` files
that already exist are skipped unless `LOOCV_OVERWRITE = True`.

### Phase 1 — Station dataset preparation

Builds and caches all inputs the LOOCV loop needs:

| Cache file | Contents |
|---|---|
| `validation/cache/station_obs.parquet` | All NOAA GSOD daily observations for VAL_YEAR_START–VAL_YEAR_END |
| `validation/cache/cerra_stations_nearest.parquet` | CERRA values at each station location via **nearest grid cell** (no interpolation) — used as the raw CERRA baseline |
| `validation/cache/station_geo.parquet` | Geographic features per station: coast_dist_km, elevation_m, is_island, dem_northness, … |
| `validation/cache/station_cerra_static.parquet` | CERRA static fields per station: orography, land-sea mask value |
| `downscaling/cache/cerra_stations.parquet` | Cubic-interpolated CERRA at station locations — same file as the main pipeline; created here if not already present |

The nearest-grid-cell extraction (CERRA baseline) uses a KDTree on the CERRA
lat/lon grid.  If a station falls on a sea cell, the nearest land cell is used
instead (LSM-aware patching), consistent with how the main pipeline handles
Delos grid points.

### Phase 2 — LOOCV training and prediction

For each station *s* in the training network:

1. **Remove** station *s* from the training set.
2. **Recompute** zone memberships and Delos-zone similarity weights using the
   remaining N−1 stations (see [Delos zone weights](#delos-zone-weights-in-loocv) below).
3. **Retrain** the full LightGBM pipeline (all variables, all quantiles) on the
   N−1 stations using a temporary directory for model artefacts (auto-deleted
   after prediction).
4. **Predict** at station *s* using its own zone membership and cubic-interpolated
   CERRA features — exactly as the production pipeline predicts at Delos grid points.
5. **Merge** predictions with observed values and nearest-index CERRA values;
   write `validation/loocv/loocv_{station_id}.parquet`.

Each parquet file contains one row per calendar day with columns:

| Column group | Columns |
|---|---|
| Observations | `obs_tmean_c`, `obs_tmin_c`, `obs_tmax_c`, `obs_rh_pct`, `obs_prcp_mm`, `obs_wdsp_ms`, `obs_gust_ms` |
| LOOCV predictions (p50) | `tmean_p50`, `tmin_p50`, `tmax_p50`, `rh_p50`, `precip_p50`, `precip_occ_prob`, `precip_mean`, `precip_best`, `wind_p50`, `gust_p50` |
| LOOCV predictions (p10/p90) | Same variables with `_p10` / `_p90` suffixes (precip also p95/p99) |
| Final precip | `precip_best_eqm` — QM(CERRA)+model blend, added in Phase 3 |
| CERRA nearest baseline | `cerra_tmean`, `cerra_tmin`, `cerra_tmax`, `cerra_rh`, `cerra_wind`, `cerra_gust`, `cerra_precip` (m), `cerra_precip_mm` |

Predicted precipitation for scoring is `precip_best_eqm`: a distribution-corrected
blend of quantile-mapped CERRA and the model's Hurdle amount `occ_prob × precip_mean`
(see the Precipitation subsection under Scientific Methods).

### Phase 3 — Skill metrics

Loads all LOOCV parquet files and computes two families of metrics for both
the LOOCV predictions and the raw CERRA baseline.  Results are written to
`validation/metrics/`.

#### Continuous metrics (R², RMSE, Bias)

Computed on **annual** and **monthly** aggregates for 13 climate indices:

| Index | Definition |
|---|---|
| `tmean` | Annual / monthly mean of daily mean temperature |
| `tmin` | Annual / monthly mean of daily minimum temperature |
| `tmax` | Annual / monthly mean of daily maximum temperature |
| `wind` | Annual / monthly mean wind speed |
| `gust` | Annual / monthly mean wind gust |
| `precip_annual` | Annual total precipitation |
| `pr20` | Annual / monthly count of days with precip > 20 mm |
| `pr50` | Annual / monthly count of days with precip > 50 mm |
| `r99p` | Annual / monthly count of days exceeding the 99th percentile of the station's own observed wet-day distribution |
| `tmax35` | Annual / monthly count of days with tmax > 35 °C |
| `tmax37` | Annual / monthly count of days with tmax > 37 °C |
| `tmin0` | Annual / monthly count of frost days (tmin < 0 °C) |
| `aridity_index` | Annual P / Thornthwaite PET (annual only) |

Output files:

| File | Contents |
|---|---|
| `metrics_loocv_continuous.csv` | R², RMSE, Bias — LOOCV predictions vs observations |
| `metrics_cerra_continuous.csv` | R², RMSE, Bias — raw CERRA baseline vs observations |

Each file has one row per (station_id, index, frequency) combination, plus a
`station_id = ALL` row with pooled metrics across all stations.

#### Contingency metrics (POD, FAR, CSI)

Computed on **daily** data for 7 threshold-exceedance events:

| Event | Definition |
|---|---|
| `wet_1mm` | Wet day: precip > 1 mm |
| `pr20` | Heavy rain: precip > 20 mm |
| `pr50` | Very heavy rain: precip > 50 mm |
| `tmax35` | Hot day: tmax > 35 °C |
| `tmax37` | Extreme heat: tmax > 37 °C |
| `r99p` | Extreme precip: exceeds station-specific 99th percentile wet-day threshold |
| `tmin0` | Frost day: tmin < 0 °C |

Metrics: **POD** (Probability of Detection = H / (H+M)), **FAR** (False Alarm
Ratio = F / (F+H)), **CSI** (Critical Success Index = H / (H+M+F)).

Output files:

| File | Contents |
|---|---|
| `metrics_loocv_contingency.csv` | POD, FAR, CSI — LOOCV predictions |
| `metrics_cerra_contingency.csv` | POD, FAR, CSI — raw CERRA baseline |

### Delos zone weights in LOOCV

The production pipeline trains models optimised for Delos conditions by
up-weighting training stations that are geographically and climatically similar
to each of the N climate zones on Delos (N-Coastal, S-Coastal, Ridge,
Transitional).  These **Delos zone similarity weights** are LightGBM sample
weights — they make the loss function care more about fitting patterns that
represent Delos, not the average Aegean station.

In the LOOCV loop, these weights are recomputed for each fold using the N−1
remaining stations, ensuring the training procedure is identical to production.
The held-out station's predictions are generated by blending the N zone models
according to the **held-out station's own zone membership**, not Delos's.

**What this means for interpreting skill scores:**

- A station with **high zone weights** (similar to Delos) is well represented
  in training even when held out, because other similar stations remain.  Its
  LOOCV skill score is the best estimate of production skill on Delos.
- A station with **low zone weights** (dissimilar to Delos, e.g. high-altitude
  mainland) was never strongly influencing the models even when included in
  training.  Poor LOOCV skill at these stations does not imply poor skill on
  Delos; it means the model was not designed for those conditions.

To inspect the weights:

```bash
python run_loocv_validation.py --inspect-weights
```

This writes `validation/zone_weights.csv` with columns
`station_id, lat, lon, elevation_m, is_island, zone_0_weight, …, dominant_zone`
and prints the table sorted by zone_0 weight.  Weights are normalised so the
mean across stations equals 1.0; values above 1 mean more representative of
that zone on Delos than average.

**Recommended workflow for interpreting results:** filter `metrics_loocv_continuous.csv`
to stations where zone weights are above 1 for their dominant zone.  Those
scores are the primary evidence of model quality for the Delos output.

### Configuration (`downscaling/validation_config.py`)

| Parameter | Default | Description |
|---|---|---|
| `VAL_YEAR_START` / `VAL_YEAR_END` | 1985 / 2005 | Year range for extraction and evaluation |
| `WET_DAY_MM` | 1.0 | Wet-day threshold for r99p and contingency events |
| `PR20_MM` / `PR50_MM` | 20.0 / 50.0 | Heavy / very heavy precipitation thresholds (mm) |
| `TMAX35_C` / `TMAX37_C` | 35.0 / 37.0 | Hot / extreme heat thresholds (°C) |
| `TMIN0_C` | 0.0 | Frost day threshold (°C) |
| `R99P_PERCENTILE` | 99 | Percentile for r99p threshold, computed per station from observed wet days |
| `OCC_PROB_THRESHOLD` | 0.5 | Minimum predicted occurrence probability to assign non-zero precipitation |
| `CERRA_PRECIP_MM_FACTOR` | 1000.0 | Converts CERRA `tp` from metres to mm |
| `LOOCV_FAST_MODE` | False | Reduce LightGBM tree count for faster development runs |
| `LOOCV_OVERWRITE` | False | Recompute existing per-station parquet files |
| `CACHE_OVERWRITE` | False | Recompute Phase 1 cache even if files exist |

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

- **Precipitation model + distribution correction**: a binary occurrence model
  plus conditional quantiles (p10/p50/p90/p95/p99) and a Gamma conditional-mean
  on wet days (daily precip > 0.1 mm).  The model amount `occ_prob × precip_mean`
  is variance-compressed, so the **final** daily value is a blend (see the
  Precipitation subsection below).
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

### Precipitation: distribution-corrected blend

The ML precipitation *amount* is variance-compressed (the Gamma conditional-mean
predicts a near-constant ~5–12 mm on every wet day), which collapses annual totals
to ~55–65 % and misses extremes.  The final field restores both while keeping
day-to-day timing:

```
precip_best_eqm = PRECIP_EQM_BLEND · QM_seasonal(CERRA → obs) + (1 − blend) · (occ_prob × precip_mean)
```

- `QM_seasonal` — interpolation quantile mapping of CERRA to the observed
  distribution (smooth ±45-day DOY windows).  Being monotonic it **preserves
  CERRA's day-matched timing** (CERRA has real daily precip skill) while correcting
  the distribution (totals and >20/>50 mm frequency).
- `occ_prob × precip_mean` — the ML Hurdle amount, added for sub-grid spatial
  structure that CERRA (1–2 cells over Delos) cannot provide.
- `PRECIP_EQM_BLEND = 0.6` (validated: monthly R² 0.17, totals 94 %, extreme-day
  CSI at or above the CERRA baseline).  Donors are station observations paired
  with their CERRA values; set `PRECIP_EQM_ENABLE = False` to fall back to the
  plain two-part `P(wet) × E[P|wet]`.

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

### Soft zone membership and donor-station similarity

The training network spans a wide range of climatic environments (Aegean
islands, mainland coast, high-altitude interior).  To prevent mainland
high-altitude stations from dominating models that are applied to a sea-level
Aegean island, the pipeline uses a two-layer weighting scheme:

**Zone membership** — each location (station or Delos grid point) receives a
soft Gaussian membership weight across N climate zones defined in
`config.ZONE_DEFINITIONS`.  Zones are characterised by coast distance, elevation,
and aspect (northness) in a normalised space.  Memberships sum to 1 and form a
partition of unity, so every location belongs fractionally to every zone.

**Zone similarity** — for each zone k, a Delos zone centroid is defined as the
membership-weighted mean of all Delos grid points belonging to zone k.  Each
training station's similarity to that centroid is computed with Gaussian kernels
on elevation, coast distance, and monthly CERRA temperature climatology.  Island
stations receive a 1.5× bonus reflecting their closer affinity to Delos
conditions.  The resulting per-zone similarity weights are used as LightGBM
sample weights during training.

**Prediction** — at any target location, the N zone models are blended using
that location's own zone memberships as ensemble weights.  A coastal Delos grid
cell draws heavily from the coastal zone model; a ridge cell draws from the
ridge model; transitional cells interpolate smoothly between them.

### Aridity Index

Annual aridity index is computed as P / PET, where P is annual total
precipitation and PET is estimated by the Thornthwaite (1948) method from
monthly mean temperatures.  Values below 0.5 indicate semi-arid conditions.

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
| `rasterio` | DEM sampling (`geo_features.py`, `raster/`) |
| `netCDF4` | Low-level NetCDF4 reading/writing (CORDEX pipeline) |
| `matplotlib`, `cartopy` | Plotting and map projections |
| `pillow`, `requests` | DEM tile download and image handling (`raster/`) |
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

- Jolliffe, I. T., and Stephenson, D. B. (eds.) (2003). *Forecast Verification:
  A Practitioner's Guide in Atmospheric Science*. Wiley, Chichester.
  (Contingency table metrics: POD, FAR, CSI.)
