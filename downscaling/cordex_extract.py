#!/usr/bin/env python3
"""
cordex_extract.py
=================
Extract CORDEX EUR-11 daily climate variables to the Delos grid points
using the same cubic spatial interpolation used for CERRA (cerra_extract.py).

WHAT THIS SCRIPT DOES
---------------------
For each CORDEX GCM-RCM pair × scenario combination:
  1. Reads daily NetCDF files from the CORDEX archive (tas, tasmin, tasmax,
     pr, sfcWind and either hurs or huss+psl for relative humidity).
  2. Spatially interpolates each variable to the Delos grid using the
     CloughTocher2D cubic interpolator (via cerra_extract helpers).
  3. Clips physical bounds (non-negative precipitation/wind, 0–100 % RH).
  4. Writes a compact NetCDF4 file (zlib compressed) per model × scenario.

CORDEX MODELS AND SCENARIOS
----------------------------
Seven GCM-RCM pairs are processed (see config.CORDEX_ALL_MODELS):
  historical  : 1950–2005  (or 1970–2005 for some models)
  rcp45       : 2006–2100
  rcp85       : 2006–2100

RELATIVE HUMIDITY HANDLING
---------------------------
Two subsets of models differ in the available humidity variable:
  hurs models : hurs (near-surface relative humidity, %) is available directly.
  huss models : only huss (specific humidity, kg/kg) and psl (surface pressure,
                Pa) are available.  RH is derived via the Magnus formula:
                  es = 611.2 · exp(17.67 · T_C / (T_C + 243.5))   [Pa]
                  e  = q · p / (0.622 + 0.378 · q)
                  RH = 100 · e / es

LAND-MASK-AWARE INTERPOLATION
------------------------------
Each CORDEX model has its own land-sea mask (sftlf_<model>.nc, %).  Points
classified as sea (sftlf ≤ 50 %) near a target are avoided during patch
selection, mirroring the LSM-aware patching in cerra_extract.py.

COMMAND-LINE USAGE
------------------
  # Process all 7 models × 3 scenarios sequentially:
  python cordex_extract.py

  # Process a single model + scenario:
  python cordex_extract.py --model ICHEC-EC-EARTH_DMI-HIRHAM5 --scenario historical

  --model and --scenario must be given together or both omitted.

OUTPUT
------
  {output_dir}/{model}_{scenario}.nc   — NetCDF4, dims: (time, point)
  {log_dir}/{model}_{scenario}.log     — per-run log file (mirrors stdout)

CONFIGURATION
-------------
All paths are read from downscaling/config.py:
  CORDEX_DIR          : root directory of the CORDEX archive
  CORDEX_OUTPUT_DIR   : directory where output NetCDF files are written
  CORDEX_LAND_MASK_DIR: directory containing sftlf_<model>.nc files
"""
import argparse
import datetime
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Tuple

# Make the downscaling package importable when this script is run directly
# (i.e. not via `python -m downscaling.cordex_extract`).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import netCDF4 as nc4
import numpy as np
import pandas as pd
from config import (
    KELVIN_OFFSET,        # 273.15 — offset to convert Kelvin to Celsius
    MIN_FREE_GB,          # abort if output disk has less than this many GB free
    CORDEX_DIR,           # root of the CORDEX archive (server path)
    CORDEX_OUTPUT_DIR,    # where output NC files go (server path)
    CORDEX_YEAR_RANGES,   # dict: scenario → (year_start, year_end)
    CORDEX_VAR_MAP,       # dict: var_key → (nc_var_name, unit_conv_fn)
    CORDEX_HURS_MODELS,   # set of model names that provide hurs directly
    CORDEX_ALL_MODELS,    # ordered list of all 7 GCM-RCM model names
    CORDEX_LAND_MASK_DIR, # directory of sftlf_<model>.nc land-fraction files
)
from cerra_extract import (
    _precompute_patches,      # build per-target patch + Delaunay triangulation
    _patch_bbox,              # compute minimal bounding sub-region over all patches
    _interpolate_subregion,   # cubic interpolation of one location at all time steps
)


# ---------------------------------------------------------------------------
# Tee: simultaneously write stdout to both the console and a log file
# ---------------------------------------------------------------------------

class _Tee:
    """
    Redirect writes to multiple file-like objects at once.

    Used to mirror all print() output to a per-run log file without
    wrapping every print() call in custom logging logic.
    """
    def __init__(self, *files):
        self._files = files

    def write(self, data):
        for f in self._files:
            f.write(data)
            f.flush()

    def flush(self):
        for f in self._files:
            f.flush()


def _ts() -> str:
    """Return the current wall-clock time as HH:MM:SS (for log prefixes)."""
    return datetime.datetime.now().strftime('%H:%M:%S')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse CLI arguments and dispatch to _run_one() for each model × scenario.

    If --model / --scenario are omitted, all combinations from config are run
    sequentially.  Each run's stdout is mirrored to a log file in log_dir so
    that long batch runs can be monitored and reviewed after the fact.
    """
    parser = argparse.ArgumentParser(description='CORDEX EUR-11 → Delos grid extraction')
    parser.add_argument('--model',    choices=CORDEX_ALL_MODELS,
                        help='Model to process (default: all)')
    parser.add_argument('--scenario', choices=list(CORDEX_YEAR_RANGES),
                        help='Scenario to process (default: all)')
    parser.add_argument('--cordex-dir',    default=str(CORDEX_DIR))
    parser.add_argument('--output-dir',    default=str(CORDEX_OUTPUT_DIR))
    parser.add_argument('--delos-csv',     default='aurehly_PCs_dilos.csv',
                        help='CSV with Delos grid points (lon, lat, VALUE)')
    parser.add_argument('--log-dir',       default=None,
                        help='Log directory (default: <script_dir>/logs)')
    parser.add_argument('--land-mask-dir', default=str(CORDEX_LAND_MASK_DIR),
                        help='Directory containing sftlf_*.nc files (default from config)')
    args = parser.parse_args()

    # Both --model and --scenario must be provided together, or both omitted.
    if bool(args.model) != bool(args.scenario):
        parser.error('--model and --scenario must be given together or both omitted')

    cordex_dir = Path(args.cordex_dir)
    output_dir = Path(args.output_dir)
    log_dir    = Path(args.log_dir) if args.log_dir else Path(_THIS_DIR) / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build the list of (model, scenario) pairs to process.
    if args.model:
        runs = [(args.model, args.scenario)]
    else:
        runs = [(m, s) for m in CORDEX_ALL_MODELS for s in CORDEX_YEAR_RANGES]

    for model, scenario in runs:
        logfile = log_dir / '{}_{}.log'.format(model, scenario)
        with open(str(logfile), 'w') as lf:
            orig_stdout = sys.stdout
            sys.stdout  = _Tee(orig_stdout, lf)
            try:
                print('{} START  {} {}'.format(_ts(), model, scenario))
                _run_one(model, scenario, cordex_dir, output_dir, args.delos_csv,
                         Path(args.land_mask_dir))
                print('{} DONE   {} {}'.format(_ts(), model, scenario))
            except Exception as exc:
                print('{} ERROR  {} {}: {}'.format(_ts(), model, scenario, exc))
            finally:
                sys.stdout = orig_stdout


# ---------------------------------------------------------------------------
# Single model + scenario run
# ---------------------------------------------------------------------------

def _run_one(model, scenario, cordex_dir, output_dir, delos_csv, land_mask_dir=None):
    # type: (str, str, Path, Path, str, Path) -> None
    """
    Extract one CORDEX model × scenario to the Delos grid and write output.

    Steps:
      1. Load Delos grid coordinates from the CSV.
      2. Load the CORDEX model grid (lat/lon 2-D arrays).
      3. Load the model's land-sea mask and precompute per-point patches.
      4. Discover and load all relevant NetCDF files for the scenario years.
      5. Interpolate and write the output NetCDF.
    """
    if land_mask_dir is None:
        land_mask_dir = CORDEX_LAND_MASK_DIR

    model_dir         = cordex_dir / model
    year_start, year_end = CORDEX_YEAR_RANGES[scenario]

    if not model_dir.exists():
        print('ERROR: model directory not found: {}'.format(model_dir))
        return

    # Abort early if the output disk is running low.
    _check_disk(output_dir)

    out_path = output_dir / '{}_{}.nc'.format(model, scenario)
    if out_path.exists():
        print('Output already exists, overwriting: {}'.format(out_path))

    # Load Delos grid coordinates; VALUE column holds elevation (metres).
    delos_lat, delos_lon, delos_elev = _load_delos(delos_csv)
    print('Delos grid: {} points  lat=[{:.4f}, {:.4f}]  lon=[{:.4f}, {:.4f}]'.format(
        len(delos_lat), delos_lat.min(), delos_lat.max(),
        delos_lon.min(), delos_lon.max()))

    # Load the 2-D geographic coordinates of the CORDEX model grid.
    lat2d, lon2d = _load_model_grid(model_dir)
    ny, nx       = lat2d.shape
    print('Model grid: {}×{}'.format(ny, nx))

    # Load the model land-sea mask and compute per-point interpolation patches.
    is_land  = _load_model_lsm(model, land_mask_dir)
    patch_info = _precompute_patches(lat2d, lon2d, np.column_stack([delos_lat, delos_lon]), is_land)
    y_min, y_max, x_min, x_max = _patch_bbox(patch_info, ny, nx)
    # Convert to sub-region-relative indices for _interpolate_subregion().
    rel_info = [(rows - y_min, cols - x_min, tri, tgt) for rows, cols, tri, tgt in patch_info]
    print('Sub-region: y=[{}:{}], x=[{}:{}]'.format(y_min, y_max, x_min, x_max))

    _extract_and_write(
        model_dir, model, scenario, year_start, year_end,
        rel_info, y_min, y_max, x_min, x_max,
        delos_lat, delos_lon, delos_elev, out_path,
    )
    print('Done → {}'.format(out_path))


# ---------------------------------------------------------------------------
# Land mask
# ---------------------------------------------------------------------------

def _load_model_lsm(model_name, land_mask_dir=None):
    # type: (str, Path) -> np.ndarray
    """
    Load the land-fraction mask for a CORDEX model and return a flat bool array.

    CORDEX land-fraction files (sftlf) use values in % (0–100).
    Cells with sftlf > 50 % are classified as land (True).

    The flat array shape (ny*nx,) matches the convention used by
    _precompute_patches() in cerra_extract.py.
    """
    if land_mask_dir is None:
        land_mask_dir = CORDEX_LAND_MASK_DIR
    lsm_path = land_mask_dir / 'sftlf_{}.nc'.format(model_name)
    if not lsm_path.exists():
        sys.exit('ERROR: land mask not found for model {}: {}'.format(model_name, lsm_path))
    ds    = nc4.Dataset(str(lsm_path))
    sftlf = np.array(ds.variables['sftlf'][:])
    ds.close()
    return (sftlf > 50.0).ravel()   # True = land


# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------

def _load_delos(csv_path):
    # type: (str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]
    """
    Read the Delos grid coordinates CSV.

    Expected columns: lat, lon, VALUE (elevation in metres above sea level).
    Returns three 1-D arrays: (delos_lat, delos_lon, delos_elev).
    """
    df = pd.read_csv(csv_path)
    return df['lat'].values, df['lon'].values, df['VALUE'].values


def _load_model_grid(model_dir):
    # type: (Path) -> Tuple[np.ndarray, np.ndarray]
    """
    Read the 2-D geographic coordinate arrays (lat, lon) from the first
    available CORDEX temperature file (tas_*.nc).

    CORDEX EUR-11 files store coordinates as 2-D 'lat' and 'lon' variables
    because the native grid is a rotated pole projection.
    """
    files = sorted(model_dir.glob('tas_*.nc'))
    if not files:
        sys.exit('ERROR: no tas_*.nc files in {}'.format(model_dir))
    ds    = nc4.Dataset(str(files[0]))
    lat2d = np.array(ds.variables['lat'][:])
    lon2d = np.array(ds.variables['lon'][:])
    ds.close()
    return lat2d, lon2d


def _interpolate_all_locs(data, rel_info):
    # type: (np.ndarray, list) -> np.ndarray
    """
    Interpolate a 3-D sub-region array to all Delos locations.

    Parameters
    ----------
    data     : (T, H_sub, W_sub) float array — spatial sub-region for T time steps.
    rel_info : list of (rel_rows, rel_cols, tri, tgt) from _precompute_patches.

    Returns
    -------
    (T, N_points) float array — interpolated value at each Delos point per time step.
    """
    return np.column_stack(
        [_interpolate_subregion(data, rows, cols, tri, tgt) for rows, cols, tri, tgt in rel_info]
    )


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _date_suffix(path):
    # type: (Path) -> str
    """
    Extract the YYYYMMDD-YYYYMMDD date suffix from a CORDEX filename.

    CORDEX filenames follow the convention:
      <var>_EUR-11_<GCM>_<scenario>_<member>_<RCM>_<version>_day_<start>-<end>.nc
    Returns the '<start>-<end>' string, or '' if not found.
    """
    m = re.search(r'_day_(\d{8}-\d{8})\.nc$', path.name)
    return m.group(1) if m else ''


def _find_files(model_dir, nc_var, scenario, year_start, year_end):
    # type: (Path, str, str, int, int) -> Dict[str, Path]
    """
    Discover CORDEX NetCDF files for one variable and scenario.

    Returns a dict mapping date_suffix → Path for all files whose year range
    overlaps [year_start, year_end].  The dict key (date suffix) is used to
    align files for different variables that cover the same time period.

    Only files whose filename year range overlaps the requested period are
    returned; files entirely outside [year_start, year_end] are skipped.
    """
    result = {}
    for p in sorted(model_dir.glob('{}_EUR-11_*_{}_*.nc'.format(nc_var, scenario))):
        # Extract start and end years from the filename date suffix.
        m = re.search(r'_day_(\d{4})\d{4}-(\d{4})\d{4}\.nc$', p.name)
        if m and int(m.group(1)) <= year_end and int(m.group(2)) >= year_start:
            result[_date_suffix(p)] = p
    return result


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------

def _load_subregion(fpath, nc_var, year_start, year_end, y_min, y_max, x_min, x_max):
    # type: (Path, str, int, int, int, int, int, int) -> tuple
    """
    Load a spatial sub-region of one CORDEX variable for the year range.

    Handles both standard and 360-day (HadGEM2) calendars via nc4.num2date.
    Filters time steps to [year_start, year_end] before loading data to
    avoid reading years that are outside the requested scenario window.

    Returns (dates, sub_array, calendar):
      dates     : list of cftime date objects for retained time steps
      sub_array : (T_filtered, H_sub, W_sub) float array
      calendar  : calendar string ('standard', '360_day', etc.)
    """
    ds       = nc4.Dataset(str(fpath))
    tvar     = ds.variables['time']
    calendar = getattr(tvar, 'calendar', 'standard')
    units    = tvar.units
    raw_times = np.array(tvar[:])
    # Convert numeric time to cftime date objects (handles 360-day calendars).
    dates = nc4.num2date(raw_times, units, calendar)

    # Filter to the requested year range.
    mask = np.array([year_start <= d.year <= year_end for d in dates])
    if not np.any(mask):
        ds.close()
        return None, None, calendar

    sub = np.array(ds.variables[nc_var][mask, y_min:y_max, x_min:x_max], dtype=float)
    ds.close()

    filtered_dates = [d for d, m in zip(dates, mask) if m]
    return filtered_dates, sub, calendar


def _rh_from_huss(q, T_K, p_Pa):
    # type: (np.ndarray, np.ndarray, np.ndarray) -> np.ndarray
    """
    Derive near-surface relative humidity from specific humidity, temperature,
    and surface pressure using the Magnus/Bolton approximation.

    Parameters
    ----------
    q    : specific humidity (kg/kg)
    T_K  : air temperature (K)
    p_Pa : surface or sea-level pressure (Pa)

    Returns
    -------
    Relative humidity (%), clipped to [0, 100].

    Method
    ------
    Saturation vapour pressure (Magnus formula):
      es = 611.2 · exp(17.67 · T_C / (T_C + 243.5))   [Pa]
    Vapour pressure from specific humidity:
      e  = q · p / (0.622 + 0.378 · q)                 [Pa]
    Relative humidity:
      RH = 100 · e / es                                 [%]
    """
    T_C   = T_K - KELVIN_OFFSET
    es_Pa = 611.2 * np.exp(17.67 * T_C / (T_C + 243.5))
    e_Pa  = q * p_Pa / (0.622 + 0.378 * q)
    return np.clip(100.0 * e_Pa / es_Pa, 0.0, 100.0)


# ---------------------------------------------------------------------------
# Main extraction and writing loop
# ---------------------------------------------------------------------------

def _extract_and_write(
    model_dir,     # type: Path
    model_name,    # type: str
    scenario,      # type: str
    year_start,    # type: int
    year_end,      # type: int
    rel_info,      # type: list
    y_min, y_max, x_min, x_max,  # type: int
    delos_lat,     # type: np.ndarray
    delos_lon,     # type: np.ndarray
    delos_elev,    # type: np.ndarray
    out_path,      # type: Path
):
    # type: (...) -> None
    """
    Iterate over CORDEX date-range files, interpolate each variable to the
    Delos grid, then write the combined time series to a NetCDF4 file.

    STRUCTURE
    ---------
    CORDEX archives are split into overlapping date-range files (e.g.
    "19700101-20051231").  We iterate over the date-range keys of the
    temperature (tas) files and use the same key to locate companion files
    for other variables.  Missing companion files produce NaN slices.

    CALENDAR HANDLING
    -----------------
    The output NetCDF preserves the source calendar (e.g. '360_day' for
    HadGEM2 models).  This means the output may have 360 time steps per year
    for those models.  Downstream code in bias_correct.py uses
    map_360_to_365() to align 360-day data with the 365-day CERRA reference.
    """
    n_points  = len(delos_lat)
    use_hurs  = model_name in CORDEX_HURS_MODELS  # True → hurs available directly

    # --- Discover all relevant files for this model/scenario -----------------
    file_index = {}  # type: Dict[str, Dict[str, Path]]
    for var_key, (nc_name, _) in CORDEX_VAR_MAP.items():
        file_index[nc_name] = _find_files(model_dir, nc_name, scenario, year_start, year_end)
    # Load the appropriate humidity variable set based on model capability.
    if use_hurs:
        file_index['hurs'] = _find_files(model_dir, 'hurs', scenario, year_start, year_end)
    else:
        file_index['huss'] = _find_files(model_dir, 'huss', scenario, year_start, year_end)
        file_index['psl']  = _find_files(model_dir, 'psl',  scenario, year_start, year_end)

    tas_file_index = file_index['tas']
    if not tas_file_index:
        print('  WARNING: no tas files for {} {} in {}-{}'.format(
            model_name, scenario, year_start, year_end))
        return

    # Accumulate time steps and interpolated arrays across all date-range files.
    all_dates    = []
    calendar_out = 'standard'
    chunks       = {v: [] for v in ['tmean', 'tmin', 'tmax', 'rh', 'precip', 'wind']}

    for suffix, tas_path in sorted(tas_file_index.items()):
        print('  {}'.format(suffix), end='', flush=True)

        # Load and interpolate temperature (tas) — always required.
        dates, tas_sub, calendar = _load_subregion(
            tas_path, 'tas', year_start, year_end, y_min, y_max, x_min, x_max)
        if dates is None:
            print()
            continue
        # Capture the calendar on the first successful load.
        if not all_dates:
            calendar_out = calendar

        all_dates.extend(dates)
        T = len(dates)   # number of time steps in this date-range file

        # Convert Kelvin → Celsius and interpolate to all Delos points.
        chunks['tmean'].append(_interpolate_all_locs(tas_sub - KELVIN_OFFSET, rel_info))

        # --- Process all other variables using CORDEX_VAR_MAP ----------------
        for var_key, (nc_name, conv) in CORDEX_VAR_MAP.items():
            if var_key == 'tmean':
                continue   # already handled above
            fpath = file_index[nc_name].get(suffix)
            if fpath is None:
                print('\n    MISSING {} for {}'.format(nc_name, suffix), end='')
                chunks[var_key].append(np.full((T, n_points), np.nan))
                continue
            dates2, sub, _ = _load_subregion(
                fpath, nc_name, year_start, year_end, y_min, y_max, x_min, x_max)
            if dates2 is None:
                chunks[var_key].append(np.full((T, n_points), np.nan))
                continue
            # conv is the unit-conversion lambda from CORDEX_VAR_MAP (e.g. kg/m²/s → mm/day).
            chunks[var_key].append(_interpolate_all_locs(conv(sub), rel_info))

        # --- Relative humidity -----------------------------------------------
        if use_hurs:
            # Model provides hurs directly as % — no conversion needed.
            fpath = file_index['hurs'].get(suffix)
            if fpath:
                _, hurs_sub, _ = _load_subregion(
                    fpath, 'hurs', year_start, year_end, y_min, y_max, x_min, x_max)
                if hurs_sub is not None:
                    chunks['rh'].append(
                        np.clip(_interpolate_all_locs(hurs_sub, rel_info), 0.0, 100.0))
                else:
                    chunks['rh'].append(np.full((T, n_points), np.nan))
            else:
                print('\n    MISSING hurs for {}'.format(suffix), end='')
                chunks['rh'].append(np.full((T, n_points), np.nan))
        else:
            # Model provides huss + psl → derive RH via Magnus formula.
            huss_path = file_index['huss'].get(suffix)
            psl_path  = file_index['psl'].get(suffix)
            if huss_path:
                _, huss_sub, _ = _load_subregion(
                    huss_path, 'huss', year_start, year_end, y_min, y_max, x_min, x_max)
                if huss_sub is not None:
                    huss_delos = _interpolate_all_locs(huss_sub, rel_info)
                    # Temperature in Kelvin (add offset back after the Celsius interpolation).
                    tas_delos  = chunks['tmean'][-1] + KELVIN_OFFSET
                    # Use actual surface pressure if available, otherwise assume ISA sea-level.
                    psl_delos = 101325.0
                    if psl_path:
                        _, psl_sub, _ = _load_subregion(
                            psl_path, 'psl', year_start, year_end, y_min, y_max, x_min, x_max)
                        if psl_sub is not None:
                            psl_delos = _interpolate_all_locs(psl_sub, rel_info)
                    chunks['rh'].append(_rh_from_huss(huss_delos, tas_delos, psl_delos))
                else:
                    chunks['rh'].append(np.full((T, n_points), np.nan))
            else:
                print('\n    MISSING huss for {}'.format(suffix), end='')
                chunks['rh'].append(np.full((T, n_points), np.nan))

        print()   # newline after the suffix progress marker

    if not all_dates:
        print('  WARNING: no data produced for {} {}'.format(model_name, scenario))
        return

    # Concatenate all date-range chunks into single arrays.
    var_arrays = {v: np.concatenate(chunks[v], axis=0).astype(np.float32)
                  for v in chunks}

    # Apply physical bounds before writing.
    var_arrays['precip'] = np.clip(var_arrays['precip'], 0.0, None)   # no negative precip
    var_arrays['wind']   = np.clip(var_arrays['wind'],   0.0, None)   # no negative wind speed
    var_arrays['rh']     = np.clip(var_arrays['rh'],     0.0, 100.0)  # RH ∈ [0, 100]

    _write_nc(out_path, all_dates, var_arrays, delos_lat, delos_lon, delos_elev, calendar_out)


# ---------------------------------------------------------------------------
# NetCDF4 output
# ---------------------------------------------------------------------------

def _write_nc(out_path, dates, var_arrays, delos_lat, delos_lon, delos_elev, calendar):
    """
    Write interpolated Delos time series to a NetCDF4 file.

    Dimensions:
      time  : number of days in the extracted period
      point : number of Delos grid points

    Coordinate variables:
      time       : numeric time values (days since 1950-01-01, source calendar)
      lat, lon   : geographic coordinates of each Delos point (degrees)
      elevation_m: elevation of each point (metres)

    Data variables (float32, zlib compressed):
      tmean, tmin, tmax : temperature (°C)
      rh                : relative humidity (%)
      precip            : precipitation (mm day⁻¹)
      wind              : 10-m wind speed (m s⁻¹)

    The source calendar is preserved in the 'time' variable attributes so that
    360-day (HadGEM2) and standard-calendar outputs are distinguishable by
    downstream code.
    """
    n_times  = len(dates)
    n_points = len(delos_lat)
    time_units = 'days since 1950-01-01'
    # nc4.date2num handles both standard and 360-day cftime objects correctly.
    time_values = nc4.date2num(dates, time_units, calendar)

    ds = nc4.Dataset(str(out_path), 'w', format='NETCDF4')
    ds.createDimension('time',  n_times)
    ds.createDimension('point', n_points)

    tv          = ds.createVariable('time', 'f8', ('time',), zlib=True, complevel=4)
    tv.units    = time_units
    tv.calendar = calendar
    tv[:]       = time_values

    # Write coordinate arrays (1-D, per-point).
    for name, data, units_str in [
        ('lat',         delos_lat,  'degrees_north'),
        ('lon',         delos_lon,  'degrees_east'),
        ('elevation_m', delos_elev, 'm'),
    ]:
        v       = ds.createVariable(name, 'f4', ('point',), zlib=True)
        v.units = units_str
        v[:]    = data

    # CF-standard units for each output variable.
    _UNITS = {
        'tmean': 'degC', 'tmin': 'degC', 'tmax': 'degC',
        'rh': '%', 'precip': 'mm day-1', 'wind': 'm s-1',
    }
    for name, arr in var_arrays.items():
        v         = ds.createVariable(name, 'f4', ('time', 'point'),
                                      zlib=True, complevel=4, fill_value=np.float32(1e20))
        v.units   = _UNITS.get(name, '')
        v[:]      = arr

    ds.close()
    print('  written {} timesteps × {} points → {}'.format(n_times, n_points, out_path))


# ---------------------------------------------------------------------------
# Disk check
# ---------------------------------------------------------------------------

def _check_disk(path):
    # type: (Path) -> None
    """
    Check available disk space at the output directory.

    Aborts with sys.exit() if free space falls below MIN_FREE_GB (config).
    Called before starting extraction to avoid writing partial files that
    consume valuable disk space on the server.
    """
    free_gb = shutil.disk_usage(str(path)).free / 1e9
    print('Disk free at {}: {:.1f} GB'.format(path, free_gb))
    if free_gb < MIN_FREE_GB:
        sys.exit('ERROR: only {:.1f} GB free — aborting'.format(free_gb))


if __name__ == '__main__':
    main()
