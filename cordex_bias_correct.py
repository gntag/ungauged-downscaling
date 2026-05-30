#!/usr/bin/env python3
"""
cordex_bias_correct.py
=======================
Apply Empirical Quantile Mapping (EQM) bias correction to CORDEX EUR-11
downscaled outputs.

WHAT THIS SCRIPT DOES
---------------------
1. Reads CORDEX NetCDF files already interpolated to the Delos grid by
   cordex_extract.py (one file per model × scenario).
2. Loads the statistical downscaling "truth" dataset from
   delos_downscaled_daily.zarr (produced by post_processing_cerra_downscaling.py).
3. Fits per-DOY quantile transfer functions (EQM) using the historical period
   1985–2005, mapping the CORDEX distribution onto the truth distribution.
4. Applies those transfer functions to historical + rcp45 + rcp85 scenarios.
5. Writes bias-corrected NetCDF files: {model}_bc_{scenario}.nc

EMPIRICAL QUANTILE MAPPING (EQM)
---------------------------------
EQM (Gudmundsson et al. 2012, HESS 16(9)) corrects systematic biases in RCM
output by matching empirical quantile distributions between the model and
observations:

  For each grid point and day-of-year (DOY), within a ±DOY_WINDOW window:
    1. Compute the empirical CDFs of the observed and modelled historical
       distributions at _N_QUANTILES quantile levels.
    2. For a new modelled value x̂, find its rank in the model CDF (u),
       then replace it with the value at rank u in the observed CDF.

  Extrapolation (model values outside the historical training range) is
  handled by a mean-delta correction:
    x_corrected = x̂ + (obs_mean − mdl_mean)

  Precipitation uses a two-step approach (Déqué 2007):
    1. Compute the wet-day fraction in both obs and model.
    2. Map only wet-day amounts using EQM (dry days are set to 0).
    The wet-day threshold is 0.1 mm/day (WMO standard).

CALENDAR HANDLING
-----------------
HadGEM2-based models (CORDEX_HADGEM2_MODELS) use a 360-day calendar (12
months × 30 days).  DOYs for these models are computed as:
  DOY = (month − 1) × 30 + day
and range from 1 to 360.  The truth DOY window always uses the Gregorian
(365-day) calendar; the map_360_to_365() helper converts HadGEM2 DOYs to
their nearest Gregorian equivalent for the truth lookup.

USAGE
-----
  # Process all 7 models sequentially:
  python cordex_bias_correct.py

  # Process a single model:
  python cordex_bias_correct.py --model ICHEC-EC-EARTH_DMI-HIRHAM5

  # Validate inputs without writing output:
  python cordex_bias_correct.py --model MOHC-HadGEM2-ES_SMHI-RCA4 --dry-run

INPUT / OUTPUT
--------------
  Input:   {input_dir}/{model}_{scenario}.nc      (from cordex_extract.py)
  Truth:   delos_downscaled_daily.zarr             (from post_processing...)
  Output:  {output_dir}/{model}_bc_{scenario}.nc

CONFIGURATION
-------------
  CORDEX_ALL_MODELS, CORDEX_HADGEM2_MODELS, CORDEX_OUTPUT_DIR: config.py
  POSTPROCESS_DAILY_ZARR: config.py

Reference: Gudmundsson et al. (2012), HESS 16(9).
"""
import argparse
import datetime
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

# Make the downscaling package importable when run as a top-level script.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE / "downscaling") not in sys.path:
    sys.path.insert(0, str(_HERE / "downscaling"))

import netCDF4 as nc4
import numpy as np
import pandas as pd
import zarr

from downscaling.bias_correct import (
    doy_window_mask,    # boolean mask selecting time steps within ±window of a DOY
    fit_eqm_grid,       # fit EQM transfer functions for all grid points simultaneously
    apply_eqm_grid,     # apply fitted transfer functions to new model values
    map_360_to_365,     # convert 360-day DOYs to the nearest Gregorian equivalent
)
from downscaling.config import (
    CORDEX_ALL_MODELS,       # ordered list of all 7 GCM-RCM model names
    CORDEX_HADGEM2_MODELS,   # subset of models using 360-day calendars
    CORDEX_OUTPUT_DIR,       # directory where cordex_extract.py wrote its outputs
    MIN_FREE_GB,             # abort if output disk has less than this many GB free
    POSTPROCESS_DAILY_ZARR,  # path to the statistical downscaling truth dataset
)

# ---------------------------------------------------------------------------
# EQM hyperparameters
# ---------------------------------------------------------------------------

# Calibration period for EQM fitting (historical overlap between obs and model).
# 1985–2005 gives 21 years × 365 days = 7 665 training samples per DOY window.
_HIST_YEARS = (1985, 2005)

_FUTURE_SCENARIOS     = ("rcp45", "rcp85")
_ALL_OUTPUT_SCENARIOS = ("historical", "rcp45", "rcp85")

# Number of quantile levels for the empirical CDF.
# 250 levels correspond to 0.4 % quantile steps — sufficient resolution for
# climate extremes without overfitting to the training sample size.
_N_QUANTILES = 250

# Half-width of the DOY smoothing window (days).
# A window of ±15 days gives ~31 days × 21 years ≈ 650 samples per point —
# enough for a stable 250-quantile fit while remaining seasonally local.
_DOY_WINDOW = 15

# WMO standard wet-day threshold (mm/day).  Days with precip ≤ 0.1 mm are
# treated as dry; EQM is fitted and applied only on wet-day sub-samples.
_WET_THRESHOLD = 0.1

# ---------------------------------------------------------------------------
# Variable mapping: CORDEX NC variable → truth Zarr variable + precip flag
# ---------------------------------------------------------------------------
# Each entry maps the model variable name to:
#   obs_var_name : the variable name in delos_downscaled_daily.zarr
#   is_precip    : True → use the two-part wet-day EQM approach
_VAR_MAP: Dict[str, Tuple[str, bool]] = {
    "tmean":  ("tmean_c",   False),
    "tmin":   ("tmin_c",    False),
    "tmax":   ("tmax_c",    False),
    "rh":     ("rh_pct",    False),
    "precip": ("precip_mm", True),
    "wind":   ("wind_ms",   False),
}

# CF-standard units for each output variable.
_UNITS = {
    "tmean":  "degC",
    "tmin":   "degC",
    "tmax":   "degC",
    "rh":     "%",
    "precip": "mm day-1",
    "wind":   "m s-1",
}


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

class _Tee:
    """Mirror all writes to multiple file-like objects (console + log file)."""
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
    return datetime.datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    print(f"{_ts()} {msg}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Parse arguments, iterate over selected models, and dispatch to _run_model().
    Per-model log files mirror stdout for post-run review.
    """
    parser = argparse.ArgumentParser(
        description="EQM bias correction for CORDEX EUR-11 outputs"
    )
    parser.add_argument(
        "--model", choices=CORDEX_ALL_MODELS, default=None,
        help="Single model to process (default: all 7 models)",
    )
    parser.add_argument(
        "--input-dir", default=str(CORDEX_OUTPUT_DIR),
        help="Directory containing {model}_{scenario}.nc files",
    )
    parser.add_argument(
        "--obs-zarr", default=str(POSTPROCESS_DAILY_ZARR),
        help="Path to delos_downscaled_daily.zarr (truth dataset)",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: {input-dir}/bias_corrected/)",
    )
    parser.add_argument(
        "--log-dir", default=None,
        help="Log directory (default: {input-dir}/logs/)",
    )
    parser.add_argument(
        "--var", nargs="+", choices=list(_VAR_MAP.keys()), default=None,
        metavar="VAR",
        help="Variables to bias-correct (default: all). Others are copied unchanged.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print plan without writing output",
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    obs_zarr   = Path(args.obs_zarr)
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "bias_corrected"
    log_dir    = Path(args.log_dir)    if args.log_dir    else input_dir / "logs"
    models     = [args.model] if args.model else list(CORDEX_ALL_MODELS)

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)
        _check_disk(output_dir)

    _log("Input dir  : {}".format(input_dir))
    _log("Obs zarr   : {}".format(obs_zarr))
    _log("Output dir : {}".format(output_dir))
    _log("Models     : {}".format(models))
    _log("Vars       : {}".format(args.var or "all"))
    _log("Dry run    : {}".format(args.dry_run))

    for model in models:
        if args.dry_run:
            try:
                _log("CHECK  {}".format(model))
                _run_model(model=model, input_dir=input_dir, obs_zarr=obs_zarr,
                           output_dir=output_dir, vars=args.var, dry_run=True)
            except Exception as exc:
                _log("ERROR  {}: {}".format(model, exc))
            continue

        logfile = log_dir / "{}_bc.log".format(model)
        with open(str(logfile), "w") as lf:
            orig = sys.stdout
            sys.stdout = _Tee(orig, lf)
            try:
                _log("START  {}".format(model))
                _run_model(model=model, input_dir=input_dir, obs_zarr=obs_zarr,
                           output_dir=output_dir, vars=args.var, dry_run=args.dry_run)
                _log(f"DONE   {model}")
            except Exception as exc:
                _log(f"ERROR  {model}: {exc}")
                import traceback
                traceback.print_exc()
            finally:
                sys.stdout = orig


# ---------------------------------------------------------------------------
# Per-model pipeline
# ---------------------------------------------------------------------------

def _run_model(
    model: str,
    input_dir: Path,
    obs_zarr: Path,
    output_dir: Path,
    dry_run: bool,
    vars: list[str] | None = None,
) -> None:
    """
    Run the complete EQM pipeline for one CORDEX model:
      1. Verify all required input files exist.
      2. Load truth (obs) data from the daily Zarr for the calibration period.
      3. Load historical CORDEX data for the same period.
      4. Load future scenario (rcp45, rcp85) CORDEX data.
      5. For each variable and each DOY, fit EQM transfer functions and apply
         them to historical + future data.
      6. Enforce physical constraints (clip negative precip, RH bounds,
         temperature ordering).
      7. Write bias-corrected NetCDF files.

    Parameters
    ----------
    model     : Model name string (one of CORDEX_ALL_MODELS).
    vars      : If not None, bias-correct only these variables; others get
                copied from the input file unchanged.
    dry_run   : If True, validate inputs only and return without writing.
    """
    is_hadgem = model in CORDEX_HADGEM2_MODELS
    # 360-day models have DOYs 1..360; all others use 1..365.
    cal_len = 360 if is_hadgem else 365

    # --- Check all required input files exist --------------------------------
    hist_path    = input_dir / f"{model}_historical.nc"
    future_paths = {sc: input_dir / f"{model}_{sc}.nc" for sc in _FUTURE_SCENARIOS}

    for path in [hist_path, *future_paths.values()]:
        if not path.exists():
            _log(f"  MISSING {path} — skipping model")
            return

    if dry_run:
        _log(f"  [dry-run] inputs OK for {model}")
        return

    # --- Load truth dataset for the calibration period -----------------------
    # The truth is the statistical downscaling output (LightGBM + CERRA),
    # which we treat as ground truth for correcting the coarser CORDEX RCM.
    _log("  Loading truth zarr …")
    obs_data, obs_doys = _load_obs_zarr(obs_zarr, _HIST_YEARS)
    n_pts = obs_data[next(iter(obs_data))].shape[1]
    _log(f"  Obs: {obs_data[next(iter(obs_data))].shape[0]} days × {n_pts} points")

    # --- Load historical model data ------------------------------------------
    _log("  Loading historical NC …")
    mdl_hist, mdl_hist_doys, hist_dates, hist_lat, hist_lon, hist_elev, hist_cal = \
        _load_cordex_nc(hist_path, _HIST_YEARS, is_hadgem)
    _log(f"  Hist: {mdl_hist[next(iter(mdl_hist))].shape[0]} days, calendar={hist_cal}")

    # --- Load future scenario data -------------------------------------------
    # historical scenario is also included in the output (uses the model's
    # own historical run corrected by the same transfer functions).
    future_data: Dict[str, Dict]        = {"historical": mdl_hist}
    future_doys: Dict[str, np.ndarray]  = {"historical": mdl_hist_doys}
    future_dates: Dict[str, list]       = {"historical": hist_dates}

    for sc in _FUTURE_SCENARIOS:
        _log("  Loading {} NC …".format(sc))
        data, doys, dates, *_ = _load_cordex_nc(future_paths[sc], None, is_hadgem)
        future_data[sc]  = data
        future_doys[sc]  = doys
        future_dates[sc] = dates
        _log("  {}: {} days".format(sc, data[next(iter(data))].shape[0]))

    # --- Determine which variables will be corrected -------------------------
    # Only correct variables that exist in both the model NC and the truth Zarr.
    available_vars = [
        v for v in _VAR_MAP
        if v in mdl_hist
        and _VAR_MAP[v][0] in obs_data
        and (vars is None or v in vars)
    ]

    # Pre-allocate output arrays filled with NaN; EQM will overwrite them
    # DOY by DOY.  Uncorrected DOYs (empty windows) remain NaN.
    corrected: Dict[str, Dict[str, np.ndarray]] = {
        sc: {
            var: np.full_like(future_data[sc][var], np.nan, dtype=np.float32)
            for var in available_vars
            if var in future_data[sc]
        }
        for sc in _ALL_OUTPUT_SCENARIOS
    }

    # --- EQM fit + apply: one variable at a time, all DOYs ------------------
    for var in available_vars:
        obs_var, is_precip = _VAR_MAP[var]
        _log(f"  Correcting {var} ({'precip' if is_precip else 'non-precip'}) …")

        obs_arr = obs_data[obs_var]   # (n_obs_days, n_pts) float64
        mdl_arr = mdl_hist[var]       # (n_mdl_days, n_pts) float32

        for d in range(1, cal_len + 1):
            # Map the native model DOY to the Gregorian DOY for truth lookup.
            # For HadGEM2 models, DOY 1..360 → nearest Gregorian DOY 1..365.
            d365 = int(map_360_to_365(np.array([d]))[0]) if is_hadgem else d

            # Select the ±DOY_WINDOW truth window (Gregorian).
            obs_mask = doy_window_mask(obs_doys, d365, _DOY_WINDOW, 365)
            # Select the ±DOY_WINDOW model window (native calendar).
            mdl_mask = doy_window_mask(mdl_hist_doys, d, _DOY_WINDOW, cal_len)

            if not np.any(obs_mask) or not np.any(mdl_mask):
                continue  # skip DOYs with no data (shouldn't happen in practice)

            obs_win = obs_arr[obs_mask, :]   # (n_obs_window, n_pts)
            mdl_win = mdl_arr[mdl_mask, :]   # (n_mdl_window, n_pts)

            # Fit the transfer functions across all grid points simultaneously.
            # Returns: model quantiles, obs quantiles, obs mean, model mean,
            # and (for precip) the model wet-day threshold value.
            mdl_qs, obs_qs, obs_mean, mdl_mean, wet_thr = fit_eqm_grid(
                obs_win, mdl_win,
                n_quantiles=_N_QUANTILES,
                is_precip=is_precip,
                wet_threshold=_WET_THRESHOLD,
            )

            # Apply the fitted transfer functions to every scenario.
            for sc in _ALL_OUTPUT_SCENARIOS:
                if var not in corrected[sc]:
                    continue
                # Exact DOY (no window) — apply to the specific day of year only.
                fut_mask = future_doys[sc] == d
                if not np.any(fut_mask):
                    continue
                block = future_data[sc][var][fut_mask, :].astype(float)
                corrected[sc][var][fut_mask, :] = apply_eqm_grid(
                    block, mdl_qs, obs_qs, obs_mean, mdl_mean, wet_thr, is_precip
                ).astype(np.float32)

    # --- Physical constraints (post-correction) ------------------------------
    # EQM can occasionally violate physical bounds due to extrapolation.
    for sc in _ALL_OUTPUT_SCENARIOS:
        c = corrected[sc]
        if "precip" in c:
            c["precip"] = np.clip(c["precip"], 0.0, None)        # no negative precip
        if "rh" in c:
            c["rh"] = np.clip(c["rh"], 0.0, 100.0)               # RH ∈ [0, 100]
        if "wind" in c:
            c["wind"] = np.clip(c["wind"], 0.0, None)            # no negative wind
        if all(v in c for v in ("tmin", "tmean", "tmax")):
            # Restore tmin ≤ tmean ≤ tmax ordering violated by independent correction.
            c["tmin"] = np.minimum(c["tmin"], c["tmean"])
            c["tmax"] = np.maximum(c["tmax"], c["tmean"])

    # --- Write output NetCDF files -------------------------------------------
    for sc in _ALL_OUTPUT_SCENARIOS:
        out_path = output_dir / "{}_bc_{}.nc".format(model, sc)
        _write_corrected_nc(
            out_path=out_path,
            corrected=corrected[sc],
            dates=future_dates[sc],
            lat=hist_lat,
            lon=hist_lon,
            elev=hist_elev,
            calendar=hist_cal,
        )


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------

def _load_obs_zarr(
    zarr_path: Path,
    hist_years: Tuple[int, int],
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """
    Load truth variables from delos_downscaled_daily.zarr for the calibration period.

    Time is stored as int64 nanoseconds (pandas convention); converted to
    DatetimeIndex for year filtering.

    DOY 366 (31 December in leap years) is clamped to 365 so it falls within
    the 1..365 EQM fitting scheme (the ±15-day window around DOY 365 already
    covers late December; mapping 366 → 365 adds it to the same window).

    Returns
    -------
    obs_data : dict of {zarr_var_name: (n_days, n_pts) float64 array}
    obs_doys : (n_days,) int array of Gregorian day-of-year values (1..365)
    """
    store   = zarr.open(str(zarr_path), mode="r")
    time_ns = store["time"][:]
    time_dt = pd.to_datetime(time_ns, unit="ns")

    year_mask = (time_dt.year >= hist_years[0]) & (time_dt.year <= hist_years[1])
    obs_doys  = time_dt[year_mask].dayofyear.values.astype(int)
    obs_doys  = np.where(obs_doys > 365, 365, obs_doys)   # clamp leap-year DOY 366

    obs_data: Dict[str, np.ndarray] = {}
    for var in _VAR_MAP:
        obs_var = _VAR_MAP[var][0]
        if obs_var in store:
            obs_data[obs_var] = store[obs_var][:][year_mask, :].astype(np.float64)

    return obs_data, obs_doys


def _load_cordex_nc(
    nc_path: Path,
    year_range: Optional[Tuple[int, int]],
    is_hadgem: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, list, np.ndarray, np.ndarray, np.ndarray, str]:
    """
    Load all variables from a CORDEX NetCDF file produced by cordex_extract.py.

    Parameters
    ----------
    nc_path    : Path to the {model}_{scenario}.nc file.
    year_range : (start, end) years to filter, or None to load all years.
    is_hadgem  : True if this model uses a 360-day calendar.

    Returns
    -------
    data     : {var_name: (n_days, n_pts) float32}
    doys     : (n_days,) int array of day-of-year values (native calendar)
    dates    : list of cftime datetime objects
    lat, lon : (n_pts,) coordinate arrays
    elev     : (n_pts,) elevation array (metres; zeros if not in file)
    calendar : calendar string from the 'time' variable ('standard', '360_day', …)

    CALENDAR NOTES
    --------------
    For HadGEM2 models (360-day calendar), DOY is computed as:
      (month - 1) × 30 + day  →  range 1..360

    For standard-calendar models, DOY is taken from timetuple()[7] (tm_yday).
    DOY 366 (leap Dec 31) is clamped to 365 for consistency with the EQM
    fitting range.
    """
    ds        = nc4.Dataset(str(nc_path))
    tvar      = ds.variables["time"]
    calendar  = getattr(tvar, "calendar", "standard")
    units     = tvar.units
    raw_times = np.array(tvar[:])
    all_dates = nc4.num2date(raw_times, units, calendar)

    # Filter to the requested year range (or take all years if None).
    if year_range is not None:
        mask = np.array([year_range[0] <= d.year <= year_range[1] for d in all_dates])
    else:
        mask = np.ones(len(all_dates), dtype=bool)

    dates = [d for d, m in zip(all_dates, mask) if m]

    if is_hadgem:
        # 360-day calendar: each month has exactly 30 days — DOY is exact.
        doys = np.array([(d.month - 1) * 30 + d.day for d in dates], dtype=int)
    else:
        # timetuple() index 7 is tm_yday (day of year).
        doys = np.array([d.timetuple()[7] for d in dates], dtype=int)
        doys = np.where(doys > 365, 365, doys)   # clamp leap-year DOY 366

    lat  = np.array(ds.variables["lat"][:])
    lon  = np.array(ds.variables["lon"][:])
    elev = (np.array(ds.variables["elevation_m"][:])
            if "elevation_m" in ds.variables else np.zeros(len(lat)))

    data: Dict[str, np.ndarray] = {}
    for var in _VAR_MAP:
        if var in ds.variables:
            data[var] = np.array(ds.variables[var][mask, :], dtype=np.float32)

    ds.close()
    return data, doys, dates, lat, lon, elev, calendar


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def _write_corrected_nc(
    out_path: Path,
    corrected: Dict[str, np.ndarray],
    dates: list,
    lat: np.ndarray,
    lon: np.ndarray,
    elev: np.ndarray,
    calendar: str,
) -> None:
    """
    Write bias-corrected arrays to a NetCDF4 file.

    If the output file already exists (e.g. from a partial previous run or
    a per-variable retry), the function opens it in "r+" mode and updates
    only the variables in 'corrected', preserving all other content.

    For new files, full metadata is written including bias-correction
    provenance attributes (method, reference, calibration period).
    """
    time_units  = "days since 1950-01-01"
    time_values = nc4.date2num(dates, time_units, calendar)

    if out_path.exists():
        # --- Update existing file --------------------------------------------
        ds = nc4.Dataset(str(out_path), "r+")
        for var, arr in corrected.items():
            if var not in ds.variables:
                v = ds.createVariable(
                    var, "f4", ("time", "point"),
                    zlib=True, complevel=4, fill_value=np.float32(1e20),
                )
            else:
                v = ds.variables[var]
            v.units          = _UNITS.get(var, "")
            v.bias_corrected = "1"
            v[:]             = arr
        ds.close()
        _log(f"  Updated {list(corrected.keys())} → {out_path}")
    else:
        # --- Create new file -------------------------------------------------
        n_times = len(dates)
        n_pts   = len(lat)
        ds      = nc4.Dataset(str(out_path), "w", format="NETCDF4")
        ds.createDimension("time",  n_times)
        ds.createDimension("point", n_pts)

        tv          = ds.createVariable("time", "f8", ("time",), zlib=True, complevel=4)
        tv.units    = time_units
        tv.calendar = calendar
        tv[:]       = time_values

        for name, data, u in [
            ("lat",         lat,  "degrees_north"),
            ("lon",         lon,  "degrees_east"),
            ("elevation_m", elev, "m"),
        ]:
            v       = ds.createVariable(name, "f4", ("point",), zlib=True)
            v.units = u
            v[:]    = data

        # Provenance metadata — aids reproducibility and downstream auditing.
        ds.bias_correction_method    = "Empirical Quantile Mapping (EQM)"
        ds.bias_correction_reference = "Gudmundsson et al. (2012), HESS 16(9)"
        ds.n_quantiles               = _N_QUANTILES
        ds.doy_window                = _DOY_WINDOW
        ds.wet_day_threshold_mm      = _WET_THRESHOLD
        ds.calibration_period        = f"{_HIST_YEARS[0]}-{_HIST_YEARS[1]}"

        for var, arr in corrected.items():
            v                = ds.createVariable(
                var, "f4", ("time", "point"),
                zlib=True, complevel=4, fill_value=np.float32(1e20),
            )
            v.units          = _UNITS.get(var, "")
            v.bias_corrected = "1"
            v[:]             = arr

        ds.close()
        _log(f"  Written {n_times} days × {n_pts} pts → {out_path}")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _check_disk(path: Path) -> None:
    """
    Abort with sys.exit() if free disk space at 'path' falls below MIN_FREE_GB.

    Called before starting the main extraction loop to avoid writing partial
    files that consume space on the server and require manual cleanup.
    """
    free_gb = shutil.disk_usage(str(path)).free / 1e9
    _log(f"Disk free at {path}: {free_gb:.1f} GB")
    if free_gb < MIN_FREE_GB:
        sys.exit(f"ERROR: only {free_gb:.1f} GB free — aborting")


if __name__ == "__main__":
    main()
