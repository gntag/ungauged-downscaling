#!/usr/bin/env python3
"""
ba_postprocessing.py
====================
Compute ensemble-mean climate indicators from bias-corrected CORDEX NetCDF files
and write one CSV per variable to the output directory.

DATA FLOW
---------
  cordex_bias_correct.py  →  <model>_bc_<scenario>.nc  →  this script  →  ba_<var>.csv
  ba_<var>.csv is then read by plot_ba_data.py to produce the 7-panel figures.

CSV SCHEMA (one row per Delos grid point)
-----------------------------------------
  lon | lat | historical | rcp45_2041_2060 | rcp45_2061_2080 | rcp45_2081_2100
                         | rcp85_2041_2060 | rcp85_2061_2080 | rcp85_2081_2100

ENSEMBLE HANDLING
-----------------
Each ensemble member contributes the mean of its complete calendar years within
each period.  HadGEM2 rcp45 ends November 2099 (~330 days), so its 2099 annual
value is excluded via MIN_DAYS_FOR_COMPLETE_YEAR=340.  An equal-weight nanmean
is then taken across all members.

ARIDITY INDEX
-------------
UNEP Aridity Index = P_annual / PET_annual
PET computed via Thornthwaite (1948):
    i_m  = max(T_monthly / 5, 0)^1.514   for each of the 12 months
    I    = Σ i_m  (annual heat index)
    a    = 6.75e-7·I³ − 7.71e-5·I² + 1.792e-2·I + 0.49239
    PET_m = 16 · (10·T_m / I)^a   if T_m > 0, else 0   [mm month⁻¹]

USAGE
-----
    python ba_postprocessing.py              # all variables
    python ba_postprocessing.py --var tmean  # single variable
    python ba_postprocessing.py --dry-run    # validate config and members only

REFERENCES
----------
    Thornthwaite, C.W. (1948). Geographical Review, 38(1), 55-94.
    UNEP (1992). World Atlas of Desertification. Edward Arnold, London.
"""

import os
import sys
import glob
import re
import argparse

import numpy as np
import xarray as xr
import pandas as pd

# ---------------------------------------------------------------------------
# Locate the downscaling package relative to this script.
# Avoids the need for a machine-specific sys.path modification.
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import downscaling.plt_config as cfg

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# HadGEM2-ES uses a 360-day calendar; these models need special handling when
# checking for complete years (a "year" is 360 days, not 365).
HADGEM2_IDS = frozenset({
    "MOHC-HadGEM2-ES_KNMI-RACMO22E",
    "MOHC-HadGEM2-ES_SMHI-RCA4",
})

HIST_PERIOD                = (1985, 2005)
FUTURE_PERIODS             = [(2041, 2060), (2061, 2080), (2081, 2100)]
SCENARIOS                  = ["rcp45", "rcp85"]

# Minimum time steps for a year to count as "complete".
# 340 excludes the truncated HadGEM2 rcp45 year 2099 (~330 steps).
MIN_DAYS_FOR_COMPLETE_YEAR = 340


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------
# Use the CF-aware time decoder so both standard and 360-day calendars decode
# into cftime datetime objects rather than numpy datetimes.
_TIME_CODER = xr.coders.CFDatetimeCoder(use_cftime=True)


def discover_members():
    """
    Scan INPUT_DIR for bias-corrected historical NetCDF files and return a
    list of (member_name, is_hadgem2) tuples sorted alphabetically.

    Files are expected to be named: <member>_bc_historical.nc
    """
    pattern = os.path.join(cfg.INPUT_DIR, "*_bc_historical.nc")
    paths   = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matching {pattern}")
    return [
        (
            re.sub(r"_bc_historical\.nc$", "", os.path.basename(p)),
            re.sub(r"_bc_historical\.nc$", "", os.path.basename(p)) in HADGEM2_IDS,
        )
        for p in paths
    ]


def open_ds(member, scenario):
    """Open one bias-corrected NetCDF file as an xarray Dataset."""
    path = os.path.join(cfg.INPUT_DIR, f"{member}_bc_{scenario}.nc")
    return xr.open_dataset(path, decode_times=_TIME_CODER)


def load_latlon():
    """
    Return (lons, lats) as float64 arrays from the first available
    bias-corrected historical NetCDF.  All members share the same Delos grid.
    """
    first = sorted(glob.glob(os.path.join(cfg.INPUT_DIR, "*_bc_historical.nc")))[0]
    with xr.open_dataset(first) as ds:
        return (ds["lon"].values.astype(np.float64),
                ds["lat"].values.astype(np.float64))


def complete_years_in_period(ds, yr_start, yr_end):
    """
    Return a list of years within [yr_start, yr_end] that have at least
    MIN_DAYS_FOR_COMPLETE_YEAR time steps.

    This filters out:
      - Years with missing data (e.g. run start/end edge effects)
      - The truncated HadGEM2 rcp45 2099 (~330 steps instead of 360)
    """
    year_vals = ds.time.dt.year.values
    return [
        yr for yr in range(yr_start, yr_end + 1)
        if np.sum(year_vals == yr) >= MIN_DAYS_FOR_COMPLETE_YEAR
    ]


# ---------------------------------------------------------------------------
# Annual statistics
# ---------------------------------------------------------------------------

def compute_annual_stat(ds, vcfg, years):
    """
    Compute annual values of one statistic for all Delos points.

    Handles:
      - annual_mean : mean of daily values for each complete year
      - annual_sum  : sum  of daily values for each complete year
      - count_gt:<N>: count of days where variable > N per year
      - count_lt:<N>: count of days where variable < N per year

    Parameters
    ----------
    ds    : xarray.Dataset  (one bias-corrected member, one scenario)
    vcfg  : variable config dict from plt_config.VARIABLES
    years : list of int — complete years to include

    Returns (n_years, n_points) float32 array.
    """
    stat_type = vcfg["stat_type"]
    assert stat_type != "aridity_index", "Use compute_annual_ai for aridity index"

    varname   = vcfg["varname"]
    year_vals = ds.time.dt.year.values
    vals_all  = ds[varname].values  # (n_days, n_points)

    threshold = (float(stat_type.split(":")[1])
                 if stat_type.startswith(("count_gt", "count_lt")) else None)

    rows = []
    for yr in years:
        v = vals_all[year_vals == yr]   # (n_days_in_year, n_points)
        if   stat_type == "annual_mean":
            rows.append(np.nanmean(v, axis=0))
        elif stat_type == "annual_sum":
            rows.append(np.nansum(v, axis=0))
        elif stat_type.startswith("count_gt"):
            rows.append(np.nansum(v > threshold, axis=0).astype(np.float32))
        else:                                   # count_lt:<N>
            rows.append(np.nansum(v < threshold, axis=0).astype(np.float32))

    return np.stack(rows, axis=0) if rows else np.empty((0, vals_all.shape[1]))


def compute_annual_ai(ds, years):
    """
    Compute the UNEP Aridity Index (AI = P / PET_TW) per year for all points.

    PET is estimated by the Thornthwaite (1948) method using monthly mean
    temperature.  AI < 0.5 defines an arid or semi-arid climate.

    Returns (n_years, n_points) float array.
    """
    year_vals  = ds.time.dt.year.values
    month_vals = ds.time.dt.month.values
    tmean_all  = ds["tmean"].values   # (n_days, n_points)
    precip_all = ds["precip"].values  # (n_days, n_points) in mm day⁻¹
    n_pts      = tmean_all.shape[1]

    rows = []
    for yr in years:
        yr_mask   = year_vals == yr
        tmean_yr  = tmean_all[yr_mask]
        precip_yr = precip_all[yr_mask]
        months_yr = month_vals[yr_mask]

        # Monthly mean temperature for Thornthwaite PET
        T_m = np.zeros((12, n_pts), dtype=float)
        for m in range(1, 13):
            mm = months_yr == m
            T_m[m - 1] = np.nanmean(tmean_yr[mm], axis=0) if mm.any() else 0.0

        # Heat index I = Σ (T_m / 5)^1.514  (sum over 12 months)
        i_m = np.where(T_m > 0.0, (T_m / 5.0) ** 1.514, 0.0)
        I   = i_m.sum(axis=0)   # (n_points,)

        # Empirical exponent a (polynomial fit to tabulated values)
        a = 6.75e-7 * I**3 - 7.71e-5 * I**2 + 1.792e-2 * I + 0.49239

        # Monthly PET [mm month⁻¹]; 0 if T_m ≤ 0 or I = 0
        PET_m = np.where(
            (T_m > 0.0) & (I > 0.0),
            16.0 * (10.0 * T_m / np.where(I > 0, I, np.nan)) ** a,
            0.0,
        )

        P_annual   = np.nansum(precip_yr, axis=0)   # mm yr⁻¹
        PET_annual = PET_m.sum(axis=0)               # mm yr⁻¹
        rows.append(np.where(PET_annual > 0, P_annual / PET_annual, np.nan))

    return np.stack(rows, axis=0) if rows else np.empty((0, n_pts))


# ---------------------------------------------------------------------------
# Extreme precipitation threshold (R99p)
# ---------------------------------------------------------------------------

def compute_r99p_threshold(ds, hist_years):
    """
    Compute the pixel-wise 99th percentile of wet-day (> 1 mm) precipitation
    over the historical calibration years.

    The threshold is later used to count days per scenario period that exceed it.
    """
    year_vals  = ds.time.dt.year.values
    precip_all = ds["precip"].values
    hist_data  = precip_all[np.isin(year_vals, hist_years)].astype(float).copy()
    hist_data[hist_data <= 1.0] = np.nan   # restrict to wet days
    return np.nanpercentile(hist_data, 99, axis=0)   # (n_points,)


def compute_r99p_counts(ds, years, threshold):
    """
    Count days per year that exceed the pixel-wise R99p threshold.

    Parameters
    ----------
    threshold : (n_points,) array from compute_r99p_threshold()
    """
    year_vals  = ds.time.dt.year.values
    precip_all = ds["precip"].values
    rows = []
    for yr in years:
        v = precip_all[year_vals == yr]
        rows.append(np.nansum(v > threshold, axis=0).astype(np.float32))
    return np.stack(rows, axis=0) if rows else np.empty((0, precip_all.shape[1]))


# ---------------------------------------------------------------------------
# Ensemble mean
# ---------------------------------------------------------------------------

def ensemble_mean(per_member_arrays):
    """
    Equal-weight nanmean across ensemble members.

    Each element of per_member_arrays is a (n_years, n_points) array.
    Returns the (n_points,) period mean, or None if no valid years exist.
    """
    period_means = [np.nanmean(a, axis=0) for a in per_member_arrays if a.shape[0] > 0]
    if not period_means:
        return None
    return np.nanmean(np.stack(period_means, axis=0), axis=0)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _col(scen, yr_s, yr_e):
    """Return the CSV column name for a scenario/period combination."""
    return f"{scen}_{yr_s}_{yr_e}"


def main():
    parser = argparse.ArgumentParser(description="Postprocess bias-adjusted CORDEX NetCDF → CSV")
    parser.add_argument("--var", nargs="+", default=None,
                        help="Variable keys to process (default: all in plt_config.VARIABLES)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Discover members and validate config, then exit without writing")
    args = parser.parse_args()

    members = discover_members()
    print(f"Found {len(members)} ensemble members:")
    for name, is_hg in members:
        print(f"  {name}  [{'360-day' if is_hg else 'standard'} calendar]")

    if args.dry_run:
        print("\nDry run complete.")
        return

    lons, lats = load_latlon()
    print(f"\nDomain  lon [{lons.min():.4f}, {lons.max():.4f}]  "
          f"lat [{lats.min():.4f}, {lats.max():.4f}]  points: {len(lons)}\n")

    var_keys = args.var if args.var else list(cfg.VARIABLES.keys())

    for var_key in var_keys:
        vcfg = cfg.VARIABLES[var_key]
        print(f"── {var_key}  ({vcfg['stat_type']}) ──")
        results = {}

        if vcfg["stat_type"] == "precip_r99p":
            # R99p uses each member's own historical threshold to count future exceedances,
            # so the threshold must be computed per-member before processing future periods.
            member_thresholds = {}
            hist_arrays = []
            for member, is_hg in members:
                ds    = open_ds(member, "historical")
                years = complete_years_in_period(ds, *HIST_PERIOD)
                thr   = compute_r99p_threshold(ds, years)
                member_thresholds[member] = thr
                hist_arrays.append(compute_r99p_counts(ds, years, thr))
                ds.close()
                print(f"  historical  {member}  years={len(years)}")
            results["historical"] = ensemble_mean(hist_arrays)

            for scen in SCENARIOS:
                for yr_s, yr_e in FUTURE_PERIODS:
                    fut_arrays = []
                    for member, is_hg in members:
                        ds    = open_ds(member, scen)
                        years = complete_years_in_period(ds, yr_s, yr_e)
                        if not years:
                            fut_arrays.append(np.empty((0, len(lats))))
                            ds.close()
                            continue
                        fut_arrays.append(
                            compute_r99p_counts(ds, years, member_thresholds[member]))
                        ds.close()
                        print(f"  {scen} {yr_s}-{yr_e}  {member}  years={len(years)}")
                    results[(scen, yr_s, yr_e)] = ensemble_mean(fut_arrays)

        else:
            # Standard statistics: process historical and all future periods uniformly
            hist_arrays = []
            for member, is_hg in members:
                ds    = open_ds(member, "historical")
                years = complete_years_in_period(ds, *HIST_PERIOD)
                arr   = (compute_annual_ai(ds, years)
                         if vcfg["stat_type"] == "aridity_index"
                         else compute_annual_stat(ds, vcfg, years))
                hist_arrays.append(arr)
                ds.close()
                print(f"  historical  {member}  years={len(years)}")
            results["historical"] = ensemble_mean(hist_arrays)

            for scen in SCENARIOS:
                for yr_s, yr_e in FUTURE_PERIODS:
                    fut_arrays = []
                    for member, is_hg in members:
                        ds    = open_ds(member, scen)
                        years = complete_years_in_period(ds, yr_s, yr_e)
                        if not years:
                            fut_arrays.append(np.empty((0, len(lats))))
                            ds.close()
                            continue
                        arr = (compute_annual_ai(ds, years)
                               if vcfg["stat_type"] == "aridity_index"
                               else compute_annual_stat(ds, vcfg, years))
                        fut_arrays.append(arr)
                        ds.close()
                        print(f"  {scen} {yr_s}-{yr_e}  {member}  years={len(years)}")
                    results[(scen, yr_s, yr_e)] = ensemble_mean(fut_arrays)

        # Write CSV — scale factor is NOT applied here; plot_ba_data.py handles it
        row = {"lon": lons, "lat": lats, "historical": results["historical"]}
        for scen in SCENARIOS:
            for yr_s, yr_e in FUTURE_PERIODS:
                row[_col(scen, yr_s, yr_e)] = results.get((scen, yr_s, yr_e))

        out_path = os.path.join(cfg.INPUT_DIR, f"ba_{var_key}.csv")
        pd.DataFrame(row).to_csv(out_path, index=False)
        print(f"  → saved: {out_path}\n")

    print("Done.")


if __name__ == "__main__":
    main()
