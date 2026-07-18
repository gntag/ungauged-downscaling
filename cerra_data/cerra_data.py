#!/usr/bin/env python3
"""
cerra_data.py
=============
Download CERRA (Copernicus European Regional ReAnalysis) single-level variables
from the Copernicus Climate Data Store (CDS) as NetCDF (.nc) files.

WHY CERRA?
----------
CERRA is a regional reanalysis at ~5 km horizontal resolution covering Europe
for 1984–2021.  It provides the predictor variables used by the statistical
downscaling LightGBM models (temperature, humidity, wind, precipitation, gust).
Its higher resolution compared to ERA5 (~31 km) means CERRA captures topographic
effects and sea-breeze signals relevant to the Aegean island environment.

CHUNKING STRATEGY
-----------------
CDS imposes per-request size limits.  Downloading monthly chunks for analysis
variables (3-hourly, ~8 steps/day × 31 days × full grid) stays well within
those limits.  Forecast variables have fewer lead times per day and are
downloaded as full yearly files.

STORAGE LAYOUT
--------------
The layout written here is expected by cerra_extract.extract_all():

  Analysis variables (3-hourly, monthly files):
    <output_dir>/<variable_name>/<year>/<variable_name>_<year>_<month>.nc

  Forecast variables (sub-daily accumulations, yearly files):
    <output_dir>/<variable_name>/<variable_name>_<year>.nc

  Static fields (one-time download):
    <output_dir>/orography/cerra_orography.nc  — model terrain elevation (m)

VARIABLES DOWNLOADED
--------------------
  Analysis (product_type="analysis"):
    10m_wind_speed              — 10-m wind speed (si10, m/s), 8 times/day
    2m_temperature              — 2-m air temperature (t2m, K), 8 times/day
    2m_relative_humidity        — 2-m relative humidity (r2, %), 8 times/day
    10m_wind_direction          — 10-m wind direction (wdir10, °), 8 times/day
    NOTE: u10/v10 are NOT available in reanalysis-cerra-single-levels (neither as
    analysis nor forecast).  cerra_extract.py derives them from si10 and wdir10.

  Forecast (product_type="forecast"):
    10m_wind_gust_since_previous_post_processing — wind gust (fg10, m/s)
    total_precipitation                          — precipitation accumulation (tp, m)

  Static (one-time, product_type="analysis", single timestep):
    orography — model terrain elevation (orog, m) — time-invariant

REQUIREMENTS
------------
  pip install cdsapi
  ~/.cdsapirc must contain your CDS API credentials:
      url: https://cds.climate.copernicus.eu/api
      key: <your-key>
"""

from __future__ import annotations

import argparse
import calendar
from pathlib import Path

try:
    import cdsapi
except ImportError as exc:
    raise SystemExit("Missing dependency 'cdsapi'. Install it with: pip install cdsapi") from exc


# CDS dataset identifier for CERRA single-level variables.
DATASET = "reanalysis-cerra-single-levels"

# Analysis-stream variables (3-hourly, 8 steps/day) stored in monthly files.
# Wind direction is included so cerra_extract.py can derive u10/v10 from
# si10 and wdir10; u10/v10 do not exist in reanalysis-cerra-single-levels.
ANALYSIS_VARIABLES = [
    "10m_wind_speed",
    "2m_temperature",
    "2m_relative_humidity",
    "10m_wind_direction",
]

# Forecast-stream variables stored as yearly files.
FORECAST_VARIABLES = [
    "10m_wind_gust_since_previous_post_processing",
    "total_precipitation",
]

MONTHS = [f"{month:02d}" for month in range(1, 13)]
# DAYS includes all possible days 01–31; CDS ignores invalid combinations
# (e.g. day 31 in a 30-day month) rather than raising an error.
DAYS = [f"{day:02d}" for day in range(1, 32)]

# 3-hourly analysis times (UTC): 00:00, 03:00, …, 21:00
ANALYSIS_TIMES = [
    "00:00", "03:00", "06:00", "09:00",
    "12:00", "15:00", "18:00", "21:00",
]

# Forecast initialisation time: 00:00 UTC (daily run).
FORECAST_TIME = ["00:00"]
LEADTIME_HOURS = ["1", "2", "3", "4", "5", "6", "9", "12", "15", "18", "21", "24"]


def build_analysis_request(variable: str, year: int, month: int) -> dict:
    """
    Build a CDS API request dict for one analysis variable × year × month.

    The request includes only the days actually in the month (e.g. 28 days
    in February) rather than a fixed 1–31 list.  This avoids CDS warnings
    and ensures the response contains the correct number of time steps.
    """
    month_str      = f"{month:02d}"
    days_in_month  = calendar.monthrange(year, month)[1]
    month_days     = [f"{day:02d}" for day in range(1, days_in_month + 1)]

    return {
        "variable":     [variable],
        "level_type":   "surface_or_atmosphere",
        "data_type":    ["reanalysis"],
        "product_type": "analysis",
        "year":         [str(year)],
        "month":        [month_str],
        "day":          month_days,
        "time":         ANALYSIS_TIMES,
        "data_format":  "netcdf",
    }


def build_forecast_request(variable: str, year: int) -> dict:
    """
    Build a CDS API request dict for one forecast variable × full year.

    Downloading all 12 months in a single request is within CDS size limits
    for forecast variables because there is only one initialisation time per
    day (00:00 UTC) and a limited set of lead times.
    """
    return {
        "variable":       [variable],
        "level_type":     "surface_or_atmosphere",
        "data_type":      ["reanalysis"],
        "product_type":   "forecast",
        "year":           [str(year)],
        "month":          MONTHS,
        "day":            DAYS,
        "time":           FORECAST_TIME,
        "leadtime_hour":  LEADTIME_HOURS,
        "data_format":    "netcdf",
    }


def download_analysis_variable_month(
    client: cdsapi.Client,
    year: int,
    month: int,
    variable: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """
    Download one analysis variable for one month and write it to disk.

    Output path: <output_dir>/<variable>/<year>/<variable>_<year>_<month>.nc
    Skips the download if the file already exists (unless overwrite=True).
    """
    variable_year_dir = output_dir / variable / str(year)
    variable_year_dir.mkdir(parents=True, exist_ok=True)
    target_file = variable_year_dir / f"{variable}_{year}_{month:02d}.nc"

    if target_file.exists() and not overwrite:
        print(f"[SKIP] {target_file} already exists.")
        return

    request = build_analysis_request(variable, year, month)
    print(f"[START] year={year} month={month:02d} variable={variable} type=analysis")
    client.retrieve(DATASET, request, str(target_file))
    print(f"[DONE] {target_file}")


def download_forecast_variable_year(
    client: cdsapi.Client,
    year: int,
    variable: str,
    output_dir: Path,
    overwrite: bool,
) -> None:
    """
    Download one forecast variable for an entire year and write it to disk.

    Output path: <output_dir>/<variable>/<variable>_<year>.nc
    Skips the download if the file already exists (unless overwrite=True).
    """
    variable_dir = output_dir / variable
    variable_dir.mkdir(parents=True, exist_ok=True)
    target_file = variable_dir / f"{variable}_{year}.nc"

    if target_file.exists() and not overwrite:
        print(f"[SKIP] {target_file} already exists.")
        return

    request = build_forecast_request(variable, year)
    print(f"[START] year={year} variable={variable} type=forecast")
    client.retrieve(DATASET, request, str(target_file))
    print(f"[DONE] {target_file}")


def download_orography(
    client: cdsapi.Client,
    output_dir: Path,
    overwrite: bool,
    year: int = 1985,
) -> None:
    """
    Download the CERRA model orography as a one-time static NetCDF.

    The orography (terrain elevation in metres) is time-invariant; a single
    analysis timestep is sufficient.  The file is written to:
      <output_dir>/orography/cerra_orography.nc

    This file is read by cerra_extract.extract_cerra_static() to compute the
    delta_elev_m feature (target elevation − CERRA grid elevation) and to
    derive CERRA-scale slope/aspect (northness, eastness) for each location.

    Parameters
    ----------
    year : Any year within the CERRA archive (1984–2021). The orography is
           identical across all years; 1985 is used by default.
    """
    orog_dir = output_dir / "orography"
    orog_dir.mkdir(parents=True, exist_ok=True)
    target_file = orog_dir / "cerra_orography.nc"

    if target_file.exists() and not overwrite:
        print(f"[SKIP] {target_file} already exists.")
        return

    request = {
        "variable":     ["orography"],
        "level_type":   "surface_or_atmosphere",
        "data_type":    ["reanalysis"],
        "product_type": "analysis",
        "year":         [str(year)],
        "month":        ["01"],
        "day":          ["01"],
        "time":         ["00:00"],
        "data_format":  "netcdf",
    }
    print(f"[START] orography (static, year={year})")
    client.retrieve(DATASET, request, str(target_file))
    print(f"[DONE] {target_file}")


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(
        description="Download CERRA single-level data per year and variable as NetCDF files."
    )
    parser.add_argument("--start-year", type=int, default=1985, help="First year to download.")
    parser.add_argument("--end-year", type=int, default=2005, help="Last year to download.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Folder where NetCDF files will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing .nc files if present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be less than or equal to --end-year.")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    client = cdsapi.Client()

    # Download the static orography once before the year loop.
    print("━━ Downloading CERRA orography (static, one-time)")
    try:
        download_orography(client, output_dir, args.overwrite)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] orography: {exc}")

    failures: list[tuple[int, str, str]] = []

    for year in range(args.start_year, args.end_year + 1):
        for i, variable in enumerate(ANALYSIS_VARIABLES, start=1):
            for j, month in enumerate(range(1, 13), start=1):
                print(
                    f"{year}/{args.end_year} --- analysis {i}/{len(ANALYSIS_VARIABLES)} "
                    f"--- month {j}/12 --- {variable}"
                )
                try:
                    download_analysis_variable_month(
                        client=client,
                        year=year,
                        month=month,
                        variable=variable,
                        output_dir=output_dir,
                        overwrite=args.overwrite,
                    )
                except Exception as exc:  # noqa: BLE001
                    failures.append((year, f"{variable}_{month:02d}", str(exc)))
                    print(f"[ERROR] year={year} month={month:02d} variable={variable}: {exc}")

        for i, variable in enumerate(FORECAST_VARIABLES, start=1):
            print(f"{year}/{args.end_year} --- forecast {i}/{len(FORECAST_VARIABLES)} --- {variable}")
            try:
                download_forecast_variable_year(
                    client=client,
                    year=year,
                    variable=variable,
                    output_dir=output_dir,
                    overwrite=args.overwrite,
                )
            except Exception as exc:  # noqa: BLE001
                failures.append((year, variable, str(exc)))
                print(f"[ERROR] year={year} variable={variable}: {exc}")

    total_tasks = (args.end_year - args.start_year + 1) * (len(ANALYSIS_VARIABLES) * 12 + len(FORECAST_VARIABLES))
    print(f"Finished. Tasks attempted: {total_tasks}")

    if failures:
        print(f"Failures: {len(failures)}")
        for year, variable, message in failures:
            print(f" - year={year}, variable={variable}, error={message}")
    else:
        print("All downloads completed successfully.")


if __name__ == "__main__":
    main()
