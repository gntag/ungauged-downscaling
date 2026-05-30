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

VARIABLES DOWNLOADED
--------------------
  Analysis (product_type="analysis"):
    10m_wind_speed          — 10-m wind speed (si10, m/s), 8 times/day
    2m_temperature          — 2-m air temperature (t2m, K), 8 times/day
    2m_relative_humidity    — 2-m relative humidity (r2, %), 8 times/day

  Forecast (product_type="forecast"):
    10m_wind_gust_since_previous_post_processing — wind gust (fg10, m/s)
    total_precipitation                          — precipitation accumulation (tp, m)

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
ANALYSIS_VARIABLES = [
    "10m_wind_speed",
    "2m_temperature",
    "2m_relative_humidity",
]

# Forecast-stream variables (accumulated/gust) stored as yearly files.
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
# Lead times in hours requested for forecast variables.
# CERRA forecasts run to 24 h from 00 UTC; gust and precip are available
# at these lead times.  Lead time 24 h provides the next-day daily total
# for precipitation (tp) and the maximum gust (fg10).
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
