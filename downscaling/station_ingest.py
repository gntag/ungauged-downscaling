"""
station_ingest.py
=================
Load and parse NOAA GSOD (Global Summary of the Day) CSV station files.

DATA SOURCE
-----------
GSOD daily summaries are downloaded by Station_Data/Station_Data.py from:
  https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<year>/

LAYOUT
------
  <station_dir>/<year>/<station_id>.csv

Each CSV has NOAA-standard column names (STATION, DATE, LATITUDE, LONGITUDE,
ELEVATION, TEMP, MIN, MAX, DEWP, PRCP, WDSP, GUST …).

UNIT CONVERSIONS
----------------
GSOD stores temperatures in Fahrenheit, precipitation in inches,
wind speed in knots.  This module converts everything to SI / climate-standard:
  Temperature : °F  → °C
  Precipitation: inches → mm  (× 25.4)
  Wind speed   : knots → m/s (× 0.514444)

MISSING VALUE SENTINELS
-----------------------
NOAA uses fixed sentinel values for missing data:
  9999.9  for temperature and dew point
   999.9  for wind speed
    99.99 for precipitation
These are replaced with NaN before conversion.

WET-DAY FILTERING
-----------------
Stations that never record measurable precipitation (PRCP > 0.1 mm) are dropped.
These are typically temperature-only stations where NCDC quality control fills
the PRCP column with 0 (PRCP_ATTRIBUTES = 'I') rather than actual gauge data.
Including them would teach the precipitation model that nearby points never rain,
severely suppressing predicted wet-day frequency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# NOAA GSOD missing-value sentinels (these exact values mean "not recorded")
_MISSING_TEMP = 9999.9
_MISSING_WIND =  999.9
_MISSING_PRCP =   99.99


def load_stations(
    station_dir: Path,
    year_start: int,
    year_end: int,
    lat_min: float = 33.0,
    lat_max: float = 43.0,
    lon_min: float = 11.0,
    lon_max: float = 32.0,
) -> pd.DataFrame:
    """
    Load and combine all GSOD CSV files in station_dir for year_start..year_end.

    Applies the geographic bounding box filter and drops precipitation-inactive
    stations before returning.

    Parameters
    ----------
    station_dir : root directory; expected layout: <dir>/<year>/<station>.csv
    year_start, year_end : inclusive year range
    lat_min/max, lon_min/max : geographic filter (drops stations outside box)

    Returns
    -------
    DataFrame with columns: station_id, date, lat, lon, elevation_m, name,
    tmean_c, tmin_c, tmax_c, rh_pct, prcp_mm, wdsp_ms, gust_ms
    """
    frames = []
    for year in range(year_start, year_end + 1):
        year_dir = Path(station_dir) / str(year)
        if not year_dir.exists():
            continue
        for path in sorted(year_dir.glob("*.csv")):
            df = _parse_gsod_file(path)
            if df is not None:
                frames.append(df)

    if not frames:
        return _empty_df()

    data = pd.concat(frames, ignore_index=True)

    # Geographic filter
    mask = data["lat"].between(lat_min, lat_max) & data["lon"].between(lon_min, lon_max)
    data = data.loc[mask].reset_index(drop=True)

    # Drop stations with no measurable precipitation (see module docstring)
    has_rain   = data.groupby("station_id")["prcp_mm"].apply(lambda x: (x > 0.1).any())
    valid_ids  = has_rain[has_rain].index
    return data[data["station_id"].isin(valid_ids)].reset_index(drop=True)


def _parse_gsod_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Parse one GSOD CSV file into a standardised DataFrame.

    Returns None if the file cannot be read or lacks required columns.
    Missing values are set to NaN using the NOAA sentinels before any conversion.
    """
    try:
        raw = pd.read_csv(path)
    except Exception:
        return None

    raw.columns = raw.columns.str.strip()
    if not {"STATION", "DATE", "LATITUDE", "LONGITUDE"}.issubset(raw.columns):
        return None

    df = pd.DataFrame()
    df["station_id"]  = raw["STATION"].astype(str).str.strip()
    df["date"]        = pd.to_datetime(raw["DATE"], errors="coerce")
    df["lat"]         = pd.to_numeric(raw["LATITUDE"],  errors="coerce")
    df["lon"]         = pd.to_numeric(raw["LONGITUDE"], errors="coerce")
    df["elevation_m"] = pd.to_numeric(
        raw["ELEVATION"] if "ELEVATION" in raw.columns else pd.Series(dtype=float),
        errors="coerce",
    )
    df["name"] = raw["NAME"].astype(str).str.strip() if "NAME" in raw.columns else ""

    # Temperature: mask sentinels, then convert °F → °C
    df["tmean_c"] = _f_to_c(_mask(raw, "TEMP", _MISSING_TEMP))
    df["tmin_c"]  = _f_to_c(_mask(raw, "MIN",  _MISSING_TEMP))
    df["tmax_c"]  = _f_to_c(_mask(raw, "MAX",  _MISSING_TEMP))

    # Relative humidity derived from temperature and dew point (Magnus formula)
    dewp_c    = _f_to_c(_mask(raw, "DEWP", _MISSING_TEMP))
    df["rh_pct"] = _rh_from_temp_dewp(df["tmean_c"], dewp_c)

    # Precipitation: inches → mm
    df["prcp_mm"] = _mask(raw, "PRCP", _MISSING_PRCP) * 25.4

    # Wind speed and gust: knots → m/s
    df["wdsp_ms"] = _mask(raw, "WDSP", _MISSING_WIND) * 0.514444
    df["gust_ms"] = _mask(raw, "GUST", _MISSING_WIND) * 0.514444

    return df.dropna(subset=["date", "lat", "lon"])


def _f_to_c(series: pd.Series) -> pd.Series:
    """Convert Fahrenheit to Celsius."""
    return (series - 32.0) * 5.0 / 9.0


def _mask(raw: pd.DataFrame, col: str, sentinel: float) -> pd.Series:
    """
    Extract a column from raw, strip trailing asterisks/spaces (NOAA flag chars),
    convert to float, and replace the NOAA missing-value sentinel with NaN.
    """
    if col not in raw.columns:
        return pd.Series(np.nan, index=raw.index)
    cleaned = raw[col].astype(str).str.rstrip("* ")
    vals    = pd.to_numeric(cleaned, errors="coerce")
    return vals.where(vals < sentinel)


def _rh_from_temp_dewp(temp_c: pd.Series, dewp_c: pd.Series) -> pd.Series:
    """
    Estimate relative humidity from temperature and dew point via the
    August-Roche-Magnus approximation:
        RH = 100 × exp(a·Td/(b+Td)) / exp(a·T/(b+T))
    with a = 17.625, b = 243.04 °C  (Alduchov & Eskridge 1996).
    """
    a, b = 17.625, 243.04
    rh = 100.0 * np.exp(a * dewp_c / (b + dewp_c)) / np.exp(a * temp_c / (b + temp_c))
    return rh.clip(0.0, 100.0)


def _empty_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected schema."""
    return pd.DataFrame(
        columns=[
            "station_id", "date", "lat", "lon", "elevation_m", "name",
            "tmean_c", "tmin_c", "tmax_c", "rh_pct", "prcp_mm", "wdsp_ms", "gust_ms",
        ]
    )
