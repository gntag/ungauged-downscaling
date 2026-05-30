#!/usr/bin/env python3
"""
Station_Data.py
===============
Download NOAA GSOD (Global Summary of the Day) station CSV files from the
NOAA Climate Data Online repository.

DATA SOURCE
-----------
NOAA GSOD files are publicly available at:
  https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/<year>/

Each year's directory contains one CSV per station.  Station IDs (filenames)
follow the WMO convention: the first two digits identify the country/region.
For the Aegean/eastern Mediterranean region, station IDs starting with "16"
cover Greece, Turkey, and surrounding countries.

STORAGE LAYOUT
--------------
Files are written to:
  <output_dir>/<year>/<station_id>.csv

This layout is expected by station_ingest.load_stations() when it walks
the station directory tree to build the training dataset.

USAGE
-----
  # Download all eastern Mediterranean stations for 1970–2020:
  python Station_Data.py

  # Custom range and prefix:
  python Station_Data.py --start-year 1985 --end-year 2005 --station-prefix 16

  # Specific output directory:
  python Station_Data.py --output-dir /data/gsod/

DOWNLOAD STRATEGY
-----------------
For each year, the script:
  1. Fetches the NOAA HTML directory listing for that year.
  2. Parses all <a href> links and filters by station_prefix and .csv extension.
  3. Downloads each matching file with automatic retry (--retries) and
     exponential back-off on transient errors.
  4. Skips existing files unless --overwrite is set.
"""

from __future__ import annotations

import argparse
import shutil
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen

# NOAA Global Summary of the Day access URL.
# Each sub-directory is a year (e.g. .../access/1985/).
BASE_URL = "https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/"


class LinkParser(HTMLParser):
    """
    Minimal HTML <a href> extractor for NOAA directory index pages.

    NOAA directory listings are plain Apache-style HTML; we only need the
    href values from anchor tags.  Using the stdlib html.parser rather than
    BeautifulSoup avoids an external dependency for this simple use case.
    """

    def __init__(self) -> None:
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.links.append(value)


def fetch_links(url: str, timeout: int) -> list[str]:
    """
    Fetch the HTML page at 'url' and return all href values.

    A custom User-Agent header prevents some NOAA servers from returning 403
    when accessed by generic scripts.
    """
    request = Request(url, headers={"User-Agent": "station-data-downloader/1.0"})
    with urlopen(request, timeout=timeout) as response:
        html = response.read().decode("utf-8", errors="ignore")

    parser = LinkParser()
    parser.feed(html)
    return parser.links


def list_station_files(base_url: str, year: int, station_prefix: str, timeout: int) -> list[str]:
    """
    Return sorted list of station CSV filenames in the NOAA year directory.

    Filters to files that:
      - End with '.csv'
      - Start with station_prefix (e.g. '16' for Aegean-area WMO stations)

    Station IDs starting with '16' cover Greece, Cyprus, Turkey, and parts
    of the Middle East in the WMO numbering scheme.
    """
    year_url = urljoin(base_url.rstrip("/") + "/", f"{year}/")
    links = fetch_links(year_url, timeout)

    matched: set[str] = set()
    for link in links:
        filename = Path(link).name
        if not filename.endswith(".csv"):
            continue
        if not filename.startswith(station_prefix):
            continue
        matched.add(filename)

    return sorted(matched)


def download_file(url: str, destination: Path, timeout: int, overwrite: bool, retries: int) -> str:
    """
    Download one file with automatic retry on transient network errors.

    Retry delay grows linearly (1 s, 2 s, 3 s, …) so a brief server outage
    doesn't flood NOAA with immediate repeated requests.

    Returns 'skipped', 'downloaded', or raises RuntimeError after all retries.
    """
    if destination.exists() and not overwrite:
        return "skipped"

    destination.parent.mkdir(parents=True, exist_ok=True)

    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "station-data-downloader/1.0"})
            with urlopen(request, timeout=timeout) as response, destination.open("wb") as output_file:
                shutil.copyfileobj(response, output_file)
            return "downloaded"
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
            time.sleep(attempt)   # linear back-off

    return "failed"


def download_stations(
    base_url: str,
    output_dir: Path,
    start_year: int,
    end_year: int,
    station_prefix: str,
    timeout: int,
    overwrite: bool,
    retries: int,
) -> tuple[int, int, int]:
    """
    Iterate over years, discover station files, and download them.

    Failures on individual files are caught and counted rather than aborting
    the whole run — a single 404 (station not present in a given year) should
    not interrupt the download of 200 other stations for that year.

    Returns (n_downloaded, n_skipped, n_failed) as a summary tuple.
    """
    downloaded = 0
    skipped    = 0
    failed     = 0

    for year in range(start_year, end_year + 1):
        try:
            station_files = list_station_files(base_url, year, station_prefix, timeout)
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            failed += 1
            print(f"[{year}] Could not read listing: {exc}")
            continue

        if not station_files:
            print(f"[{year}] No matching station files.")
            continue

        year_downloaded = 0
        year_skipped    = 0
        year_failed     = 0
        year_url = urljoin(base_url.rstrip("/") + "/", f"{year}/")

        for i, filename in enumerate(station_files, start=1):
            print(f"{year}/{end_year} --- {i}/{len(station_files)} --- {filename}")
            file_url    = urljoin(year_url, filename)
            destination = output_dir / str(year) / filename
            try:
                status = download_file(file_url, destination, timeout, overwrite, retries)
            except RuntimeError as exc:
                year_failed += 1
                print(f"[{year}] ERROR {filename}: {exc}")
                continue

            if status == "downloaded":
                year_downloaded += 1
            else:
                year_skipped += 1

        downloaded += year_downloaded
        skipped    += year_skipped
        failed     += year_failed
        print(
            f"[{year}] matched={len(station_files)} "
            f"downloaded={year_downloaded} skipped={year_skipped} failed={year_failed}"
        )

    return downloaded, skipped, failed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download NOAA GSOD station CSV files by station prefix and year range "
            "from https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/."
        )
    )
    parser.add_argument("--base-url", default=BASE_URL, help="Base NOAA access URL.")
    parser.add_argument("--start-year", type=int, default=1970, help="First year to download.")
    parser.add_argument("--end-year", type=int, default=2020, help="Last year to download.")
    parser.add_argument("--station-prefix", default="16", help="Station filename prefix.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "downloaded_station_data"),
        help="Directory where files are written (default: downloaded_station_data/ next to this script).",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per file.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must be <= --end-year")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    downloaded, skipped, failed = download_stations(
        base_url=args.base_url,
        output_dir=output_dir,
        start_year=args.start_year,
        end_year=args.end_year,
        station_prefix=args.station_prefix,
        timeout=args.timeout,
        overwrite=args.overwrite,
        retries=args.retries,
    )

    print(
        "Done. "
        f"downloaded={downloaded} "
        f"skipped_existing={skipped} "
        f"failed={failed}"
    )


if __name__ == "__main__":
    main()
