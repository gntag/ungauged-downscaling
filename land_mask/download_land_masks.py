#!/usr/bin/env python3
"""
download_land_masks.py
======================
Download native land-sea masks for CERRA and all 7 CORDEX EUR-11 models
from the Copernicus Climate Data Store (CDS).

WHY LAND-SEA MASKS ARE NEEDED
------------------------------
CERRA and each CORDEX model define "land" differently.  Interpolating a value
for a small island (like Delos, 5 km²) from a coarse grid that classifies the
nearest cell as sea would introduce systematic biases from ocean SST-driven
surface fluxes.  By loading each model's own land fraction file, the spatial
interpolation in cerra_extract.py and cordex_extract.py can select the nearest
*land* cells as interpolation donors instead.

OUTPUTS
-------
Written to the land_mask/ directory alongside this script (or --out-dir):
  cerra_lsm.nc                                    — CERRA land-sea mask (binary)
  sftlf_MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009.nc    — land area fraction (%)
  sftlf_ICHEC-EC-EARTH_CLMcom-CCLM4-8-17.nc
  sftlf_MOHC-HadGEM2-ES_SMHI-RCA4.nc
  sftlf_MOHC-HadGEM2-ES_KNMI-RACMO22E.nc
  sftlf_CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E.nc
  sftlf_MPI-M-MPI-ESM-LR_SMHI-RCA4.nc
  sftlf_ICHEC-EC-EARTH_DMI-HIRHAM5.nc

REQUIREMENTS
------------
  pip install cdsapi
  ~/.cdsapirc must contain your CDS credentials:
      url: https://cds.climate.copernicus.eu/api
      key: <your-key>

USAGE
-----
  python download_land_masks.py                     # download everything
  python download_land_masks.py --skip-cerra        # only CORDEX masks
  python download_land_masks.py --skip-cordex       # only CERRA mask
  python download_land_masks.py --model ICHEC-EC-EARTH_DMI-HIRHAM5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cdsapi

LAND_MASK_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# CERRA land-sea mask
# ---------------------------------------------------------------------------

def download_cerra_lsm(client: cdsapi.Client, out_dir: Path, force: bool = False) -> None:
    out_path = out_dir / "cerra_lsm.nc"
    if out_path.exists() and not force:
        print(f"  SKIP (exists): {out_path.name}")
        return

    print("Downloading CERRA land-sea mask ...")

    # The land-sea mask is a static (time-invariant) field in CERRA-Land.
    # The CDS API still requires time selectors even for static variables;
    # any single valid time step works since the field never changes.
    base: dict = {
        "variable": "land_sea_mask",
        "level_type": "surface",
        "product_type": "analysis",
        "year": "1985",
        "month": "01",
        "day": "01",
        "time": "00:00",
    }

    attempts = [
        {**base, "data_format": "netcdf"},
        {**base, "data_format": "netcdf_legacy"},
        {**base, "format": "netcdf"},           # old CDS API key name
    ]

    for i, req in enumerate(attempts):
        try:
            client.retrieve("reanalysis-cerra-land", req, str(out_path))
            out_path = _unzip_if_needed(out_path)
            print(f"  Saved: {out_path.name}")
            _verify_nc(out_path, expected_vars=["lsm", "land_sea_mask"])
            return
        except Exception as exc:
            if out_path.exists():
                out_path.unlink()
            if i < len(attempts) - 1:
                print(f"  Attempt {i + 1} failed ({exc}), retrying ...")
            else:
                print(f"  ERROR: all attempts failed: {exc}")


# ---------------------------------------------------------------------------
# CORDEX sftlf (land area fraction) — one file per model
# ---------------------------------------------------------------------------

# CDS API identifiers for each CORDEX model.
# Ensemble members and versions are taken directly from the server filenames:
#   r1i1p1  / v1   MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009
#   r12i1p1 / v1   ICHEC-EC-EARTH_CLMcom-CCLM4-8-17
#   r1i1p1  / v1   MOHC-HadGEM2-ES_SMHI-RCA4
#   r1i1p1  / v2   MOHC-HadGEM2-ES_KNMI-RACMO22E
#   r1i1p1  / v2   CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E
#   r1i1p1  / v1a  MPI-M-MPI-ESM-LR_SMHI-RCA4
#   r3i1p1  / v2   ICHEC-EC-EARTH_DMI-HIRHAM5
_CORDEX_MODELS: dict[str, dict[str, str]] = {
    # sftlf ensemble members confirmed via CDS constraints API — they differ
    # from the simulation run members because fixed fields use r0i0p0 in CDS
    # for models where the RCM was run from multiple GCM members.
    "MPI-M-MPI-ESM-LR_MPI-CSC-REMO2009": dict(
        gcm_model="mpi_m_mpi_esm_lr",
        rcm_model="mpi_csc_remo2009",
        ensemble_member="r0i0p0",
    ),
    "ICHEC-EC-EARTH_CLMcom-CCLM4-8-17": dict(
        gcm_model="ichec_ec_earth",
        rcm_model="clmcom_clm_cclm4_8_17",
        ensemble_member="r0i0p0",
    ),
    "MOHC-HadGEM2-ES_SMHI-RCA4": dict(
        gcm_model="mohc_hadgem2_es",
        rcm_model="smhi_rca4",
        ensemble_member="r0i0p0",
    ),
    "MOHC-HadGEM2-ES_KNMI-RACMO22E": dict(
        gcm_model="mohc_hadgem2_es",
        rcm_model="knmi_racmo22e",
        ensemble_member="r1i1p1",
    ),
    "CNRM-CERFACS-CNRM-CM5_KNMI-RACMO22E": dict(
        gcm_model="cnrm_cerfacs_cm5",
        rcm_model="knmi_racmo22e",
        ensemble_member="r1i1p1",
    ),
    "MPI-M-MPI-ESM-LR_SMHI-RCA4": dict(
        gcm_model="mpi_m_mpi_esm_lr",
        rcm_model="smhi_rca4",
        ensemble_member="r0i0p0",
    ),
    "ICHEC-EC-EARTH_DMI-HIRHAM5": dict(
        gcm_model="ichec_ec_earth",
        rcm_model="dmi_hirham5",
        ensemble_member="r3i1p1",
    ),
}


def download_cordex_sftlf(
    client: cdsapi.Client,
    model_name: str,
    params: dict[str, str],
    out_dir: Path,
    force: bool = False,
) -> None:
    out_path = out_dir / f"sftlf_{model_name}.nc"
    if out_path.exists() and not force:
        print(f"  SKIP (exists): {out_path.name}")
        return

    print(f"Downloading CORDEX sftlf for {model_name} ...")

    # sftlf is time-invariant; CDS requires temporal_resolution='fixed' and
    # rejects start_year/end_year for fixed-frequency fields.
    req: dict = {
        "domain": "europe",
        "horizontal_resolution": "0_11_degree_x_0_11_degree",
        "temporal_resolution": "fixed",
        "experiment": "historical",
        "variable": "land_area_fraction",
        "gcm_model": params["gcm_model"],
        "rcm_model": params["rcm_model"],
        "ensemble_member": params["ensemble_member"],
        "data_format": "netcdf_legacy",
    }

    try:
        client.retrieve("projections-cordex-domains-single-levels", req, str(out_path))
        out_path = _unzip_if_needed(out_path)
        print(f"  Saved: {out_path.name}")
        _verify_nc(out_path, expected_vars=["sftlf", "land_area_fraction"])
    except Exception as exc:
        if out_path.exists():
            out_path.unlink()
        print(f"  ERROR: {model_name}: {exc}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unzip_if_needed(path: Path) -> Path:
    """If path is a zip, extract the single .nc inside and replace the zip."""
    import zipfile
    # Use magic bytes rather than zipfile.is_zipfile(), which can return False
    # for valid zips depending on the read cursor state after a download.
    if path.read_bytes()[:2] != b"PK":
        return path
    with zipfile.ZipFile(str(path)) as z:
        members = z.namelist()
        nc_members = [m for m in members if m.endswith(".nc")]
        if not nc_members:
            raise RuntimeError(f"Zip contains no .nc file: {members}")
        extracted = z.extract(nc_members[0], path.parent)
    path.unlink()
    result = path.parent / path.name
    Path(extracted).rename(result)
    return result


def _verify_nc(path: Path, expected_vars: list[str]) -> None:
    """Print the variables found in the downloaded NetCDF file."""
    try:
        import netCDF4 as nc4
        ds = nc4.Dataset(str(path))
        found = list(ds.variables.keys())
        ds.close()
        matched = [v for v in expected_vars if v in found]
        if matched:
            print(f"    OK — found variable(s): {matched}  |  all vars: {found}")
        else:
            print(f"    WARNING: expected one of {expected_vars} but got {found}")
    except Exception as exc:
        print(f"    WARNING: could not verify file ({exc})")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download CERRA and CORDEX land-sea masks from CDS"
    )
    parser.add_argument("--skip-cerra", action="store_true", help="Skip CERRA download")
    parser.add_argument("--skip-cordex", action="store_true", help="Skip all CORDEX downloads")
    parser.add_argument("--force", action="store_true", help="Re-download even if file already exists")
    parser.add_argument(
        "--model",
        metavar="MODEL",
        help="Download only this CORDEX model (exact pipeline name)",
    )
    parser.add_argument(
        "--out-dir",
        default=str(LAND_MASK_DIR),
        help=f"Output directory (default: {LAND_MASK_DIR})",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}\n")

    client = cdsapi.Client()

    if not args.skip_cerra:
        download_cerra_lsm(client, out_dir, force=args.force)
        print()

    if not args.skip_cordex:
        models = _CORDEX_MODELS
        if args.model:
            if args.model not in models:
                sys.exit(
                    f"ERROR: unknown model '{args.model}'. Available:\n  "
                    + "\n  ".join(models)
                )
            models = {args.model: models[args.model]}

        for model_name, params in models.items():
            download_cordex_sftlf(client, model_name, params, out_dir, force=args.force)
            print()

    print("Done.")


if __name__ == "__main__":
    main()
