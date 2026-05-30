"""
Delos island DEM pipeline
--------------------------
1. Determine island extent from the S2Coast 2023 shoreline shapefile.
2. Download Mapzen Global Terrain tiles (zoom 15, ~4 m resolution) for that extent.
3. Decode Terrarium RGB encoding to metres.
4. Clip the assembled mosaic to the S2Coast island polygon.
5. Save GeoTIFF + elevation PNG and print an ASCII terrain map.

Outputs (written to the same directory as this script):
  delos_mapzen_s2coast.tif
  delos_mapzen_s2coast.png
"""

import io
import math
import os
import struct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import rasterio
from PIL import Image
from rasterio.crs import CRS
from rasterio.io import MemoryFile
from rasterio.mask import mask as rio_mask
from rasterio.transform import from_bounds
from shapely.geometry import LineString, mapping
from shapely.ops import linemerge, polygonize, unary_union

# ── Paths ─────────────────────────────────────────────────────────────────────
HERE         = os.path.dirname(os.path.abspath(__file__))
CSV_PATH     = r"G:\Dilos\aurehly_PCs_dilos.csv"
S2COAST_SHP  = r"G:\S2Coast2023_ShapeFile_vector\S2Coast-2023_Polyline_diss.shp"
OUT_TIF      = os.path.join(HERE, "delos_mapzen_s2coast.tif")
OUT_PNG      = os.path.join(HERE, "delos_mapzen_s2coast.png")

# ── Mapzen settings ───────────────────────────────────────────────────────────
ZOOM     = 15
TILE_PX  = 256
TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"

# Tight bbox around Delos only (excludes Rinia to the west)
LON0, LON1 = 25.260, 25.285
LAT0, LAT1 = 37.370, 37.420


# ── S2Coast binary scanner (mirrors plot_ba_data._scan_shp_lines) ─────────────
def _scan_shp_lines(shp_path, lon0, lon1, lat0, lat1):
    segments = []
    with open(shp_path, "rb") as f:
        f.seek(100)
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            _, content_len_words = struct.unpack(">II", header)
            content_bytes = content_len_words * 2
            shape_raw = f.read(4)
            if len(shape_raw) < 4:
                break
            shape_type = struct.unpack("<i", shape_raw)[0]
            if shape_type not in (3, 5):
                f.seek(content_bytes - 4, 1)
                continue
            bbox_raw = f.read(32)
            if len(bbox_raw) < 32:
                break
            xmin, ymin, xmax, ymax = struct.unpack("<dddd", bbox_raw)
            remaining = content_bytes - 36
            if xmax < lon0 or xmin > lon1 or ymax < lat0 or ymin > lat1:
                f.seek(remaining, 1)
                continue
            rest = f.read(remaining)
            if len(rest) < 8:
                continue
            n_parts  = struct.unpack("<i", rest[0:4])[0]
            n_points = struct.unpack("<i", rest[4:8])[0]
            if n_parts < 1 or n_points < 2:
                continue
            parts_end = 8 + n_parts * 4
            parts     = list(struct.unpack(f"<{n_parts}i", rest[8:parts_end]))
            pts_start = parts_end
            pts_bytes = rest[pts_start: pts_start + n_points * 16]
            if len(pts_bytes) < n_points * 16:
                continue
            points = np.frombuffer(pts_bytes, dtype="<f8").reshape(n_points, 2)
            for p_i in range(n_parts):
                p_start = parts[p_i]
                p_end   = parts[p_i + 1] if p_i + 1 < n_parts else n_points
                if p_end - p_start < 2:
                    continue
                xy = points[p_start:p_end]
                segments.append((xy[:, 0], xy[:, 1]))
    return segments


# ── Tile helpers ──────────────────────────────────────────────────────────────
def _lon_lat_to_tile(lon, lat, z):
    n   = 2 ** z
    x   = int((lon + 180.0) / 360.0 * n)
    lr  = math.radians(lat)
    y   = int((1.0 - math.log(math.tan(lr) + 1.0 / math.cos(lr)) / math.pi) / 2.0 * n)
    return x, y


def _tile_top_left(x, y, z):
    n   = 2 ** z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon, lat


def _terrarium_decode(arr):
    R = arr[:, :, 0].astype(np.float32)
    G = arr[:, :, 1].astype(np.float32)
    B = arr[:, :, 2].astype(np.float32)
    return R * 256.0 + G + B / 256.0 - 32768.0


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — S2Coast island polygon
# ══════════════════════════════════════════════════════════════════════════════
print("Step 1: Loading S2Coast shoreline ...", flush=True)
raw_segs = _scan_shp_lines(S2COAST_SHP, LON0, LON1, LAT0, LAT1)
lines = [
    LineString(zip(lx, ly))
    for lx, ly in raw_segs
    if not (lx.max() < LON0 or lx.min() > LON1 or ly.max() < LAT0 or ly.min() > LAT1)
]
print(f"  Segments in bbox : {len(lines)}")

merged = linemerge(unary_union(lines))
polys  = list(polygonize(merged))
if polys:
    island_poly = max(polys, key=lambda p: p.area)
    print(f"  Island polygon   : {island_poly.bounds}")
else:
    print("  No closed polygon — falling back to buffered lines.")
    island_poly = unary_union(lines).buffer(0.0003)

# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Download Mapzen tiles
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 2: Downloading Mapzen tiles (zoom 15) ...", flush=True)

# Expand slightly so the island is fully covered
buf = 0.002
tx_min, ty_max = _lon_lat_to_tile(LON0 - buf, LAT0 - buf, ZOOM)
tx_max, ty_min = _lon_lat_to_tile(LON1 + buf, LAT1 + buf, ZOOM)

n_cols = tx_max - tx_min + 1
n_rows = ty_max - ty_min + 1
print(f"  Tile grid        : {n_cols} x {n_rows} = {n_cols * n_rows} tiles")

mosaic = np.full((n_rows * TILE_PX, n_cols * TILE_PX), np.nan, dtype=np.float32)
headers = {"User-Agent": "python-requests/mapzen-delos-dem"}

for row_i, ty in enumerate(range(ty_min, ty_max + 1)):
    for col_i, tx in enumerate(range(tx_min, tx_max + 1)):
        url = TILE_URL.format(z=ZOOM, x=tx, y=ty)
        r   = requests.get(url, headers=headers, timeout=30)
        if r.status_code != 200:
            print(f"  WARNING: {tx}/{ty} -> HTTP {r.status_code}")
            continue
        img  = Image.open(io.BytesIO(r.content)).convert("RGB")
        elev = _terrarium_decode(np.array(img))
        r0, r1 = row_i * TILE_PX, (row_i + 1) * TILE_PX
        c0, c1 = col_i * TILE_PX, (col_i + 1) * TILE_PX
        mosaic[r0:r1, c0:c1] = elev
        print(f"  tile {tx}/{ty}  [{elev.min():.1f}, {elev.max():.1f}] m", flush=True)

# Georeferenced extent of the assembled mosaic
left,  top    = _tile_top_left(tx_min,     ty_min,     ZOOM)
right, bottom = _tile_top_left(tx_max + 1, ty_max + 1, ZOOM)
h_px, w_px    = mosaic.shape
transform_mosaic = from_bounds(left, bottom, right, top, w_px, h_px)

res_x   = (right - left)   / w_px
res_y   = (top   - bottom) / h_px
lat_c   = (LAT0 + LAT1) / 2
res_m_x = res_x * 111320 * math.cos(math.radians(lat_c))
res_m_y = res_y * 110540
print(f"  Resolution       : ~{res_m_x:.2f} m x {res_m_y:.2f} m  "
      f"({res_x:.7f} deg x {res_y:.7f} deg)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Clip mosaic to S2Coast polygon and save
# ══════════════════════════════════════════════════════════════════════════════
print("\nStep 3: Clipping to S2Coast island polygon ...", flush=True)

crs = CRS.from_epsg(4326)
mosaic_meta = {
    "driver": "GTiff", "dtype": "float32",
    "count": 1, "crs": crs,
    "height": h_px, "width": w_px,
    "transform": transform_mosaic, "nodata": float("nan"),
}

with MemoryFile() as memfile:
    with memfile.open(**mosaic_meta) as mem:
        mem.write(mosaic[np.newaxis, :, :])
    with memfile.open() as mem:
        out_img, out_transform = rio_mask(
            mem, [mapping(island_poly)], crop=True, nodata=np.nan
        )
        out_meta = mem.meta.copy()
        out_meta.update({
            "height"   : out_img.shape[1],
            "width"    : out_img.shape[2],
            "transform": out_transform,
        })

# The S2Coast 2023 shoreline traces the instantaneous waterline, so the clip
# boundary sits right at sea level. Mapzen Terrarium pixels that straddle the
# shoreline can therefore decode to slightly negative values (artefact of
# bilinear interpolation near the coast, not true below-sea elevations).
# Clamp only those negative pixels to 0 — do NOT shift the whole raster.
abs_min = float(np.nanmin(out_img))
if abs_min < 0:
    n_neg = int(np.sum((out_img < 0) & ~np.isnan(out_img)))
    out_img = np.where(np.isnan(out_img), np.nan,
                       np.maximum(out_img, 0.0)).astype(np.float32)
    print(f"  Clamped {n_neg} negative coastal pixels to 0 m  (min was {abs_min:.4f} m)")

os.makedirs(HERE, exist_ok=True)
with rasterio.open(OUT_TIF, "w", **out_meta) as dst:
    dst.write(out_img)
print(f"  Saved: {OUT_TIF}  ({out_img.shape[2]} cols x {out_img.shape[1]} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — Elevation stats + ASCII map + PNG
# ══════════════════════════════════════════════════════════════════════════════
elev  = out_img[0]
valid = elev[~np.isnan(elev)]
e_min, e_max, e_mean = float(valid.min()), float(valid.max()), float(valid.mean())

print(f"\nElevation stats:")
print(f"  Min  : {e_min:.2f} m")
print(f"  Max  : {e_max:.2f} m")
print(f"  Mean : {e_mean:.2f} m")
print(f"  Res  : ~{res_m_x:.2f} m x {res_m_y:.2f} m")

# ASCII map
W, H   = 80, 30
re, ce = elev.shape
rs     = max(1, re // H)
cs     = max(1, ce // W)
block  = elev[::rs, ::cs][:H, :W]
SHADES = " .,-~:;=!*#$@"

print("\n+" + "-" * W + "+")
for row in range(block.shape[0]):
    line = "|"
    for col in range(block.shape[1]):
        v = block[row, col]
        if np.isnan(v):
            line += " "
        else:
            idx   = int((v - e_min) / (e_max - e_min + 1e-9) * (len(SHADES) - 1))
            line += SHADES[idx]
    line += "|"
    print(line)
print("+" + "-" * W + "+")
print(f"  {e_min:.0f} m {'.'*34} {e_max:.0f} m  (low -> high)")

# PNG
fig, ax = plt.subplots(figsize=(6, 10))
im = ax.imshow(elev, cmap="terrain", vmin=e_min, vmax=e_max)
plt.colorbar(im, ax=ax, label="Elevation (m)")
ax.set_title(f"Delos - Mapzen DEM  (~{res_m_x:.1f} m)  |  S2Coast 2023 boundary")
ax.axis("off")
fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nElevation map  : {OUT_PNG}")
print("Done.")
