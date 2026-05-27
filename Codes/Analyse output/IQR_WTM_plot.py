#!/usr/bin/env python3
"""
Plot yearly IQR maps (IQR_WTD12Z_2001.tif ... IQR_WTD12Z_2025.tif) in ONE multi-panel figure,
with a "Blues" colorbar style, and with:
- Greenland filled light gray + black outline
- NA boundary (without Greenland) outlined in black
- Save as PNG (dpi=1500) and PDF

Edits (this version):
- Colorbar moved to the BOTTOM (horizontal) like your example figure
- Colorbar values shown with 2 decimals
- Colorbar domain is fixed by vmin/vmax (edit below if needed)

Edit ONLY: IQR_DIR (where your GeoTIFFs live).
"""

import os
import re
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter

# =========================
# USER PATHS
# =========================
IQR_DIR = "/home/mohammad/Desktop/1/8"   # <-- IMPORTANT: put your folder path here

GREENLAND_SHP = "/home/mohammad/Desktop/N_America_shapefile/Greenland.shp"
NA_BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

OUT_DIR = "/home/mohammad/Desktop/1/8"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "IQR_WTD12Z_2001_2025_panel.png")
OUT_PDF = os.path.join(OUT_DIR, "IQR_WTD12Z_2001_2025_panel.pdf")

# =========================
# FIND + SORT FILES
# =========================
pat = re.compile(r"IQR_WTD12Z_(\d{4})\.tif$")
tif_paths = sorted(glob.glob(os.path.join(IQR_DIR, "IQR_WTD12Z_*.tif")))

pairs = []
for p in tif_paths:
    m = pat.search(os.path.basename(p))
    if m:
        year = int(m.group(1))
        pairs.append((year, p))

pairs = sorted(pairs, key=lambda x: x[0])

if not pairs:
    raise FileNotFoundError(
        f"No files found like IQR_WTD12Z_YYYY.tif in: {IQR_DIR}\n"
        f"Example expected: {os.path.join(IQR_DIR, 'IQR_WTD12Z_2001.tif')}"
    )

# If you specifically want 2001–2025 only:
pairs = [(y, p) for (y, p) in pairs if 2001 <= y <= 2025]
if not pairs:
    raise FileNotFoundError("Found TIFFs, but none in the year range 2001–2025.")

# =========================
# READ SHAPEFILES ONCE
# =========================
g_green = gpd.read_file(GREENLAND_SHP)
g_na = gpd.read_file(NA_BOUNDARY_SHP)

# =========================
# COLORBAR DOMAIN (FIXED)
# =========================
vmin = 0.0
vmax = 1.0   # change to 1.5 if you want

cmap = plt.cm.Blues
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# =========================
# FIGURE LAYOUT (maps + bottom colorbar row)
# =========================
n = len(pairs)
ncols = 5
nrows = int(np.ceil(n / ncols))

fig_w = 3.1 * ncols
fig_h = 2.6 * nrows + 0.9  # extra space for bottom colorbar
fig = plt.figure(figsize=(fig_w, fig_h))

# Add an extra row at the bottom for the colorbar
gs = fig.add_gridspec(
    nrows=nrows + 1,
    ncols=ncols,
    height_ratios=[1] * nrows + [0.08],  # last row is thin for colorbar
    wspace=0.02,
    hspace=0.15
)

# =========================
# PLOT EACH YEAR
# =========================
for i, (year, tif) in enumerate(pairs):
    r = i // ncols
    c = i % ncols
    ax = fig.add_subplot(gs[r, c])

    with rasterio.open(tif) as src:
        band = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            band[band == nodata] = np.nan
        band[~np.isfinite(band)] = np.nan

        raster_crs = src.crs
        if raster_crs is None:
            raise RuntimeError(f"Raster has no CRS: {tif}")

        na = g_na.to_crs(raster_crs)
        green = g_green.to_crs(raster_crs)

        mask_in = geometry_mask(
            geometries=na.geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )
        band[~mask_in] = np.nan

        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]

    ax.imshow(
        band,
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    green.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1.0, zorder=5)
    na.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

    ax.set_title(str(year), fontsize=10, pad=2)
    ax.set_axis_off()

# Hide empty panels if any (e.g., last row not full)
for j in range(len(pairs), nrows * ncols):
    r = j // ncols
    c = j % ncols
    ax = fig.add_subplot(gs[r, c])
    ax.set_axis_off()

# =========================
# BOTTOM COLORBAR (horizontal)
# =========================
cax = fig.add_subplot(gs[-1, :])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")

cb.set_label("IQR of Standardized WTD", fontsize=14, labelpad=6)

nt = 6
ticks = np.linspace(vmin, vmax, nt)
cb.set_ticks(ticks)
cb.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
cb.ax.tick_params(labelsize=11)

# =========================
# SAVE
# =========================
fig.savefig(OUT_PNG, dpi=1500, bbox_inches="tight")
fig.savefig(OUT_PDF, dpi=1500, bbox_inches="tight")
plt.close(fig)

print("Saved:")
print(" -", OUT_PNG)
print(" -", OUT_PDF)
print(f"Color scale: vmin={vmin}, vmax={vmax}")
