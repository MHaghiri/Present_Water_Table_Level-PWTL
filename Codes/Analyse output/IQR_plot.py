#!/usr/bin/env python3
"""
Plot yearly IQR maps (IQR_WTD12Z_2001.tif ... IQR_WTD12Z_2025.tif) in ONE multi-panel figure,
styled like your screenshot:
- Green sequential colormap
- Bottom horizontal colorbar with label:
    "IQR of Standardized WB12 (P − ET)"
- Fixed colorbar domain: 0 to 2.5 with ticks every 0.5
- Greenland filled light gray + black outline
- NA boundary (without Greenland) outlined in black
- Save as PNG (dpi=1500) and PDF

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
from matplotlib.ticker import FuncFormatter

# =========================
# USER PATHS
# =========================
IQR_DIR = "/home/mohammad/Desktop/1/3"   # <-- put your folder path here

GREENLAND_SHP   = "/home/mohammad/Desktop/N_America_shapefile/Greenland.shp"
NA_BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

OUT_DIR = "/home/mohammad/Desktop/1/3"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_PNG = os.path.join(OUT_DIR, "IQR_WB12Z_2001_2025_panel.png")
OUT_PDF = os.path.join(OUT_DIR, "IQR_WB12Z_2001_2025_panel.pdf")

# =========================
# FIND + SORT FILES
# =========================
pat = re.compile(r"IQR_WB12Z_(\d{4})\.tif$")
tif_paths = sorted(glob.glob(os.path.join(IQR_DIR, "IQR_WB12Z_*.tif")))

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

# Keep only 2001–2025
pairs = [(y, p) for (y, p) in pairs if 2001 <= y <= 2025]
if not pairs:
    raise FileNotFoundError("Found TIFFs, but none in the year range 2001–2025.")

# =========================
# READ SHAPEFILES ONCE
# =========================
g_green = gpd.read_file(GREENLAND_SHP)
g_na    = gpd.read_file(NA_BOUNDARY_SHP)

# =========================
# COLORBAR DOMAIN (FIXED LIKE YOUR FIGURE)
# =========================
vmin = 0.0
vmax = 2.5

# Green sequential like your screenshot
cmap = plt.cm.Greens
norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# tick formatter: integers as "0,1,2" and halves as "0.5,1.5,2.5"
def half_step_fmt(x, pos):
    if abs(x - round(x)) < 1e-9:
        return f"{int(round(x))}"
    return f"{x:.1f}".rstrip("0").rstrip(".")

# =========================
# FIGURE LAYOUT (5 columns + bottom colorbar row)
# =========================
n = len(pairs)
ncols = 5
nrows = int(np.ceil(n / ncols))

fig_w = 3.1 * ncols
fig_h = 2.6 * nrows + 1.2  # extra for bottom colorbar
fig = plt.figure(figsize=(fig_w, fig_h))

# add an extra row at bottom for colorbar
gs = fig.add_gridspec(
    nrows=nrows + 1,
    ncols=ncols,
    height_ratios=[1]*nrows + [0.12],
    wspace=0.02,
    hspace=0.10
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

        # reproject vectors to raster CRS
        na    = g_na.to_crs(raster_crs)
        green = g_green.to_crs(raster_crs)

        # mask outside NA boundary (transparent outside)
        mask_in = geometry_mask(
            geometries=na.geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True
        )
        band[~mask_in] = np.nan

        # extent for imshow
        left, bottom, right, top = src.bounds
        extent = [left, right, bottom, top]

    ax.imshow(
        band,
        extent=extent,
        cmap=cmap,
        norm=norm,
        interpolation="nearest"
    )

    # Greenland: light gray fill + black outline
    green.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1.0, zorder=5)

    # NA boundary outline (black)
    na.boundary.plot(ax=ax, color="black", linewidth=1.0, zorder=6)

    ax.set_title(str(year), fontsize=10, pad=2)
    ax.set_axis_off()

# =========================
# COLORBAR (BOTTOM HORIZONTAL like your screenshot)
# =========================
cax = fig.add_subplot(gs[-1, :])
cb = mpl.colorbar.ColorbarBase(cax, cmap=cmap, norm=norm, orientation="horizontal")

cb.set_label("IQR of Standardized WB12 (P − ET)", fontsize=13, labelpad=6)

ticks = np.arange(vmin, vmax + 0.001, 0.5)  # 0, 0.5, ..., 2.5
cb.set_ticks(ticks)
cb.ax.xaxis.set_major_formatter(FuncFormatter(half_step_fmt))
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
