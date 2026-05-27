#!/usr/bin/env python3
"""
Yearly IQR maps (Dispersion) for WTD based on monthly WTD GeoTIFFs:

- WTD12(t) = 12-month rolling MEAN (recommended for a state variable like WTD)
- Z(t) = (WTD12 - mean) / std  (per-pixel baseline, computed over BASELINE_START..BASELINE_END)
- For each year: IQR across the 12 monthly Z values in that year: IQR = Q75 - Q25

Computations are restricted to:
  /home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp
Boundary drawn in black (lw=1)

Greenland overlay (light gray fill + black edge) using:
  /home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp

Outputs:
- GeoTIFF per year:  IQR_WTD12Z_YYYY.tif   (masked outside NA boundary)
- ONE combined figure: IQR_WTD12Z_all_years.png and .pdf (dpi=1500)
"""

import os
import re
import glob
import math
from collections import deque

import numpy as np
import rasterio
from rasterio.windows import Window
from rasterio.features import geometry_mask

import geopandas as gpd
import matplotlib.pyplot as plt


# =========================
# USER SETTINGS
# =========================
WTD_DIR = "/media/mohammad/My Book/WTM_Result/Monthly/3"
OUT_DIR = "/home/mohammad/Desktop/1"

START_YEAR = 2000
END_YEAR   = 2025

BASELINE_START = 2000
BASELINE_END   = 2025

BLOCK_SIZE = 512
DPI = 1500

NA_BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"
GREENLAND_SHP   = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"

# Colorbar similar to your example figure
COLORMAP = "Blues"
# =========================


# Extract YYYYMM from filename (first 6-digit token that looks like YYYYMM)
YYYYMM_RE = re.compile(r"(19\d{2}(0[1-9]|1[0-2])|20\d{2}(0[1-9]|1[0-2]))")  # 190001..209912


def yyyymm_to_year_month(yyyymm: str):
    return int(yyyymm[:4]), int(yyyymm[4:6])


def list_monthly_wtd_files(wtd_dir):
    """
    Returns a sorted list of tuples: (yyyymm, filepath)
    Keeps only months within START_YEAR..END_YEAR.
    """
    tifs = glob.glob(os.path.join(wtd_dir, "*.tif"))
    items = []
    for f in tifs:
        base = os.path.basename(f)
        m = YYYYMM_RE.search(base)
        if not m:
            continue
        yyyymm = m.group(1)
        y, mo = yyyymm_to_year_month(yyyymm)
        if START_YEAR <= y <= END_YEAR:
            items.append((yyyymm, f))

    items.sort(key=lambda x: x[0])
    if not items:
        raise RuntimeError(
            "No monthly WTD GeoTIFFs found with a YYYYMM in filename.\n"
            "Your files must include something like 200001, 200002, ... in the name."
        )
    return items


def iter_windows(width, height, block):
    for r in range(0, height, block):
        h = min(block, height - r)
        for c in range(0, width, block):
            w = min(block, width - c)
            yield Window(c, r, w, h)


def read_block(path, window, nodata):
    with rasterio.open(path) as src:
        a = src.read(1, window=window).astype(np.float32)
    if nodata is not None:
        a[(a == nodata) | (~np.isfinite(a))] = np.nan
    else:
        a[~np.isfinite(a)] = np.nan
    return a


def make_inside_mask_for_window(window: Window, window_transform, shapes):
    """Returns boolean array True for pixels INSIDE shapes."""
    inside = geometry_mask(
        shapes,
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform,
        invert=True,
        all_touched=False
    )
    return inside


def welford_update(mean, m2, count, x):
    valid = np.isfinite(x)
    if not np.any(valid):
        return
    count[valid] += 1
    c = count[valid].astype(np.float32)
    delta = x[valid] - mean[valid]
    mean[valid] += delta / c
    delta2 = x[valid] - mean[valid]
    m2[valid] += delta * delta2


def compute_baseline_mean_std(wtd_items, baseline_start, baseline_end, block, na_shapes):
    """
    Baseline mean/std of WTD12 (rolling 12-month mean), per pixel.
    """
    ref_path = wtd_items[0][1]
    with rasterio.open(ref_path) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata
        height, width = ref.height, ref.width

    mean = np.full((height, width), 0.0, dtype=np.float32)
    m2   = np.full((height, width), 0.0, dtype=np.float32)
    cnt  = np.zeros((height, width), dtype=np.int32)

    for win in iter_windows(width, height, block):
        wh, ww = int(win.height), int(win.width)

        with rasterio.open(ref_path) as ref:
            w_transform = ref.window_transform(win)
        inside_na = make_inside_mask_for_window(win, w_transform, na_shapes)

        w_mean = np.zeros((wh, ww), dtype=np.float32)
        w_m2   = np.zeros((wh, ww), dtype=np.float32)
        w_cnt  = np.zeros((wh, ww), dtype=np.int32)

        wtd_deque = deque(maxlen=12)

        for yyyymm, fpath in wtd_items:
            y, _ = yyyymm_to_year_month(yyyymm)

            wtd = read_block(fpath, win, nodata)
            wtd[~inside_na] = np.nan

            wtd_deque.append(wtd)
            if len(wtd_deque) < 12:
                continue

            # WTD12: 12-month rolling MEAN
            wtd12 = np.nanmean(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)

            if baseline_start <= y <= baseline_end:
                welford_update(w_mean, w_m2, w_cnt, wtd12)

        r0, c0 = int(win.row_off), int(win.col_off)
        r1, c1 = r0 + wh, c0 + ww
        mean[r0:r1, c0:c1] = w_mean
        m2[r0:r1, c0:c1]   = w_m2
        cnt[r0:r1, c0:c1]  = w_cnt

    std = np.full_like(mean, np.nan, dtype=np.float32)
    ok = cnt > 1
    std[ok] = np.sqrt(m2[ok] / (cnt[ok].astype(np.float32) - 1.0))
    std[(std == 0) | (~np.isfinite(std))] = np.nan
    mean[~np.isfinite(mean)] = np.nan
    return mean, std, profile, nodata


def compute_yearly_iqr_tifs(wtd_items, years, mean, std, profile, nodata, out_dir, block, na_shapes):
    """
    For each year:
      - build Z for each month of that year (from WTD12),
      - compute IQR across those 12 monthly Z values,
      - write IQR tif.
    """
    height, width = mean.shape

    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")

    ref_path = wtd_items[0][1]

    for year in years:
        out_tif = os.path.join(out_dir, f"IQR_WTD12Z_{year}.tif")
        print(f"\nYear {year}: computing IQR tif...")

        with rasterio.open(ref_path) as ref, rasterio.open(out_tif, "w", **profile_out) as dst:
            for win in iter_windows(width, height, block):
                wh, ww = int(win.height), int(win.width)

                w_transform = ref.window_transform(win)
                inside_na = make_inside_mask_for_window(win, w_transform, na_shapes)

                r0, c0 = int(win.row_off), int(win.col_off)
                r1, c1 = r0 + wh, c0 + ww

                w_mean = mean[r0:r1, c0:c1]
                w_std  = std[r0:r1, c0:c1]

                wtd_deque = deque(maxlen=12)
                z_months = []

                for yyyymm, fpath in wtd_items:
                    y, _ = yyyymm_to_year_month(yyyymm)

                    wtd = read_block(fpath, win, nodata)
                    wtd[~inside_na] = np.nan

                    wtd_deque.append(wtd)
                    if len(wtd_deque) < 12:
                        continue

                    wtd12 = np.nanmean(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)

                    if y == year:
                        valid = np.isfinite(wtd12) & np.isfinite(w_mean) & np.isfinite(w_std)
                        z = np.full((wh, ww), np.nan, dtype=np.float32)
                        z[valid] = (wtd12[valid] - w_mean[valid]) / w_std[valid]
                        z[~inside_na] = np.nan
                        z_months.append(z)

                    if y > year:
                        break

                if len(z_months) == 0:
                    iqr = np.full((wh, ww), np.nan, dtype=np.float32)
                else:
                    stack = np.stack(z_months, axis=0)
                    q25 = np.nanpercentile(stack, 25, axis=0)
                    q75 = np.nanpercentile(stack, 75, axis=0)
                    iqr = (q75 - q25).astype(np.float32)
                    iqr[~inside_na] = np.nan

                dst.write(iqr, 1, window=win)

        print(f"Saved tif: {out_tif}")


def robust_global_vmax(tif_paths, sample_step=8):
    vals = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            a = src.read(1)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        a = a[::sample_step]
        vals.append(a)
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    return float(np.nanpercentile(allv, 99))


def plot_all_years_one_figure(years, tif_paths, out_png, out_pdf,
                              na_gdf, greenland_gdf, dpi, cmap, vmin, vmax):
    n = len(years)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6*ncols, 3.0*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    with rasterio.open(tif_paths[0]) as src0:
        bounds = src0.bounds
        extent = [bounds.left, bounds.right, bounds.bottom, bounds.top]

    mappable = None

    for i, year in enumerate(years):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        with rasterio.open(tif_paths[i]) as src:
            arr = src.read(1)

        im = ax.imshow(arr, extent=extent, origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        mappable = im

        if greenland_gdf is not None and len(greenland_gdf) > 0:
            greenland_gdf.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1)

        if na_gdf is not None and len(na_gdf) > 0:
            na_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        ax.set_title(str(year), fontsize=10, pad=2)
        ax.set_axis_off()

    for j in range(n, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_axis_off()

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("IQR of standardized WTD12", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    if not os.path.isdir(WTD_DIR):
        raise SystemExit(f"WTD folder not found: {WTD_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    wtd_items = list_monthly_wtd_files(WTD_DIR)
    print(f"Found {len(wtd_items)} monthly WTD files.")
    print(f"First: {wtd_items[0][0]}   Last: {wtd_items[-1][0]}")

    # Reference CRS from first WTD raster
    ref_path = wtd_items[0][1]
    with rasterio.open(ref_path) as ref:
        raster_crs = ref.crs

    # Load shapefiles and reproject to raster CRS
    if not os.path.exists(NA_BOUNDARY_SHP):
        raise SystemExit(f"NA boundary shapefile not found: {NA_BOUNDARY_SHP}")
    if not os.path.exists(GREENLAND_SHP):
        raise SystemExit(f"Greenland shapefile not found: {GREENLAND_SHP}")

    na_gdf = gpd.read_file(NA_BOUNDARY_SHP)
    greenland_gdf = gpd.read_file(GREENLAND_SHP)

    if na_gdf.crs is None:
        raise SystemExit("NA boundary shapefile has no CRS. Please define it (e.g., EPSG:4326) then rerun.")
    if greenland_gdf.crs is None:
        raise SystemExit("Greenland shapefile has no CRS. Please define it then rerun.")

    na_gdf = na_gdf.to_crs(raster_crs)
    greenland_gdf = greenland_gdf.to_crs(raster_crs)

    na_shapes = [geom for geom in na_gdf.geometry if geom is not None]
    years = list(range(START_YEAR, END_YEAR + 1))

    print("\nPASS 1: computing baseline mean/std of WTD12 (masked to NA boundary) ...")
    mean, std, profile, nodata = compute_baseline_mean_std(
        wtd_items, BASELINE_START, BASELINE_END, BLOCK_SIZE, na_shapes
    )

    print("\nPASS 2: computing yearly IQR GeoTIFFs (masked to NA boundary) ...")
    compute_yearly_iqr_tifs(
        wtd_items, years, mean, std, profile, nodata, OUT_DIR, BLOCK_SIZE, na_shapes
    )

    tif_paths = [os.path.join(OUT_DIR, f"IQR_WTD12Z_{y}.tif") for y in years]
    missing = [p for p in tif_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Some output tifs are missing (first few):\n" + "\n".join(missing[:5]))

    vmin = 0.0
    vmax = robust_global_vmax(tif_paths, sample_step=8)
    if vmax <= vmin:
        vmax = vmin + 1.0

    out_png = os.path.join(OUT_DIR, "IQR_WTD12Z_all_years.png")
    out_pdf = os.path.join(OUT_DIR, "IQR_WTD12Z_all_years.pdf")

    print(f"\nPlotting ONE combined figure for {len(years)} years...")
    plot_all_years_one_figure(
        years=years,
        tif_paths=tif_paths,
        out_png=out_png,
        out_pdf=out_pdf,
        na_gdf=na_gdf,
        greenland_gdf=greenland_gdf,
        dpi=DPI,
        cmap=COLORMAP,
        vmin=vmin,
        vmax=vmax,
    )

    print("\nDone.")
    print(f"Combined figure saved:\n  {out_png}\n  {out_pdf}")


if __name__ == "__main__":
    main()
