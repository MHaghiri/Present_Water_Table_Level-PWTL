#!/usr/bin/env python3
"""
Yearly IQR maps (Dispersion) like the example figure, but based on precipitation & evaporation:

- Index: WB(t)=P-E, WB12=12-month rolling sum, Z=(WB12-mean)/std (per-pixel baseline),
- For each year: IQR across the 12 monthly Z values in that year: IQR = Q75 - Q25.

Your requested changes:
1) Colormap similar to your figure (green sequential).
2) Put ALL yearly maps in ONE figure (no axis text).
3) Do ALL calculations only inside:
   /home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp
   and draw that boundary in black (lw=1).
4) Draw Greenland boundary (fill light gray, edge black lw=1) using:
   /home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp

Outputs:
- GeoTIFF per year:  IQR_WB12Z_YYYY.tif   (masked outside NA boundary)
- ONE combined figure: IQR_WB12Z_all_years.png and .pdf (dpi=1500)
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
PREC_DIR = "/media/mohammad/My Book/0-2025/Monthly/pr/CMIP6/monthly/downscaled/N_America/1"
EVAP_DIR = "/media/mohammad/My Book/0-2025/Monthly/evap/CMIP6/monthly/downscaled/N_America/1"
OUT_DIR  = "/home/mohammad/Desktop/1"

START_YEAR = 2001
END_YEAR   = 2025

BASELINE_START = 2001
BASELINE_END   = 2025

BLOCK_SIZE = 512
DPI = 1500

NA_BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"
GREENLAND_SHP   = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"

# Plot look
COLORMAP = "Greens"         # close to your example
# =========================


PAT_P = re.compile(r"^N_America_(\d{6})_precipitation\.tif$")
PAT_E = re.compile(r"^N_America_(\d{6})_evaporation\.tif$")


def yyyymm_to_year_month(yyyymm: str):
    return int(yyyymm[:4]), int(yyyymm[4:6])


def list_common_months(prec_dir, evap_dir):
    p_files = glob.glob(os.path.join(prec_dir, "N_America_??????_precipitation.tif"))
    e_files = glob.glob(os.path.join(evap_dir, "N_America_??????_evaporation.tif"))

    p_months = set()
    for f in p_files:
        m = PAT_P.match(os.path.basename(f))
        if m:
            p_months.add(m.group(1))

    e_months = set()
    for f in e_files:
        m = PAT_E.match(os.path.basename(f))
        if m:
            e_months.add(m.group(1))

    common = sorted(p_months.intersection(e_months))
    if not common:
        raise RuntimeError("No common YYYYMM found between precip and evaporation folders.")
    return common


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
    """
    Returns boolean array True for pixels INSIDE shapes.
    """
    # geometry_mask returns True for OUTSIDE by default; use invert=True for inside
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


def compute_baseline_mean_std(months, prec_dir, evap_dir, baseline_start, baseline_end,
                              block, na_shapes):
    # reference raster
    ref_p = os.path.join(prec_dir, f"N_America_{months[0]}_precipitation.tif")
    with rasterio.open(ref_p) as ref:
        profile = ref.profile.copy()
        nodata = ref.nodata
        height, width = ref.height, ref.width

    mean = np.full((height, width), 0.0, dtype=np.float32)
    m2   = np.full((height, width), 0.0, dtype=np.float32)
    cnt  = np.zeros((height, width), dtype=np.int32)

    for win in iter_windows(width, height, block):
        wh, ww = int(win.height), int(win.width)

        # window transform for correct masking
        with rasterio.open(ref_p) as ref:
            w_transform = ref.window_transform(win)
        inside_na = make_inside_mask_for_window(win, w_transform, na_shapes)

        w_mean = np.zeros((wh, ww), dtype=np.float32)
        w_m2   = np.zeros((wh, ww), dtype=np.float32)
        w_cnt  = np.zeros((wh, ww), dtype=np.int32)

        pe_deque = deque(maxlen=12)

        for yyyymm in months:
            y, _ = yyyymm_to_year_month(yyyymm)

            p_path = os.path.join(prec_dir, f"N_America_{yyyymm}_precipitation.tif")
            e_path = os.path.join(evap_dir, f"N_America_{yyyymm}_evaporation.tif")

            p = read_block(p_path, win, nodata)
            e = read_block(e_path, win, nodata)
            pe = p - e

            # restrict computation to NA boundary only
            pe[~inside_na] = np.nan

            pe_deque.append(pe)
            if len(pe_deque) < 12:
                continue

            wb12 = np.nansum(np.stack(pe_deque, axis=0), axis=0).astype(np.float32)

            if baseline_start <= y <= baseline_end:
                welford_update(w_mean, w_m2, w_cnt, wb12)

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


def compute_yearly_iqr_tifs(months, years, prec_dir, evap_dir, mean, std, profile, nodata,
                            out_dir, block, na_shapes):

    height, width = mean.shape

    profile_out = profile.copy()
    profile_out.update(dtype="float32", count=1, nodata=np.nan, compress="deflate")

    # reference raster for window transforms
    ref_p = os.path.join(prec_dir, f"N_America_{months[0]}_precipitation.tif")

    for year in years:
        out_tif = os.path.join(out_dir, f"IQR_WB12Z_{year}.tif")
        print(f"\nYear {year}: computing IQR tif...")

        with rasterio.open(ref_p) as ref, rasterio.open(out_tif, "w", **profile_out) as dst:
            for win in iter_windows(width, height, block):
                wh, ww = int(win.height), int(win.width)

                w_transform = ref.window_transform(win)
                inside_na = make_inside_mask_for_window(win, w_transform, na_shapes)

                r0, c0 = int(win.row_off), int(win.col_off)
                r1, c1 = r0 + wh, c0 + ww

                w_mean = mean[r0:r1, c0:c1]
                w_std  = std[r0:r1, c0:c1]

                pe_deque = deque(maxlen=12)
                z_months = []

                for yyyymm in months:
                    y, _ = yyyymm_to_year_month(yyyymm)

                    p_path = os.path.join(prec_dir, f"N_America_{yyyymm}_precipitation.tif")
                    e_path = os.path.join(evap_dir, f"N_America_{yyyymm}_evaporation.tif")

                    p = read_block(p_path, win, nodata)
                    e = read_block(e_path, win, nodata)
                    pe = p - e
                    pe[~inside_na] = np.nan

                    pe_deque.append(pe)
                    if len(pe_deque) < 12:
                        continue

                    wb12 = np.nansum(np.stack(pe_deque, axis=0), axis=0).astype(np.float32)

                    if y == year:
                        valid = np.isfinite(wb12) & np.isfinite(w_mean) & np.isfinite(w_std)
                        z = np.full((wh, ww), np.nan, dtype=np.float32)
                        z[valid] = (wb12[valid] - w_mean[valid]) / w_std[valid]
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
    """
    Estimate a single vmax for all panels using a robust percentile across all years.
    Uses subsampling to stay fast.
    """
    vals = []
    for p in tif_paths:
        with rasterio.open(p) as src:
            a = src.read(1)
        a = a[np.isfinite(a)]
        if a.size == 0:
            continue
        # subsample
        a = a[::sample_step]
        vals.append(a)
    if not vals:
        return 1.0
    allv = np.concatenate(vals)
    return float(np.nanpercentile(allv, 99))  # robust upper limit


def plot_all_years_one_figure(years, tif_paths, out_png, out_pdf,
                              na_gdf, greenland_gdf, dpi, cmap, vmin, vmax):
    n = len(years)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    # Figure size: tune for readability; dpi is huge so keep inches moderate
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6*ncols, 3.0*nrows))
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    # Read one raster for extent
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

        # Greenland overlay (fill light gray + black edge)
        if greenland_gdf is not None and len(greenland_gdf) > 0:
            greenland_gdf.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1)

        # NA boundary overlay (black edge, no fill)
        if na_gdf is not None and len(na_gdf) > 0:
            na_gdf.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=1)

        ax.set_title(str(year), fontsize=10, pad=2)
        ax.set_axis_off()

    # Turn off empty panels
    for j in range(n, nrows*ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].set_axis_off()

    # One shared colorbar
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.set_label("IQR of standardized WB12 (P−E)", rotation=90)

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    # Checks
    if not os.path.isdir(PREC_DIR):
        raise SystemExit(f"Precip folder not found: {PREC_DIR}")
    if not os.path.isdir(EVAP_DIR):
        raise SystemExit(f"Evap folder not found: {EVAP_DIR}")
    os.makedirs(OUT_DIR, exist_ok=True)

    months = list_common_months(PREC_DIR, EVAP_DIR)
    print(f"Found {len(months)} common months.")
    print(f"First: {months[0]}   Last: {months[-1]}")

    # Reference CRS
    ref_p = os.path.join(PREC_DIR, f"N_America_{months[0]}_precipitation.tif")
    with rasterio.open(ref_p) as ref:
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

    # For masking computations: use NA boundary geometries
    na_shapes = [geom for geom in na_gdf.geometry if geom is not None]

    years = list(range(START_YEAR, END_YEAR + 1))

    print("\nPASS 1: computing baseline mean/std of WB12 (masked to NA boundary) ...")
    mean, std, profile, nodata = compute_baseline_mean_std(
        months, PREC_DIR, EVAP_DIR, BASELINE_START, BASELINE_END, BLOCK_SIZE, na_shapes
    )

    print("\nPASS 2: computing yearly IQR GeoTIFFs (masked to NA boundary) ...")
    compute_yearly_iqr_tifs(
        months, years, PREC_DIR, EVAP_DIR, mean, std, profile, nodata,
        out_dir=OUT_DIR, block=BLOCK_SIZE, na_shapes=na_shapes
    )

    # Build combined figure (all years)
    tif_paths = [os.path.join(OUT_DIR, f"IQR_WB12Z_{y}.tif") for y in years]
    missing = [p for p in tif_paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Some output tifs are missing (first few):\n" + "\n".join(missing[:5]))

    # Choose a consistent color range across all panels
    vmin = 0.0
    vmax = robust_global_vmax(tif_paths, sample_step=8)
    if vmax <= vmin:
        vmax = vmin + 1.0

    out_png = os.path.join(OUT_DIR, "IQR_WB12Z_all_years.png")
    out_pdf = os.path.join(OUT_DIR, "IQR_WB12Z_all_years.pdf")

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

