#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
import rasterio.mask
import matplotlib.pyplot as plt

# ============================================================
# USER PATHS
# ============================================================
TIF_DIR = "/media/mohammad/My Book/WTM_Result/1800-2015/fix/1"
OUT_DIR = "/home/mohammad/Desktop/1/13"

SHAPE_A = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"
SHAPE_WATERSHED = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_level2_watershed_without_greenland.shp"
WATERSHED_NAME_FIELD = "name3"

# ============================================================
# MODE SWITCH (GROUNDWATER)
# If your TIFF groundwater values are already NEGATIVE and you want to keep them negative:
#   VALUE_MULTIPLIER = 1.0
# If you ever want to flip sign (plot positive depth): set to -1.0
# ============================================================
VALUE_MULTIPLIER = 1.0
Y_LABEL = "Mean groundwater depth (m)"   # will be a single global Y title

# ============================================================
# Y-AXIS LOCATION OPTIONS
#   "left"  -> ticks/labels on left (default)
#   "right" -> ticks/labels on right
# ============================================================
Y_AXIS_SIDE = "left"   # change to "right" if you want

# ============================================================
# FIGURE STYLE (same as your script)
# ============================================================
FIG_W, FIG_H = 16, 9
NROWS, NCOLS = 4, 3

COLOR_OBS   = "#2b0a3d"   # deep purple
COLOR_TREND = "#ff7f0e"   # orange
BAND_COLOR  = "#f4a3a3"   # light pink band
BAND_ALPHA  = 0.35

LW_OBS   = 2.0
MS_OBS   = 4.5
LW_TREND = 2.0
LS_TREND = "--"

FS_PANEL  = 14
FS_AXIS   = 12
FS_TICKS  = 10
FS_LEGEND = 16

DPI = 300

# ============================================================
# HELPERS
# ============================================================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def parse_year_from_filename(fp: str) -> int:
    """
    Example: N_America_001990_petsc_000000001.tif -> 1990, then +10 -> 2000
    """
    base = os.path.basename(fp)
    m = re.search(r"N_America_(\d{6})_petsc", base)
    if not m:
        raise ValueError(f"Could not parse year from filename: {base}")
    y = int(m.group(1))
    return y + 10

def to_raster_crs(gdf: gpd.GeoDataFrame, raster_crs) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("Shapefile CRS is undefined. Please define CRS before running.")
    if gdf.crs != raster_crs:
        return gdf.to_crs(raster_crs)
    return gdf

def zonal_mean_raster(tif_path: str, geom) -> float:
    """
    Compute mean of raster pixels inside polygon geometry.
    Returns np.nan if polygon has no valid pixels.
    """
    with rasterio.open(tif_path) as src:
        nodata = src.nodata
        out_img, _ = rasterio.mask.mask(src, [geom], crop=True, filled=False)
        arr = out_img[0].astype("float64")

        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)

        if np.ma.isMaskedArray(arr):
            arr = arr.filled(np.nan)

        if np.all(np.isnan(arr)):
            return np.nan

        return float(np.nanmean(arr))

def linear_trend_with_ci(x_years: np.ndarray, y: np.ndarray):
    """
    OLS y = a + b*x, with 95% CI band for mean prediction.
    """
    x = x_years.astype(float)
    y = y.astype(float)

    # remove NaNs (important if a polygon has missing pixels in some years)
    msk = np.isfinite(x) & np.isfinite(y)
    x = x[msk]
    y = y[msk]

    n = len(x)
    if n < 3:
        # not enough points for trend
        return np.full_like(x_years, np.nan, dtype=float), None, None, msk

    xbar = x.mean()
    ybar = y.mean()

    Sxx = np.sum((x - xbar) ** 2)
    Sxy = np.sum((x - xbar) * (y - ybar))

    b = Sxy / Sxx
    a = ybar - b * xbar

    # predictions on full x_years
    xfull = x_years.astype(float)
    yhat_full = a + b * xfull

    resid = y - (a + b * x)
    s2 = np.sum(resid ** 2) / (n - 2)
    s = np.sqrt(s2)

    try:
        from scipy.stats import t
        tcrit = t.ppf(0.975, df=n - 2)
    except Exception:
        tcrit = 2.0

    se_mean_full = s * np.sqrt((1.0 / n) + ((xfull - xbar) ** 2) / Sxx)
    lower_full = yhat_full - tcrit * se_mean_full
    upper_full = yhat_full + tcrit * se_mean_full

    return yhat_full, lower_full, upper_full, msk

def apply_yaxis_side(ax, side: str):
    side = side.lower().strip()
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

# ============================================================
# MAIN
# ============================================================
def main():
    ensure_outdir(OUT_DIR)

    tif_files = sorted(glob.glob(os.path.join(TIF_DIR, "*.tif")))
    if len(tif_files) == 0:
        raise FileNotFoundError(f"No .tif files found in: {TIF_DIR}")

    # Build year -> tif mapping
    year_to_tif = {}
    for fp in tif_files:
        yr = parse_year_from_filename(fp)
        year_to_tif[yr] = fp

    years = np.array(sorted(year_to_tif.keys()), dtype=int)

    # Raster CRS
    with rasterio.open(year_to_tif[years[0]]) as src0:
        raster_crs = src0.crs

    # Load shapes
    gdf_a = gpd.read_file(SHAPE_A)
    gdf_w = gpd.read_file(SHAPE_WATERSHED)

    gdf_a = to_raster_crs(gdf_a, raster_crs)
    gdf_w = to_raster_crs(gdf_w, raster_crs)

    geom_a = gdf_a.unary_union

    # dissolve watersheds by name3 and sort
    if WATERSHED_NAME_FIELD not in gdf_w.columns:
        raise ValueError(f"Field '{WATERSHED_NAME_FIELD}' not found in watershed shapefile attributes.")

    gdf_w2 = gdf_w[[WATERSHED_NAME_FIELD, "geometry"]].copy()
    gdf_w2[WATERSHED_NAME_FIELD] = gdf_w2[WATERSHED_NAME_FIELD].astype(str)

    dissolved = gdf_w2.dissolve(by=WATERSHED_NAME_FIELD, as_index=True).reset_index()
    dissolved = dissolved.sort_values(WATERSHED_NAME_FIELD).reset_index(drop=True)

    if len(dissolved) < 11:
        raise ValueError(f"Watershed dissolve produced only {len(dissolved)} regions, but panels b–l need 11.")

    dissolved_11 = dissolved.iloc[:11].copy()

    region_names = ["North America"] + dissolved_11[WATERSHED_NAME_FIELD].tolist()
    region_geoms  = [geom_a] + dissolved_11["geometry"].tolist()

    # Compute time series
    data = {"year": years}
    for name, geom in zip(region_names, region_geoms):
        vals = []
        for yr in years:
            v = zonal_mean_raster(year_to_tif[yr], geom)
            v = VALUE_MULTIPLIER * v
            vals.append(v)
        data[name] = vals

    df = pd.DataFrame(data)

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "timeseries_groundwater_a_to_l.csv")
    df.to_csv(csv_path, index=False)

    # Plot
    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H), sharex=True)
    axes = axes.flatten()
    panel_letters = list("abcdefghijkl")

    for i in range(12):
        ax = axes[i]
        apply_yaxis_side(ax, Y_AXIS_SIDE)

        series_name = region_names[i]
        y = df[series_name].values.astype(float)
        x = df["year"].values.astype(int)

        yhat, lo, hi, valid_mask = linear_trend_with_ci(x, y)

        # Confidence band & trend (only if we have them)
        if lo is not None:
            ax.fill_between(x, lo, hi, color=BAND_COLOR, alpha=BAND_ALPHA, label="95 % Confidence Band")
            ax.plot(x, yhat, color=COLOR_TREND, linestyle=LS_TREND, linewidth=LW_TREND, label="Linear Trend")

        # Observed
        ax.plot(x, y, color=COLOR_OBS, marker="o", markersize=MS_OBS, linewidth=LW_OBS, label="Observed")

        ax.text(0.01, 0.92, panel_letters[i], transform=ax.transAxes, fontsize=FS_PANEL)
        ax.tick_params(labelsize=FS_TICKS)

        # IMPORTANT: do NOT set per-axis ylabel (we do one global ylabel)

    # X label only on bottom row
    for c in range(NCOLS):
        axes[(NROWS - 1) * NCOLS + c].set_xlabel("Year", fontsize=FS_AXIS)

    # One global Y label (left side of the whole figure by default)
    # If you move y-axis ticks to right, you can also move the global y label to right by setting x=0.99
    if Y_AXIS_SIDE.lower() == "right":
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS, x=0.99)
    else:
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS, x=0.01)

    # Single legend at bottom center
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=FS_LEGEND, frameon=True)

    # Layout
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_png = os.path.join(OUT_DIR, "panel_a_to_l_groundwater.png")
    out_pdf = os.path.join(OUT_DIR, "panel_a_to_l_groundwater.pdf")
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("DONE")
    print("PNG:", out_png)
    print("PDF:", out_pdf)
    print("CSV:", csv_path)

if __name__ == "__main__":
    main()