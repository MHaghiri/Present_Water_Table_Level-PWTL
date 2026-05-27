#!/usr/bin/env python3
import os, re, glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

# ============================================================
# PATHS
# ============================================================
GREENLAND_SHP = "/home/mohammad/Desktop/N_America_shapefile/Greenland.shp"
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"
BOUNDARY_SHP  = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

PREC_DIR  = "/media/mohammad/My Book/0-2025/Monthly/pr/CMIP6/monthly/downscaled/N_America/1"
PREC_GLOB = os.path.join(PREC_DIR, "N_America_*_precipitation.tif")

# Your PET folder (kept your filename pattern)
PET_DIR  = "/media/mohammad/My Book/0-2025/Monthly/owe/CMIP6/monthly/downscaled/tiff/N_America/1"
PET_GLOB = os.path.join(PET_DIR, "N_America_*_open_water_evaporation.tif")

OUT_DIR  = "/home/mohammad/Desktop/1"
os.makedirs(OUT_DIR, exist_ok=True)

LW = 0.8

# ============================================================
# PERIOD (YYYYMM ints)
# ============================================================
START_YYYYMM = 200001
END_YYYYMM   = 202512

# ============================================================
# UNITS
# ============================================================
P_SCALE   = 1.0
PET_SCALE = 1.0   # if PET is mm/day and P mm/month -> 30.4375

# ============================================================
# VALIDITY
# ============================================================
MIN_VALID_FRAC = 0.60
PET_MIN_OK = 0.0
TREAT_ZERO_AS_NODATA_FOR_PET = False

# ============================================================
# CLASS BINS (dryness proxy AI* = sum(P)/sum(PET))
# Smaller AI => drier.
# To make MORE arid, we INCREASE thresholds.
# ============================================================
BIN_PRESET = "very_dry"   # <<<<<< change here

BINS_PRESETS = {
    # original (wet-friendly)
    "paper":       [0.0, 0.05, 0.2, 0.5, 0.65, np.inf],

    # already drier than paper
    "drier":       [0.0, 0.08, 0.30, 0.65, 0.85, np.inf],
    "much_drier":  [0.0, 0.10, 0.40, 0.80, 1.00, np.inf],

    # NEW: push a lot more area into Hyper/Arid/Semi-arid
    # Hyper-arid : AI < 0.20
    # Arid       : 0.20–0.60
    # Semi-arid  : 0.60–1.10
    # Sub-humid  : 1.10–1.50
    # Humid      : > 1.50
    "very_dry":    [0.0, 0.20, 0.50, 0.70, 1.50, np.inf],

    # Even stronger (use only if very_dry still not enough)
    # Hyper-arid : AI < 0.30
    # Arid       : 0.30–0.80
    # Semi-arid  : 0.80–1.40
    # Sub-humid  : 1.40–1.80
    # Humid      : > 1.80
    "extreme_dry": [0.0, 0.30, 0.80, 1.40, 1.80, np.inf],
}

AI_BINS = BINS_PRESETS[BIN_PRESET]
AI_LABELS = ["Hyper-arid", "Arid", "Semi-arid", "Sub-humid", "Humid"]

AI_COLORS = ["#b98a3c", "#d7b36a", "#efe3c8", "#a7c6b1", "#4f9b8e"]
cmap = ListedColormap(AI_COLORS, name="ai_ref_5")
cmap.set_bad(color="white", alpha=1.0)
norm = BoundaryNorm([0, 1, 2, 3, 4, 5], cmap.N)

# ============================================================
# HELPERS
# ============================================================
def parse_yyyymm(fp: str) -> tuple[int, int]:
    m = re.search(r"N_America_(\d{6})_", os.path.basename(fp))
    if not m:
        raise ValueError(f"Could not parse YYYYMM from filename: {fp}")
    yyyymm = m.group(1)
    return int(yyyymm[:4]), int(yyyymm[4:])

def yyyymm_int(key: tuple[int,int]) -> int:
    y, m = key
    return y * 100 + m

def list_sorted_files(glob_pattern: str) -> list[str]:
    files = glob.glob(glob_pattern)
    if not files:
        raise FileNotFoundError(f"No files found: {glob_pattern}")
    return sorted(files, key=lambda f: parse_yyyymm(f))

def classify_ai(ai2d: np.ndarray):
    cls = np.digitize(ai2d, AI_BINS) - 1
    cls[~np.isfinite(ai2d)] = -1
    return cls.astype("int16")

def add_overlays(ax, boundary, watersheds, greenland, lw=0.8):
    if len(greenland) > 0:
        greenland.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=lw, zorder=6)
    if len(watersheds) > 0:
        watersheds.boundary.plot(ax=ax, color="black", linewidth=lw, zorder=7)
    if len(boundary) > 0:
        boundary.boundary.plot(ax=ax, color="black", linewidth=lw, zorder=8)

def read_on_ref(fp, ref_crs, ref_transform, ref_width, ref_height, win, out_shape, inside_mask,
                treat_zero_as_nodata=False):
    with rasterio.open(fp) as src:
        nod = src.nodata
        with WarpedVRT(
            src,
            crs=ref_crs,
            transform=ref_transform,
            width=ref_width,
            height=ref_height,
            resampling=Resampling.bilinear
        ) as vrt:
            arr = vrt.read(1, window=win, out_shape=out_shape).astype("float32")

    if nod is not None:
        arr[arr == nod] = np.nan
    else:
        arr[arr == -9999] = np.nan
        arr[arr < -1e30] = np.nan

    if treat_zero_as_nodata:
        arr[arr == 0] = np.nan

    arr[~inside_mask] = np.nan
    return arr

# ============================================================
# MAIN
# ============================================================
def main():
    boundary   = gpd.read_file(BOUNDARY_SHP)
    watersheds = gpd.read_file(WATERSHED_SHP)
    greenland  = gpd.read_file(GREENLAND_SHP)

    P_files   = list_sorted_files(PREC_GLOB)
    PET_files = list_sorted_files(PET_GLOB)

    P_map   = {parse_yyyymm(f): f for f in P_files}
    PET_map = {parse_yyyymm(f): f for f in PET_files}

    P_keys = sorted(P_map.keys(), key=yyyymm_int)

    chosen = [k for k in P_keys if (START_YYYYMM <= yyyymm_int(k) <= END_YYYYMM) and (k in PET_map)]
    print(f"Months matched: {len(chosen)} | BIN_PRESET={BIN_PRESET} | AI_BINS={AI_BINS}")
    if not chosen:
        raise RuntimeError("No overlapping months in that period.")

    # Reference grid
    ref_fp = P_map[chosen[0]]
    with rasterio.open(ref_fp) as ref:
        ref_crs = ref.crs
        ref_transform = ref.transform
        ref_width = ref.width
        ref_height = ref.height

    if boundary.crs != ref_crs:   boundary = boundary.to_crs(ref_crs)
    if watersheds.crs != ref_crs: watersheds = watersheds.to_crs(ref_crs)
    if greenland.crs != ref_crs:  greenland = greenland.to_crs(ref_crs)

    # Clip watershed polygons to boundary so they do not exceed the boundary line
    boundary_union = boundary.unary_union
    watersheds = gpd.clip(watersheds, boundary_union)

    minx, miny, maxx, maxy = boundary.total_bounds
    win = from_bounds(minx, miny, maxx, maxy, transform=ref_transform)
    win = win.round_offsets().round_lengths()
    out_shape = (int(win.height), int(win.width))

    w_transform = rasterio.windows.transform(win, ref_transform)
    inside_mask = geometry_mask(boundary.geometry, out_shape=out_shape, transform=w_transform, invert=True)

    xmin = w_transform.c
    ymax = w_transform.f
    xmax = xmin + w_transform.a * out_shape[1]
    ymin = ymax + w_transform.e * out_shape[0]
    extent = (xmin, xmax, ymin, ymax)

    sumP   = np.zeros(out_shape, dtype="float64")
    sumPET = np.zeros(out_shape, dtype="float64")
    cnt    = np.zeros(out_shape, dtype="int32")

    for key in chosen:
        p = read_on_ref(P_map[key], ref_crs, ref_transform, ref_width, ref_height, win, out_shape, inside_mask) * float(P_SCALE)
        pet = read_on_ref(PET_map[key], ref_crs, ref_transform, ref_width, ref_height, win, out_shape, inside_mask,
                          treat_zero_as_nodata=TREAT_ZERO_AS_NODATA_FOR_PET) * float(PET_SCALE)

        m = np.isfinite(p) & np.isfinite(pet) & (pet >= PET_MIN_OK)
        sumP[m]   += p[m]
        sumPET[m] += pet[m]
        cnt[m]    += 1

    n_months = len(chosen)
    min_valid = int(np.ceil(MIN_VALID_FRAC * n_months))

    AI = np.full(out_shape, np.nan, dtype="float32")
    good = (cnt >= min_valid) & (sumPET > 0)
    AI[good] = (sumP[good] / sumPET[good]).astype("float32")

    cls = classify_ai(AI)
    show = cls.astype("float32")
    show[show < 0] = np.nan

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    ax.set_facecolor("white")

    ax.imshow(show, extent=extent, origin="upper", cmap=cmap, norm=norm, interpolation="nearest")
    add_overlays(ax, boundary, watersheds, greenland, lw=LW)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"", loc="left")

    handles = [
        plt.Line2D([0],[0], marker='s', linestyle='', markersize=10,
                   markerfacecolor=AI_COLORS[i], markeredgecolor='black', label=AI_LABELS[i])
        for i in range(5)
    ]
    ax.legend(handles=handles, ncols=5, loc="lower center",
              bbox_to_anchor=(0.5, -0.02), frameon=False)

    out_png = os.path.join(OUT_DIR, f"Fig6a_P_over_PET_bins_{BIN_PRESET}.png")
    out_pdf = os.path.join(OUT_DIR, f"Fig6a_P_over_PET_bins_{BIN_PRESET}.pdf")
    fig.savefig(out_png, dpi=1500, bbox_inches="tight", facecolor="white")
    fig.savefig(out_pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print("Saved:")
    print(" ", out_png)
    print(" ", out_pdf)

    if np.isfinite(AI).any():
        q = np.nanquantile(AI, [0.01, 0.1, 0.5, 0.9, 0.99])
        print("AI* quantiles (1%,10%,50%,90%,99%):", q)

if __name__ == "__main__":
    main()
