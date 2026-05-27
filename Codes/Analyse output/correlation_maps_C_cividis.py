#!/usr/bin/env python3
"""
FIG 2 + FIG 3 from MONTHLY precipitation + evaporation GeoTIFFs.

Fixes:
- Correct rolling buffers for k=12 via BUF_LEN=KMAX+1 (already working for you)
New:
- Better contrast for panel (d) by using a robust colorbar domain (vmin/vmax)
  computed from k=12 significant r values (percentile-based).
- Save FIG2 in 4 different colormaps:
    (1) YlGnBu (your current) + better domain
    (2) viridis + better domain
    (3) cividis + better domain
    (4) magma   + better domain
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_window, geometry_mask
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import PowerNorm
from scipy.stats import t as student_t

# ============================================================
# PATHS
# ============================================================
GREENLAND_SHP = "/home/mohammad/Desktop/N_America_shapefile/Greenland.shp"
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"
BOUNDARY_SHP  = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

PREC_DIR = "/media/mohammad/My Book/0-2025/Monthly/pr/CMIP6/monthly/downscaled/N_America/1"
EVAP_DIR = "/media/mohammad/My Book/0-2025/Monthly/evap/CMIP6/monthly/downscaled/N_America/1"
OUT_DIR  = "/home/mohammad/Desktop/1"
os.makedirs(OUT_DIR, exist_ok=True)

PREC_GLOB = os.path.join(PREC_DIR, "N_America_*_precipitation.tif")
EVAP_GLOB = os.path.join(EVAP_DIR, "N_America_*_evaporation.tif")

# Rolling windows (months)
TIMESCALES = [1, 3, 6, 12]
KMAX = max(TIMESCALES)
BUF_LEN = KMAX + 1  # critical for k==KMAX

# Seasons
SEASONS = {
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall":   [9, 10, 11],
    "Winter": [12, 1, 2],
}
SEASON_NAMES = list(SEASONS.keys())

# ============================================================
# OPTIONS
# ============================================================
START_YYYYMM = 200001
END_YYYYMM   = 202512

P_THRESHOLD = 0.05
RELAX_EMPTY_PANEL = True
P_THRESHOLD_RELAXED = 0.10
MIN_PANEL_COVERAGE = 0.005  # 0.5%

# Partial rolling windows
MIN_VALID_FRAC_BY_K = {1: 0.90, 3: 0.80, 6: 0.70, 12: 0.50}
SCALE_PARTIAL_WINDOWS = True

# --- Visualization: better contrast ---
# We'll compute vmin/vmax from k=12 distribution (percentiles), then apply to ALL panels.
AUTO_DOMAIN_FROM_K = 12
DOMAIN_USE_SIGNIFICANT_ONLY = True
DOMAIN_P_LOW  = 5     # 5th percentile
DOMAIN_P_HIGH = 99    # 99th percentile
DOMAIN_PAD = 0.00     # optional small padding (e.g., 0.01)

# Colormaps to save (4 outputs)
FIG2_CMAPS = [
    ("YlGnBu", "A_YlGnBu_domain"),   # your current colors
    ("viridis", "B_viridis_domain"),
    ("cividis", "C_cividis_domain"),
    ("magma", "D_magma_domain"),
]

# Contrast shaping inside the chosen domain:
# gamma > 1 emphasizes high-end differences (good when r clustered near 1)
GAMMA = 2.0
LW = 0.8

FILL_NONSIG_WITH_LOW = True
LOW_FILL_VALUE = np.nan  # set NaN so nonsignificant is transparent/white in map

# Boxplot
BOXPLOT_YLIM = (0.6, 1.0)
SEASON_COLORS = {
    "Spring": "#66c2a5",
    "Summer": "#fc8d62",
    "Fall":   "#8da0cb",
    "Winter": "#e78ac3",
}

# ============================================================
# HELPERS
# ============================================================
def parse_yyyymm(fp: str) -> tuple[int, int]:
    m = re.search(r"N_America_(\d{6})_", os.path.basename(fp))
    if not m:
        raise ValueError(f"Could not parse YYYYMM from filename: {fp}")
    yyyymm = m.group(1)
    return int(yyyymm[:4]), int(yyyymm[4:])

def yyyymm_int(key: tuple[int, int]) -> int:
    return key[0] * 100 + key[1]

def list_sorted_files(glob_pattern: str) -> list[str]:
    files = glob.glob(glob_pattern)
    files = sorted(files, key=lambda f: parse_yyyymm(f))
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {glob_pattern}")
    return files

def season_of_month(mo: int) -> str:
    for sname, smonths in SEASONS.items():
        if mo in smonths:
            return sname
    return "UNKNOWN"

def add_overlays(ax, boundary_gdf, watersheds_gdf, greenland_gdf):
    if greenland_gdf is not None and len(greenland_gdf) > 0:
        greenland_gdf.plot(ax=ax, facecolor="lightgray", edgecolor="black",
                           linewidth=LW, zorder=6)
    if watersheds_gdf is not None and len(watersheds_gdf) > 0:
        watersheds_gdf.boundary.plot(ax=ax, color="black", linewidth=LW, zorder=7)
    if boundary_gdf is not None and len(boundary_gdf) > 0:
        boundary_gdf.boundary.plot(ax=ax, color="black", linewidth=LW, zorder=8)

def compute_extent_from_transform(transform, width, height):
    xmin = transform.c
    ymax = transform.f
    xmax = xmin + transform.a * width
    ymin = ymax + transform.e * height
    return (xmin, xmax, ymin, ymax)

def read_window_masked(fp, window, inside_mask):
    with rasterio.open(fp) as src:
        arr = src.read(1, window=window).astype("float32")
        nod = src.nodata
        if nod is not None:
            arr[arr == nod] = np.nan
        else:
            arr[arr == -9999] = np.nan
    arr[~inside_mask] = np.nan
    return arr

def welford_update(mean, M2, n, x):
    m = np.isfinite(x)
    if not np.any(m):
        return mean, M2, n
    n_new = n[m] + 1.0
    delta = x[m] - mean[m]
    mean[m] += delta / n_new
    delta2 = x[m] - mean[m]
    M2[m] += delta * delta2
    n[m] = n_new
    return mean, M2, n

def safe_std_from_M2(M2, n):
    out = np.full_like(M2, np.nan, dtype="float64")
    m = n > 1
    out[m] = np.sqrt(M2[m] / (n[m] - 1.0))
    out[out == 0] = np.nan
    return out

def corr_from_sums(n, sx, sy, sxx, syy, sxy):
    n = n.astype("float64")
    denom_x = n * sxx - sx * sx
    denom_y = n * syy - sy * sy
    prod = np.maximum(denom_x * denom_y, 0.0)
    denom = np.sqrt(prod)
    r = (n * sxy - sx * sy) / denom
    r[denom <= 0] = np.nan
    return r.astype("float32")

def p_from_r_n(r, n):
    n = n.astype("float64")
    p = np.full_like(r, np.nan, dtype="float32")
    m = np.isfinite(r) & (n >= 3)
    if not np.any(m):
        return p
    rr = r[m].astype("float64")
    nn = n[m]
    denom = np.maximum(1e-15, 1.0 - rr * rr)
    tval = rr * np.sqrt((nn - 2.0) / denom)
    pv = 2.0 * (1.0 - student_t.cdf(np.abs(tval), df=nn - 2.0))
    p[m] = pv.astype("float32")
    return p

def robust_domain_from_values(vals, p_low=5, p_high=99, pad=0.0):
    vals = np.asarray(vals, dtype="float64")
    vals = vals[np.isfinite(vals)]
    if vals.size < 100:
        return 0.0, 1.0
    vmin = float(np.percentile(vals, p_low))
    vmax = float(np.percentile(vals, p_high))
    vmin = max(0.0, vmin - pad)
    vmax = min(1.0, vmax + pad)
    if vmax <= vmin:
        return 0.0, 1.0
    return vmin, vmax

# ============================================================
# MAIN
# ============================================================
def main():
    boundary   = gpd.read_file(BOUNDARY_SHP)
    watersheds = gpd.read_file(WATERSHED_SHP)
    greenland  = gpd.read_file(GREENLAND_SHP)

    prec_files = list_sorted_files(PREC_GLOB)
    evap_files = list_sorted_files(EVAP_GLOB)

    prec_map = {parse_yyyymm(f): f for f in prec_files}
    evap_map = {parse_yyyymm(f): f for f in evap_files}

    common = sorted(set(prec_map.keys()).intersection(set(evap_map.keys())), key=yyyymm_int)
    common = [k for k in common if START_YYYYMM <= yyyymm_int(k) <= END_YYYYMM]
    if not common:
        raise RuntimeError("No matching YYYYMM after filter.")

    dates = pd.DatetimeIndex([pd.Timestamp(y, m, 15) for (y, m) in common])
    months = dates.month.values
    seasons = np.array([season_of_month(mo) for mo in months], dtype=object)

    print("Matched months:", len(common), "from", common[0], "to", common[-1])
    print("Using BUF_LEN =", BUF_LEN)

    # ---- fixed window + inside mask ----
    first_fp = prec_map[common[0]]
    with rasterio.open(first_fp) as src0:
        rcrs = src0.crs
        if boundary.crs != rcrs:   boundary   = boundary.to_crs(rcrs)
        if watersheds.crs != rcrs: watersheds = watersheds.to_crs(rcrs)
        if greenland.crs != rcrs:  greenland  = greenland.to_crs(rcrs)

        window = geometry_window(src0, boundary.geometry, pad_x=0, pad_y=0)
        w_transform = src0.window_transform(window)
        H = int(window.height)
        W = int(window.width)

        inside_mask = geometry_mask(boundary.geometry, out_shape=(H, W),
                                    transform=w_transform, invert=True)
        extent = compute_extent_from_transform(w_transform, W, H)

    min_valid = {k: max(1, int(np.ceil(k * MIN_VALID_FRAC_BY_K[k]))) for k in TIMESCALES}
    print("min_valid months per k:", min_valid)

    # ========================================================
    # PASS 1: mean/std of rolling sums
    # ========================================================
    print("\nPASS 1/2: computing mean/std of rolling sums...")

    P_buf   = [None] * BUF_LEN
    WB_buf  = [None] * BUF_LEN
    mP_buf  = [None] * BUF_LEN
    mWB_buf = [None] * BUF_LEN

    rollP  = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    rollNP = {k: np.zeros((H, W), dtype="int16")  for k in TIMESCALES}
    rollWB = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    rollNW = {k: np.zeros((H, W), dtype="int16")  for k in TIMESCALES}

    meanP = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    M2P   = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    nP    = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}

    meanW = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    M2W   = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}
    nW    = {k: np.zeros((H, W), dtype="float64") for k in TIMESCALES}

    buf_idx = 0
    t_idx = 0

    for key in common:
        p = read_window_masked(prec_map[key], window, inside_mask)
        e = read_window_masked(evap_map[key], window, inside_mask)
        wb = p - e

        validP  = np.isfinite(p)
        validWB = np.isfinite(p) & np.isfinite(e)

        P_buf[buf_idx]   = p
        WB_buf[buf_idx]  = wb
        mP_buf[buf_idx]  = validP
        mWB_buf[buf_idx] = validWB

        for k in TIMESCALES:
            rollP[k][validP]   += p[validP]
            rollNP[k][validP]  += 1
            rollWB[k][validWB] += wb[validWB]
            rollNW[k][validWB] += 1

            if t_idx >= k:
                old_idx = (buf_idx - k) % BUF_LEN
                oldP = P_buf[old_idx]
                oldWB = WB_buf[old_idx]
                oldmP = mP_buf[old_idx]
                oldmW = mWB_buf[old_idx]

                rollP[k][oldmP]   -= oldP[oldmP]
                rollNP[k][oldmP]  -= 1
                rollWB[k][oldmW]  -= oldWB[oldmW]
                rollNW[k][oldmW]  -= 1

            if t_idx >= k - 1:
                okP = rollNP[k] >= min_valid[k]
                okW = rollNW[k] >= min_valid[k]

                if SCALE_PARTIAL_WINDOWS:
                    scaleP = np.where(okP, (k / np.maximum(1, rollNP[k])).astype("float64"), np.nan)
                    scaleW = np.where(okW, (k / np.maximum(1, rollNW[k])).astype("float64"), np.nan)
                    xP = (rollP[k]  * scaleP).astype("float32")
                    xW = (rollWB[k] * scaleW).astype("float32")
                else:
                    xP = np.where(okP, rollP[k],  np.nan).astype("float32")
                    xW = np.where(okW, rollWB[k], np.nan).astype("float32")

                meanP[k], M2P[k], nP[k] = welford_update(meanP[k], M2P[k], nP[k], xP)
                meanW[k], M2W[k], nW[k] = welford_update(meanW[k], M2W[k], nW[k], xW)

        buf_idx = (buf_idx + 1) % BUF_LEN
        t_idx += 1
        if t_idx % 24 == 0:
            print(f"  processed {t_idx}/{len(common)} months...")

    stdP = {k: safe_std_from_M2(M2P[k], nP[k]).astype("float32") for k in TIMESCALES}
    stdW = {k: safe_std_from_M2(M2W[k], nW[k]).astype("float32") for k in TIMESCALES}
    meanP = {k: meanP[k].astype("float32") for k in TIMESCALES}
    meanW = {k: meanW[k].astype("float32") for k in TIMESCALES}

    for k in TIMESCALES:
        print(f"k={k}: finite stdP={np.isfinite(stdP[k]).sum()}  finite stdW={np.isfinite(stdW[k]).sum()}")

    # ========================================================
    # PASS 2: correlations
    # ========================================================
    print("\nPASS 2/2: computing correlations...")
    r_all_maps = {}
    p_all_maps = {}
    fig3_data = {k: {s: [] for s in SEASON_NAMES} for k in TIMESCALES}

    for k in TIMESCALES:
        print(f"\nTimescale k={k} months...")

        rollP_k  = np.zeros((H, W), dtype="float64")
        rollNP_k = np.zeros((H, W), dtype="int16")
        rollW_k  = np.zeros((H, W), dtype="float64")
        rollNW_k = np.zeros((H, W), dtype="int16")

        P_buf   = [None] * BUF_LEN
        WB_buf  = [None] * BUF_LEN
        mP_buf  = [None] * BUF_LEN
        mW_buf  = [None] * BUF_LEN

        buf_idx = 0
        t_idx = 0

        groups = ["ALL"] + SEASON_NAMES
        n   = {g: np.zeros((H, W), dtype="uint16") for g in groups}
        sx  = {g: np.zeros((H, W), dtype="float32") for g in groups}
        sy  = {g: np.zeros((H, W), dtype="float32") for g in groups}
        sxx = {g: np.zeros((H, W), dtype="float32") for g in groups}
        syy = {g: np.zeros((H, W), dtype="float32") for g in groups}
        sxy = {g: np.zeros((H, W), dtype="float32") for g in groups}

        for (key, sname) in zip(common, seasons):
            p = read_window_masked(prec_map[key], window, inside_mask)
            e = read_window_masked(evap_map[key], window, inside_mask)
            wb = p - e

            validP  = np.isfinite(p)
            validWB = np.isfinite(p) & np.isfinite(e)

            P_buf[buf_idx]  = p
            WB_buf[buf_idx] = wb
            mP_buf[buf_idx] = validP
            mW_buf[buf_idx] = validWB

            rollP_k[validP]   += p[validP]
            rollNP_k[validP]  += 1
            rollW_k[validWB]  += wb[validWB]
            rollNW_k[validWB] += 1

            if t_idx >= k:
                old_idx = (buf_idx - k) % BUF_LEN
                oldP = P_buf[old_idx]
                oldW = WB_buf[old_idx]
                oldmP = mP_buf[old_idx]
                oldmW = mW_buf[old_idx]

                rollP_k[oldmP]   -= oldP[oldmP]
                rollNP_k[oldmP]  -= 1
                rollW_k[oldmW]   -= oldW[oldmW]
                rollNW_k[oldmW]  -= 1

            if t_idx >= k - 1:
                okP = rollNP_k >= min_valid[k]
                okW = rollNW_k >= min_valid[k]

                if SCALE_PARTIAL_WINDOWS:
                    scaleP = np.where(okP, (k / np.maximum(1, rollNP_k)).astype("float64"), np.nan)
                    scaleW = np.where(okW, (k / np.maximum(1, rollNW_k)).astype("float64"), np.nan)
                    Pk  = (rollP_k * scaleP).astype("float32")
                    WBk = (rollW_k * scaleW).astype("float32")
                else:
                    Pk  = np.where(okP, rollP_k, np.nan).astype("float32")
                    WBk = np.where(okW, rollW_k, np.nan).astype("float32")

                x = (Pk  - meanP[k]) / stdP[k]
                y = (WBk - meanW[k]) / stdW[k]

                m = np.isfinite(x) & np.isfinite(y)
                if np.any(m):
                    n["ALL"][m]   += 1
                    sx["ALL"][m]  += x[m]
                    sy["ALL"][m]  += y[m]
                    sxx["ALL"][m] += x[m] * x[m]
                    syy["ALL"][m] += y[m] * y[m]
                    sxy["ALL"][m] += x[m] * y[m]

                    if sname in SEASON_NAMES:
                        n[sname][m]   += 1
                        sx[sname][m]  += x[m]
                        sy[sname][m]  += y[m]
                        sxx[sname][m] += x[m] * x[m]
                        syy[sname][m] += y[m] * y[m]
                        sxy[sname][m] += x[m] * y[m]

            buf_idx = (buf_idx + 1) % BUF_LEN
            t_idx += 1
            if t_idx % 24 == 0:
                print(f"  k={k}: processed {t_idx}/{len(common)} months...")

        r_all = corr_from_sums(n["ALL"], sx["ALL"], sy["ALL"], sxx["ALL"], syy["ALL"], sxy["ALL"])
        p_all = p_from_r_n(r_all, n["ALL"])
        r_all_maps[k] = r_all
        p_all_maps[k] = p_all

        print(f"  k={k}: finite r pixels={np.isfinite(r_all).sum()} | pixels with n>=3 = {(n['ALL']>=3).sum()}")

        for sn in SEASON_NAMES:
            r_s = corr_from_sums(n[sn], sx[sn], sy[sn], sxx[sn], syy[sn], sxy[sn])
            p_s = p_from_r_n(r_s, n[sn])
            vals = r_s[(p_s < P_THRESHOLD) & np.isfinite(r_s)]
            if vals.size > 0:
                fig3_data[k][sn].extend(vals.astype("float32").tolist())

    # ========================================================
    # Build plot maps + store significance masks
    # ========================================================
    r_plot_maps = {}
    used_thresh = {}
    sig_masks = {}

    for k in TIMESCALES:
        r_all = r_all_maps[k]
        p_all = p_all_maps[k]

        mask_sig = (p_all < P_THRESHOLD) & np.isfinite(r_all)
        denom = float(np.count_nonzero(np.isfinite(r_all)) + 1e-9)
        coverage = float(np.count_nonzero(mask_sig)) / denom

        thr = P_THRESHOLD
        if RELAX_EMPTY_PANEL and (coverage < MIN_PANEL_COVERAGE):
            thr = P_THRESHOLD_RELAXED
            mask_sig = (p_all < thr) & np.isfinite(r_all)

        used_thresh[k] = thr
        sig_masks[k] = mask_sig

        # For plotting: show ONLY significant r (others transparent)
        out = np.full_like(r_all, np.nan, dtype="float32")
        out[mask_sig] = r_all[mask_sig]
        r_plot_maps[k] = out

        print(f"k={k}: used p<{thr:.2f} | kept pixels={np.isfinite(out).sum()}")

    # ========================================================
    # Auto better domain (vmin/vmax) from k=12 distribution
    # ========================================================
    kdom = AUTO_DOMAIN_FROM_K
    if DOMAIN_USE_SIGNIFICANT_ONLY:
        vals = r_all_maps[kdom][sig_masks[kdom] & np.isfinite(r_all_maps[kdom])]
    else:
        vals = r_all_maps[kdom][np.isfinite(r_all_maps[kdom])]

    vmin, vmax = robust_domain_from_values(vals, DOMAIN_P_LOW, DOMAIN_P_HIGH, DOMAIN_PAD)
    print(f"\nAuto colorbar domain from k={kdom}: vmin={vmin:.3f}, vmax={vmax:.3f} (percentiles {DOMAIN_P_LOW}-{DOMAIN_P_HIGH})")

    # ========================================================
    # FIG 2: save 4 versions (different colormaps, same domain)
    # ========================================================
    ks = [1, 3, 6, 12]
    panel_text = {1: "30 days\n1 month", 3: "90 days\n3 months", 6: "180 days\n6 months", 12: "360 days\n12 months"}

    for cmap_name, tag in FIG2_CMAPS:
        fig, axes = plt.subplots(2, 2, figsize=(12, 7))
        cmap = plt.get_cmap(cmap_name).copy()
        cmap.set_bad(color="white", alpha=0.0)  # outside-domain / nonsig transparent

        # PowerNorm within the NEW domain
        norm = PowerNorm(gamma=GAMMA, vmin=vmin, vmax=vmax)

        im = None
        for idx, (ax, k) in enumerate(zip(axes.ravel(), ks)):
            im = ax.imshow(r_plot_maps[k], extent=extent, origin="upper", cmap=cmap, norm=norm)
            add_overlays(ax, boundary, watersheds, greenland)

            ax.text(0.02, 0.98, f"({chr(97+idx)})", transform=ax.transAxes,
                    ha="left", va="top", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="none", alpha=0.85))

            ax.text(0.03, 0.06, panel_text[k], transform=ax.transAxes,
                    ha="left", va="bottom", fontsize=16,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85))

            ax.set_xticks([])
            ax.set_yticks([])

        fig.subplots_adjust(bottom=0.12, right=0.98, left=0.02, top=0.98, wspace=0.06, hspace=0.08)
        cax = fig.add_axes([0.18, 0.05, 0.64, 0.03])
        cbar = fig.colorbar(im, cax=cax, orientation="horizontal")

        # ticks: include the domain endpoints + some helpful anchors
        ticks = sorted(set([
            round(vmin, 2),
            0.50, 0.60, 0.70, 0.80, 0.90,
            round(vmax, 2)
        ]))
        ticks = [t for t in ticks if (vmin <= t <= vmax)]
        cbar.set_ticks(ticks)

        cbar.set_label(
            f"Correlation coefficient (r)  | significant pixels "
            f"(p<{P_THRESHOLD}; auto-relax to p<{P_THRESHOLD_RELAXED} if needed) | "
            f"domain [{vmin:.2f}, {vmax:.2f}]"
        )

        fig2_png = os.path.join(OUT_DIR, f"Fig2_correlation_maps_{tag}.png")
        fig2_pdf = os.path.join(OUT_DIR, f"Fig2_correlation_maps_{tag}.pdf")
        fig.savefig(fig2_png, dpi=1500, bbox_inches="tight")
        fig.savefig(fig2_pdf, bbox_inches="tight")
        plt.close(fig)

        print("Saved:", fig2_png)
        print("Saved:", fig2_pdf)

    # ========================================================
    # FIG 3 (legend inside bottom of panel c)
    # ========================================================
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    panel_labels = {1: "1 month", 3: "3 months", 6: "6 months", 12: "12 months"}

    legend_handles = [Patch(facecolor=SEASON_COLORS[s], edgecolor="black", label=s, alpha=0.70)
                      for s in SEASON_NAMES]

    for idx, (ax, k) in enumerate(zip(axes.ravel(), ks)):
        data = [fig3_data[k][s] for s in SEASON_NAMES]
        bp = ax.boxplot(data, labels=SEASON_NAMES, showfliers=False, patch_artist=True)

        for box, s in zip(bp["boxes"], SEASON_NAMES):
            box.set_facecolor(SEASON_COLORS[s])
            box.set_alpha(0.70)

        ax.set_ylabel("Correlation coefficient (r)")
        ax.set_title(f"({chr(97+idx)})  {panel_labels[k]}", loc="left")
        ax.grid(True, axis="y", alpha=0.3)
        ax.set_ylim(*BOXPLOT_YLIM)

        if idx == 2:  # panel (c)
            ax.legend(handles=legend_handles, loc="lower center",
                      bbox_to_anchor=(0.5, 0.02), ncol=4, frameon=False)

    fig3_png = os.path.join(OUT_DIR, "Fig3_seasonal_boxplots_correlation.png")
    fig3_pdf = os.path.join(OUT_DIR, "Fig3_seasonal_boxplots_correlation.pdf")
    fig.savefig(fig3_png, dpi=1500, bbox_inches="tight")
    fig.savefig(fig3_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nDONE ✅")
    print("Fig3:", fig3_png)
    print("Fig3:", fig3_pdf)
    print("\nPanel thresholds used:")
    for k in ks:
        print(f"  k={k}: p<{used_thresh[k]:.2f}")

if __name__ == "__main__":
    main()
