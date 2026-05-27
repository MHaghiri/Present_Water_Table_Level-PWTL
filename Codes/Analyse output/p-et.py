#!/usr/bin/env python3
"""
12-panel P-ET figure (a–l) — 4 rows × 3 columns (like your layout)

Row1: a  b  c
Row2: d  e  f
Row3: g  h  i
Row4: j  k  l

(a) P-ET map for PANEL_A_MONTH in meters
(b–l) DIFFERENCE of P-ET for specified time-pairs (NEW - OLD) in cm
      + shared diverging colorbar with white around ~0

Shapefiles:
- NA boundary: edge only (no fill)
- Watersheds: edge only (no fill)
- Greenland: light gray fill

Outputs:
- PNG + PDF saved to OUT_DIR
"""

import os
import numpy as np
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt

from rasterio.plot import plotting_extent
from rasterio.features import geometry_mask

from matplotlib.colors import ListedColormap, BoundaryNorm, Normalize, TwoSlopeNorm
from matplotlib.cm import get_cmap, ScalarMappable
from matplotlib.gridspec import GridSpec


# ============================================================
# OUTPUT
# ============================================================
OUT_DIR = "/home/mohammad/Desktop/1/14"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_BASE = "P_minus_ET_12panel"


# ============================================================
# SHAPEFILES
# ============================================================
NA_BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_boundery_without_greenland.shp"
GREENLAND_SHP   = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/Greenland.shp"
WATERSHED_SHP   = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_level2_watershed_without_greenland.shp"


# ============================================================
# INPUT PATHS
# ============================================================
PRECIP_DIR = "/media/mohammad/My Book1/0-2025/Monthly/pr/CMIP6/monthly/downscaled/N_America/1"
EVAP_DIR   = "/media/mohammad/My Book1/0-2025/Monthly/evap/CMIP6/monthly/downscaled/N_America/1"

# Month used for panel (a)
PANEL_A_MONTH = "202501"


# ============================================================
# SAME TIME-PAIRS AS YOUR OLD CODE
# (b–l): old_YYYYMM -> new_YYYYMM
# ============================================================
PANEL_DIFF_MONTHS = {
    "b": ("202501", "202502"),
    "c": ("202401", "202403"),
    "d": ("202401", "202404"),
    "e": ("202401", "202405"),
    "f": ("202401", "202406"),
    "g": ("202401", "202407"),
    "h": ("202401", "202408"),
    "i": ("202401", "202409"),
    "j": ("202401", "202410"),
    "k": ("202401", "202411"),
    "l": ("202501", "202511"),
}


# ============================================================
# FILE NAME PATTERNS
# ============================================================
# Precip is already known:
PRECIP_SUFFIX = "precipitation"

# Evap filenames can vary; we try these suffixes in order
EVAP_SUFFIX_CANDIDATES = [
    "evaporation",
    "evap",
    "evspsbl",
    "et",
    "ET",
    "Evaporation",
    "EVAP",
]

def precip_file(yyyymm: str) -> str:
    return os.path.join(PRECIP_DIR, f"N_America_{yyyymm}_{PRECIP_SUFFIX}.tif")

def evap_file(yyyymm: str) -> str:
    """Try multiple common evap suffixes; return the first existing match."""
    tried = []
    for suf in EVAP_SUFFIX_CANDIDATES:
        p = os.path.join(EVAP_DIR, f"N_America_{yyyymm}_{suf}.tif")
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not find an evap file for month "
        f"{yyyymm}. Tried:\n  " + "\n  ".join(tried)
    )


# ============================================================
# LAYOUT CONTROLS (4x3)
# ============================================================
NROWS, NCOLS = 4, 3
WSPACE = 0.02
HSPACE = 0.02

COL_WIDTHS  = [1.0, 1.0, 1.0]
ROW_HEIGHTS = [1.0, 1.0, 1.0, 1.0]

# auto figure size from a "base panel size"
BASE_PANEL_W_IN = 4.0
BASE_PANEL_H_IN = 2.2
EXTRA_W_IN = 1.2
EXTRA_H_IN = 0.8

# Panel (a) target size in inches (ONLY panel a)
PANEL_A_SIZE_IN = (3.25, 2.80)  # (width_in, height_in)  <-- CHANGE THIS

# Borders
PANEL_BORDER_ON = True
PANEL_BORDER_LW = 0.8
PANEL_BORDER_COLOR = "black"

# outlines
WATERSHED_LW = 0.5
NA_LW = 0.6
GREENLAND_LW = 0.6

# letters
PANEL_LETTER_FONTSIZE = 13

# save
PNG_DPI = 1500
PDF_DPI = 300


# ============================================================
# PANEL (a) P-ET SETTINGS (m)
# ============================================================
# P-ET can be negative/positive -> diverging is usually better
A_VMIN = -2.0
A_VMAX =  2.0
A_CENTER = 0.0
A_CMAP_NAME = "BrBG"

# inset colorbar inside panel a
USE_INSET_CBAR_A = True
CBAR_A_INSET_RECT = [0.85, 0.025, 0.02, 0.65]  # [x0,y0,w,h] in AXES coords

A_CBAR_TICKS = [-2, -1, 0, 1, 2]
A_CBAR_LABEL_TEXT = "P − ET (m)"
A_CBAR_LABEL_FONTSIZE = 8
A_CBAR_LABEL_XY = (6.6, 0.50)  # in colorbar-axes coordinates
A_CBAR_TICK_LABELSIZE = 8


# ============================================================
# (b–l) DIFF SETTINGS (cm) for Δ(P-ET)
# ============================================================
# Discrete ticks (shown) + white around zero
MAP_TICKS = [-100, -50, -10, 0, 10, 50, 100]
WHITE_LOW = -2
WHITE_HIGH = 2

# Shared cbar (b–l) manual position (tweak if needed)
USE_MANUAL_CBAR_DIFF = True
DIFF_CBAR_RECT = [0.91, 0.2, 0.015, 0.5]   # [left,bottom,width,height] figure coords
DIFF_CBAR_ORIENTATION = "vertical"
CBAR_TICKS = [-100, -50, -10, 0, 10, 50, 100]
DIFF_CBAR_LABEL = "Δ(P − ET) (cm)"


# ============================================================
# HELPERS
# ============================================================
def _check_exists(p):
    if not os.path.exists(p):
        raise FileNotFoundError(f"File not found: {p}")

def _read_raster(tif_path):
    with rasterio.open(tif_path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            arr = np.ma.masked_equal(arr, nodata)
        else:
            arr = np.ma.masked_invalid(arr)
        extent = plotting_extent(src)
        meta = {
            "crs": src.crs,
            "transform": src.transform,
            "height": src.height,
            "width": src.width,
            "nodata": nodata
        }
    return arr, extent, meta

def _ensure_same_crs(gdf, target_crs):
    if gdf.crs is None or target_crs is None:
        return gdf
    if str(gdf.crs) != str(target_crs):
        return gdf.to_crs(target_crs)
    return gdf

def _make_outside_mask(poly_gdf, raster_meta):
    shapes = [geom for geom in poly_gdf.geometry if geom is not None and not geom.is_empty]
    if len(shapes) == 0:
        raise ValueError("Boundary shapefile has no valid geometries.")
    outside = geometry_mask(
        geometries=shapes,
        out_shape=(raster_meta["height"], raster_meta["width"]),
        transform=raster_meta["transform"],
        invert=False,
        all_touched=False
    )
    return outside

def _style_panel_border(ax):
    if not PANEL_BORDER_ON:
        ax.set_axis_off()
        return
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(PANEL_BORDER_LW)
        spine.set_edgecolor(PANEL_BORDER_COLOR)

def _panel_letter(ax, letter):
    ax.text(
        0.02, 0.02, letter,
        transform=ax.transAxes,
        fontsize=PANEL_LETTER_FONTSIZE,
        fontweight="bold",
        ha="left", va="bottom",
        zorder=30
    )

def _set_axes_size_in_inches(ax, w_in, h_in, fig):
    if w_in <= 0 or h_in <= 0:
        raise ValueError("Panel size in inches must be > 0.")
    fig_w_in, fig_h_in = fig.get_size_inches()
    bb = ax.get_position()
    cx = 0.5 * (bb.x0 + bb.x1)
    cy = 0.5 * (bb.y0 + bb.y1)
    w_frac = w_in / fig_w_in
    h_frac = h_in / fig_h_in
    ax.set_position([cx - w_frac/2, cy - h_frac/2, w_frac, h_frac])


# ============================================================
# (b–l) COLORMAP BUILDERS (RdYlBu + white at 0)
# ============================================================
def build_diff_cmap_norm():
    neg = sorted([float(t) for t in MAP_TICKS if t < 0])
    pos = sorted([float(t) for t in MAP_TICKS if t > 0])

    # Force a white band near 0
    levels = np.array(neg + [WHITE_LOW, WHITE_HIGH] + pos, dtype=float)

    cont = get_cmap("RdYlBu")
    mids = (levels[:-1] + levels[1:]) / 2.0
    t = (mids - levels[0]) / (levels[-1] - levels[0])
    cols = [cont(v) for v in t]

    # Force the bin that straddles 0 to white
    zidx = np.where((levels[:-1] < 0) & (levels[1:] > 0))[0]
    if zidx.size == 1:
        cols[int(zidx[0])] = (1, 1, 1, 1)

    cmap = ListedColormap(cols, name="rdylbu_white_zero_DIFF")
    cmap.set_under(cols[0])
    cmap.set_over(cols[-1])
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm, levels

def build_fake_colorbar_mappable(levels):
    cmap, norm, _ = build_diff_cmap_norm()
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm


# ============================================================
# CORE CALC: P-ET (meters)
# ============================================================
def compute_p_minus_et(yyyymm: str):
    """Return (P-ET array, extent, meta) in meters for a given month."""
    p_tif = precip_file(yyyymm)
    e_tif = evap_file(yyyymm)

    _check_exists(p_tif)
    _check_exists(e_tif)

    p_arr, p_extent, p_meta = _read_raster(p_tif)
    e_arr, e_extent, e_meta = _read_raster(e_tif)

    # Require same grid for pixel-wise P-ET
    if (p_meta["height"], p_meta["width"]) != (e_meta["height"], e_meta["width"]) or p_meta["transform"] != e_meta["transform"]:
        raise ValueError(
            f"Grid mismatch for month {yyyymm}:\n"
            f"  P : {p_tif}\n"
            f"  ET: {e_tif}\n"
            "They must have same shape and transform."
        )

    pet = (p_arr - e_arr).astype("float32")
    return pet, p_extent, p_meta


# ============================================================
# MAIN
# ============================================================
def main():
    # check shapefiles
    _check_exists(NA_BOUNDARY_SHP)
    _check_exists(GREENLAND_SHP)
    _check_exists(WATERSHED_SHP)

    # Check panel-a inputs
    _check_exists(precip_file(PANEL_A_MONTH))
    _ = evap_file(PANEL_A_MONTH)  # will raise if not found

    # Check all diff months exist (both P and ET for both months)
    for L, (old_m, new_m) in PANEL_DIFF_MONTHS.items():
        _check_exists(precip_file(old_m))
        _check_exists(precip_file(new_m))
        _ = evap_file(old_m)
        _ = evap_file(new_m)

    # Compute panel A P-ET first to get CRS target
    a_pet, a_extent, a_meta = compute_p_minus_et(PANEL_A_MONTH)
    target_crs = a_meta["crs"]

    # Load shapefiles
    na_gdf = _ensure_same_crs(gpd.read_file(NA_BOUNDARY_SHP), target_crs)
    gr_gdf = _ensure_same_crs(gpd.read_file(GREENLAND_SHP), target_crs)
    ws_gdf = _ensure_same_crs(gpd.read_file(WATERSHED_SHP), target_crs)

    def overlay_shapes(ax):
        na_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=NA_LW, zorder=5)
        gr_gdf.plot(ax=ax, edgecolor="black", facecolor="lightgray", linewidth=GREENLAND_LW, zorder=6)
        ws_gdf.plot(ax=ax, edgecolor="black", facecolor="none", linewidth=WATERSHED_LW, zorder=7)

    def finalize_ax(ax):
        ax.set_aspect("equal")
        _style_panel_border(ax)

    # Panel A colormap (diverging around 0)
    a_cmap = get_cmap(A_CMAP_NAME)
    a_norm = TwoSlopeNorm(vmin=A_VMIN, vcenter=A_CENTER, vmax=A_VMAX)

    # Diff colormap for (b–l)
    diff_cmap, diff_norm, diff_levels = build_diff_cmap_norm()
    sm_cbar = build_fake_colorbar_mappable(diff_levels)

    # Auto figure size
    fig_w = NCOLS * BASE_PANEL_W_IN + EXTRA_W_IN
    fig_h = NROWS * BASE_PANEL_H_IN + EXTRA_H_IN
    fig = plt.figure(figsize=(fig_w, fig_h))

    gs = GridSpec(
        nrows=NROWS, ncols=NCOLS, figure=fig,
        width_ratios=COL_WIDTHS, height_ratios=ROW_HEIGHTS,
        wspace=WSPACE, hspace=HSPACE
    )

    # Positions for 4x3
    positions = {
        "a": (0, 0), "b": (0, 1), "c": (0, 2),
        "d": (1, 0), "e": (1, 1), "f": (1, 2),
        "g": (2, 0), "h": (2, 1), "i": (2, 2),
        "j": (3, 0), "k": (3, 1), "l": (3, 2),
    }
    axes = {k: fig.add_subplot(gs[r, c]) for k, (r, c) in positions.items()}

    # ---- Panel (a): P-ET (m) ----
    outside_a = _make_outside_mask(na_gdf, a_meta)

    a_filled = np.array(a_pet, dtype="float32")
    a_filled[outside_a] = np.nan
    if np.ma.isMaskedArray(a_pet):
        a_filled[a_pet.mask] = np.nan

    ax0 = axes["a"]
    im_a = ax0.imshow(a_filled, cmap=a_cmap, norm=a_norm, extent=a_extent, zorder=1)
    overlay_shapes(ax0)
    finalize_ax(ax0)
    _panel_letter(ax0, "a")

    # Panel A inset colorbar
    cax_a = ax0.inset_axes(CBAR_A_INSET_RECT)
    cbar_a = fig.colorbar(im_a, cax=cax_a, orientation="vertical", ticks=A_CBAR_TICKS)
    cbar_a.ax.tick_params(labelsize=A_CBAR_TICK_LABELSIZE)
    cbar_a.set_label(A_CBAR_LABEL_TEXT, rotation=270, labelpad=10, fontsize=A_CBAR_LABEL_FONTSIZE)
    cbar_a.ax.yaxis.set_label_coords(A_CBAR_LABEL_XY[0], A_CBAR_LABEL_XY[1])

    # ---- Panels (b–l): Δ(P-ET) in cm ----
    for L, (old_m, new_m) in PANEL_DIFF_MONTHS.items():
        ax = axes[L]

        pet_old, extent_old, meta_old = compute_p_minus_et(old_m)
        pet_new, extent_new, meta_new = compute_p_minus_et(new_m)

        # Enforce same grid old vs new
        if (meta_old["height"], meta_old["width"]) != (meta_new["height"], meta_new["width"]) or meta_old["transform"] != meta_new["transform"]:
            raise ValueError(
                f"Grid mismatch in panel {L} between months:\n"
                f"  OLD month: {old_m}\n"
                f"  NEW month: {new_m}\n"
                "They must have same shape and transform."
            )

        outside = _make_outside_mask(na_gdf, meta_old)

        diff_cm = (pet_new - pet_old).astype("float32") * 100.0  # meters -> cm

        diff_filled = np.array(diff_cm, dtype="float32")
        diff_filled[outside] = np.nan
        if np.ma.isMaskedArray(pet_old):
            diff_filled[pet_old.mask] = np.nan
        if np.ma.isMaskedArray(pet_new):
            diff_filled[pet_new.mask] = np.nan

        ax.imshow(diff_filled, cmap=diff_cmap, norm=diff_norm, extent=extent_old, zorder=1)
        overlay_shapes(ax)
        finalize_ax(ax)
        _panel_letter(ax, L)

    # Force Panel (a) size in inches
    _set_axes_size_in_inches(axes["a"], PANEL_A_SIZE_IN[0], PANEL_A_SIZE_IN[1], fig)

    # ---- Shared colorbar for (b–l) ----
    if USE_MANUAL_CBAR_DIFF:
        cax_d = fig.add_axes(DIFF_CBAR_RECT)
        cbar_d = fig.colorbar(
            sm_cbar,
            cax=cax_d,
            orientation=DIFF_CBAR_ORIENTATION,
            extend="both",
            ticks=CBAR_TICKS,
            boundaries=diff_levels,
            spacing="uniform"
        )
    else:
        cbar_d = fig.colorbar(
            sm_cbar,
            ax=[axes[x] for x in "bcdefghijkl"],
            location="bottom",
            orientation="horizontal",
            fraction=0.05,
            pad=0.06,
            shrink=0.85,
            extend="both",
            ticks=CBAR_TICKS,
            boundaries=diff_levels,
            spacing="uniform"
        )

    if DIFF_CBAR_ORIENTATION == "vertical":
        cbar_d.ax.yaxis.set_ticks_position("right")
        cbar_d.ax.yaxis.set_label_position("right")
        cbar_d.set_label(DIFF_CBAR_LABEL, rotation=270, labelpad=16)
    else:
        cbar_d.ax.xaxis.set_ticks_position("bottom")
        cbar_d.ax.xaxis.set_label_position("bottom")
        cbar_d.set_label(DIFF_CBAR_LABEL)

    cbar_d.ax.tick_params(labelsize=11)
    cbar_d.outline.set_linewidth(0.9)

    # ---- Save ----
    out_png = os.path.join(OUT_DIR, f"{OUT_BASE}.png")
    out_pdf = os.path.join(OUT_DIR, f"{OUT_BASE}.pdf")

    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=PDF_DPI, bbox_inches="tight")

    print(f"Saved:\n  {out_png}\n  {out_pdf}")
    plt.show()


if __name__ == "__main__":
    main()