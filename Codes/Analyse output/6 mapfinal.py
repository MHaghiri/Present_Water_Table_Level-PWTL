#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.cm import get_cmap, ScalarMappable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import FuncFormatter
from PIL import Image

import geopandas as gpd
import rasterio
from rasterio.plot import plotting_extent
from rasterio.warp import reproject, Resampling
from rasterio.features import geometry_mask


# =========================================================
# ====================== USER INPUT =======================
# =========================================================

# ---------------- Panel A input ----------------
CSV_FILE = "/home/mohammad/Desktop/1/15/monthly_WTD_2001_2025.csv"

COL_YEAR  = "year"
COL_MONTH = "month"
COL_DATE  = "date"
COL_WTD   = "WTD"
COL_P     = "P"
COL_ET    = "ET"
COL_PET   = "P-ET"

# ---------------- Panel B input ----------------
MAP_PANELS = [
    {
        "label": "c",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002000_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002005_petsc_000000001.tif",
        "pie_values": (61, 39),

        "profile_xlim_right": (-400, 400),
        "profile_ylim_bottom": (-400, 400),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,

        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2000-2005",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
    {
        "label": "d",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002005_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002010_petsc_000000001.tif",
        "pie_values": (31, 69),

        "profile_xlim_right": (-400, 400),
        "profile_ylim_bottom": (-400, 400),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,

        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2005-2010",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
    {
        "label": "e",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002010_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002015_petsc_000000001.tif",
        "pie_values": (30, 70),

        "profile_xlim_right": (-400, 400),
        "profile_ylim_bottom": (-400, 400),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,

        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2010-2015",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
    {
        "label": "f",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002015_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002020_petsc_000000001.tif",
        "pie_values": (56, 44),

        "profile_xlim_right": (-400, 400),
        "profile_ylim_bottom": (-400, 400),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,

        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2015-2020",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
    {
        "label": "g",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002020_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002025_petsc_000000001.tif",
        "pie_values": (53, 47),

        "profile_xlim_right": (-400, 400),
        "profile_ylim_bottom": (-400, 400),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,
        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2020-2025",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
    {
        "label": "h",
        "map1": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002000_petsc_000000001.tif",
        "map2": "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1/N_America_002025_petsc_000000001.tif",
        "pie_values": (42, 58),

        "profile_xlim_right": (-700, 700),
        "profile_ylim_bottom": (-600, 600),

        "right_tick_divisor": 400.0,
        "bottom_tick_divisor": 400.0,

        "right_axis_prefix": "",
        "bottom_axis_prefix": "",

        # year label controls
        "year_text": "2000-2025",
        "year_label_x": 0.80,
        "year_label_y": 0.40,
        "year_label_fontsize": 11,
        "year_label_ha": "center",
        "year_label_va": "top",
    },
]

# ---------------- Shapefiles ----------------
BOUNDARY_SHP  = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_boundery_without_greenland.shp"
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_level2_watershed_without_greenland.shp"
GREENLAND_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/Greenland.shp"

# ---------------- Output ----------------
OUTPUT_PNG = "/home/mohammad/Desktop/1/15/25_year_diff.png"
OUTPUT_PDF = "/home/mohammad/Desktop/1/15/25_year_diff.pdf"

SAVE_PNG = True
SAVE_PDF = True

OUTPUT_DPI = 1500

# Use a lower DPI for PDF fallback image export to avoid failure
PDF_FALLBACK_DPI = 400

TEMP_PDF_PNG = OUTPUT_PDF.replace(".pdf", "_temp_export.png")

# ---------------- Difference settings ----------------
DIFF_MULTIPLIER = 100.0
DIFF_SIGN = 1
RESAMPLING_METHOD = Resampling.bilinear

# Mask options
MASK_TO_BOUNDARY = True
MASK_TO_WATERSHED = False

# ---------------- Map domain ----------------
LON_MIN = -180
LON_MAX = -7
LAT_MIN = 7
LAT_MAX = 85

# =========================================================
# ====== DIFFERENCE MAP COLOR CLASSIFICATION SETTINGS ======
# =========================================================
MAP_TICKS = [-50, -30, -10, -5, 0, 5, 10, 30, 50]
WHITE_LOW = -0.1
WHITE_HIGH = 0.1

CBAR_TICKS = [-30, -20, -10, -3, 0, 3, 10, 20, 30]
CBAR_LABEL = "Change in Water Table Depth (cm)"

# ---------------- Figure controls ----------------
FIGSIZE = (10, 15)

OUTER_WSPACE = 0.1
OUTER_HSPACE = 0.1

PANEL_A_ROW1_HEIGHT = 1.50
PANEL_A_ROW2_HEIGHT = 1.50

# ---------------- Map/profile size controls ----------------
MAP_WIDTH_RATIO = 6.0
RIGHT_PROFILE_WIDTH_RATIO = 1.25

MAP_HEIGHT_RATIO = 4.0
BOTTOM_PROFILE_HEIGHT_RATIO = 1.15

INNER_WSPACE = 0.02
INNER_HSPACE = 0.04

RIGHT_PROFILE_WIDTH_SCALE = 0.8
RIGHT_PROFILE_HEIGHT_SCALE = 0.975

BOTTOM_PROFILE_WIDTH_SCALE = 1.0
BOTTOM_PROFILE_HEIGHT_SCALE = 1.00

# ---------------- Colorbar position controls ----------------
CBAR_X = 0.905
CBAR_Y = 0.32
CBAR_W = 0.018
CBAR_H = 0.16

# ---------------- Boundary / Watershed style ----------------
SHOW_BOUNDARY = True
SHOW_WATERSHED = True
SHOW_GREENLAND = True

BOUNDARY_LW = 0.6
BOUNDARY_COLOR = "black"

WATERSHED_LW = 0.35
WATERSHED_COLOR = "black"

GREENLAND_LW = 0.5
GREENLAND_FACE = "#d9d9d9"
GREENLAND_EDGE = "black"

# ---------------- Pie legend controls ----------------
SHOW_PIE_LEGEND = True
PIE_LEGEND_PANEL = "c"
PIE_LEGEND_X = 2.0
PIE_LEGEND_Y = -0.18
PIE_LEGEND_FONTSIZE = 10

# ---------------- Shared line legend controls ----------------
SHOW_SHARED_LINE_LEGEND = True
LINE_LEGEND_X = 0.50
LINE_LEGEND_Y = 0.69
LINE_LEGEND_NCOL = 4
LINE_LEGEND_FONTSIZE = 10
LINE_LEGEND_LOC = "upper center"
LINE_LEGEND_FRAMEON = True

# ---------------- Line chart colors/styles ----------------
WTD_COLOR = "purple"
WTD_LS = "--"
WTD_LW = 1.8

P_COLOR = "blue"
P_LS = "-"
P_LW = 0.8

ET_COLOR = "orange"
ET_LS = "--"
ET_LW = 0.8

PET_COLOR = "red"
PET_LS = "-"
PET_LW = 0.8

# Make WTD axis match WTD line color
WTD_AXIS_COLOR = WTD_COLOR

# ---------------- Line chart domain controls ----------------
TOP_X_MIN = None
TOP_X_MAX = None
TOP_Y1_MIN = -9.25
TOP_Y1_MAX = -9.00
TOP_Y2_MIN = 0.0
TOP_Y2_MAX = 1.5

BOTTOM_X_MIN = None
BOTTOM_X_MAX = None
BOTTOM_Y1_MIN = -9.2
BOTTOM_Y1_MAX = -9.08
BOTTOM_Y2_MIN = 0.0
BOTTOM_Y2_MAX = 1

# ---------------- Show x tick labels ----------------
SHOW_TOP_X_TICKLABELS = False
SHOW_BOTTOM_X_TICKLABELS = True

# ---------------- Remove background grid in line charts ----------------
SHOW_LINE_GRID = False

# ---------------- Shared axis titles for line charts ----------------
SHARED_LEFT_Y_LABEL = "Water Table Depth (m)"
SHARED_RIGHT_Y_LABEL = "P, ET, P-ET (m)"

SHOW_SHARED_Y_LABELS = True

LEFT_Y_LABEL_X = 0.06
LEFT_Y_LABEL_Y = 0.775
RIGHT_Y_LABEL_X = 0.96
RIGHT_Y_LABEL_Y = 0.775

LEFT_Y_LABEL_ROTATION = 90
RIGHT_Y_LABEL_ROTATION = -90

LEFT_Y_LABEL_FONTSIZE = 11
RIGHT_Y_LABEL_FONTSIZE = 11

# ---------------- Panel A labels ----------------
PANEL_A_TOP_LABEL = "a"
PANEL_A_BOTTOM_LABEL = "b"

PANEL_A_TOP_LABEL_X = 0.015
PANEL_A_TOP_LABEL_Y = 0.95

PANEL_A_BOTTOM_LABEL_X = 0.015
PANEL_A_BOTTOM_LABEL_Y = 0.95

PANEL_A_LABEL_FONTSIZE = 18

# ---------------- Panel B map-label position ----------------
MAP_LABEL_X = 0.015
MAP_LABEL_Y = 0.98
MAP_LABEL_HA = "left"
MAP_LABEL_VA = "top"
MAP_LABEL_FONTSIZE = 18

# ---------------- X-axis labels ----------------
TOP_X_LABEL = ""
BOTTOM_X_LABEL = ""

# ---------------- Pie ----------------
PIE_COLORS = ["#2b6cb0", "#e03131"]
PIE_LABELS = ["WT Rise %", "WT Drop %"]

# ---------------- Profile line settings ----------------
PROFILE_LINEWIDTH = 0.8

# ---------------------------------------------------------
# HIGHLIGHT BOX (Panel A line charts: top monthly + bottom annual)
# Draws a vertical band behind the lines for a chosen year range.
# ---------------------------------------------------------
HIGHLIGHT_BOX = {
    "show": True,           # turn the highlight box on/off
    "year_start": 2011.5,     # first year of highlight (inclusive)
    "year_end": 2017,       # last year of highlight (inclusive)
    "color": "red",  # any matplotlib color name, hex (e.g. "#90ee90"), or rgb tuple
    "alpha": 0.25,          # transparency 0.0 - 1.0
    "edgecolor": "none",    # set to e.g. "darkgreen" for a border
    "linewidth": 0.0,       # border width (only matters if edgecolor != "none")
    "zorder": 0,            # behind the lines (lines default to ~2)
    "label": None,          # e.g. "Drought period" to add to legend, or None
}

# ---------------- Matplotlib style ----------------
mpl.rcParams["font.size"] = 10
mpl.rcParams["axes.titlesize"] = 11
mpl.rcParams["axes.labelsize"] = 10


# =========================================================
# =================== HELPER FUNCTIONS ====================
# =========================================================

def ensure_output_dir(path):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)


def check_file_exists(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found:\n{path}")


def load_csv_data(csv_file):
    df = pd.read_csv(csv_file)

    if COL_DATE in df.columns:
        df["date_dt"] = pd.to_datetime(df[COL_DATE], errors="coerce")
    else:
        df["date_dt"] = pd.to_datetime(
            df[COL_YEAR].astype(str) + "-" + df[COL_MONTH].astype(str).str.zfill(2) + "-01",
            errors="coerce"
        )

    annual = (
        df.groupby(COL_YEAR)[[COL_WTD, COL_P, COL_ET, COL_PET]]
        .mean()
        .reset_index()
    )

    return df, annual


def build_diff_cmap_norm():
    neg = sorted([float(t) for t in MAP_TICKS if t < 0])
    pos = sorted([float(t) for t in MAP_TICKS if t > 0])

    levels = np.array(neg + [WHITE_LOW, WHITE_HIGH] + pos, dtype=float)

    cont = get_cmap("RdYlBu")
    mids = (levels[:-1] + levels[1:]) / 2.0
    t = (mids - levels[0]) / (levels[-1] - levels[0])
    cols = [cont(v) for v in t]

    zero_idx = np.where((levels[:-1] < 0) & (levels[1:] > 0))[0]
    if zero_idx.size == 1:
        cols[int(zero_idx[0])] = (1, 1, 1, 1)

    cmap = ListedColormap(cols, name="rdylbu_white_zero")
    cmap.set_under(cols[0])
    cmap.set_over(cols[-1])
    cmap.set_bad(color="white", alpha=1.0)

    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=False)
    return cmap, norm, levels


def build_fake_colorbar_mappable():
    cbar_bounds = np.array(
        [-2, -1.5, -1, -0.5, WHITE_LOW, WHITE_HIGH, 0.5, 1, 1.5, 2],
        dtype=float
    )
    cont = get_cmap("RdYlBu")
    mids = (cbar_bounds[:-1] + cbar_bounds[1:]) / 2.0
    t = (mids - cbar_bounds[0]) / (cbar_bounds[-1] - cbar_bounds[0])
    cols = [cont(v) for v in t]

    zero_idx = np.where((cbar_bounds[:-1] < 0) & (cbar_bounds[1:] > 0))[0]
    if zero_idx.size == 1:
        cols[int(zero_idx[0])] = (1, 1, 1, 1)

    cmap = ListedColormap(cols, name="rdylbu_white_zero_fake")
    cmap.set_bad(color="white", alpha=1.0)
    norm = BoundaryNorm(cbar_bounds, ncolors=cmap.N, clip=False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    return sm, cbar_bounds


def read_raster(raster_path):
    with rasterio.open(raster_path) as src:
        arr = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        extent = plotting_extent(src)
        nodata = src.nodata

    if nodata is not None:
        arr[arr == nodata] = np.nan

    return arr, profile, transform, crs, bounds, extent, nodata


def reproject_to_match(source_arr, source_transform, source_crs,
                       target_shape, target_transform, target_crs,
                       src_nodata=np.nan):
    dst = np.full(target_shape, np.nan, dtype=np.float32)

    reproject(
        source=source_arr,
        destination=dst,
        src_transform=source_transform,
        src_crs=source_crs,
        dst_transform=target_transform,
        dst_crs=target_crs,
        src_nodata=src_nodata,
        dst_nodata=np.nan,
        resampling=RESAMPLING_METHOD
    )
    return dst


def mask_array_to_shape(arr, transform, shape_gdf):
    mask_inside = geometry_mask(
        shape_gdf.geometry,
        transform=transform,
        invert=True,
        out_shape=arr.shape
    )
    out = arr.copy()
    out[~mask_inside] = np.nan
    return out


def prepare_shape_to_raster_crs(gdf, raster_crs):
    if gdf is None or gdf.empty:
        return gdf
    if gdf.crs != raster_crs:
        return gdf.to_crs(raster_crs)
    return gdf


def calculate_difference_map(map1_path, map2_path, boundary_gdf=None, watershed_gdf=None):
    arr1, profile1, transform1, crs1, bounds1, extent1, nodata1 = read_raster(map1_path)
    arr2, profile2, transform2, crs2, bounds2, extent2, nodata2 = read_raster(map2_path)

    if arr1.shape != arr2.shape or transform1 != transform2 or crs1 != crs2:
        arr2_match = reproject_to_match(
            source_arr=arr2,
            source_transform=transform2,
            source_crs=crs2,
            target_shape=arr1.shape,
            target_transform=transform1,
            target_crs=crs1,
            src_nodata=np.nan
        )
    else:
        arr2_match = arr2

    diff = DIFF_SIGN * (arr2_match - arr1) * DIFF_MULTIPLIER
    diff = diff.astype(np.float32)

    boundary_use = prepare_shape_to_raster_crs(boundary_gdf, crs1)
    watershed_use = prepare_shape_to_raster_crs(watershed_gdf, crs1)

    if MASK_TO_BOUNDARY and boundary_use is not None and not boundary_use.empty:
        diff = mask_array_to_shape(diff, transform1, boundary_use)

    if MASK_TO_WATERSHED and watershed_use is not None and not watershed_use.empty:
        diff = mask_array_to_shape(diff, transform1, watershed_use)

    return diff, extent1, bounds1, transform1, crs1, boundary_use, watershed_use


def compute_profiles(arr, bounds):
    nrows, ncols = arr.shape
    row_mean = np.nanmean(arr, axis=1)
    col_mean = np.nanmean(arr, axis=0)

    lat_vals = np.linspace(bounds.top, bounds.bottom, nrows)
    lon_vals = np.linspace(bounds.left, bounds.right, ncols)

    return row_mean, col_mean, lat_vals, lon_vals


def format_pie_value(v):
    if abs(v - round(v)) < 1e-9:
        return f"{int(round(v))}"
    return f"{v:.1f}"


def pie_autopct_from_values(values):
    total = sum(values)

    def _autopct(pct):
        val = pct * total / 100.0
        return format_pie_value(val)
    return _autopct


def apply_axis_limits(ax, xmin=None, xmax=None, ymin=None, ymax=None):
    if xmin is not None or xmax is not None:
        ax.set_xlim(left=xmin, right=xmax)
    if ymin is not None or ymax is not None:
        ax.set_ylim(bottom=ymin, top=ymax)


def make_target_max_formatter(real_min, real_max, shown_max, prefix=""):
    real_abs_max = max(abs(real_min), abs(real_max))

    if real_abs_max == 0 or shown_max is None:
        scale = 1.0
    else:
        scale = float(shown_max) / float(real_abs_max)

    def _formatter(x, pos):
        val = x * scale
        if abs(val - round(val)) < 1e-9:
            txt = f"{int(round(val))}"
        else:
            txt = f"{val:.1f}"
        return f"{prefix}{txt}"

    return FuncFormatter(_formatter)


def set_min_zero_max_ticks(ax, axis="x", vmin=None, vmax=None):
    if vmin is None or vmax is None:
        return

    ticks = [vmin, 0, vmax]

    ticks_unique = []
    for t in ticks:
        if not any(np.isclose(t, existing) for existing in ticks_unique):
            ticks_unique.append(t)

    if axis == "x":
        ax.set_xticks(ticks_unique)
    else:
        ax.set_yticks(ticks_unique)


def style_left_wtd_axis(ax):
    ax.tick_params(axis="y", colors=WTD_AXIS_COLOR)
    ax.tick_params(axis="x", colors="black")
    ax.yaxis.label.set_color(WTD_AXIS_COLOR)
    ax.xaxis.label.set_color("black")

    ax.spines["left"].set_color(WTD_AXIS_COLOR)
    ax.spines["bottom"].set_color("black")
    ax.spines["top"].set_color("black")

    if "right" in ax.spines:
        ax.spines["right"].set_color("black")


def add_highlight_box(ax, mode="monthly"):
    """
    Draw a vertical highlight band behind the lines, between
    HIGHLIGHT_BOX['year_start'] and HIGHLIGHT_BOX['year_end'].

    mode = "monthly": x-axis is datetime
    mode = "annual" : x-axis is integer year
    """
    if not HIGHLIGHT_BOX["show"]:
        return None

    y0 = int(HIGHLIGHT_BOX["year_start"])
    y1 = int(HIGHLIGHT_BOX["year_end"])

    # ensure correct ordering
    if y0 > y1:
        y0, y1 = y1, y0

    if mode == "monthly":
        # Cover full years: Jan 1 of y0 through Dec 31 of y1
        x_start = pd.Timestamp(year=y0, month=1, day=1)
        x_end = pd.Timestamp(year=y1, month=12, day=31)
    elif mode == "annual":
        # Cover annual data points y0..y1 with a small pad so the band
        # actually covers those points visually (annual x is integer year).
        x_start = y0 - 0.5
        x_end = y1 + 0.5
    else:
        raise ValueError("mode must be 'monthly' or 'annual'")

    patch = ax.axvspan(
        x_start, x_end,
        facecolor=HIGHLIGHT_BOX["color"],
        alpha=HIGHLIGHT_BOX["alpha"],
        edgecolor=HIGHLIGHT_BOX["edgecolor"],
        linewidth=HIGHLIGHT_BOX["linewidth"],
        zorder=HIGHLIGHT_BOX["zorder"],
        label=HIGHLIGHT_BOX["label"] if HIGHLIGHT_BOX["label"] else None,
    )
    return patch


def plot_dual_axis_timeseries(
    ax, x, wtd, p, et, pet, title, xlab,
    x_min=None, x_max=None,
    y1_min=None, y1_max=None,
    y2_min=None, y2_max=None,
    show_x_ticklabels=True,
    left_ylabel="",
    right_ylabel="",
    highlight_mode=None
):
    # Draw the highlight band first so it sits behind the lines
    hl_patch = None
    if highlight_mode is not None:
        hl_patch = add_highlight_box(ax, mode=highlight_mode)

    ax2 = ax.twinx()

    line1, = ax.plot(
        x, wtd, label="WTD",
        color=WTD_COLOR, linestyle=WTD_LS, linewidth=WTD_LW
    )
    line2, = ax2.plot(
        x, p, label="Precipitation",
        color=P_COLOR, linestyle=P_LS, linewidth=P_LW
    )
    line3, = ax2.plot(
        x, et, label="Evapotranspiration",
        color=ET_COLOR, linestyle=ET_LS, linewidth=ET_LW
    )
    line4, = ax2.plot(
        x, pet, label="P - ET",
        color=PET_COLOR, linestyle=PET_LS, linewidth=PET_LW
    )

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(left_ylabel)
    ax2.set_ylabel(right_ylabel)

    if SHOW_LINE_GRID:
        ax.grid(True, alpha=0.3)
    else:
        ax.grid(False)
        ax2.grid(False)

    apply_axis_limits(ax, xmin=x_min, xmax=x_max, ymin=y1_min, ymax=y1_max)
    apply_axis_limits(ax2, ymin=y2_min, ymax=y2_max)

    if not show_x_ticklabels:
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")

    style_left_wtd_axis(ax)

    ax2.tick_params(axis="y", colors="black")
    ax2.yaxis.label.set_color("black")
    ax2.spines["right"].set_color("black")
    ax2.spines["top"].set_color("black")
    ax2.spines["bottom"].set_color("black")

    return ax, ax2, [line1, line2, line3, line4], hl_patch


def plot_panel_a(ax1, ax2, df, annual):
    ax1, ax1r, lines1, hl1 = plot_dual_axis_timeseries(
        ax1,
        df["date_dt"], df[COL_WTD], df[COL_P], df[COL_ET], df[COL_PET],
        "",
        TOP_X_LABEL,
        x_min=TOP_X_MIN, x_max=TOP_X_MAX,
        y1_min=TOP_Y1_MIN, y1_max=TOP_Y1_MAX,
        y2_min=TOP_Y2_MIN, y2_max=TOP_Y2_MAX,
        show_x_ticklabels=SHOW_TOP_X_TICKLABELS,
        left_ylabel="",
        right_ylabel="",
        highlight_mode="monthly"
    )

    ax2, ax2r, lines2, hl2 = plot_dual_axis_timeseries(
        ax2,
        annual[COL_YEAR], annual[COL_WTD], annual[COL_P], annual[COL_ET], annual[COL_PET],
        "",
        BOTTOM_X_LABEL,
        x_min=BOTTOM_X_MIN, x_max=BOTTOM_X_MAX,
        y1_min=BOTTOM_Y1_MIN, y1_max=BOTTOM_Y1_MAX,
        y2_min=BOTTOM_Y2_MIN, y2_max=BOTTOM_Y2_MAX,
        show_x_ticklabels=SHOW_BOTTOM_X_TICKLABELS,
        left_ylabel="",
        right_ylabel="",
        highlight_mode="annual"
    )

    ax1.text(
        PANEL_A_TOP_LABEL_X, PANEL_A_TOP_LABEL_Y, PANEL_A_TOP_LABEL,
        transform=ax1.transAxes, fontsize=PANEL_A_LABEL_FONTSIZE,
        ha="left", va="top"
    )
    ax2.text(
        PANEL_A_BOTTOM_LABEL_X, PANEL_A_BOTTOM_LABEL_Y, PANEL_A_BOTTOM_LABEL,
        transform=ax2.transAxes, fontsize=PANEL_A_LABEL_FONTSIZE,
        ha="left", va="top"
    )

    # Build the list of legend handles. If a highlight label was provided,
    # include the patch from the top chart so it appears in the shared legend.
    legend_handles = list(lines1)
    if hl1 is not None and HIGHLIGHT_BOX["label"]:
        legend_handles.append(hl1)

    return legend_handles


def scale_axes_box(ax, sx=1.0, sy=1.0):
    pos = ax.get_position()
    cx = 0.5 * (pos.x0 + pos.x1)
    cy = 0.5 * (pos.y0 + pos.y1)
    w = pos.width * sx
    h = pos.height * sy
    new_pos = [cx - w / 2.0, cy - h / 2.0, w, h]
    ax.set_position(new_pos)


def style_profile_axes(ax):
    ax.tick_params(labelsize=8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)


def plot_panel_map(ax_map, ax_right, ax_bottom, panel_cfg, cmap, norm,
                   boundary_gdf, watershed_gdf, greenland_gdf):
    diff_arr, full_extent, bounds, transform, crs, boundary_use, watershed_use = calculate_difference_map(
        panel_cfg["map1"], panel_cfg["map2"], boundary_gdf=boundary_gdf, watershed_gdf=watershed_gdf
    )

    label = panel_cfg["label"]
    pie_values = panel_cfg["pie_values"]

    diff_ma = np.ma.masked_invalid(diff_arr)

    im = ax_map.imshow(
        diff_ma,
        cmap=cmap,
        norm=norm,
        extent=full_extent,
        origin="upper",
        interpolation="none",
        resample=False,
        aspect="auto"
    )

    if SHOW_GREENLAND and greenland_gdf is not None and not greenland_gdf.empty:
        greenland_use = greenland_gdf.to_crs(crs) if greenland_gdf.crs != crs else greenland_gdf
        greenland_use.plot(
            ax=ax_map,
            facecolor=GREENLAND_FACE,
            edgecolor=GREENLAND_EDGE,
            linewidth=GREENLAND_LW,
            zorder=4
        )

    if SHOW_WATERSHED and watershed_use is not None and not watershed_use.empty:
        watershed_use.boundary.plot(
            ax=ax_map,
            color=WATERSHED_COLOR,
            linewidth=WATERSHED_LW,
            zorder=5
        )

    if SHOW_BOUNDARY and boundary_use is not None and not boundary_use.empty:
        boundary_use.boundary.plot(
            ax=ax_map,
            color=BOUNDARY_COLOR,
            linewidth=BOUNDARY_LW,
            zorder=6
        )

    ax_map.set_xlim(LON_MIN, LON_MAX)
    ax_map.set_ylim(LAT_MIN, LAT_MAX)
    ax_map.set_xticks([])
    ax_map.set_yticks([])
    ax_map.set_facecolor("white")

    ax_map.text(
        MAP_LABEL_X, MAP_LABEL_Y, label,
        transform=ax_map.transAxes,
        fontsize=MAP_LABEL_FONTSIZE,
        ha=MAP_LABEL_HA, va=MAP_LABEL_VA
    )

    if "year_text" in panel_cfg and str(panel_cfg["year_text"]).strip() != "":
        ax_map.text(
            panel_cfg.get("year_label_x", 0.50),
            panel_cfg.get("year_label_y", 0.94),
            panel_cfg["year_text"],
            transform=ax_map.transAxes,
            fontsize=panel_cfg.get("year_label_fontsize", 11),
            ha=panel_cfg.get("year_label_ha", "center"),
            va=panel_cfg.get("year_label_va", "top"),
            color="black"
        )

    pie_ax = inset_axes(
        ax_map,
        width="28%",
        height="42%",
        loc="lower left",
        borderpad=0.8
    )

    wedges, texts, autotexts = pie_ax.pie(
        pie_values,
        colors=PIE_COLORS,
        startangle=90,
        autopct=pie_autopct_from_values(pie_values),
        textprops={"fontsize": 9, "color": "black"},
        wedgeprops={"edgecolor": "black", "linewidth": 0.5}
    )
    pie_ax.set_aspect("equal")
    pie_ax.set_facecolor("white")

    if SHOW_PIE_LEGEND and label == PIE_LEGEND_PANEL:
        pie_ax.legend(
            wedges,
            PIE_LABELS,
            loc="lower left",
            bbox_to_anchor=(PIE_LEGEND_X, PIE_LEGEND_Y),
            fontsize=PIE_LEGEND_FONTSIZE,
            frameon=True,
            borderpad=0.3,
            handlelength=1.5,
            handletextpad=0.4
        )

    row_mean, col_mean, lat_vals, lon_vals = compute_profiles(diff_arr, bounds)

    ax_right.plot(row_mean, lat_vals, color="black", linewidth=PROFILE_LINEWIDTH)
    ax_right.axvline(0, color="gray", linewidth=0.6)
    ax_right.grid(False)
    ax_right.set_ylim(LAT_MIN, LAT_MAX)
    ax_right.set_xlim(*panel_cfg["profile_xlim_right"])
    set_min_zero_max_ticks(
        ax_right, axis="x",
        vmin=panel_cfg["profile_xlim_right"][0],
        vmax=panel_cfg["profile_xlim_right"][1]
    )
    ax_right.xaxis.set_major_formatter(
        make_target_max_formatter(
            real_min=panel_cfg["profile_xlim_right"][0],
            real_max=panel_cfg["profile_xlim_right"][1],
            shown_max=panel_cfg.get("right_tick_divisor", None),
            prefix=panel_cfg.get("right_axis_prefix", "")
        )
    )
    ax_right.set_yticks([])
    style_profile_axes(ax_right)

    ax_bottom.plot(lon_vals, col_mean, color="black", linewidth=PROFILE_LINEWIDTH)
    ax_bottom.axhline(0, color="gray", linewidth=0.6)
    ax_bottom.grid(False)
    ax_bottom.set_xlim(LON_MIN, LON_MAX)
    ax_bottom.set_ylim(*panel_cfg["profile_ylim_bottom"])
    set_min_zero_max_ticks(
        ax_bottom, axis="y",
        vmin=panel_cfg["profile_ylim_bottom"][0],
        vmax=panel_cfg["profile_ylim_bottom"][1]
    )
    ax_bottom.yaxis.set_major_formatter(
        make_target_max_formatter(
            real_min=panel_cfg["profile_ylim_bottom"][0],
            real_max=panel_cfg["profile_ylim_bottom"][1],
            shown_max=panel_cfg.get("bottom_tick_divisor", None),
            prefix=panel_cfg.get("bottom_axis_prefix", "")
        )
    )
    ax_bottom.set_xticks([])
    style_profile_axes(ax_bottom)

    return im


def add_shared_line_legend(fig, lines):
    if not SHOW_SHARED_LINE_LEGEND:
        return
    labels = [ln.get_label() for ln in lines]
    fig.legend(
        lines, labels,
        loc=LINE_LEGEND_LOC,
        bbox_to_anchor=(LINE_LEGEND_X, LINE_LEGEND_Y),
        ncol=LINE_LEGEND_NCOL,
        fontsize=LINE_LEGEND_FONTSIZE,
        frameon=LINE_LEGEND_FRAMEON
    )


def add_shared_y_axis_titles(fig):
    if not SHOW_SHARED_Y_LABELS:
        return

    fig.text(
        LEFT_Y_LABEL_X, LEFT_Y_LABEL_Y, SHARED_LEFT_Y_LABEL,
        rotation=LEFT_Y_LABEL_ROTATION,
        va="center", ha="center",
        fontsize=LEFT_Y_LABEL_FONTSIZE,
        color="black"
    )

    fig.text(
        RIGHT_Y_LABEL_X, RIGHT_Y_LABEL_Y, SHARED_RIGHT_Y_LABEL,
        rotation=RIGHT_Y_LABEL_ROTATION,
        va="center", ha="center",
        fontsize=RIGHT_Y_LABEL_FONTSIZE,
        color="black"
    )


def save_pdf_direct(fig, pdf_path):
    fig.savefig(
        pdf_path,
        format="pdf",
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none"
    )


def save_pdf_from_png(fig, pdf_path, temp_png_path, dpi=300):
    fig.savefig(
        temp_png_path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none"
    )

    with Image.open(temp_png_path) as im:
        rgb = im.convert("RGB")
        rgb.save(pdf_path, "PDF", resolution=dpi)

    if os.path.exists(temp_png_path):
        os.remove(temp_png_path)


def save_figure_outputs(fig):
    if SAVE_PNG:
        ensure_output_dir(OUTPUT_PNG)
        fig.savefig(
            OUTPUT_PNG,
            dpi=OUTPUT_DPI,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none"
        )

    if SAVE_PDF:
        ensure_output_dir(OUTPUT_PDF)

        try:
            save_pdf_direct(fig, OUTPUT_PDF)
            print(f"PDF figure saved directly to:\n{OUTPUT_PDF}")
        except Exception as e:
            print("Direct PDF save failed.")
            print(f"Reason: {e}")
            print("Trying fallback PNG-to-PDF export...")

            save_pdf_from_png(
                fig=fig,
                pdf_path=OUTPUT_PDF,
                temp_png_path=TEMP_PDF_PNG,
                dpi=PDF_FALLBACK_DPI
            )
            print(f"PDF figure saved by fallback method to:\n{OUTPUT_PDF}")


def check_required_files():
    check_file_exists(CSV_FILE)
    check_file_exists(BOUNDARY_SHP)

    if SHOW_WATERSHED or MASK_TO_WATERSHED:
        check_file_exists(WATERSHED_SHP)

    if SHOW_GREENLAND and GREENLAND_SHP:
        check_file_exists(GREENLAND_SHP)

    for panel in MAP_PANELS:
        check_file_exists(panel["map1"])
        check_file_exists(panel["map2"])


# =========================================================
# ======================== MAIN ===========================
# =========================================================

def main():
    check_required_files()

    df, annual = load_csv_data(CSV_FILE)

    boundary_gdf = gpd.read_file(BOUNDARY_SHP)

    watershed_gdf = None
    if SHOW_WATERSHED or MASK_TO_WATERSHED:
        watershed_gdf = gpd.read_file(WATERSHED_SHP)

    greenland_gdf = gpd.read_file(GREENLAND_SHP) if (SHOW_GREENLAND and os.path.exists(GREENLAND_SHP)) else None

    diff_cmap, diff_norm, diff_levels = build_diff_cmap_norm()
    sm_cbar, cbar_bounds = build_fake_colorbar_mappable()

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    fig.patch.set_facecolor("white")

    outer = GridSpec(
        nrows=5, ncols=2, figure=fig,
        height_ratios=[PANEL_A_ROW1_HEIGHT, PANEL_A_ROW2_HEIGHT, 2.6, 2.6, 2.6],
        hspace=OUTER_HSPACE,
        wspace=OUTER_WSPACE
    )

    axA1 = fig.add_subplot(outer[0, :])
    axA2 = fig.add_subplot(outer[1, :])

    shared_lines = plot_panel_a(axA1, axA2, df, annual)

    positions = [
        outer[2, 0], outer[2, 1],
        outer[3, 0], outer[3, 1],
        outer[4, 0], outer[4, 1],
    ]

    panel_axes = []

    for panel_cfg, pos in zip(MAP_PANELS, positions):
        sub = pos.subgridspec(
            2, 2,
            width_ratios=[MAP_WIDTH_RATIO, RIGHT_PROFILE_WIDTH_RATIO],
            height_ratios=[MAP_HEIGHT_RATIO, BOTTOM_PROFILE_HEIGHT_RATIO],
            wspace=INNER_WSPACE,
            hspace=INNER_HSPACE
        )

        ax_map = fig.add_subplot(sub[0, 0])
        ax_right = fig.add_subplot(sub[0, 1])
        ax_bottom = fig.add_subplot(sub[1, 0])
        ax_blank = fig.add_subplot(sub[1, 1])
        ax_blank.axis("off")

        plot_panel_map(
            ax_map=ax_map,
            ax_right=ax_right,
            ax_bottom=ax_bottom,
            panel_cfg=panel_cfg,
            cmap=diff_cmap,
            norm=diff_norm,
            boundary_gdf=boundary_gdf,
            watershed_gdf=watershed_gdf,
            greenland_gdf=greenland_gdf
        )

        panel_axes.append((ax_map, ax_right, ax_bottom))

    for ax_map, ax_right, ax_bottom in panel_axes:
        scale_axes_box(ax_right, sx=RIGHT_PROFILE_WIDTH_SCALE, sy=RIGHT_PROFILE_HEIGHT_SCALE)
        scale_axes_box(ax_bottom, sx=BOTTOM_PROFILE_WIDTH_SCALE, sy=BOTTOM_PROFILE_HEIGHT_SCALE)

    cax = fig.add_axes([CBAR_X, CBAR_Y, CBAR_W, CBAR_H])
    cb = fig.colorbar(
        sm_cbar,
        cax=cax,
        orientation="vertical",
        extend="both",
        boundaries=cbar_bounds,
        spacing="uniform"
    )

    cbar_tick_positions = np.linspace(cbar_bounds[0], cbar_bounds[-1], len(CBAR_TICKS))
    cb.set_ticks(cbar_tick_positions)
    cb.set_ticklabels([str(v) for v in CBAR_TICKS])

    cb.set_label(CBAR_LABEL)
    cb.ax.tick_params(labelsize=9)

    add_shared_line_legend(fig, shared_lines)
    add_shared_y_axis_titles(fig)

    save_figure_outputs(fig)

    plt.show()

    if SAVE_PNG:
        print(f"PNG figure saved to:\n{OUTPUT_PNG}")
    if SAVE_PDF:
        print(f"PDF figure saved to:\n{OUTPUT_PDF}")


if __name__ == "__main__":
    main()
