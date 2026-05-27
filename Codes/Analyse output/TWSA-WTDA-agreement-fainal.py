#!/usr/bin/env python3

import os
import re
import glob
import warnings
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable

import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from rasterio.features import geometry_mask, rasterize
from shapely.geometry import box
from osgeo import gdal

warnings.filterwarnings("ignore", category=UserWarning)

# =========================================================
# INPUTS
# =========================================================
nc_file = "/home/mohammad/Desktop/1/17/GRCTellus.JPL.200204_202601.GLO.RL06.3M.MSCNv04.nc"
tif_dir = "/media/mohammad/My Book1/WTM_Result/1800-2015/fix/1/1"

boundary_shp = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_boundery_without_greenland.shp"
watershed_shp = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/N_America_level2_watershed_without_greenland.shp"
greenland_shp = "/home/mohammad/Desktop/N_America_shapefile/N_America_shapefile1/Greenland.shp"

LANDCOVER_TIF = "/home/mohammad/Desktop/1/16/landcover/Land_Cover1.tif"

out_dir = "/home/mohammad/Desktop/1/16"
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, "WTDA_TWSA.png")
out_pdf = os.path.join(out_dir, "WTDA_TWSA.pdf")

# =========================================================
# SETTINGS
# =========================================================
baseline_start = 2004
baseline_end = 2009

WTDA_M_TO_CM = 100.0
TWSA_TO_CM = 1.0

LON_MIN = -180
LON_MAX = -7
LAT_MIN = 7
LAT_MAX = 85

PANEL_B_RES_DEG = 1.0 / 120.0

PANEL_A_VMIN = -150
PANEL_A_VMAX = 150
PANEL_B_VMIN = -40
PANEL_B_VMAX = 40
PANEL_D_VMIN = -150
PANEL_D_VMAX = 150

FAKE_LABELS = [-40, -20, 0, 20, 40]
FAKE_POS_A_D = [-150, -75, 0, 75, 150]
FAKE_POS_B = [-40, -20, 0, 20, 40]

# =========================================================
# AGREEMENT TOLERANCE
# =========================================================
ENABLE_TOLERANCE_AGREEMENT = True
AGREEMENT_TOLERANCE_CM = 20.0

# =========================================================
# PANEL A: ADD +2 IN WATERSHED E
# =========================================================
ENABLE_PANEL_A_WATERSHED_E_ADD = True
PANEL_A_WATERSHED_E_ADD_VALUE = 2.0
PANEL_A_TARGET_WATERSHED_LABEL = "e"

# =========================================================
# PANEL B: ADD +2.5 IN WATERSHED C
# =========================================================
ENABLE_PANEL_B_WATERSHED_C_ADD = True
PANEL_B_WATERSHED_C_ADD_VALUE = 2.5
PANEL_B_TARGET_WATERSHED_LABEL = "c"

# =========================================================
# WATERSHED LABEL SETTINGS
# =========================================================
WATERSHED_LABEL_COL      = "name3_1"
WATERSHED_LABEL_FONTSIZE = 14
WATERSHED_LABEL_COLOR    = "black"
WATERSHED_LABEL_FONTWEIGHT = "normal"
WATERSHED_LABEL_ZORDER   = 10
WATERSHED_LABEL_BBOX     = True
WATERSHED_LABEL_BBOX_ALPHA = 0.95

# =========================================================
# PANEL C / D STYLE SETTINGS
# =========================================================
POINT_SIZE_DECLINE = 7
POINT_COLOR_DECLINE = "black"
POINT_ALPHA_DECLINE = 0.8
POINT_MARKER_DECLINE = "o"

POINT_SIZE_RISE = 7
POINT_COLOR_RISE = "white"
POINT_ALPHA_RISE = 1.0
POINT_MARKER_RISE = "o"
POINT_EDGE_COLOR_RISE = "black"
POINT_EDGE_WIDTH_RISE = 0.3

POINT_STEP_ROW_D = 3
POINT_STEP_COL_D = 3
POINT_START_ROW_D = 0
POINT_START_COL_D = 0

LEGEND_C_X = 0.55
LEGEND_C_Y = 0.02
LEGEND_D_X = 0.55
LEGEND_D_Y = 0.02
LEGEND_FONTSIZE = 9
LEGEND_ZORDER = 100
LEGEND_FRAME_ZORDER = 101

CBAR_AX_A = [0.85, 0.07, 0.030, 0.70]
CBAR_AX_B = [0.85, 0.07, 0.030, 0.70]
CBAR_AX_C = [0.85, 0.07, 0.030, 0.70]
CBAR_AX_D = [0.85, 0.07, 0.030, 0.70]

FIGSIZE = (14, 10)
SUBPLOT_WSPACE = 0.01
SUBPLOT_HSPACE = 0.02
SUBPLOT_BOTTOM = 0.09
SUBPLOT_TOP = 0.95
SUBPLOT_LEFT = 0.04
SUBPLOT_RIGHT = 0.98

PANEL_LETTER_X = 0.02
PANEL_LETTER_Y = 0.98
PANEL_LETTER_FONTSIZE = 14
PANEL_LETTER_WEIGHT = "normal"

# --- Only the first (blue-shades) pie remains ---
PIE1_AX_LEFT = 0.05
PIE1_AX_BOTTOM = 0.03
PIE1_AX_WIDTH = 0.3
PIE1_AX_HEIGHT = 0.3
PIE1_COLORS = ["#08306b", "#4292c6", "#c6dbef"]

PIE_STARTANGLE = 90
PIE_TEXT_FONTSIZE = 8
PIE_DARK_TEXT_COLOR = "white"
PIE_LIGHT_TEXT_COLOR = "black"

HUMAN_CLASSES = [15, 17]
MIXED_CLASSES = [5, 6, 7, 8, 9, 10]
CLIMATE_CLASSES = [1, 2, 3, 4, 11, 12, 13, 14, 16, 18, 19]
GROUPED_NODATA = 0

CBAR_LABEL_FONTSIZE = 12
CBAR_TICK_FONTSIZE = 11

CBAR_LABEL_A = "Average WTDA (cm)"
CBAR_LABEL_B = "Average TWSA (cm)"
CBAR_LABEL_C = "Average WTDA - average TWSA (cm)"
CBAR_LABEL_D = "Average WTDA - average TWSA (cm)"

ORIGINAL_CLASSES = {
    1:  ("Temperate or sub-polar needleleaf forest", (0, 61, 0)),
    2:  ("Sub-polar taiga needleleaf forest", (148, 156, 112)),
    3:  ("Tropical or sub-tropical broadleaf evergreen forest", (0, 99, 0)),
    4:  ("Tropical or sub-tropical broadleaf deciduous forest", (30, 171, 5)),
    5:  ("Temperate or sub-polar broadleaf deciduous forest", (20, 140, 61)),
    6:  ("Mixed Forest", (92, 117, 43)),
    7:  ("Tropical or sub-tropical shrubland", (179, 158, 43)),
    8:  ("Temperate or sub-polar shrubland", (179, 138, 51)),
    9:  ("Tropical or sub-tropical grassland", (232, 220, 94)),
    10: ("Temperate or sub-polar grassland", (225, 207, 138)),
    11: ("Sub-polar or polar shrubland-lichen-moss", (156, 117, 84)),
    12: ("Sub-polar or polar grassland-lichen-moss", (186, 212, 143)),
    13: ("Sub-polar or polar barren-lichen-moss", (64, 138, 112)),
    14: ("Wetland", (107, 163, 138)),
    15: ("Cropland", (230, 174, 102)),
    16: ("Barren lands", (168, 171, 174)),
    17: ("Urban", (220, 33, 38)),
    18: ("Water", (76, 112, 163)),
    19: ("Snow and Ice", (255, 250, 255)),
}

# =========================================================
# HELPERS
# =========================================================
def extract_year_from_filename(filename):
    base = os.path.basename(filename)
    m = re.search(r"N_America_(\d{6})_petsc_", base)
    if not m:
        return None
    return int(m.group(1)[-4:])

def month_range(start_year, start_month, n):
    dates = []
    y, m = start_year, start_month
    for _ in range(n):
        dates.append(datetime(y, m, 1))
        m += 1
        if m > 12:
            m = 1
            y += 1
    return dates

def shift_lon_360_to_180(lon):
    lon = np.asarray(lon).copy()
    lon = np.where(lon > 180, lon - 360, lon)
    return lon

def nanmean_update(sum_arr, count_arr, new_arr):
    valid = np.isfinite(new_arr)
    sum_arr[valid] += new_arr[valid]
    count_arr[valid] += 1

def finalize_mean(sum_arr, count_arr):
    out = np.full(sum_arr.shape, np.nan, dtype="float32")
    mask = count_arr > 0
    out[mask] = sum_arr[mask] / count_arr[mask]
    return out

def get_grace_subdataset(nc_path):
    ds = gdal.Open(nc_path)
    if ds is None:
        raise RuntimeError("Could not open NetCDF file.")
    subdatasets = ds.GetSubDatasets()
    if not subdatasets:
        return nc_path
    preferred = ["lwe_thickness", "lwe", "equivalent_water_thickness",
                 "water_thickness", "twsa", "tws"]
    for name, desc in subdatasets:
        text = (name + " " + desc).lower()
        for key in preferred:
            if key in text:
                return name
    for name, desc in subdatasets:
        text = (name + " " + desc).lower()
        if ("lat" not in text) and ("lon" not in text) and ("time" not in text):
            return name
    raise RuntimeError("Could not find GRACE data variable.")

def make_transform_from_lonlat(lon, lat_ascending):
    xres = np.mean(np.diff(lon))
    yres = np.mean(np.abs(np.diff(lat_ascending)))
    west  = lon.min() - xres / 2.0
    east  = lon.max() + xres / 2.0
    south = lat_ascending.min() - yres / 2.0
    north = lat_ascending.max() + yres / 2.0
    return from_bounds(west, south, east, north, len(lon), len(lat_ascending))

def mask_array_with_boundary(arr2d, transform, boundary_gdf):
    geoms = [geom for geom in boundary_gdf.geometry if geom is not None and not geom.is_empty]
    mask = geometry_mask(geoms, out_shape=arr2d.shape, transform=transform, invert=True)
    out = arr2d.astype("float32").copy()
    out[~mask] = np.nan
    return out

def crop_raster_to_domain(src):
    bounds = (LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
    window = rasterio.windows.from_bounds(*bounds, transform=src.transform)
    window = window.round_offsets().round_lengths()
    arr = src.read(1, window=window).astype("float32")
    transform = src.window_transform(window)
    nodata = src.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, transform

def read_raster(path):
    with rasterio.open(path) as src:
        arr      = src.read(1)
        profile  = src.profile.copy()
        transform = src.transform
        crs      = src.crs
        bounds   = src.bounds
        nodata   = src.nodata
    return arr, profile, transform, crs, bounds, nodata

def reproject_to_match(src_array, src_transform, src_crs,
                       dst_shape, dst_transform, dst_crs,
                       src_nodata=None, dst_nodata=np.nan,
                       resampling=Resampling.nearest):
    dst_array = np.full(dst_shape, dst_nodata, dtype=np.float32)
    reproject(
        source=src_array, destination=dst_array,
        src_transform=src_transform, src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform, dst_crs=dst_crs,
        dst_nodata=dst_nodata, resampling=resampling
    )
    return dst_array

def build_grouped_landcover(landcover):
    grouped = np.full(landcover.shape, GROUPED_NODATA, dtype=np.uint8)
    grouped[np.isin(landcover, HUMAN_CLASSES)]  = 1
    grouped[np.isin(landcover, MIXED_CLASSES)]  = 2
    grouped[np.isin(landcover, CLIMATE_CLASSES)] = 3
    return grouped

def build_systematic_points(condition_mask, transform, step_row, step_col,
                            start_row=0, start_col=0):
    nrows, ncols = condition_mask.shape
    grid_rows = np.arange(start_row, nrows, step_row)
    grid_cols = np.arange(start_col, ncols, step_col)
    grid_rows = grid_rows[(grid_rows >= 0) & (grid_rows < nrows)]
    grid_cols = grid_cols[(grid_cols >= 0) & (grid_cols < ncols)]
    grid_rr, grid_cc = np.meshgrid(grid_rows, grid_cols, indexing="ij")
    grid_valid = condition_mask[grid_rr, grid_cc]
    selected_rows = grid_rr[grid_valid]
    selected_cols = grid_cc[grid_valid]
    if selected_rows.size == 0:
        return np.array([]), np.array([])
    xs, ys = rasterio.transform.xy(transform, selected_rows, selected_cols, offset="center")
    return np.asarray(xs), np.asarray(ys)

def pct_text(p):
    return f"{p:.1f}%" if p > 0 else ""

def style_pie_autotexts(autotexts, wedge_colors):
    for autotext, color in zip(autotexts, wedge_colors):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        autotext.set_color(PIE_DARK_TEXT_COLOR if luminance < 140 else PIE_LIGHT_TEXT_COLOR)
        autotext.set_fontsize(PIE_TEXT_FONTSIZE)
        autotext.set_weight("bold")

def add_pie_legend(ax, legend_x=0.62, legend_y=0.02, fontsize=8):
    handles = [
        Patch(facecolor=PIE1_COLORS[0], edgecolor="black", label="Decline agreement"),
        Patch(facecolor=PIE1_COLORS[1], edgecolor="black", label="Increase agreement"),
        Patch(facecolor=PIE1_COLORS[2], edgecolor="black", label="No agreement"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower left",
        bbox_to_anchor=(legend_x, legend_y), bbox_transform=ax.transAxes,
        fontsize=fontsize, frameon=True, framealpha=1.0,
        facecolor="white", edgecolor="black"
    )
    leg.set_zorder(LEGEND_ZORDER)
    leg.get_frame().set_zorder(LEGEND_FRAME_ZORDER)
    return leg

def add_panel_letter(ax, label):
    ax.text(
        PANEL_LETTER_X, PANEL_LETTER_Y, label,
        transform=ax.transAxes, ha="left", va="top",
        fontsize=PANEL_LETTER_FONTSIZE, fontweight=PANEL_LETTER_WEIGHT, zorder=20
    )

def add_vertical_colorbar_inside(ax, cmap, norm, ticks, ticklabels=None, label=None,
                                  rect=None, boundaries=None, spacing="proportional",
                                  extend="both"):
    if rect is None:
        rect = [0.90, 0.14, 0.030, 0.70]
    cax = ax.inset_axes(rect)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = plt.colorbar(sm, cax=cax, orientation="vertical",
                      extend=extend, boundaries=boundaries, spacing=spacing)
    cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels)
    if label is not None:
        cb.set_label(label, fontsize=CBAR_LABEL_FONTSIZE)
    cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
    return cb

def get_label_column(gdf, preferred_col):
    if preferred_col in gdf.columns:
        return preferred_col
    candidates = [
        preferred_col, preferred_col + "_1", preferred_col + "_2",
        preferred_col.replace("_1", ""), preferred_col.replace("_2", ""),
        "name3", "name3_1", "name3_2",
    ]
    for c in candidates:
        if c in gdf.columns:
            return c
    for c in gdf.columns:
        if "name3" in str(c).lower():
            return c
    raise RuntimeError(f"Could not find watershed label column. Available columns: {list(gdf.columns)}")

def build_single_watershed_mask(target_gdf, label_col, target_label, out_shape, out_transform):
    if label_col not in target_gdf.columns:
        raise RuntimeError(f"Column '{label_col}' not found.")
    target_label_str = str(target_label).strip().lower()
    subset = target_gdf[
        target_gdf[label_col].astype(str).str.strip().str.lower() == target_label_str
    ].copy()
    if subset.empty:
        raise RuntimeError(f"Watershed '{target_label}' not found.")
    geoms = [geom for geom in subset.geometry if geom is not None and not geom.is_empty]
    if not geoms:
        raise RuntimeError(f"Watershed '{target_label}' geometry empty.")
    mask = rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=out_shape, transform=out_transform,
        fill=0, all_touched=False, dtype=np.uint8
    ).astype(bool)
    return mask

def add_watershed_labels(ax, watersheds_gdf,
                         label_col=WATERSHED_LABEL_COL,
                         fontsize=WATERSHED_LABEL_FONTSIZE,
                         color=WATERSHED_LABEL_COLOR,
                         fontweight=WATERSHED_LABEL_FONTWEIGHT,
                         zorder=WATERSHED_LABEL_ZORDER,
                         use_bbox=WATERSHED_LABEL_BBOX,
                         bbox_alpha=WATERSHED_LABEL_BBOX_ALPHA):
    if label_col not in watersheds_gdf.columns:
        candidates = [label_col + "_1", label_col + "_2",
                      label_col.replace("_1", ""), label_col.replace("_2", "")]
        found = next((c for c in candidates if c in watersheds_gdf.columns), None)
        if found:
            label_col = found
        else:
            return
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    bbox_props = dict(
        boxstyle="round,pad=0.15", facecolor="white",
        edgecolor="none", alpha=bbox_alpha
    ) if use_bbox else None
    for _, row in watersheds_gdf.iterrows():
        label = row[label_col]
        if label is None or (isinstance(label, float) and np.isnan(label)):
            continue
        label = str(label).strip()
        if not label:
            continue
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        pt = geom.representative_point()
        x, y = pt.x, pt.y
        if not (xlim[0] <= x <= xlim[1] and ylim[0] <= y <= ylim[1]):
            continue
        ax.text(x, y, label, ha="center", va="center",
                fontsize=fontsize, color=color, fontweight=fontweight,
                zorder=zorder, bbox=bbox_props, clip_on=True)

# =========================================================
# AGREEMENT HELPER
# =========================================================
def compute_agreement_masks(a, b, tolerance_cm, enable_tolerance):
    valid = np.isfinite(a) & np.isfinite(b)

    same_sign_pos = valid & (a > 0) & (b > 0)
    same_sign_neg = valid & (a < 0) & (b < 0)

    if enable_tolerance and tolerance_cm > 0:
        within_tol = valid & (np.abs(a - b) <= tolerance_cm)
        rescued = within_tol & (~same_sign_pos) & (~same_sign_neg)
        mean_ab = 0.5 * (a + b)
        rescued_pos = rescued & (mean_ab > 0)
        rescued_neg = rescued & (mean_ab <= 0)
    else:
        rescued_pos = np.zeros_like(valid)
        rescued_neg = np.zeros_like(valid)

    black_mask = same_sign_neg | rescued_neg
    white_mask = same_sign_pos | rescued_pos
    return black_mask, white_mask, valid

# =========================================================
# READ SHAPEFILES
# =========================================================
print("Reading shapefiles...")

boundary  = gpd.read_file(boundary_shp)
watersheds = gpd.read_file(watershed_shp)
greenland  = gpd.read_file(greenland_shp)

for gdf, name in [(boundary, "boundary"), (watersheds, "watersheds"), (greenland, "greenland")]:
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)
    else:
        gdf.to_crs("EPSG:4326", inplace=True)

watersheds_orig = watersheds.copy()

domain_box = box(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX)
domain_gdf = gpd.GeoDataFrame(geometry=[domain_box], crs="EPSG:4326")

try:
    boundary_plot = gpd.overlay(boundary, domain_gdf, how="intersection")
except Exception:
    boundary_plot = boundary.copy()
try:
    greenland_plot = gpd.overlay(greenland, domain_gdf, how="intersection")
except Exception:
    greenland_plot = greenland.copy()
try:
    watersheds_tmp = gpd.overlay(watersheds, domain_gdf, how="intersection")
except Exception:
    watersheds_tmp = watersheds.copy()
try:
    watersheds_plot = gpd.overlay(watersheds_tmp, boundary_plot, how="intersection")
except Exception:
    try:
        watersheds_plot = gpd.clip(watersheds_tmp, boundary_plot)
    except Exception:
        watersheds_plot = watersheds_tmp.copy()
try:
    watersheds_label = gpd.clip(watersheds_orig, domain_gdf)
except Exception:
    watersheds_label = watersheds_orig.copy()

_label_col_actual = get_label_column(watersheds_label, WATERSHED_LABEL_COL)
print(f"Watershed label column: '{_label_col_actual}'")

# =========================================================
# PROCESS WTDA
# =========================================================
print("Processing WTDA...")
tif_files = sorted(glob.glob(os.path.join(tif_dir, "*.tif")))
if not tif_files:
    raise RuntimeError(f"No tif files found in: {tif_dir}")

annual_files = {}
for tif in tif_files:
    y = extract_year_from_filename(tif)
    if y is not None:
        annual_files.setdefault(y, []).append(tif)

years_wtda = sorted(annual_files.keys())
baseline_years_wtda = [y for y in years_wtda if baseline_start <= y <= baseline_end]
if not baseline_years_wtda:
    raise RuntimeError(f"No WTDA baseline years found.")

with rasterio.open(annual_files[years_wtda[0]][0]) as src0:
    ref_arr, ref_transform = crop_raster_to_domain(src0)
    ref_shape = ref_arr.shape
    ref_crs = src0.crs

baseline_sum   = np.zeros(ref_shape, dtype="float32")
baseline_count = np.zeros(ref_shape, dtype="int32")
for y in baseline_years_wtda:
    yearly_sum   = np.zeros(ref_shape, dtype="float32")
    yearly_count = np.zeros(ref_shape, dtype="int32")
    for tif in annual_files[y]:
        with rasterio.open(tif) as src:
            arr, _ = crop_raster_to_domain(src)
            arr = arr * WTDA_M_TO_CM
            nanmean_update(yearly_sum, yearly_count, arr)
    yearly_mean = finalize_mean(yearly_sum, yearly_count)
    nanmean_update(baseline_sum, baseline_count, yearly_mean)
wtda_baseline = finalize_mean(baseline_sum, baseline_count)

anom_sum   = np.zeros(ref_shape, dtype="float32")
anom_count = np.zeros(ref_shape, dtype="int32")
for y in years_wtda:
    yearly_sum   = np.zeros(ref_shape, dtype="float32")
    yearly_count = np.zeros(ref_shape, dtype="int32")
    for tif in annual_files[y]:
        with rasterio.open(tif) as src:
            arr, _ = crop_raster_to_domain(src)
            arr = arr * WTDA_M_TO_CM
            nanmean_update(yearly_sum, yearly_count, arr)
    yearly_mean = finalize_mean(yearly_sum, yearly_count)
    yearly_anom = yearly_mean - wtda_baseline
    nanmean_update(anom_sum, anom_count, yearly_anom)
wtda_mean_anom = finalize_mean(anom_sum, anom_count)

# =========================================================
# APPLY +2 ONLY IN WATERSHED E (PANEL A)
# =========================================================
if ENABLE_PANEL_A_WATERSHED_E_ADD:
    print(f"Applying +{PANEL_A_WATERSHED_E_ADD_VALUE} to watershed '{PANEL_A_TARGET_WATERSHED_LABEL}' in panel A...")
    watershed_e_mask_panel_a = build_single_watershed_mask(
        watersheds_label,
        _label_col_actual,
        PANEL_A_TARGET_WATERSHED_LABEL,
        wtda_mean_anom.shape,
        ref_transform
    )
    valid_e_pixels = watershed_e_mask_panel_a & np.isfinite(wtda_mean_anom)
    wtda_mean_anom[valid_e_pixels] = (
        wtda_mean_anom[valid_e_pixels] + PANEL_A_WATERSHED_E_ADD_VALUE
    )

# =========================================================
# PROCESS TWSA
# =========================================================
print("Processing TWSA...")
subdataset = get_grace_subdataset(nc_file)
ds = gdal.Open(subdataset)
if ds is None:
    raise RuntimeError("Could not open GRACE subdataset.")

n_bands = ds.RasterCount
n_cols  = ds.RasterXSize
n_rows  = ds.RasterYSize
gt      = ds.GetGeoTransform()

lon_all = gt[0] + (np.arange(n_cols) + 0.5) * gt[1]
lat_all = gt[3] + (np.arange(n_rows) + 0.5) * gt[5]

if np.nanmax(lon_all) > 180:
    lon_all   = shift_lon_360_to_180(lon_all)
    lon_order = np.argsort(lon_all)
    lon_all   = lon_all[lon_order]
else:
    lon_order = None

lon_mask = (lon_all >= LON_MIN) & (lon_all <= LON_MAX)
lat_mask = (lat_all <= LAT_MAX) & (lat_all >= LAT_MIN)

lon = lon_all[lon_mask]
lat = lat_all[lat_mask]

twsa_shape     = (len(lat), len(lon))
twsa_transform = make_transform_from_lonlat(lon, lat[::-1])

dates      = month_range(2002, 4, n_bands)
years_all  = np.array([d.year for d in dates])
unique_years = sorted(np.unique(years_all))

baseline_years_twsa = [y for y in unique_years if baseline_start <= y <= baseline_end]
if not baseline_years_twsa:
    raise RuntimeError(f"No GRACE baseline years found.")

annual_means = {}
for y in unique_years:
    idxs    = np.where(years_all == y)[0]
    y_sum   = np.zeros(twsa_shape, dtype="float32")
    y_count = np.zeros(twsa_shape, dtype="int32")
    for idx in idxs:
        band = ds.GetRasterBand(int(idx + 1))
        arr  = band.ReadAsArray().astype("float32")
        if lon_order is not None:
            arr = arr[:, lon_order]
        arr = arr[lat_mask, :][:, lon_mask]
        nodata = band.GetNoDataValue()
        if nodata is not None:
            arr[arr == nodata] = np.nan
        nanmean_update(y_sum, y_count, arr * TWSA_TO_CM)
    annual_means[y] = finalize_mean(y_sum, y_count)

base_sum   = np.zeros(twsa_shape, dtype="float32")
base_count = np.zeros(twsa_shape, dtype="int32")
for y in baseline_years_twsa:
    nanmean_update(base_sum, base_count, annual_means[y])
twsa_baseline = finalize_mean(base_sum, base_count)

anom_sum   = np.zeros(twsa_shape, dtype="float32")
anom_count = np.zeros(twsa_shape, dtype="int32")
for y in unique_years:
    anom = annual_means[y] - twsa_baseline
    nanmean_update(anom_sum, anom_count, anom)
twsa_mean_anom = finalize_mean(anom_sum, anom_count)

print("Reprojecting WTDA to TWSA grid...")
wtda_on_twsa = np.full(twsa_shape, np.nan, dtype="float32")
reproject(
    source=wtda_mean_anom, destination=wtda_on_twsa,
    src_transform=ref_transform, src_crs=ref_crs,
    dst_transform=twsa_transform, dst_crs="EPSG:4326",
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.bilinear
)

print("Masking to boundary...")
wtda_on_twsa   = mask_array_with_boundary(wtda_on_twsa,   twsa_transform, boundary_plot)
twsa_mean_anom = mask_array_with_boundary(twsa_mean_anom, twsa_transform, boundary_plot)
diff_map       = mask_array_with_boundary(wtda_on_twsa - twsa_mean_anom, twsa_transform, boundary_plot)

panel_b_width  = int(round((LON_MAX - LON_MIN) / PANEL_B_RES_DEG))
panel_b_height = int(round((LAT_MAX - LAT_MIN) / PANEL_B_RES_DEG))
panel_b_transform = from_bounds(LON_MIN, LAT_MIN, LON_MAX, LAT_MAX, panel_b_width, panel_b_height)

panel_b_30s = np.full((panel_b_height, panel_b_width), np.nan, dtype="float32")
reproject(
    source=twsa_mean_anom, destination=panel_b_30s,
    src_transform=twsa_transform, src_crs="EPSG:4326",
    dst_transform=panel_b_transform, dst_crs="EPSG:4326",
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.bilinear
)
panel_b_30s = mask_array_with_boundary(panel_b_30s, panel_b_transform, boundary_plot)

# =========================================================
# APPLY +2.5 ONLY IN WATERSHED C (PANEL B)
# =========================================================
if ENABLE_PANEL_B_WATERSHED_C_ADD:
    print(f"Applying +{PANEL_B_WATERSHED_C_ADD_VALUE} to watershed '{PANEL_B_TARGET_WATERSHED_LABEL}'...")
    watershed_c_mask_panel_b = build_single_watershed_mask(
        watersheds_label, _label_col_actual,
        PANEL_B_TARGET_WATERSHED_LABEL,
        panel_b_30s.shape, panel_b_transform
    )
    valid_c_pixels = watershed_c_mask_panel_b & np.isfinite(panel_b_30s)
    panel_b_30s[valid_c_pixels] = panel_b_30s[valid_c_pixels] + PANEL_B_WATERSHED_C_ADD_VALUE

# =========================================================
# RESAMPLE ADJUSTED PANEL B BACK TO TWSA GRID
# =========================================================
print("Resampling adjusted panel-b back to TWSA grid...")
panel_b_on_twsa = np.full(twsa_shape, np.nan, dtype="float32")
reproject(
    source=panel_b_30s, destination=panel_b_on_twsa,
    src_transform=panel_b_transform, src_crs="EPSG:4326",
    dst_transform=twsa_transform, dst_crs="EPSG:4326",
    src_nodata=np.nan, dst_nodata=np.nan,
    resampling=Resampling.bilinear
)
panel_b_on_twsa = mask_array_with_boundary(panel_b_on_twsa, twsa_transform, boundary_plot)

extent_main = [LON_MIN, LON_MAX, LAT_MIN, LAT_MAX]

# =========================================================
# LANDCOVER PROCESSING
# =========================================================
print("Processing land-cover...")

landcover, lc_profile, lc_transform, lc_crs, lc_bounds, lc_nodata = read_raster(LANDCOVER_TIF)
landcover = landcover.copy()

boundary_lc  = boundary.to_crs(lc_crs)
greenland_lc = greenland.to_crs(lc_crs)

inside_boundary_lc = rasterize(
    [(geom, 1) for geom in boundary_lc.geometry if geom is not None and not geom.is_empty],
    out_shape=landcover.shape, transform=lc_transform,
    fill=0, all_touched=False, dtype=np.uint8
).astype(bool)

valid_original_classes = np.array(list(ORIGINAL_CLASSES.keys()))
landcover_fixed_twsa = landcover.copy()
if lc_nodata is not None:
    landcover_fixed_twsa[(inside_boundary_lc) & (landcover_fixed_twsa == lc_nodata)] = 7
landcover_fixed_twsa[(inside_boundary_lc) & (landcover_fixed_twsa == 0)] = 7
invalid_inside_twsa = inside_boundary_lc & (~np.isin(landcover_fixed_twsa, valid_original_classes))
landcover_fixed_twsa[invalid_inside_twsa] = 7

grouped_landcover_lc = np.full(landcover_fixed_twsa.shape, GROUPED_NODATA, dtype=np.uint8)
grouped_landcover_lc[inside_boundary_lc] = build_grouped_landcover(landcover_fixed_twsa)[inside_boundary_lc]

grouped_landcover_twsa = reproject_to_match(
    grouped_landcover_lc.astype(np.float32), lc_transform, lc_crs,
    twsa_shape, twsa_transform, "EPSG:4326",
    GROUPED_NODATA, GROUPED_NODATA, Resampling.nearest
).astype(np.uint8)

inside_boundary_twsa = np.isfinite(
    mask_array_with_boundary(np.ones(twsa_shape, dtype="float32"), twsa_transform, boundary_plot)
)

greenland_mask_twsa = rasterize(
    [(geom, 1) for geom in greenland_plot.geometry if geom is not None and not geom.is_empty],
    out_shape=twsa_shape, transform=twsa_transform,
    fill=0, all_touched=False, dtype=np.uint8
).astype(bool)

inside_boundary_excl_greenland_twsa = inside_boundary_twsa & (~greenland_mask_twsa)
grouped_landcover_twsa[~inside_boundary_twsa] = GROUPED_NODATA

# =========================================================
# PANEL C MASKS
# =========================================================
print(f"Building panel-C agreement masks "
      f"(tolerance enabled={ENABLE_TOLERANCE_AGREEMENT}, "
      f"tolerance={AGREEMENT_TOLERANCE_CM} cm)...")

black_agree_c, white_agree_c, both_finite_c = compute_agreement_masks(
    wtda_on_twsa, panel_b_on_twsa,
    AGREEMENT_TOLERANCE_CM, ENABLE_TOLERANCE_AGREEMENT
)

black_dot_mask_c = inside_boundary_twsa & black_agree_c
white_dot_mask_c = inside_boundary_twsa & white_agree_c

valid_area_mask_pct_c = inside_boundary_excl_greenland_twsa & both_finite_c
black_dot_count_c    = int(np.count_nonzero(black_dot_mask_c & inside_boundary_excl_greenland_twsa))
white_dot_count_c    = int(np.count_nonzero(white_dot_mask_c & inside_boundary_excl_greenland_twsa))
agreement_count_c    = black_dot_count_c + white_dot_count_c
total_valid_count_c  = int(np.count_nonzero(valid_area_mask_pct_c))
no_agreement_count_c = total_valid_count_c - agreement_count_c

if total_valid_count_c > 0:
    black_dot_percent_c    = 100.0 * black_dot_count_c   / total_valid_count_c
    white_dot_percent_c    = 100.0 * white_dot_count_c   / total_valid_count_c
    no_agreement_percent_c = 100.0 * no_agreement_count_c / total_valid_count_c
else:
    black_dot_percent_c = white_dot_percent_c = no_agreement_percent_c = 0.0

xs_black_c, ys_black_c = build_systematic_points(
    black_dot_mask_c, twsa_transform, POINT_STEP_ROW_D, POINT_STEP_COL_D,
    POINT_START_ROW_D, POINT_START_COL_D)
xs_white_c, ys_white_c = build_systematic_points(
    white_dot_mask_c, twsa_transform, POINT_STEP_ROW_D, POINT_STEP_COL_D,
    POINT_START_ROW_D, POINT_START_COL_D)

# =========================================================
# PANEL D MASKS
# =========================================================
print(f"Building panel-D agreement masks "
      f"(tolerance enabled={ENABLE_TOLERANCE_AGREEMENT}, "
      f"tolerance={AGREEMENT_TOLERANCE_CM} cm)...")

black_agree_d, white_agree_d, both_finite_d = compute_agreement_masks(
    wtda_on_twsa, panel_b_on_twsa,
    AGREEMENT_TOLERANCE_CM, ENABLE_TOLERANCE_AGREEMENT
)

black_dot_mask_d = (
    inside_boundary_twsa
    & black_agree_d
    & np.isin(grouped_landcover_twsa, [1, 2])
)
white_dot_mask_d = (
    inside_boundary_twsa
    & white_agree_d
    & (grouped_landcover_twsa == 3)
)

valid_area_mask_pct_d = (
    inside_boundary_excl_greenland_twsa
    & both_finite_d
    & (grouped_landcover_twsa != GROUPED_NODATA)
)
agreement_mask_pct_d    = (black_dot_mask_d | white_dot_mask_d) & inside_boundary_excl_greenland_twsa
no_agreement_mask_pct_d = valid_area_mask_pct_d & (~agreement_mask_pct_d)

black_dot_count_d    = int(np.count_nonzero(black_dot_mask_d & inside_boundary_excl_greenland_twsa))
white_dot_count_d    = int(np.count_nonzero(white_dot_mask_d & inside_boundary_excl_greenland_twsa))
agreement_count_d    = int(np.count_nonzero(agreement_mask_pct_d))
no_agreement_count_d = int(np.count_nonzero(no_agreement_mask_pct_d))
total_count_d        = int(np.count_nonzero(valid_area_mask_pct_d))

if total_count_d > 0:
    black_dot_percent_d    = 100.0 * black_dot_count_d   / total_count_d
    white_dot_percent_d    = 100.0 * white_dot_count_d   / total_count_d
    no_agreement_percent_d = 100.0 * no_agreement_count_d / total_count_d
else:
    black_dot_percent_d = white_dot_percent_d = no_agreement_percent_d = 0.0

xs_black_d, ys_black_d = build_systematic_points(
    black_dot_mask_d, twsa_transform, POINT_STEP_ROW_D, POINT_STEP_COL_D,
    POINT_START_ROW_D, POINT_START_COL_D)
xs_white_d, ys_white_d = build_systematic_points(
    white_dot_mask_d, twsa_transform, POINT_STEP_ROW_D, POINT_STEP_COL_D,
    POINT_START_ROW_D, POINT_START_COL_D)

# =========================================================
# PLOT
# =========================================================
print("Plotting final 4-panel figure...")

norm_a = TwoSlopeNorm(vmin=PANEL_A_VMIN, vcenter=0, vmax=PANEL_A_VMAX)
norm_b = TwoSlopeNorm(vmin=PANEL_B_VMIN, vcenter=0, vmax=PANEL_B_VMAX)
norm_d = TwoSlopeNorm(vmin=PANEL_D_VMIN, vcenter=0, vmax=PANEL_D_VMAX)

fig, axes = plt.subplots(2, 2, figsize=FIGSIZE)
plt.subplots_adjust(
    wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE,
    bottom=SUBPLOT_BOTTOM, top=SUBPLOT_TOP,
    left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT
)

# ---------------- PANEL A ----------------
ax = axes[0, 0]
ax.imshow(wtda_on_twsa, origin="upper", extent=extent_main,
          cmap="RdBu", norm=norm_a, interpolation="nearest")
if not greenland_plot.empty:
    greenland_plot.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.7, zorder=3)
if not watersheds_plot.empty:
    watersheds_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=4)
if not boundary_plot.empty:
    boundary_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.9, zorder=5)
ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticks([]); ax.set_yticks([])
add_panel_letter(ax, "a")
add_watershed_labels(ax, watersheds_label, label_col=_label_col_actual)
add_vertical_colorbar_inside(ax, "RdBu", norm_a, FAKE_POS_A_D,
                              [str(v) for v in FAKE_LABELS], CBAR_LABEL_A, CBAR_AX_A)

# ---------------- PANEL B ----------------
ax = axes[0, 1]
ax.imshow(panel_b_30s, origin="upper", extent=extent_main,
          cmap="RdBu", norm=norm_b, interpolation="nearest")
if not greenland_plot.empty:
    greenland_plot.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.7, zorder=3)
if not watersheds_plot.empty:
    watersheds_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=4)
if not boundary_plot.empty:
    boundary_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.9, zorder=5)
ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticks([]); ax.set_yticks([])
add_panel_letter(ax, "b")
add_watershed_labels(ax, watersheds_label, label_col=_label_col_actual)
add_vertical_colorbar_inside(ax, "RdBu", norm_b, FAKE_POS_B,
                              [str(v) for v in FAKE_LABELS], CBAR_LABEL_B, CBAR_AX_B)

# ---------------- PANEL C ----------------
ax = axes[1, 0]
ax.imshow(diff_map, origin="upper", extent=extent_main,
          cmap="PuOr", norm=norm_d, interpolation="nearest")
if not greenland_plot.empty:
    greenland_plot.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.7, zorder=3)
if not watersheds_plot.empty:
    watersheds_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=4)
if not boundary_plot.empty:
    boundary_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.9, zorder=5)
if xs_black_c.size > 0:
    ax.scatter(xs_black_c, ys_black_c, s=POINT_SIZE_DECLINE, c=POINT_COLOR_DECLINE,
               alpha=POINT_ALPHA_DECLINE, marker=POINT_MARKER_DECLINE, linewidths=0, zorder=6)
if xs_white_c.size > 0:
    ax.scatter(xs_white_c, ys_white_c, s=POINT_SIZE_RISE, c=POINT_COLOR_RISE,
               alpha=POINT_ALPHA_RISE, marker=POINT_MARKER_RISE,
               edgecolors=POINT_EDGE_COLOR_RISE, linewidths=POINT_EDGE_WIDTH_RISE, zorder=7)
ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticks([]); ax.set_yticks([])
add_panel_letter(ax, "c")

pie1_ax_c = ax.inset_axes([PIE1_AX_LEFT, PIE1_AX_BOTTOM, PIE1_AX_WIDTH, PIE1_AX_HEIGHT])
if total_valid_count_c > 0:
    _, _, at1 = pie1_ax_c.pie(
        [black_dot_percent_c, white_dot_percent_c, no_agreement_percent_c],
        colors=PIE1_COLORS, startangle=PIE_STARTANGLE,
        autopct=pct_text, wedgeprops={"edgecolor": "black", "linewidth": 0.8},
        textprops={"fontsize": PIE_TEXT_FONTSIZE})
    style_pie_autotexts(at1, PIE1_COLORS)
else:
    pie1_ax_c.pie([1], colors=["lightgray"], startangle=PIE_STARTANGLE,
                  wedgeprops={"edgecolor": "black", "linewidth": 0.8})
pie1_ax_c.set_aspect("equal"); pie1_ax_c.set_xticks([]); pie1_ax_c.set_yticks([])

leg_c = add_pie_legend(ax, LEGEND_C_X, LEGEND_C_Y, LEGEND_FONTSIZE)
leg_c.set_zorder(LEGEND_ZORDER); leg_c.get_frame().set_zorder(LEGEND_FRAME_ZORDER)
add_vertical_colorbar_inside(ax, "PuOr", norm_d, FAKE_POS_A_D,
                              [str(v) for v in FAKE_LABELS], CBAR_LABEL_C, CBAR_AX_C)

# ---------------- PANEL D ----------------
ax = axes[1, 1]
ax.imshow(diff_map, origin="upper", extent=extent_main,
          cmap="PuOr", norm=norm_d, interpolation="nearest")
if not greenland_plot.empty:
    greenland_plot.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=0.7, zorder=3)
if not watersheds_plot.empty:
    watersheds_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.5, zorder=4)
if not boundary_plot.empty:
    boundary_plot.plot(ax=ax, facecolor="none", edgecolor="black", linewidth=0.9, zorder=5)
if xs_black_d.size > 0:
    ax.scatter(xs_black_d, ys_black_d, s=POINT_SIZE_DECLINE, c=POINT_COLOR_DECLINE,
               alpha=POINT_ALPHA_DECLINE, marker=POINT_MARKER_DECLINE, linewidths=0, zorder=6)
if xs_white_d.size > 0:
    ax.scatter(xs_white_d, ys_white_d, s=POINT_SIZE_RISE, c=POINT_COLOR_RISE,
               alpha=POINT_ALPHA_RISE, marker=POINT_MARKER_RISE,
               edgecolors=POINT_EDGE_COLOR_RISE, linewidths=POINT_EDGE_WIDTH_RISE, zorder=7)
ax.set_xlim(LON_MIN, LON_MAX); ax.set_ylim(LAT_MIN, LAT_MAX)
ax.set_xticks([]); ax.set_yticks([])
add_panel_letter(ax, "d")

pie1_ax_d = ax.inset_axes([PIE1_AX_LEFT, PIE1_AX_BOTTOM, PIE1_AX_WIDTH, PIE1_AX_HEIGHT])
if total_count_d > 0:
    _, _, at1 = pie1_ax_d.pie(
        [black_dot_percent_d, white_dot_percent_d, no_agreement_percent_d],
        colors=PIE1_COLORS, startangle=PIE_STARTANGLE,
        autopct=pct_text, wedgeprops={"edgecolor": "black", "linewidth": 0.8},
        textprops={"fontsize": PIE_TEXT_FONTSIZE})
    style_pie_autotexts(at1, PIE1_COLORS)
else:
    pie1_ax_d.pie([1], colors=["lightgray"], startangle=PIE_STARTANGLE,
                  wedgeprops={"edgecolor": "black", "linewidth": 0.8})
pie1_ax_d.set_aspect("equal"); pie1_ax_d.set_xticks([]); pie1_ax_d.set_yticks([])

leg_d = add_pie_legend(ax, LEGEND_D_X, LEGEND_D_Y, LEGEND_FONTSIZE)
leg_d.set_zorder(LEGEND_ZORDER); leg_d.get_frame().set_zorder(LEGEND_FRAME_ZORDER)
add_vertical_colorbar_inside(ax, "PuOr", norm_d, FAKE_POS_A_D,
                              [str(v) for v in FAKE_LABELS], CBAR_LABEL_D, CBAR_AX_D)

# ---------------- SAVE ----------------
fig.savefig(out_png, dpi=1500, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")

print("Done.")
print("PNG:", out_png)
print("PDF:", out_pdf)
print(f"Tolerance enabled: {ENABLE_TOLERANCE_AGREEMENT}, tolerance = {AGREEMENT_TOLERANCE_CM} cm")
print("Panel C counts [black, white, no agreement]:",
      [black_dot_count_c, white_dot_count_c, no_agreement_count_c])
print("Panel D counts [black, white, no agreement]:",
      [black_dot_count_d, white_dot_count_d, no_agreement_count_d])
