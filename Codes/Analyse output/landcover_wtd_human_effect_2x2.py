import os
import numpy as np
import rasterio
from rasterio.plot import plotting_extent
from rasterio.warp import reproject, Resampling
from rasterio.features import rasterize
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from matplotlib.colorbar import ColorbarBase
import geopandas as gpd

# ============================================================
# USER SETTINGS
# ============================================================

# ---------------- INPUT FILES ----------------
LANDCOVER_TIF = r"c:\Users\mhaghi2\Desktop\Article7\landcover\Land_Cover1.tif"

WTD_2000_TIF = r"c:\Users\mhaghi2\Desktop\1800-2015\N_America_002000_petsc_000000001.tif"
WTD_2025_TIF = r"c:\Users\mhaghi2\Desktop\1800-2015\N_America_002025_petsc_000000001.tif"

GREENLAND_SHP = r"c:\Users\mhaghi2\Desktop\data\1\shape\shape\greenland.shp"
WATERSHED_SHP = r"c:\Users\mhaghi2\Desktop\1\N_America_shapefile\N_America_level2_watershed_without_greenland.shp"
BOUNDARY_SHP  = r"c:\Users\mhaghi2\Desktop\1\N_America_shapefile\N_America_boundery_without_greenland.shp"

# ---------------- OUTPUT ----------------
OUTPUT_DIR = r"C:\Users\mhaghi2\Desktop\data\map_for_paper\watershed_map\final map\New folder"
OUTPUT_BASENAME = "landcover_wtd_human_effect_2x2"

SAVE_GROUPED_TIF = True
SAVE_DIFF_TIF = True

# ---------------- FIGURE ----------------
FIG_WIDTH = 18
FIG_HEIGHT = 12
FIG_DPI = 1500

PANEL_A_TITLE = "(a)"
PANEL_B_TITLE = "(b) "
PANEL_C_TITLE = "(c) "
PANEL_D_TITLE = ""

SHOW_AXIS_LABELS = True
SHOW_PLOT = True

# ============================================================
# DIFFERENCE SETTINGS
# ============================================================

# If WTD maps are in meters and you want cm:
WTD_TO_CM_FACTOR = 100.0

# diff = (WTD_2025 - WTD_2000) * factor
DECLINE_THRESHOLD_CM = 0.0   # points where diff < 0

# ============================================================
# POINT SETTINGS
# ============================================================

# Put points where class is 1 or 2 AND WTD decreases
POINT_CLASSES_TO_SHOW = [1, 2]

# Systematic grid spacing in raster rows/cols
# Larger values = fewer points
POINT_STEP_ROW = 160
POINT_STEP_COL = 160

# Grid start location
POINT_START_ROW = 0
POINT_START_COL = 0

# Point appearance
POINT_SIZE = 8
POINT_COLOR = "black"
POINT_ALPHA = 0.8

# ============================================================
# GROUPED CLASS DEFINITIONS
# ============================================================

HUMAN_CLASSES = [15, 17]
MIXED_CLASSES = [5, 6, 7, 8, 9, 10]
CLIMATE_CLASSES = [1, 2, 3, 4, 11, 12, 13, 14, 16, 18, 19]

GROUPED_NODATA = 0

# ============================================================
# SHAPE STYLE
# ============================================================

GREENLAND_EDGE_COLOR = "black"
GREENLAND_FACE_COLOR = "lightgray"
GREENLAND_LINEWIDTH = 0.8

WATERSHED_EDGE_COLOR = "black"
WATERSHED_FACE_COLOR = "none"
WATERSHED_LINEWIDTH = 0.4

BOUNDARY_EDGE_COLOR = "black"
BOUNDARY_FACE_COLOR = "none"
BOUNDARY_LINEWIDTH = 1.0

# ============================================================
# PANEL D LEGEND SETTINGS
# ============================================================

# Legend 1
LEGEND1_BBOX_X = 0.00
LEGEND1_BBOX_Y = 1.00
LEGEND1_LOC = "upper left"
LEGEND1_FONTSIZE = 8
LEGEND1_TITLE_FONTSIZE = 10
LEGEND1_NCOL = 2
LEGEND1_TITLE = "Panel (a): Original land-cover classes"

# Legend 2
LEGEND2_BBOX_X = 0.00
LEGEND2_BBOX_Y = 0.60
LEGEND2_LOC = "upper left"
LEGEND2_FONTSIZE = 10
LEGEND2_TITLE_FONTSIZE = 10
LEGEND2_NCOL = 1
LEGEND2_TITLE = "Panels (b) and (c)"

# ============================================================
# COLORBAR SETTINGS
# ============================================================

CBAR_LABEL = "Difference Water Table Depth (cm)"
CBAR_LABEL_FONTSIZE = 12
CBAR_TICK_FONTSIZE = 11

CBAR_TICKS = [-30, -20, -10, -3, 0, 3, 10, 20, 30]
CBAR_BOUNDARIES = [-30, -20, -10, -3, 0, 3, 10, 20, 30]

DIFF_CBAR_COLORS = [
    "#a50026",  # lower extension
    "#d73027",  # -30 to -20
    "#f46d43",  # -20 to -10
    "#fdae61",  # -10 to -3
    "#fee08b",  # -3 to 0
    "#e6e6e6",  # 0 to 3
    "#cfdad4",  # 3 to 10
    "#91bfdb",  # 10 to 20
    "#4575b4",  # 20 to 30
    "#313695",  # upper extension
]

# Colorbar location now in PANEL C
CBAR_AX_LEFT = 0.87
CBAR_AX_BOTTOM = 0.05
CBAR_AX_WIDTH = 0.035
CBAR_AX_HEIGHT = 0.70

# ============================================================
# ORIGINAL LAND-COVER DEFINITIONS
# ============================================================

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

GROUPED_COLORS = [
    "#d73027",
    "#fdae61",
    "#1a9850"
]

# ============================================================
# CREATE OUTPUT FOLDER
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

FIG_PNG = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + ".png")
FIG_PDF = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + ".pdf")
GROUPED_TIF = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + "_grouped_landcover.tif")
DIFF_TIF = os.path.join(OUTPUT_DIR, OUTPUT_BASENAME + "_wtd_diff_cm.tif")

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
        bounds = src.bounds
        nodata = src.nodata
    return arr, profile, transform, crs, bounds, nodata


def reproject_to_match(src_array, src_transform, src_crs, dst_shape, dst_transform, dst_crs,
                       src_nodata=None, dst_nodata=np.nan):
    dst_array = np.full(dst_shape, dst_nodata, dtype=np.float32)

    reproject(
        source=src_array,
        destination=dst_array,
        src_transform=src_transform,
        src_crs=src_crs,
        src_nodata=src_nodata,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        dst_nodata=dst_nodata,
        resampling=Resampling.bilinear
    )
    return dst_array


def build_grouped_landcover(landcover):
    grouped = np.full(landcover.shape, GROUPED_NODATA, dtype=np.uint8)

    grouped[np.isin(landcover, HUMAN_CLASSES)] = 1
    grouped[np.isin(landcover, MIXED_CLASSES)] = 2
    grouped[np.isin(landcover, CLIMATE_CLASSES)] = 3

    return grouped


def rgb255_to_mpl(rgb):
    return tuple(v / 255.0 for v in rgb)

# ============================================================
# READ LAND-COVER
# ============================================================

landcover, lc_profile, lc_transform, lc_crs, lc_bounds, lc_nodata = read_raster(LANDCOVER_TIF)
landcover = landcover.copy()

# ============================================================
# READ SHAPEFILES
# ============================================================

greenland_gdf = gpd.read_file(GREENLAND_SHP)
watershed_gdf = gpd.read_file(WATERSHED_SHP)
boundary_gdf = gpd.read_file(BOUNDARY_SHP)

if lc_crs is not None:
    if greenland_gdf.crs != lc_crs:
        greenland_gdf = greenland_gdf.to_crs(lc_crs)
    if watershed_gdf.crs != lc_crs:
        watershed_gdf = watershed_gdf.to_crs(lc_crs)
    if boundary_gdf.crs != lc_crs:
        boundary_gdf = boundary_gdf.to_crs(lc_crs)

watershed_clipped = gpd.clip(watershed_gdf, boundary_gdf)

# ============================================================
# BUILD BOUNDARY MASK
# ============================================================

boundary_mask = rasterize(
    [(geom, 1) for geom in boundary_gdf.geometry if geom is not None],
    out_shape=landcover.shape,
    transform=lc_transform,
    fill=0,
    all_touched=False,
    dtype=np.uint8
).astype(bool)

# ============================================================
# FIX LAND-COVER INSIDE BOUNDARY
# ============================================================

inside_boundary = boundary_mask.copy()

landcover_fixed = landcover.copy()

landcover_fixed[(inside_boundary) & (landcover_fixed == 0)] = 7

if lc_nodata is not None:
    landcover_fixed[(inside_boundary) & (landcover_fixed == lc_nodata)] = 7

valid_original_classes = np.array(list(ORIGINAL_CLASSES.keys()))
invalid_inside = inside_boundary & (~np.isin(landcover_fixed, valid_original_classes))
landcover_fixed[invalid_inside] = 7

# ============================================================
# BUILD GROUPED LAND-COVER ONLY FOR BOUNDARY AREA
# ============================================================

grouped_landcover = np.full(landcover_fixed.shape, GROUPED_NODATA, dtype=np.uint8)
grouped_landcover[inside_boundary] = build_grouped_landcover(landcover_fixed)[inside_boundary]

if SAVE_GROUPED_TIF:
    grouped_profile = lc_profile.copy()
    grouped_profile.update(
        dtype=rasterio.uint8,
        nodata=GROUPED_NODATA,
        count=1,
        compress="lzw"
    )
    with rasterio.open(GROUPED_TIF, "w", **grouped_profile) as dst:
        dst.write(grouped_landcover, 1)

# ============================================================
# READ WTD MAPS
# ============================================================

wtd_2000, _, wtd2000_transform, wtd2000_crs, _, wtd2000_nodata = read_raster(WTD_2000_TIF)
wtd_2025, _, wtd2025_transform, wtd2025_crs, _, wtd2025_nodata = read_raster(WTD_2025_TIF)

wtd_2000 = wtd_2000.astype(np.float32)
wtd_2025 = wtd_2025.astype(np.float32)

if wtd2000_nodata is not None:
    wtd_2000[wtd_2000 == wtd2000_nodata] = np.nan
if wtd2025_nodata is not None:
    wtd_2025[wtd_2025 == wtd2025_nodata] = np.nan

if (wtd2000_crs != lc_crs) or (wtd2000_transform != lc_transform) or (wtd_2000.shape != landcover.shape):
    wtd_2000 = reproject_to_match(
        src_array=wtd_2000,
        src_transform=wtd2000_transform,
        src_crs=wtd2000_crs,
        dst_shape=landcover.shape,
        dst_transform=lc_transform,
        dst_crs=lc_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )

if (wtd2025_crs != lc_crs) or (wtd2025_transform != lc_transform) or (wtd_2025.shape != landcover.shape):
    wtd_2025 = reproject_to_match(
        src_array=wtd_2025,
        src_transform=wtd2025_transform,
        src_crs=wtd2025_crs,
        dst_shape=landcover.shape,
        dst_transform=lc_transform,
        dst_crs=lc_crs,
        src_nodata=np.nan,
        dst_nodata=np.nan
    )

# ============================================================
# CALCULATE WTD DIFFERENCE ONLY INSIDE BOUNDARY
# ============================================================

wtd_diff_cm = (wtd_2025 - wtd_2000) * WTD_TO_CM_FACTOR
wtd_diff_cm[~inside_boundary] = np.nan

if SAVE_DIFF_TIF:
    diff_profile = lc_profile.copy()
    diff_profile.update(
        dtype=rasterio.float32,
        nodata=np.nan,
        count=1,
        compress="lzw"
    )
    with rasterio.open(DIFF_TIF, "w", **diff_profile) as dst:
        dst.write(wtd_diff_cm.astype(np.float32), 1)

# ============================================================
# PREPARE MAP ARRAYS
# ============================================================

original_colors = [rgb255_to_mpl(ORIGINAL_CLASSES[i][1]) for i in range(1, 20)]
original_cmap = ListedColormap(original_colors)
original_norm = BoundaryNorm(np.arange(0.5, 20.5, 1), original_cmap.N)

original_plot_arr = np.ma.masked_where(~inside_boundary, landcover_fixed)

grouped_plot_arr = np.ma.masked_where((grouped_landcover == GROUPED_NODATA) | (~inside_boundary), grouped_landcover)
grouped_cmap = ListedColormap(GROUPED_COLORS)
grouped_norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], grouped_cmap.N)

diff_cmap = ListedColormap(DIFF_CBAR_COLORS)
diff_norm = BoundaryNorm(CBAR_BOUNDARIES, diff_cmap.N, extend="both")
diff_plot_arr = np.ma.masked_invalid(wtd_diff_cm)

# ============================================================
# CREATE SYSTEMATIC GRID POINTS FOR CLASS 1 OR 2 + DECLINE
# ============================================================

point_condition = (
    inside_boundary
    & np.isin(grouped_landcover, POINT_CLASSES_TO_SHOW)
    & np.isfinite(wtd_diff_cm)
    & (wtd_diff_cm < DECLINE_THRESHOLD_CM)
)

nrows, ncols = grouped_landcover.shape

grid_rows = np.arange(POINT_START_ROW, nrows, POINT_STEP_ROW)
grid_cols = np.arange(POINT_START_COL, ncols, POINT_STEP_COL)

grid_rows = grid_rows[(grid_rows >= 0) & (grid_rows < nrows)]
grid_cols = grid_cols[(grid_cols >= 0) & (grid_cols < ncols)]

grid_rr, grid_cc = np.meshgrid(grid_rows, grid_cols, indexing="ij")
grid_valid = point_condition[grid_rr, grid_cc]

selected_rows = grid_rr[grid_valid]
selected_cols = grid_cc[grid_valid]

xs, ys = rasterio.transform.xy(lc_transform, selected_rows, selected_cols, offset="center")
xs = np.asarray(xs)
ys = np.asarray(ys)

# ============================================================
# PLOT FIGURE
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT))
extent = plotting_extent(landcover_fixed, lc_transform)

# ---------------- PANEL A ----------------
ax = axes[0, 0]
ax.imshow(
    original_plot_arr,
    cmap=original_cmap,
    norm=original_norm,
    extent=extent,
    origin="upper"
)

greenland_gdf.plot(ax=ax, facecolor=GREENLAND_FACE_COLOR, edgecolor=GREENLAND_EDGE_COLOR,
                   linewidth=GREENLAND_LINEWIDTH, zorder=3)
watershed_clipped.plot(ax=ax, facecolor=WATERSHED_FACE_COLOR, edgecolor=WATERSHED_EDGE_COLOR,
                       linewidth=WATERSHED_LINEWIDTH, zorder=4)
boundary_gdf.plot(ax=ax, facecolor=BOUNDARY_FACE_COLOR, edgecolor=BOUNDARY_EDGE_COLOR,
                  linewidth=BOUNDARY_LINEWIDTH, zorder=5)

ax.set_title(PANEL_A_TITLE, fontsize=13, pad=8)
ax.set_xlim(lc_bounds.left, lc_bounds.right)
ax.set_ylim(lc_bounds.bottom, lc_bounds.top)

if SHOW_AXIS_LABELS:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
else:
    ax.set_xticks([])
    ax.set_yticks([])

# ---------------- PANEL B ----------------
ax = axes[0, 1]
ax.imshow(
    grouped_plot_arr,
    cmap=grouped_cmap,
    norm=grouped_norm,
    extent=extent,
    origin="upper"
)

greenland_gdf.plot(ax=ax, facecolor=GREENLAND_FACE_COLOR, edgecolor=GREENLAND_EDGE_COLOR,
                   linewidth=GREENLAND_LINEWIDTH, zorder=3)
watershed_clipped.plot(ax=ax, facecolor=WATERSHED_FACE_COLOR, edgecolor=WATERSHED_EDGE_COLOR,
                       linewidth=WATERSHED_LINEWIDTH, zorder=4)
boundary_gdf.plot(ax=ax, facecolor=BOUNDARY_FACE_COLOR, edgecolor=BOUNDARY_EDGE_COLOR,
                  linewidth=BOUNDARY_LINEWIDTH, zorder=5)

ax.set_title(PANEL_B_TITLE, fontsize=13, pad=8)
ax.set_xlim(lc_bounds.left, lc_bounds.right)
ax.set_ylim(lc_bounds.bottom, lc_bounds.top)

if SHOW_AXIS_LABELS:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
else:
    ax.set_xticks([])
    ax.set_yticks([])

# ---------------- PANEL C ----------------
ax = axes[1, 0]
im_c = ax.imshow(
    diff_plot_arr,
    cmap=diff_cmap,
    norm=diff_norm,
    extent=extent,
    origin="upper"
)

greenland_gdf.plot(ax=ax, facecolor=GREENLAND_FACE_COLOR, edgecolor=GREENLAND_EDGE_COLOR,
                   linewidth=GREENLAND_LINEWIDTH, zorder=3)
watershed_clipped.plot(ax=ax, facecolor=WATERSHED_FACE_COLOR, edgecolor=WATERSHED_EDGE_COLOR,
                       linewidth=WATERSHED_LINEWIDTH, zorder=4)
boundary_gdf.plot(ax=ax, facecolor=BOUNDARY_FACE_COLOR, edgecolor=BOUNDARY_EDGE_COLOR,
                  linewidth=BOUNDARY_LINEWIDTH, zorder=5)

if xs.size > 0:
    ax.scatter(
        xs,
        ys,
        s=POINT_SIZE,
        c=POINT_COLOR,
        alpha=POINT_ALPHA,
        linewidths=0,
        zorder=6
    )

ax.set_title(PANEL_C_TITLE, fontsize=13, pad=8)
ax.set_xlim(lc_bounds.left, lc_bounds.right)
ax.set_ylim(lc_bounds.bottom, lc_bounds.top)

if SHOW_AXIS_LABELS:
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
else:
    ax.set_xticks([])
    ax.set_yticks([])

# ---- COLORBAR MOVED TO PANEL C ----
cax = ax.inset_axes([CBAR_AX_LEFT, CBAR_AX_BOTTOM, CBAR_AX_WIDTH, CBAR_AX_HEIGHT])

cb = ColorbarBase(
    cax,
    cmap=diff_cmap,
    norm=diff_norm,
    boundaries=CBAR_BOUNDARIES,
    ticks=CBAR_TICKS,
    spacing="proportional",
    orientation="vertical",
    extend="both"
)

cb.set_label(CBAR_LABEL, fontsize=CBAR_LABEL_FONTSIZE)
cb.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)

# ---------------- PANEL D ----------------
ax = axes[1, 1]
ax.set_title(PANEL_D_TITLE, fontsize=13, pad=8)
ax.axis("off")

original_handles = []
for class_value in range(1, 20):
    class_name, rgb = ORIGINAL_CLASSES[class_value]
    original_handles.append(
        Patch(facecolor=rgb255_to_mpl(rgb), edgecolor="black", label=f"{class_value}. {class_name}")
    )

legend1 = ax.legend(
    handles=original_handles,
    loc=LEGEND1_LOC,
    bbox_to_anchor=(LEGEND1_BBOX_X, LEGEND1_BBOX_Y),
    fontsize=LEGEND1_FONTSIZE,
    frameon=True,
    title=LEGEND1_TITLE,
    title_fontsize=LEGEND1_TITLE_FONTSIZE,
    ncol=LEGEND1_NCOL,
    borderaxespad=0.0
)
ax.add_artist(legend1)

grouped_handles = [
    Patch(facecolor=GROUPED_COLORS[0], edgecolor="black", label="1. Strong human effect"),
    Patch(facecolor=GROUPED_COLORS[1], edgecolor="black", label="2. Mixed human and climate effect"),
    Patch(facecolor=GROUPED_COLORS[2], edgecolor="black", label="3. Mostly climate-driven"),
    Patch(facecolor=GREENLAND_FACE_COLOR, edgecolor="black", label="Greenland"),
    Patch(facecolor="white", edgecolor="black", label="Watersheds"),
    Patch(facecolor="white", edgecolor="black", label="Boundary"),
    Patch(facecolor="black", edgecolor="black", label="Systematic grid point: class 1 or 2 + WTD decline")
]

legend2 = ax.legend(
    handles=grouped_handles,
    loc=LEGEND2_LOC,
    bbox_to_anchor=(LEGEND2_BBOX_X, LEGEND2_BBOX_Y),
    fontsize=LEGEND2_FONTSIZE,
    frameon=True,
    title=LEGEND2_TITLE,
    title_fontsize=LEGEND2_TITLE_FONTSIZE,
    ncol=LEGEND2_NCOL,
    borderaxespad=0.0
)
ax.add_artist(legend2)

# ============================================================
# SAVE FIGURE
# ============================================================

plt.tight_layout()
plt.savefig(FIG_PNG, dpi=FIG_DPI, bbox_inches="tight")
plt.savefig(FIG_PDF, dpi=FIG_DPI, bbox_inches="tight")

if SHOW_PLOT:
    plt.show()
else:
    plt.close()

# ============================================================
# PRINT INFO
# ============================================================

print("Done.")
print(f"Figure PNG saved to: {FIG_PNG}")
print(f"Figure PDF saved to: {FIG_PDF}")

if SAVE_GROUPED_TIF:
    print(f"Grouped land-cover TIFF saved to: {GROUPED_TIF}")

if SAVE_DIFF_TIF:
    print(f"WTD difference TIFF saved to: {DIFF_TIF}")

print("\nLand-cover fixing rules:")
print("1) Inside boundary, land-cover value 0 -> 4")
print("2) Inside boundary, land-cover nodata -> 4")
print("3) Outside boundary is excluded from calculation and plotting")

print("\nGrouped class values:")
print("1 = Strong human effect")
print("2 = Mixed human and climate effect")
print("3 = Mostly climate-driven")

print("\nSystematic point rule:")
print(f"Classes used for points = {POINT_CLASSES_TO_SHOW}")
print(f"WTD decline threshold = diff < {DECLINE_THRESHOLD_CM} cm")
print(f"Point grid step = every {POINT_STEP_ROW} rows and {POINT_STEP_COL} cols")
print(f"Point grid start = row {POINT_START_ROW}, col {POINT_START_COL}")
print(f"Point size = {POINT_SIZE}")