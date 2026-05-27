#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.lines import Line2D

# ============================================================
# USER OPTIONS (SIZES + FONTS)
# ============================================================
FIG_W = 18
FIG_H = 12

PANEL_A_HEIGHT_RATIO = 2.6
PANEL_B_HEIGHT_RATIO = 6

HSPACE = 0.15
WSPACE = 0.15

FS_PANEL_LABEL = 16
FS_AXIS_LABEL  = 12
FS_TICKS_A     = 12
FS_TICKS_B     = 8
FS_SUBLETTER   = 13
FS_LEGEND      = 10

LW_MAIN = 1.2
LW_A_SUB = 1.0

# ============================================================
# COLORS
# ============================================================
COLOR_TOTAL = "darkblue"     # Total SLE (all panels)
COLOR_TREND = "orange"       # Trend line (all panels)

# Panel A only: distributions
COLOR_GW = "blue"            # Groundwater distribution (Panel A only)
COLOR_SW = "black"           # Surface water distribution (Panel A only)

# Arrows
ARROW_ORANGE = "green"       # Recharge
ARROW_PURPLE = "red"         # Depletion

TREND_LS = "--"
TREND_LW = 2.0
TREND_ALPHA = 1.0

# ============================================================
# UNIT CONVERSION: cm -> mm
# ============================================================
CM_TO_MM = 10.0

# ============================================================
# TREND SLOPE CONTROL
# ============================================================
# Used ONLY when force_slope=True (Panel A + Panel B-a)
# units: mm/year
TREND_SLOPE_MM_PER_YEAR = -0.0005 * CM_TO_MM  # = -0.005 mm/year

# ============================================================
# TREND SLOPE % TEXT (ONLY Panel A)
# ============================================================
SHOW_TREND_PERCENT_TEXT = True

TREND_PCT_REF = "absmean"  # "mean", "first", "absmean"
TREND_TEXT_FMT = "Trend Line Slope: {pct:+.4f}%/yr"
TREND_TEXT_FONTSIZE = 11
TREND_TEXT_X = 0.03
TREND_TEXT_Y = 0.963

# ============================================================
# LEGEND POSITION + SPACING CONTROL
# ============================================================
LEGEND_X = 0.5
LEGEND_Y = 0.06

BOTTOM_MARGIN = 0.12
LEGEND_GAP = 0.02

# ============================================================
# PANEL-B Y-LABEL POSITION
# ============================================================
PANELB_YLABEL_X = 0.089
PANELB_YLABEL_Y = 0.37
PANELB_LEFT_MARGIN = 0.06

# ============================================================
# PANEL LABEL "B" POSITION
# ============================================================
B_LABEL_X = 0.02
B_LABEL_Y = 1.2

# ============================================================
# PATHS
# ============================================================
output_folder = "/home/mohammad/Desktop/1/11"

CSV_PANEL_A  = os.path.join(output_folder, "SLE_timeseries_PanelA_North_America_No_Greenland.csv")
CSV_PANEL_BA = os.path.join(output_folder, "SLE_timeseries_PanelB_a_boundary.csv")

# Outputs
FIG_OUT_PNG = os.path.join(output_folder, "Fig_Total_SLE_With_TrendM1.png")
FIG_OUT_PDF = os.path.join(output_folder, "Fig_Total_SLE_With_TrendM1.pdf")

# ============================================================
# EXPORT / "RESOLUTION" CONTROLS
# ============================================================
PNG_DPI = 1500
PDF_DPI = 1500
RASTERIZE_PDF = False

# ============================================================
# ARROWS (EDIT THESE)
# y and dy values are now in mm (converted from cm x10)
# ============================================================
ORANGE_ARROWS = [
    {"x": datetime(2000, 10, 20), "y": -0.164 * CM_TO_MM, "dx_days": 360*1.0, "dy": 0.054 * CM_TO_MM},
    {"x": datetime(2001, 11, 1),  "y": -0.112 * CM_TO_MM, "dx_days": 365*3,   "dy": 0.000 * CM_TO_MM},
    {"x": datetime(2004, 11, 1),  "y": -0.110 * CM_TO_MM, "dx_days": 325*1,   "dy": 0.037 * CM_TO_MM},
    {"x": datetime(2008, 10, 25), "y": -0.148 * CM_TO_MM, "dx_days": 340*1,   "dy": 0.045 * CM_TO_MM},
    {"x": datetime(2010, 9, 20),  "y": -0.145 * CM_TO_MM, "dx_days": 340*1,   "dy": 0.070 * CM_TO_MM},
    {"x": datetime(2014, 10, 20), "y": -0.168 * CM_TO_MM, "dx_days": 370*1,   "dy": 0.048 * CM_TO_MM},
    {"x": datetime(2016, 9, 25),  "y": -0.134 * CM_TO_MM, "dx_days": 370*1,   "dy": 0.028 * CM_TO_MM},
    {"x": datetime(2019, 10, 20), "y": -0.120 * CM_TO_MM, "dx_days": 371*2.9, "dy": 0.064 * CM_TO_MM},
    {"x": datetime(2024, 9, 20),  "y": -0.175 * CM_TO_MM, "dx_days": 365*1.0, "dy": 0.039 * CM_TO_MM},
]

PURPLE_ARROWS = [
    {"x": datetime(2000, 1, 1),   "y": -0.014 * CM_TO_MM, "dx_days": 260*1,    "dy": -0.156 * CM_TO_MM},
    {"x": datetime(2005, 9, 25),  "y": -0.076 * CM_TO_MM, "dx_days": 365*3.05, "dy": -0.074 * CM_TO_MM},
    {"x": datetime(2009, 10, 20), "y": -0.106 * CM_TO_MM, "dx_days": 315*1,    "dy": -0.042 * CM_TO_MM},
    {"x": datetime(2011, 9, 10),  "y": -0.077 * CM_TO_MM, "dx_days": 375*3,    "dy": -0.093 * CM_TO_MM},
    {"x": datetime(2015, 11, 1),  "y": -0.120 * CM_TO_MM, "dx_days": 320*1,    "dy": -0.015 * CM_TO_MM},
    {"x": datetime(2017, 10, 20), "y": -0.108 * CM_TO_MM, "dx_days": 365*1.95, "dy": -0.012 * CM_TO_MM},
    {"x": datetime(2022, 10, 15), "y": -0.058 * CM_TO_MM, "dx_days": 365*1.88, "dy": -0.119 * CM_TO_MM},
]

ARROW_LW = 2.0
ARROW_HEAD = 14
ARROW_ALPHA = 0.95

# ============================================================
# HELPERS
# ============================================================
def parse_date_label_to_dt(s):
    y, m = s.split("_")
    return datetime(int(y), int(m), 1)

def read_panelA_csv(csv_path):
    df = pd.read_csv(csv_path)
    required = ["date", "Total_SLE_cm", "Groundwater_SLE_cm", "SurfaceWater_SLE_cm"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")
    df["dt"] = df["date"].astype(str).apply(parse_date_label_to_dt)
    return (
        df["date"].astype(str).tolist(),
        df["dt"].tolist(),
        df["Total_SLE_cm"].to_numpy(dtype=float) * CM_TO_MM,
        df["Groundwater_SLE_cm"].to_numpy(dtype=float) * CM_TO_MM,
        df["SurfaceWater_SLE_cm"].to_numpy(dtype=float) * CM_TO_MM,
    )

def read_total_only_csv(csv_path):
    df = pd.read_csv(csv_path)
    if "date" not in df.columns or "Total_SLE_cm" not in df.columns:
        raise ValueError(f"Missing 'date' or 'Total_SLE_cm' in {csv_path}")
    df["dt"] = df["date"].astype(str).apply(parse_date_label_to_dt)
    return (
        df["date"].astype(str).tolist(),
        df["dt"].tolist(),
        df["Total_SLE_cm"].to_numpy(dtype=float) * CM_TO_MM,
    )

def add_trend_line(ax, x_dt, y, label_for_legend=None, force_slope=False):
    if len(x_dt) < 2:
        return None, None, None

    x_num = mdates.date2num(x_dt)  # days
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x_num) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None, None

    if force_slope:
        slope_mm_per_day = TREND_SLOPE_MM_PER_YEAR / 365.25
        x0 = np.mean(x_num[mask])
        y0 = np.mean(y[mask])
        y_fit = slope_mm_per_day * (x_num - x0) + y0
        slope_used_mm_per_year = TREND_SLOPE_MM_PER_YEAR
    else:
        a, b = np.polyfit(x_num[mask], y[mask], 1)
        y_fit = a * x_num + b
        slope_used_mm_per_year = a * 365.25

    line, = ax.plot(
        x_dt, y_fit,
        color=COLOR_TREND,
        linestyle=TREND_LS,
        linewidth=TREND_LW,
        alpha=TREND_ALPHA,
        label=label_for_legend if label_for_legend is not None else "_nolegend_"
    )
    return line, slope_used_mm_per_year, None

def slope_to_percent_per_year(slope_mm_per_year, y_series, ref_mode="absmean"):
    y = np.asarray(y_series, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return None

    if ref_mode == "mean":
        denom = np.mean(y)
    elif ref_mode == "first":
        denom = y[0]
    elif ref_mode == "absmean":
        denom = np.mean(np.abs(y))
    else:
        denom = np.mean(np.abs(y))

    if denom == 0 or not np.isfinite(denom):
        return None

    return (slope_mm_per_year / denom) * 100.0

def add_arrows(ax, arrows, color):
    for a in arrows:
        x0 = a["x"]
        y0 = a["y"]
        dx = a["dx_days"]
        dy = a["dy"]

        x1_num = mdates.date2num(x0) + float(dx)
        x1 = mdates.num2date(x1_num)

        ax.annotate(
            "", xy=(x1, y0 + dy), xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="-|>",
                color=color,
                lw=ARROW_LW,
                mutation_scale=ARROW_HEAD,
                alpha=ARROW_ALPHA,
                shrinkA=0,
                shrinkB=0,
            ),
            annotation_clip=True
        )

# ============================================================
# COLLECT PANEL B FILES IN ORDER a..l
# ============================================================
letters = list("abcdefghijkl")
panelB_files = {l: None for l in letters}

if os.path.exists(CSV_PANEL_BA):
    panelB_files["a"] = CSV_PANEL_BA
else:
    raise FileNotFoundError(f"Missing Panel B(a) CSV: {CSV_PANEL_BA}")

for l in letters[1:]:
    pattern = os.path.join(output_folder, f"SLE_timeseries_PanelB_{l}_*.csv")
    matches = sorted(glob.glob(pattern))
    if len(matches) == 0:
        raise FileNotFoundError(f"Missing CSV for PanelB_{l}. Expected pattern: {pattern}")
    panelB_files[l] = matches[0]

# ============================================================
# READ PANEL A
# ============================================================
if not os.path.exists(CSV_PANEL_A):
    raise FileNotFoundError(f"Missing Panel A CSV: {CSV_PANEL_A}")

A_lbl, A_dt, A_total, A_gw, A_sw = read_panelA_csv(CSV_PANEL_A)

# ============================================================
# BUILD FIGURE
# ============================================================
fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=150)

gs = fig.add_gridspec(
    nrows=2, ncols=1,
    height_ratios=[PANEL_A_HEIGHT_RATIO, PANEL_B_HEIGHT_RATIO],
    hspace=HSPACE
)

axA = fig.add_subplot(gs[0, 0])

gsB = gs[1, 0].subgridspec(4, 3, hspace=HSPACE, wspace=WSPACE)
axesB = [fig.add_subplot(gsB[r, c]) for r in range(4) for c in range(3)]

# ============================================================
# PANEL A
# ============================================================
line_total_A, = axA.plot(A_dt, A_total, color=COLOR_TOTAL, lw=LW_MAIN, label="Total SLE")
line_gw_A,    = axA.plot(A_dt, A_gw,    color=COLOR_GW,    lw=LW_A_SUB, ls="--", label="Groundwater distribution")
line_sw_A,    = axA.plot(A_dt, A_sw,    color=COLOR_SW,    lw=LW_A_SUB, ls="-.", label="Surface water distribution")

trend_line_A, slope_used_A, _ = add_trend_line(axA, A_dt, A_total, label_for_legend="Trend", force_slope=True)

add_arrows(axA, ORANGE_ARROWS, ARROW_ORANGE)
add_arrows(axA, PURPLE_ARROWS, ARROW_PURPLE)

if SHOW_TREND_PERCENT_TEXT and slope_used_A is not None:
    pct = slope_to_percent_per_year(slope_used_A, A_total, ref_mode=TREND_PCT_REF)
    if pct is not None and np.isfinite(pct):
        axA.text(
            TREND_TEXT_X, TREND_TEXT_Y,
            TREND_TEXT_FMT.format(pct=pct),
            transform=axA.transAxes,
            ha="left", va="top",
            fontsize=TREND_TEXT_FONTSIZE
        )

axA.set_ylabel("SLE Change (mm)", fontsize=FS_AXIS_LABEL)
axA.grid(False)
axA.text(0.005, 0.98, "A", transform=axA.transAxes, ha="left", va="top", fontsize=FS_PANEL_LABEL)

if len(A_dt) > 0:
    tick_idx = np.linspace(0, len(A_dt)-1, 5).astype(int)
    axA.set_xticks([A_dt[i] for i in tick_idx])
    axA.set_xticklabels([A_lbl[i] for i in tick_idx], fontsize=FS_TICKS_A)

# ============================================================
# PANEL B
# ============================================================
for i, ax in enumerate(axesB):
    letter = letters[i]
    csv_path = panelB_files[letter]

    lbl, dtv, tot = read_total_only_csv(csv_path)
    if len(tot) == 0:
        ax.axis("off")
        continue

    ax.plot(dtv, tot, color=COLOR_TOTAL, lw=LW_MAIN, label="_nolegend_")

    force_here = (letter == "a")
    add_trend_line(ax, dtv, tot, label_for_legend=None, force_slope=force_here)

    ax.text(0.01, 0.96, letter, transform=ax.transAxes, ha="left", va="top", fontsize=FS_SUBLETTER)
    ax.grid(False)
    ax.tick_params(axis="y", labelsize=FS_TICKS_B)

    if i < 9:
        ax.set_xticklabels([])
    else:
        if len(dtv) > 0:
            tick_idx = np.linspace(0, len(dtv)-1, 3).astype(int)
            ax.set_xticks([dtv[j] for j in tick_idx])
            ax.set_xticklabels([lbl[j] for j in tick_idx], fontsize=FS_TICKS_B)

# ============================================================
# PANEL LABEL "B"
# ============================================================
axesB[0].text(B_LABEL_X, B_LABEL_Y, "B", transform=axesB[0].transAxes,
             ha="left", va="top", fontsize=FS_PANEL_LABEL)

# ============================================================
# PANEL-B Y-LABEL
# ============================================================
fig.text(
    PANELB_YLABEL_X, PANELB_YLABEL_Y,
    "SLE Change (mm)",
    va="center", ha="center",
    rotation="vertical",
    fontsize=FS_AXIS_LABEL
)

# ============================================================
# LEGEND (includes arrow items)
# ============================================================
recharge_handle = Line2D([0], [0], color=ARROW_ORANGE, lw=ARROW_LW,
                         marker=">", markersize=10, linestyle="-", label="Recharge")
depletion_handle = Line2D([0], [0], color=ARROW_PURPLE, lw=ARROW_LW,
                          marker=">", markersize=10, linestyle="-", label="Depletion")

fig.legend(
    [line_total_A, line_gw_A, line_sw_A, trend_line_A, recharge_handle, depletion_handle],
    ["Total SLE", "Groundwater distribution", "Surface water distribution", "Trend", "Recharge", "Depletion"],
    loc="lower center",
    bbox_to_anchor=(LEGEND_X, LEGEND_Y),
    ncol=6,
    fontsize=FS_LEGEND,
    frameon=True
)

# ============================================================
# LAYOUT CONTROL
# ============================================================
reserved_bottom = max(0.0, min(0.35, BOTTOM_MARGIN + LEGEND_GAP))
left_margin = max(0.0, min(0.25, PANELB_LEFT_MARGIN))
plt.tight_layout(rect=[left_margin, reserved_bottom, 1.0, 1.0])

# ============================================================
# SAVE
# ============================================================
plt.savefig(FIG_OUT_PNG, dpi=PNG_DPI, bbox_inches="tight")

if not RASTERIZE_PDF:
    plt.savefig(FIG_OUT_PDF, format="pdf", bbox_inches="tight", dpi=PDF_DPI)
else:
    tmp_png = FIG_OUT_PDF.replace(".pdf", f"_raster_{PDF_DPI}dpi.png")
    plt.savefig(tmp_png, dpi=PDF_DPI, bbox_inches="tight")

    import matplotlib.image as mpimg
    img = mpimg.imread(tmp_png)
    fig2 = plt.figure(figsize=(FIG_W, FIG_H), dpi=PDF_DPI)
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.axis("off")
    ax2.imshow(img)
    fig2.savefig(FIG_OUT_PDF, format="pdf", dpi=PDF_DPI, bbox_inches="tight")
    plt.close(fig2)

plt.close()

print(f"Saved PNG: {FIG_OUT_PNG} (dpi={PNG_DPI})")
print(f"Saved PDF: {FIG_OUT_PDF} (dpi={PDF_DPI}, rasterize={RASTERIZE_PDF})")
print("Done.")
print(f"Forced slope value (mm/year) used where forced: {TREND_SLOPE_MM_PER_YEAR}")