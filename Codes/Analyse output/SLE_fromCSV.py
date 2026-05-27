#!/usr/bin/env python3
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

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
FS_TICKS_A     = 11
FS_TICKS_B     = 8
FS_SUBLETTER   = 10
FS_LEGEND      = 10

LW_A_MAIN = 1
LW_A_SUB  = 1
LW_B      = 1

# ============================================================
# COLORS (UPDATED)
# ============================================================
COLOR_TOTAL = "#756bb1"  
COLOR_GW    = "#bcbddc"  
COLOR_SW    = "#efedf5"   

# ============================================================
# LEGEND POSITION + SPACING CONTROL
# ============================================================
LEGEND_X = 0.5
LEGEND_Y = 0.06

BOTTOM_MARGIN = 0.12
LEGEND_GAP = 0.02

# ============================================================
# PANEL-B Y-LABEL POSITION + DISTANCE OPTIONS
# ============================================================
PANELB_YLABEL_X = 0.085
PANELB_YLABEL_Y = 0.37
PANELB_LEFT_MARGIN = 0.06
PANELB_YLABEL_PAD = 0

# ============================================================
# PANEL LABEL "B" POSITION OPTIONS
# ============================================================
B_LABEL_X = 0.02
B_LABEL_Y = 1.2

# ============================================================
# YOUR PATHS (CSV INPUT)
# ============================================================
output_folder = "/home/mohammad/Desktop/1/11"

CSV_PANEL_A  = os.path.join(output_folder, "SLE_timeseries_PanelA_North_America_No_Greenland.csv")
CSV_PANEL_BA = os.path.join(output_folder, "SLE_timeseries_PanelB_a_boundary.csv")

FIG_OUT = os.path.join(output_folder, "Fig_SLE_PanelA_PanelB.png")

# ============================================================
# HELPERS (CSV)
# ============================================================
def parse_date_label_to_dt(s):
    y, m = s.split("_")
    return datetime(int(y), int(m), 1)

def read_sle_csv(csv_path):
    df = pd.read_csv(csv_path)

    required = ["date", "Total_SLE_cm", "Groundwater_SLE_cm", "SurfaceWater_SLE_cm"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column '{c}' in {csv_path}")

    df["dt"] = df["date"].astype(str).apply(parse_date_label_to_dt)

    return (
        df["date"].astype(str).tolist(),
        df["dt"].tolist(),
        df["Total_SLE_cm"].to_numpy(),
        df["Groundwater_SLE_cm"].to_numpy(),
        df["SurfaceWater_SLE_cm"].to_numpy(),
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

A_lbl, A_dt, A_total, A_gw, A_sw = read_sle_csv(CSV_PANEL_A)

# ============================================================
# BUILD FIGURE LAYOUT
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
# PLOT PANEL A
# ============================================================
axA.plot(A_dt, A_total, color=COLOR_TOTAL, lw=LW_A_MAIN)
axA.plot(A_dt, A_gw,    color=COLOR_GW,    lw=LW_A_SUB, ls="--")
axA.plot(A_dt, A_sw,    color=COLOR_SW,    lw=LW_A_SUB, ls="-.")

axA.set_ylabel("SLE Change (cm)", fontsize=FS_AXIS_LABEL)
axA.grid(False)
axA.text(0.005, 0.98, "A", transform=axA.transAxes,
         ha="left", va="top", fontsize=FS_PANEL_LABEL)

if len(A_dt) > 0:
    tick_idx = np.linspace(0, len(A_dt)-1, 5).astype(int)
    axA.set_xticks([A_dt[i] for i in tick_idx])
    axA.set_xticklabels([A_lbl[i] for i in tick_idx], fontsize=FS_TICKS_A)

# ============================================================
# PLOT PANEL B
# ============================================================
for i, ax in enumerate(axesB):
    letter = letters[i]
    csv_path = panelB_files[letter]

    lbl, dtv, tot, gw, sw = read_sle_csv(csv_path)

    if len(tot) == 0:
        ax.axis("off")
        continue

    ax.plot(dtv, tot, color=COLOR_TOTAL, lw=LW_B)
    ax.plot(dtv, gw,  color=COLOR_GW,    lw=LW_B, ls="--")
    ax.plot(dtv, sw,  color=COLOR_SW,    lw=LW_B, ls="-.")

    ax.text(0.01, 0.96, letter,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=FS_SUBLETTER)

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
axesB[0].text(
    B_LABEL_X, B_LABEL_Y, "B",
    transform=axesB[0].transAxes,
    ha="left", va="top",
    fontsize=FS_PANEL_LABEL
)

# ============================================================
# PANEL-B Y-LABEL
# ============================================================
panelb_ylabel = fig.text(
    PANELB_YLABEL_X, PANELB_YLABEL_Y,
    "SLE Change (cm)",
    va="center", ha="center",
    rotation="vertical",
    fontsize=FS_AXIS_LABEL
)

# ============================================================
# LEGEND (updated colors)
# ============================================================
h1, = axA.plot([], [], color=COLOR_TOTAL, lw=LW_A_MAIN)
h2, = axA.plot([], [], color=COLOR_GW,    lw=LW_A_SUB, ls="--")
h3, = axA.plot([], [], color=COLOR_SW,    lw=LW_A_SUB, ls="-.")

fig.legend(
    [h1, h2, h3],
    ["Total SLE Change", "Groundwater Contribution", "Surface Water Contribution"],
    loc="lower center",
    bbox_to_anchor=(LEGEND_X, LEGEND_Y),
    ncol=3,
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
# SAVE FIGURE
# ============================================================
plt.savefig(FIG_OUT, dpi=200, bbox_inches="tight")
plt.close()

print(f"Saved Figure: {FIG_OUT}")
print("Done: Updated line colors.")
