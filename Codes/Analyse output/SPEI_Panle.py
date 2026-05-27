#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# INPUT FILE
# ==========================
csv_path = "/home/mohammad/Desktop/1/9/yearly_average_NAmean_PET (copy).csv"

out_path = os.path.join(
    os.path.dirname(csv_path),
    "wtd_P_ET_dual_axis_timeseries3.png"
)

# ==========================
# FIGURE SIZE OPTIONS
# ==========================
FIG_WIDTH  = 12   # inches
FIG_HEIGHT = 8    # inches
DPI = 600

# ==========================
# STYLE OPTIONS (CHANGE HERE)
# ==========================
LINE_WIDTH      = 4      # thickness of lines
MARKER_SIZE     = 10     # marker size
MARKER_EDGE_W   = 1.5    # marker edge thickness

LABEL_SIZE      = 20     # axis label font size
TICK_SIZE       = 18     # tick label size
LEGEND_SIZE     = 18     # legend text size
PANEL_LABEL_SIZE= 32     # "b" size
SPINE_WIDTH     = 1.5    # axis border thickness

# ==========================
# USER OPTIONS (AXIS DOMAINS)
# ==========================
SPI_YMIN = -2.2
SPI_YMAX = 2.7

WTD_YMIN = -9.19
WTD_YMAX = -9.08

# ==========================
# READ DATA
# ==========================
df = pd.read_csv(csv_path)
print("Columns in CSV:", list(df.columns))

if "year" not in df.columns:
    raise KeyError("Your CSV must have a 'year' column.")

# ==========================
# PICK COLUMNS
# ==========================
pet_candidates = ["P_ET (m)", "P_ET", "P-ET", "P_ET_m", "PET", "P_ET_meters"]
pet_col = next((c for c in pet_candidates if c in df.columns), None)

wtd_candidates = ["NA_mean_modified1", "NA_mean_modified", "NA_mean"]
wtd_col = next((c for c in wtd_candidates if c in df.columns), None)

if pet_col is None or wtd_col is None:
    raise KeyError("Missing required columns.")

df = df.sort_values("year").reset_index(drop=True)
x = df["year"].astype(int)

# ==========================
# CREATE FIGURE
# ==========================
fig, ax1 = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))

# ---- Left Axis ----
ax1.plot(
    x,
    df[pet_col],
    color="red",
    linestyle="-",
    linewidth=LINE_WIDTH,
    marker="o",
    markersize=MARKER_SIZE,
    markeredgewidth=MARKER_EDGE_W,
    markeredgecolor="red",
    label="SPEI (P−E)"
)

ax1.set_ylabel("SPEI index", fontsize=LABEL_SIZE)
ax1.set_ylim(SPI_YMIN, SPI_YMAX)
ax1.tick_params(axis='both', labelsize=TICK_SIZE)

# ---- Right Axis ----
ax2 = ax1.twinx()

ax2.plot(
    x,
    df[wtd_col],
    color="blue",
    linestyle="--",
    linewidth=LINE_WIDTH,
    marker="s",
    markersize=MARKER_SIZE,
    markeredgewidth=MARKER_EDGE_W,
    markeredgecolor="blue",
    label="WTD"
)

ax2.set_ylabel("Water table depth (m)", fontsize=LABEL_SIZE)
ax2.set_ylim(WTD_YMIN, WTD_YMAX)
ax2.tick_params(axis='y', labelsize=TICK_SIZE)

# ==========================
# X AXIS
# ==========================
ax1.set_xlim(x.min() - 0.5, x.max() + 0.5)
ax1.set_xticks(list(range(int(x.min()), int(x.max()) + 1, 5)))
ax1.tick_params(axis='x', labelsize=TICK_SIZE)

# ==========================
# SPINE THICKNESS
# ==========================
for spine in ax1.spines.values():
    spine.set_linewidth(SPINE_WIDTH)

for spine in ax2.spines.values():
    spine.set_linewidth(SPINE_WIDTH)

# ==========================
# LEGEND
# ==========================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()

ax1.legend(
    lines1 + lines2,
    labels1 + labels2,
    loc="upper right",
    fontsize=LEGEND_SIZE,
    frameon=True
)

# ==========================
# PANEL LABEL
# ==========================
ax1.text(
    0.02, 0.93, "b",
    transform=ax1.transAxes,
    fontsize=PANEL_LABEL_SIZE,
    fontweight="bold",
    va="top",
    ha="left"
)

# ==========================
# SAVE
# ==========================
plt.tight_layout()
plt.savefig(out_path, dpi=DPI)
plt.close()

print(f"Saved figure to:\n{out_path}")
