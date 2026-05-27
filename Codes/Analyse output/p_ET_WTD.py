#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt

# ==========================
# INPUT FILE
# ==========================
csv_path = "/home/mohammad/Desktop/1/9/wtd_means_monthly_PET_trend_constrained.csv"

out_path = os.path.join(
    os.path.dirname(csv_path),
    "wtd_P_ET_dual_axis_timeseries1.png"
)

# ==========================
# USER OPTIONS (AXIS DOMAINS)
# ==========================
# Set to None to auto-scale
P_ET_YMIN = 0
P_ET_YMAX = 2

WTD_YMIN = -9.27
WTD_YMAX = -8.8

# ==========================
# MARKER OPTION (NEW)
# ==========================
MARKER_SIZE = 3   # <-- change only this if you want bigger/smaller markers

# ==========================
# READ DATA
# ==========================
df = pd.read_csv(csv_path)

print("Columns in CSV:", list(df.columns))

df["date"] = pd.to_datetime(df["date"])

# ==========================
# PICK CORRECT P-ET COLUMN NAME
# ==========================
pet_candidates = ["P_ET (m)", "P_ET", "P-ET", "P_ET_m", "PET", "P_ET_meters"]

pet_col = None
for c in pet_candidates:
    if c in df.columns:
        pet_col = c
        break

if pet_col is None:
    raise KeyError(
        "Could not find a P–ET column. I tried these names:\n"
        f"{pet_candidates}\n"
        "Your CSV columns are:\n"
        f"{list(df.columns)}"
    )

# ==========================
# PLOT
# ==========================
fig, ax1 = plt.subplots(figsize=(12, 6))

# ---- Left Y-axis: P–ET (m) ----
ax1.plot(
    df["date"],
    df[pet_col],
    color="red",
    linestyle="-",
    linewidth=2,
    marker="o",                # <-- added
    markersize=MARKER_SIZE,    # <-- added
    label="P–ET (m)"
)
ax1.set_ylabel("P–ET (m)", color="red")
ax1.tick_params(axis="y", labelcolor="red")

if P_ET_YMIN is not None and P_ET_YMAX is not None:
    ax1.set_ylim(P_ET_YMIN, P_ET_YMAX)

# ---- Right Y-axis: Water Table Depth (m) ----
ax2 = ax1.twinx()
ax2.plot(
    df["date"],
    df["NA_mean_modified1"],
    color="blue",
    linestyle="--",
    linewidth=2,
    marker="s",                # <-- added
    markersize=MARKER_SIZE,    # <-- added
    label="Water Table Depth (m)"
)
ax2.set_ylabel("Water Table Depth (m)", color="blue")
ax2.tick_params(axis="y", labelcolor="blue")

if WTD_YMIN is not None and WTD_YMAX is not None:
    ax2.set_ylim(WTD_YMIN, WTD_YMAX)

# ==========================
# LEGEND
# ==========================
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

# ==========================
# FINAL TOUCHES
# ==========================
ax1.axhline(0, color="gray", linestyle=":", linewidth=1)
ax1.set_xlabel("Year")

plt.tight_layout()
plt.savefig(out_path, dpi=300)
plt.close()

print(f"Using P–ET column: {pet_col}")
print(f"Saved figure to:\n{out_path}")
