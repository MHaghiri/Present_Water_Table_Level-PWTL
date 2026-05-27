#!/usr/bin/env python3

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# =========================================================
# INPUT CSV FILES
# =========================================================
whole_csv = "/home/mohammad/Desktop/1/17/wtda/whole_area_TWSA_WTDA_annual_anomaly.csv"
watershed_csv = "/home/mohammad/Desktop/1/17/wtda/watersheds_TWSA_WTDA_annual_anomaly.csv"
monthly_csv = "/home/mohammad/Desktop/1/17/wtda/GRACE_NAmerica_monthly_average.csv"

# =========================================================
# OUTPUT
# =========================================================
out_dir = "/home/mohammad/Desktop/1/17/wtda"
os.makedirs(out_dir, exist_ok=True)

out_png = os.path.join(out_dir, "TWSA_WTDA_12panel.png")
out_pdf = os.path.join(out_dir, "TWSA_WTDA_12panel.pdf")

# =========================================================
# SETTINGS
# =========================================================
PLOT_ONLY_OVERLAP_YEARS = True
DEFAULT_YMIN = -20
DEFAULT_YMAX = 20
PANEL_A_YMIN = -20
PANEL_A_YMAX = 20

panel_domains = {
    "a": (-20, 20),
    "b": (-45, 45),
    "d": (-40, 40),
    "c": (-30, 30),
    "e": (-30, 30),
    "f": (-40, 40),
    "g": (-30, 30),
    "h": (-20, 20),
    "i": (-20, 20),
    "j": (-20, 20),
    "k": (-20, 20),
    "l": (-20, 20),
}


def find_first_existing_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def build_datetime_from_columns(df):
    time_col = find_first_existing_column(df, ["time", "date", "datetime", "Time", "Date", "DATE"])
    if time_col is not None:
        return pd.to_datetime(df[time_col], errors="coerce")

    year_col = find_first_existing_column(df, ["year", "Year", "YEAR"])
    month_col = find_first_existing_column(df, ["month", "Month", "MONTH"])

    if year_col is not None and month_col is not None:
        return pd.to_datetime(
            dict(
                year=pd.to_numeric(df[year_col], errors="coerce"),
                month=pd.to_numeric(df[month_col], errors="coerce"),
                day=1,
            ),
            errors="coerce",
        )

    return pd.Series([pd.NaT] * len(df), index=df.index)


def compute_corr(x, y):
    s = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    if len(s) < 2 or s["x"].std() == 0 or s["y"].std() == 0:
        return None
    return s["x"].corr(s["y"])


def corr_label(r):
    return "r = n/a" if r is None else f"r = {r:.2f}"


def annotate_corr(ax, r, fontsize=11):
    ax.text(
        0.98, 0.95,
        corr_label(r),
        transform=ax.transAxes,
        fontsize=fontsize,
        fontweight="bold",
        va="top",
        ha="right",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="gray", alpha=0.8),
    )


# READ DATA
ndf_whole = pd.read_csv(whole_csv)
df_ws = pd.read_csv(watershed_csv)
df_all = pd.concat([ndf_whole, df_ws], ignore_index=True)

df_all["region"] = df_all["region"].astype(str).str.strip()
df_all["year"] = pd.to_numeric(df_all["year"], errors="coerce")
df_all = df_all.dropna(subset=["year"])
df_all["year"] = df_all["year"].astype(int)

whole_name = "Whole Area"
watershed_names = sorted([r for r in df_all["region"].unique() if r != whole_name])[:11]

# MOVE panel positions so layout is:
# first=a, second=b, third=d, fourth=c, then e-l
# while labels stay alphabetical by displayed position
base_regions = [whole_name] + watershed_names
reordered_regions = [
    base_regions[0],  # a
    base_regions[1],  # b
    base_regions[3],  # d moved to 3rd position
    base_regions[2],  # c moved to 4th position
] + base_regions[4:]

panel_letters_small = list("abcdefghijkl")
panel_info = []
for p, r in zip(panel_letters_small, reordered_regions):
    panel_info.append({"panel": p, "region": r})

plot_df = df_all.copy()
if PLOT_ONLY_OVERLAP_YEARS:
    plot_df = plot_df[plot_df["wtda_anomaly_cm"].notna() & plot_df["twsa_anomaly_cm"].notna()].copy()

# MONTHLY DATA
_df_monthly = pd.read_csv(monthly_csv)
_df_monthly["plot_date"] = build_datetime_from_columns(_df_monthly)

monthly_twsa_col = find_first_existing_column(_df_monthly, ["twsa_anomaly_cm", "TWSA_anomaly_cm", "twsa_cm", "TWSA_cm", "twsa", "TWSA"])
monthly_wtda_col = find_first_existing_column(_df_monthly, ["wtda_anomaly_cm", "WTDA_anomaly_cm", "wtda_cm", "WTDA_cm", "wtda", "WTDA"])

if monthly_twsa_col:
    _df_monthly[monthly_twsa_col] = pd.to_numeric(_df_monthly[monthly_twsa_col], errors="coerce")
if monthly_wtda_col:
    _df_monthly[monthly_wtda_col] = pd.to_numeric(_df_monthly[monthly_wtda_col], errors="coerce")

_df_monthly = _df_monthly.dropna(subset=["plot_date"]).sort_values("plot_date")

# FIGURE
fig = plt.figure(figsize=(20, 16))
gs = GridSpec(4, 4, figure=fig, height_ratios=[1.3, 1, 1, 1], hspace=0.14, wspace=0.14)

ax_top = fig.add_subplot(gs[0, :])
axes = [fig.add_subplot(gs[r, c]) for r in range(1, 4) for c in range(4)]

# TOP PANEL = A
if monthly_twsa_col:
    ax_top.plot(_df_monthly["plot_date"], _df_monthly[monthly_twsa_col], marker="o", markersize=3, linewidth=1.5, label="Terrestrial Water Storage Anomaly (TWSA)")
if monthly_wtda_col:
    ax_top.plot(_df_monthly["plot_date"], _df_monthly[monthly_wtda_col], marker="s", markersize=3, linewidth=1.5, label="Water Table Depth Anomaly (WTDA)")

ax_top.axhline(0, linestyle="--", linewidth=1)
ax_top.set_ylim(PANEL_A_YMIN, PANEL_A_YMAX)
ax_top.grid(True, alpha=0.3)
ax_top.text(0.02, 0.95, "A", transform=ax_top.transAxes, fontsize=18, fontweight="bold", va="top", ha="left")
annotate_corr(ax_top, 0.87, fontsize=13)
ax_top.legend(loc="lower left", fontsize=10, frameon=True)
ax_top.set_ylabel("Anomaly (cm)", fontsize=12)

# SMALL PANELS
for i, ax in enumerate(axes):
    if i >= len(panel_info):
        ax.axis("off")
        continue

    panel_letter = panel_info[i]["panel"]
    region_name = panel_info[i]["region"]
    d = plot_df[plot_df["region"] == region_name].copy().sort_values("year")

    if not d.empty:
        ax.plot(d["year"], d["twsa_anomaly_cm"], marker="o", linewidth=1.8)
        ax.plot(d["year"], d["wtda_anomaly_cm"], marker="s", linewidth=1.8)
        annotate_corr(ax, compute_corr(d["twsa_anomaly_cm"], d["wtda_anomaly_cm"]), fontsize=10)
    else:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")

    ax.axhline(0, linestyle="--", linewidth=1)
    ymin, ymax = panel_domains.get(panel_letter, (DEFAULT_YMIN, DEFAULT_YMAX))
    ax.set_ylim(ymin, ymax)
    ax.text(0.02, 0.95, panel_letter, transform=ax.transAxes, fontsize=16, fontweight="bold", va="top", ha="left")
    ax.grid(True, alpha=0.3)

for idx in [8, 9, 10, 11]:
    if idx < len(axes):
        axes[idx].set_xlabel("Year", fontsize=11)
for idx in [0, 4, 8]:
    if idx < len(axes):
        axes[idx].set_ylabel("Anomaly (cm)", fontsize=11)

plt.tight_layout()
fig.savefig(out_png, dpi=1500, bbox_inches="tight")
fig.savefig(out_pdf, bbox_inches="tight")
plt.show()

print(f"Saved PNG: {out_png}")
print(f"Saved PDF: {out_pdf}")
