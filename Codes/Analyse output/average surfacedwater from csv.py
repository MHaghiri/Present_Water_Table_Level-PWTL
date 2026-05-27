#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# INPUT / OUTPUT
# ============================================================
CSV_PATH = "/home/mohammad/Desktop/1/13/timeseries_surfacewater_a_to_l.csv"
OUT_DIR  = "/home/mohammad/Desktop/1/13"
OUT_BASE = "panel_a_to_l_from_csv2"

# Global Y label for the figure (CHANGE HERE)
Y_LABEL = "Mean groundwater depth (m)"

# Y-axis tick/label side: "left" or "right"
Y_AXIS_SIDE = "left"

# ============================================================
# FIGURE SIZE
# ============================================================
FIG_W, FIG_H = 16, 9
NROWS, NCOLS = 4, 3
DPI = 300

# ============================================================
# SPACING OPTIONS (CHANGE HERE)
# Smaller = tighter, Larger = more space
# Typical good ranges: 0.05–0.35
# ============================================================
HSPACE = -0.15   # vertical space between rows
WSPACE = 4.01   # horizontal space between columns

# ============================================================
# DECIMAL FORMAT OPTIONS (CHANGE HERE)
# ============================================================
Y_TICK_DECIMALS = 2   # 2 decimals after point (e.g., -12.34)

# ============================================================
# COLORS / STYLES (same as before)
# ============================================================
COLOR_OBS   = "#2b0a3d"   # deep purple
COLOR_TREND = "#ff7f0e"   # orange
BAND_COLOR  = "#f4a3a3"   # light pink band
BAND_ALPHA  = 0.35

LW_OBS   = 2.0
MS_OBS   = 4.5
LW_TREND = 2.0
LS_TREND = "--"

# ============================================================
# TEXT / FONT OPTIONS (CHANGE HERE)
# ============================================================
FS_PANEL_LETTER = 14   # size of a,b,c,... letters
FS_AXIS_LABEL   = 13   # "Year" + global Y label
FS_TICKS        = 12   # tick labels
FS_LEGEND       = 16   # legend

# Panel letter location in Axes coordinates (0..1)
PANEL_LETTER_X = 0.01
PANEL_LETTER_Y = 0.90

# Optional panel letter background box (helps readability)
PANEL_LETTER_BOX = False   # True / False
PANEL_LETTER_BOX_ALPHA = 0.6

# ============================================================
# HELPERS
# ============================================================
def ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def apply_yaxis_side(ax, side: str):
    side = side.lower().strip()
    if side == "right":
        ax.yaxis.tick_right()
        ax.yaxis.set_label_position("right")
    else:
        ax.yaxis.tick_left()
        ax.yaxis.set_label_position("left")

def apply_y_tick_format(ax, decimals: int):
    fmt = f"%.{decimals}f"
    ax.yaxis.set_major_formatter(FormatStrFormatter(fmt))

def linear_trend_with_ci(x_years: np.ndarray, y: np.ndarray):
    """
    OLS y = a + b*x, with 95% CI band for mean prediction.
    Handles NaNs by fitting only on valid points.
    Returns:
      yhat_full, lower_full, upper_full
    """
    xfull = x_years.astype(float)
    yfull = y.astype(float)

    msk = np.isfinite(xfull) & np.isfinite(yfull)
    x = xfull[msk]
    y = yfull[msk]

    n = len(x)
    if n < 3:
        return np.full_like(xfull, np.nan), None, None

    xbar = x.mean()
    ybar = y.mean()

    Sxx = np.sum((x - xbar) ** 2)
    Sxy = np.sum((x - xbar) * (y - ybar))

    b = Sxy / Sxx
    a = ybar - b * xbar

    yhat_full = a + b * xfull

    resid = y - (a + b * x)
    s2 = np.sum(resid ** 2) / (n - 2)
    s = np.sqrt(s2)

    # 95% t critical (exact if scipy exists; fallback otherwise)
    try:
        from scipy.stats import t
        tcrit = t.ppf(0.975, df=n - 2)
    except Exception:
        tcrit = 2.0

    se_mean_full = s * np.sqrt((1.0 / n) + ((xfull - xbar) ** 2) / Sxx)
    lower_full = yhat_full - tcrit * se_mean_full
    upper_full = yhat_full + tcrit * se_mean_full

    return yhat_full, lower_full, upper_full

def add_panel_letter(ax, letter: str):
    if PANEL_LETTER_BOX:
        bbox = dict(facecolor="white", edgecolor="none", alpha=PANEL_LETTER_BOX_ALPHA, pad=1.5)
    else:
        bbox = None

    ax.text(
        PANEL_LETTER_X, PANEL_LETTER_Y, letter,
        transform=ax.transAxes,
        fontsize=FS_PANEL_LETTER,
        bbox=bbox
    )

# ============================================================
# MAIN
# ============================================================
def main():
    ensure_outdir(OUT_DIR)

    df = pd.read_csv(CSV_PATH)
    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")

    # Expect 12 series columns: North America + 11 watersheds
    series_cols = [c for c in df.columns if c != "year"]
    if len(series_cols) != 12:
        raise ValueError(
            f"Expected 12 time-series columns (besides 'year'), found {len(series_cols)}:\n{series_cols}"
        )

    x = df["year"].values.astype(int)

    fig, axes = plt.subplots(NROWS, NCOLS, figsize=(FIG_W, FIG_H), sharex=True)
    axes = axes.flatten()
    panel_letters = list("abcdefghijkl")

    for i, col in enumerate(series_cols):
        ax = axes[i]
        apply_yaxis_side(ax, Y_AXIS_SIDE)
        apply_y_tick_format(ax, Y_TICK_DECIMALS)

        y = df[col].values.astype(float)
        yhat, lo, hi = linear_trend_with_ci(x, y)

        if lo is not None:
            ax.fill_between(x, lo, hi, color=BAND_COLOR, alpha=BAND_ALPHA, label="95 % Confidence Band")
            ax.plot(x, yhat, color=COLOR_TREND, linestyle=LS_TREND, linewidth=LW_TREND, label="Linear Trend")

        ax.plot(x, y, color=COLOR_OBS, marker="o", markersize=MS_OBS, linewidth=LW_OBS, label="Observed")

        add_panel_letter(ax, panel_letters[i])
        ax.tick_params(labelsize=FS_TICKS)

    # X label on bottom row only
    for c in range(NCOLS):
        axes[(NROWS - 1) * NCOLS + c].set_xlabel("Year", fontsize=FS_AXIS_LABEL)

    # One global Y label
    if Y_AXIS_SIDE.lower() == "right":
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS_LABEL, x=0.99)
    else:
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS_LABEL, x=0.01)

    # One legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=FS_LEGEND, frameon=True)

    # ---- Control spacing between panels ----
    # (We use subplots_adjust instead of tight_layout spacing, for direct control)
    fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    # Leave space for legend at bottom
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_png = os.path.join(OUT_DIR, f"{OUT_BASE}.png")
    out_pdf = os.path.join(OUT_DIR, f"{OUT_BASE}.pdf")
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_pdf)
    plt.close(fig)

    print("DONE")
    print("PNG:", out_png)
    print("PDF:", out_pdf)

if __name__ == "__main__":
    main()