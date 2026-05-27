#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import matplotlib as mpl

# ============================================================
# HIGH-QUALITY PDF OUTPUT SETTINGS
# ============================================================
mpl.rcParams["pdf.fonttype"] = 42   # embed TrueType fonts
mpl.rcParams["ps.fonttype"]  = 42
mpl.rcParams["savefig.dpi"]  = 1500

# (Optional) helps avoid some PDF transparency artifacts in some viewers
mpl.rcParams["path.simplify"] = False

# ============================================================
# INPUT / OUTPUT
# ============================================================
CSV_PATH = "/home/mohammad/Desktop/1/13/surfacewater_timeseries_.csv"
OUT_DIR  = "/home/mohammad/Desktop/1/13"
OUT_BASE = "depletion_2000_2025_SW"

Y_LABEL = "Mean groundwater depth (m)"
Y_AXIS_SIDE = "left"

# ============================================================
# FIGURE SIZE (THIS CONTROLS PDF PHYSICAL SIZE)
# ============================================================
FIG_W, FIG_H = 16, 9
NROWS, NCOLS = 4, 3

# For PNG (raster)
PNG_DPI = 1500

# For PDF (vector; DPI only affects raster elements)
PDF_DPI = 1500

# ============================================================
# SPACING OPTIONS
# ============================================================
HSPACE = -0.0
WSPACE = 4.01

# ============================================================
# Y-AXIS FORMAT OPTIONS
# ============================================================
Y_TICK_DECIMALS = 2
Y_TICK_COUNT = 4

# ============================================================
# COLORS / STYLES
# ============================================================
COLOR_OBS   = "#2b0a3d"
COLOR_TREND = "#ff7f0e"
BAND_COLOR  = "#f4a3a3"
BAND_ALPHA  = 0.35

LW_OBS   = 2.0
MS_OBS   = 4.5
LW_TREND = 2.0
LS_TREND = "--"

# ============================================================
# TEXT / FONT OPTIONS
# ============================================================
FS_PANEL_LETTER = 14
FS_AXIS_LABEL   = 13
FS_TICKS        = 12
FS_LEGEND       = 16

PANEL_LETTER_X = 0.005
PANEL_LETTER_Y = 0.89
PANEL_LETTER_BOX = False
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
    ax.yaxis.set_major_locator(MaxNLocator(nbins=Y_TICK_COUNT, prune=None))

def linear_trend_with_ci(x_years: np.ndarray, y: np.ndarray):
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
        bbox = dict(facecolor="white", edgecolor="none",
                    alpha=PANEL_LETTER_BOX_ALPHA, pad=1.5)
    else:
        bbox = None

    ax.text(PANEL_LETTER_X, PANEL_LETTER_Y, letter,
            transform=ax.transAxes,
            fontsize=FS_PANEL_LETTER,
            bbox=bbox)

# ============================================================
# MAIN
# ============================================================
def main():
    ensure_outdir(OUT_DIR)

    df = pd.read_csv(CSV_PATH)
    if "year" not in df.columns:
        raise ValueError("CSV must contain a 'year' column.")

    series_cols = [c for c in df.columns if c != "year"]
    if len(series_cols) != 12:
        raise ValueError(
            f"Expected 12 time-series columns (besides 'year'), found {len(series_cols)}"
        )

    x = df["year"].values.astype(int)

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(FIG_W, FIG_H),
        sharex=True
    )

    axes = axes.flatten()
    panel_letters = list("abcdefghijkl")

    for i, col in enumerate(series_cols):
        ax = axes[i]

        apply_yaxis_side(ax, Y_AXIS_SIDE)
        apply_y_tick_format(ax, Y_TICK_DECIMALS)

        y = df[col].values.astype(float)
        yhat, lo, hi = linear_trend_with_ci(x, y)

        # ✅ FIX: keep band VECTOR in PDF (NO rasterized=True)
        if lo is not None:
            ax.fill_between(
                x, lo, hi,
                color=BAND_COLOR,
                alpha=BAND_ALPHA,
                edgecolor="none",
                linewidth=0,
                antialiased=True,
                label="95% Confidence Band",
                zorder=1
            )

            ax.plot(
                x, yhat,
                color=COLOR_TREND,
                linestyle=LS_TREND,
                linewidth=LW_TREND,
                label="Linear Trend",
                zorder=2
            )

        ax.plot(
            x, y,
            color=COLOR_OBS,
            marker="o",
            markersize=MS_OBS,
            linewidth=LW_OBS,
            label="Mean simulated value",
            zorder=3
        )

        add_panel_letter(ax, panel_letters[i])
        ax.tick_params(labelsize=FS_TICKS)

    # X labels bottom row
    for c in range(NCOLS):
        axes[(NROWS - 1) * NCOLS + c].set_xlabel("Year", fontsize=FS_AXIS_LABEL)

    # Global Y label
    if Y_AXIS_SIDE.lower() == "right":
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS_LABEL, x=0.99)
    else:
        fig.supylabel(Y_LABEL, fontsize=FS_AXIS_LABEL, x=0.01)

    # Global legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="lower center",
        ncol=3,
        fontsize=FS_LEGEND,
        frameon=True
    )

    fig.subplots_adjust(hspace=HSPACE, wspace=WSPACE)

    # keep space for legend
    plt.tight_layout(rect=[0, 0.07, 1, 1])

    out_png = os.path.join(OUT_DIR, f"{OUT_BASE}.png")
    out_pdf = os.path.join(OUT_DIR, f"{OUT_BASE}.pdf")

    # ---- Save PNG ----
    fig.savefig(out_png, dpi=PNG_DPI, bbox_inches="tight", pad_inches=0.02)

    # ---- Save PDF (vector; band stays vector now) ----
    fig.savefig(out_pdf, format="pdf", dpi=PDF_DPI, bbox_inches="tight", pad_inches=0.02)

    plt.close(fig)

    print("DONE")
    print("PNG:", out_png)
    print("PDF:", out_pdf)

if __name__ == "__main__":
    main()