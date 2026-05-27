#!/usr/bin/env python3
"""
Wet vs Dry Conditions heatmap (Year x Month) from monthly WTD GeoTIFFs.

Your WTD path:
  WTD dir: /media/mohammad/My Book/WTM_Result/Monthly/3
  Output : /home/mohammad/Desktop/1

Expected WTD filenames (adjust the pattern below if yours differ):
  N_America_YYYYMM_*.tif
Examples:
  N_America_200001_petsc_000000001.tif
  N_America_201507_petsc_000000001.tif
  ...

What it does (same logic as your P-E script, but for WTD):
  1) WTD12(t) = 12-month rolling sum of WTD per pixel
  2) Baseline mean/std per pixel over baseline years (default 2000–2020)
  3) z = (WTD12 - mean)/std
  4) dominance = 100*(%wet - %dry) where wet=z>0, dry=z<0  -> [-100, +100]
     (Interpretation: positive => spatially more "above baseline" WTD12; negative => more "below baseline" WTD12)
  5) Heatmap: rows=years, cols=months

Outputs (dpi=1500):
  /home/mohammad/Desktop/1/wtd_wet_dry_dominance_heatmap.png
  /home/mohammad/Desktop/1/wtd_wet_dry_dominance_heatmap.pdf
"""

import os
import re
import glob
import argparse
from collections import deque, defaultdict

import numpy as np
import rasterio
from rasterio.windows import Window

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


# --------- FIXED PATHS (your paths) ----------
WTD_DIR_DEFAULT = "/media/mohammad/My Book/WTM_Result/Monthly/3"
OUT_DIR_DEFAULT = "/home/mohammad/Desktop/1"
# --------------------------------------------

# Matches N_America_YYYYMM_anything.tif
DATE_RE = re.compile(r"^N_America_(\d{6})_.*\.tif$")


def parse_yyyymm(path: str) -> str:
    fn = os.path.basename(path)
    m = DATE_RE.match(fn)
    if not m:
        raise ValueError(f"Bad filename (expected N_America_YYYYMM_*.tif): {fn}")
    return m.group(1)


def month_index_to_year_month(yyyymm: str):
    y = int(yyyymm[:4])
    m = int(yyyymm[4:6])
    return y, m


def list_time_steps(wtd_dir: str):
    """Return sorted list of yyyymm that exist in WTD folder."""
    files = glob.glob(os.path.join(wtd_dir, "N_America_??????_*.tif"))
    if not files:
        raise RuntimeError(f"No files found matching N_America_YYYYMM_*.tif in:\n  {wtd_dir}")

    dates = sorted({parse_yyyymm(p) for p in files})
    return dates


def build_wtd_path_for_date(wtd_dir: str, yyyymm: str):
    """
    Find the WTD GeoTIFF for a given YYYYMM.
    If multiple match, uses the first sorted match.
    """
    matches = sorted(glob.glob(os.path.join(wtd_dir, f"N_America_{yyyymm}_*.tif")))
    if not matches:
        raise FileNotFoundError(f"Missing WTD file for {yyyymm} in {wtd_dir}")
    return matches[0]


def open_reference_raster(ref_path: str):
    with rasterio.open(ref_path) as src:
        nodata = src.nodata
        height, width = src.height, src.width
    return nodata, height, width


def iter_windows(width: int, height: int, block_size: int):
    for row_off in range(0, height, block_size):
        win_h = min(block_size, height - row_off)
        for col_off in range(0, width, block_size):
            win_w = min(block_size, width - col_off)
            yield Window(col_off=col_off, row_off=row_off, width=win_w, height=win_h)


def read_block(path: str, window: Window, nodata):
    with rasterio.open(path) as src:
        arr = src.read(1, window=window).astype(np.float32)

    if nodata is not None:
        valid = (arr != nodata) & np.isfinite(arr)
    else:
        valid = np.isfinite(arr)

    arr[~valid] = np.nan
    return arr


def compute_baseline_mean_std(
    dates,
    wtd_dir,
    baseline_start,
    baseline_end,
    block_size,
    nodata,
    height,
    width,
):
    """
    PASS 1:
      - compute WTD12 per pixel (rolling 12-month sum of WTD)
      - update Welford mean/std per pixel for baseline months only
    """
    mean = np.zeros((height, width), dtype=np.float32)
    m2   = np.zeros((height, width), dtype=np.float32)
    cnt  = np.zeros((height, width), dtype=np.int32)

    for window in iter_windows(width, height, block_size):
        wtd_deque = deque(maxlen=12)
        wtd12 = None

        wh = int(window.height)
        ww = int(window.width)

        w_mean = np.zeros((wh, ww), dtype=np.float32)
        w_m2   = np.zeros((wh, ww), dtype=np.float32)
        w_cnt  = np.zeros((wh, ww), dtype=np.int32)

        for yyyymm in dates:
            y, _ = month_index_to_year_month(yyyymm)

            w_path = build_wtd_path_for_date(wtd_dir, yyyymm)
            wtd = read_block(w_path, window, nodata)

            if len(wtd_deque) < 12:
                wtd_deque.append(wtd)
                if len(wtd_deque) == 12:
                    wtd12 = np.nansum(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)
                continue
            else:
                old = wtd_deque[0]
                wtd_deque.append(wtd)  # drops oldest automatically
                dropped = old

                # If NaNs involved, recompute to stay correct
                if np.isnan(wtd).any() or np.isnan(dropped).any():
                    wtd12 = np.nansum(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)
                else:
                    wtd12 = (wtd12 + wtd - dropped).astype(np.float32)

            if baseline_start <= y <= baseline_end:
                x = wtd12
                valid = np.isfinite(x)
                if not np.any(valid):
                    continue

                w_cnt[valid] += 1
                c = w_cnt[valid].astype(np.float32)

                delta = x[valid] - w_mean[valid]
                w_mean[valid] += delta / c
                delta2 = x[valid] - w_mean[valid]
                w_m2[valid] += delta * delta2

        r0 = int(window.row_off)
        c0 = int(window.col_off)
        r1 = r0 + int(window.height)
        c1 = c0 + int(window.width)

        mean[r0:r1, c0:c1] = w_mean
        m2[r0:r1, c0:c1]   = w_m2
        cnt[r0:r1, c0:c1]  = w_cnt

    std = np.full_like(mean, np.nan, dtype=np.float32)
    ok = cnt > 1
    std[ok] = np.sqrt(m2[ok] / (cnt[ok].astype(np.float32) - 1.0))

    std[(std == 0) | (~np.isfinite(std))] = np.nan
    mean[~np.isfinite(mean)] = np.nan

    return mean, std


def compute_dominance_matrix(
    dates,
    wtd_dir,
    mean,
    std,
    block_size,
    nodata,
):
    """
    PASS 2:
      - compute WTD12 again
      - compute z per pixel
      - dominance = 100*(%wet - %dry)
    """
    year_to_months = defaultdict(set)
    ym_map = {}
    for d in dates:
        y, m = month_index_to_year_month(d)
        year_to_months[y].add(m)
        ym_map[d] = (y, m)

    years_sorted = sorted(year_to_months.keys())
    dom = np.full((len(years_sorted), 12), np.nan, dtype=np.float32)
    year_index = {y: i for i, y in enumerate(years_sorted)}

    height, width = mean.shape

    wet_counts = {d: 0 for d in dates}
    dry_counts = {d: 0 for d in dates}
    tot_counts = {d: 0 for d in dates}

    for window in iter_windows(width, height, block_size):
        wtd_deque = deque(maxlen=12)
        wtd12 = None

        r0 = int(window.row_off)
        c0 = int(window.col_off)
        r1 = r0 + int(window.height)
        c1 = c0 + int(window.width)

        w_mean = mean[r0:r1, c0:c1]
        w_std  = std[r0:r1, c0:c1]

        for d in dates:
            w_path = build_wtd_path_for_date(wtd_dir, d)
            wtd = read_block(w_path, window, nodata)

            if len(wtd_deque) < 12:
                wtd_deque.append(wtd)
                if len(wtd_deque) == 12:
                    wtd12 = np.nansum(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)
                continue
            else:
                old = wtd_deque[0]
                wtd_deque.append(wtd)
                dropped = old
                if np.isnan(wtd).any() or np.isnan(dropped).any():
                    wtd12 = np.nansum(np.stack(wtd_deque, axis=0), axis=0).astype(np.float32)
                else:
                    wtd12 = (wtd12 + wtd - dropped).astype(np.float32)

            valid = np.isfinite(wtd12) & np.isfinite(w_mean) & np.isfinite(w_std)
            if not np.any(valid):
                continue

            z = (wtd12[valid] - w_mean[valid]) / w_std[valid]
            wet = np.count_nonzero(z > 0)
            dry = np.count_nonzero(z < 0)
            tot = z.size

            wet_counts[d] += wet
            dry_counts[d] += dry
            tot_counts[d] += tot

    for d in dates:
        tot = tot_counts[d]
        if tot <= 0:
            continue
        wet_pct = 100.0 * wet_counts[d] / tot
        dry_pct = 100.0 * dry_counts[d] / tot
        dominance = wet_pct - dry_pct  # [-100, +100]

        y, m = ym_map[d]
        dom[year_index[y], m - 1] = dominance

    return years_sorted, dom


def plot_heatmap(years, dom, out_png, out_pdf, dpi, title, baseline_start, baseline_end):
    months = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

    fig_w = 10
    fig_h = max(8, 0.22 * len(years))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # red = negative, blue = positive, white=0 (same style you used)
    norm = TwoSlopeNorm(vmin=-100, vcenter=0, vmax=100)
    im = ax.imshow(dom, aspect="auto", cmap="RdBu", norm=norm)

    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")

    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(months)

    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(y) for y in years])

    ax.set_xticks(np.arange(-.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(years), 1), minor=True)
    ax.grid(which="minor", linewidth=0.4)
    ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Wet–Dry Dominance (%)", rotation=90)
    cbar.set_ticks([-75, -50, -25, 0, 25, 50, 75])

    ax.text(
        0.0, -0.08,
        f"Index: z-score of 12-month sum(WTD), baseline = {baseline_start}–{baseline_end}",
        transform=ax.transAxes,
        fontsize=10,
        va="top"
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wtd_dir", default=WTD_DIR_DEFAULT, help="WTD GeoTIFF folder")
    ap.add_argument("--out_dir", default=OUT_DIR_DEFAULT, help="Output folder")
    ap.add_argument("--baseline_start", type=int, default=2000, help="Baseline start year")
    ap.add_argument("--baseline_end",   type=int, default=2020, help="Baseline end year")
    ap.add_argument("--block_size", type=int, default=512, help="Block size (256/512/1024)")
    ap.add_argument("--dpi", type=int, default=1500, help="DPI for PNG/PDF")
    ap.add_argument("--title", default="WTD Wet vs. Dry Conditions: Spatial coverage", help="Plot title")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_png = os.path.join(args.out_dir, "wtd_wet_dry_dominance_heatmap.png")
    out_pdf = os.path.join(args.out_dir, "wtd_wet_dry_dominance_heatmap.pdf")

    dates = list_time_steps(args.wtd_dir)
    print(f"Found {len(dates)} monthly steps: {dates[0]} -> {dates[-1]}")

    ref = build_wtd_path_for_date(args.wtd_dir, dates[0])
    nodata, height, width = open_reference_raster(ref)
    print(f"Raster size: {width} x {height} | nodata = {nodata}")

    print("PASS 1/2: Computing baseline mean/std (WTD12) ...")
    mean, std = compute_baseline_mean_std(
        dates=dates,
        wtd_dir=args.wtd_dir,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
        block_size=args.block_size,
        nodata=nodata,
        height=height,
        width=width,
    )

    print("PASS 2/2: Computing dominance matrix ...")
    years, dom = compute_dominance_matrix(
        dates=dates,
        wtd_dir=args.wtd_dir,
        mean=mean,
        std=std,
        block_size=args.block_size,
        nodata=nodata,
    )

    print(f"Saving:\n  {out_png}\n  {out_pdf}")
    plot_heatmap(
        years=years,
        dom=dom,
        out_png=out_png,
        out_pdf=out_pdf,
        dpi=args.dpi,
        title=args.title,
        baseline_start=args.baseline_start,
        baseline_end=args.baseline_end,
    )

    print("Done.")


if __name__ == "__main__":
    main()
