#!/usr/bin/env python3

"""
Monthly SPI / SPEI-style indices (2000–2025)

Inputs
------
- PRECIP_DIR : folder with one monthly TIFF per time step for precipitation
- EVAP_DIR   : folder with one monthly TIFF per time step for evaporation (or PET)
- BOUNDARY_SHP : shapefile for the domain (used to mask rasters)

Assumptions
-----------
- Files in PRECIP_DIR and EVAP_DIR are in the SAME order after sorting
  (e.g., filenames contain YYYYMM so that sorting works).
- Data are monthly totals (or means) from 2000-01 to 2025-12.

Method
------
1) Compute domain-mean time series:
       P_dom(t)   = mean over masked pixels of P(x,y,t)
       climate(t) = P_dom(t) - E_dom(t)
2) For SPI:
       SPI = z-score of rolling sum of P_dom over WINDOW months
3) For SPEI:
       SPEI = z-score of rolling sum of climate over WINDOW months
"""

import os
import glob
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
import matplotlib.pyplot as plt
from matplotlib import dates as mdates
import pandas as pd

# =================== USER CONFIG ===================

PRECIP_DIR = "/media/mohammad/My Book/1800/evap/downscaled/tif/N_America"
EVAP_DIR   = "/media/mohammad/My Book/1800/pr/downscaled/tif/N_America"
BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

OUT_PNG = "/home/mohammad/Desktop/1/SPI_SPEI_monthly_2000_2025.png"
OUT_PDF = "/home/mohammad/Desktop/1/SPI_SPEI_monthly_2000_2025.pdf"

START_DATE = "2000-01-01"   # first month of your series
WINDOW     = 12             # aggregation window in months (12 = 1-year SPI/SPEI)
Y_LIM      = 3              # +/- limit for y-axis
DPI_FIG    = 1500           # dpi for PNG (and for PDF export)

# =================== HELPERS ===================

def build_file_list(folder, pattern="*.tif"):
    files = sorted(glob.glob(os.path.join(folder, pattern)))
    if not files:
        raise FileNotFoundError(f"No files found in {folder} with pattern {pattern}")
    print(f"Found {len(files)} files in {folder}")
    return files


def domain_mean_series(file_list, boundary_gdf):
    """
    For each raster in file_list:
      - mask to boundary
      - compute nanmean over land pixels.
    Return 1D numpy array with length = number of files.
    """
    # Metadata from first file
    with rasterio.open(file_list[0]) as src0:
        height, width = src0.height, src0.width
        transform = src0.transform
        nodata = src0.nodata
        crs = src0.crs

    # Ensure boundary CRS matches raster CRS
    if boundary_gdf.crs != crs:
        boundary_gdf = boundary_gdf.to_crs(crs)

    mask = geometry_mask(
        [g for g in boundary_gdf.geometry],
        out_shape=(height, width),
        transform=transform,
        invert=True
    )

    series = np.full(len(file_list), np.nan, dtype="float32")

    for i, f in enumerate(file_list):
        print("Reading:", f)
        with rasterio.open(f) as src:
            arr = src.read(1).astype("float32")
            nd = src.nodata if src.nodata is not None else nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            arr = np.where(mask, arr, np.nan)
            series[i] = np.nanmean(arr)

    return series


def rolling_z_index(values, window):
    """
    SPI/SPEI-like index:
      - rolling sum over 'window'
      - then z-score over all valid sums.
    Returns numpy array with NaN for the first (window-1) steps.
    """
    s = pd.Series(values)
    roll_sum = s.rolling(window=window, min_periods=window).sum()

    valid = roll_sum.dropna()
    if valid.empty:
        raise ValueError("No valid rolling sums; window may be too large.")

    mu = valid.mean()
    sigma = valid.std()
    if sigma == 0 or np.isnan(sigma):
        raise ValueError("Standard deviation of rolling sums is zero or NaN; cannot standardize.")

    z = (roll_sum - mu) / sigma
    return z.to_numpy(dtype="float32")


def plot_spi_spei(dates, spi, spei, out_png, out_pdf, y_lim=3, dpi_fig=1500):
    """
    Two-panel bar plot:
    - Top: SPI-like (from P)
    - Bottom: SPEI-like (from P - E)
    Saves both PNG and PDF.
    """
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)

    def plot_panel(ax, idx_values, title):
        idx = np.array(idx_values, dtype="float32")
        pos = np.where(idx > 0, idx, 0)
        neg = np.where(idx < 0, idx, 0)

        # width ~ 20 days so bars almost touch
        ax.bar(dates, pos, width=20, color="blue", align="center")
        ax.bar(dates, neg, width=20, color="red",  align="center")

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylim(-y_lim, y_lim)
        ax.set_ylabel("Index")
        ax.set_title(title, fontsize=11)

        # Ticks every 2 years for readability
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plot_panel(axes[0], spi,  "STANDARDIZED PRECIPITATION INDEX (SPI-like)")
    plot_panel(axes[1], spei, "STANDARDIZED PRECIPITATION EVAPOTRANSPIRATION INDEX (SPEI-like)")

    axes[1].set_xlabel("Year")

    plt.tight_layout()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    os.makedirs(os.path.dirname(out_pdf), exist_ok=True)

    # Save PNG and PDF
    plt.savefig(out_png, dpi=dpi_fig, bbox_inches="tight")
    plt.savefig(out_pdf, dpi=dpi_fig, bbox_inches="tight")

    plt.close(fig)
    print(f"\nSaved PNG: {out_png}")
    print(f"Saved PDF: {out_pdf}")

# =================== MAIN ===================

def main():
    # File lists
    P_files = build_file_list(PRECIP_DIR, "*.tif")
    E_files = build_file_list(EVAP_DIR, "*.tif")

    if len(P_files) != len(E_files):
        raise ValueError(f"Different number of P ({len(P_files)}) and E ({len(E_files)}) files")

    n_time = len(P_files)
    print(f"Number of monthly time steps: {n_time}")

    # Build monthly datetime index
    dates = pd.date_range(start=START_DATE, periods=n_time, freq="MS")

    # Boundary shapefile
    boundary = gpd.read_file(BOUNDARY_SHP)

    # Domain mean P and E
    P_dom = domain_mean_series(P_files, boundary)
    E_dom = domain_mean_series(E_files, boundary)

    print("\nDomain-mean P (first 5 months):", P_dom[:5])
    print("Domain-mean E (first 5 months):", E_dom[:5])

    # SPI from P only
    spi = rolling_z_index(P_dom, WINDOW)

    # SPEI from climate = P - E
    climate = P_dom - E_dom
    spei = rolling_z_index(climate, WINDOW)

    # Plot & save both formats
    plot_spi_spei(dates, spi, spei, OUT_PNG, OUT_PDF, y_lim=Y_LIM, dpi_fig=DPI_FIG)


if __name__ == "__main__":
    main()
