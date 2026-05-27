#!/usr/bin/env python3

"""
Monthly SPI / SPEI-style indices (2000–2025) per watershed

For each Level-2 watershed polygon in WATERSHED_SHP:
  - Compute domain-mean monthly P and E
  - Build 12-month rolling SPI-like (from P) and SPEI-like (from P-E)
  - Plot all watersheds in one figure (rows = watersheds, 2 columns = SPI/SPEI)

Outputs:
  - High-resolution PNG + PDF with all watersheds
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

PRECIP_DIR = "/media/mohammad/My Book/0-2025/Monthly/pr/CMIP6/monthly/downscaled/N_America/1"
EVAP_DIR   = "/media/mohammad/My Book/0-2025/Monthly/evap/CMIP6/monthly/downscaled/N_America/1"

# Shapefile with 11 Level-2 watersheds (no Greenland)
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"

# Name field in the watershed shapefile (EDIT if needed).
# If this column is not found, generic names "WS 1", "WS 2", ... will be used.
WATERSHED_NAME_FIELD = "name1"   # <-- change to your actual name column, or set to None

OUT_PNG = "/home/mohammad/Desktop/1/SPI_SPEI_monthly_2000_2025_watersheds.png"
OUT_PDF = "/home/mohammad/Desktop/1/SPI_SPEI_monthly_2000_2025_watersheds.pdf"

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
      - mask to boundary_gdf polygon(s)
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
        print("  Reading:", f)
        with rasterio.open(f) as src:
            arr = src.read(1).astype("float32")
            nd = src.nodata if src.nodata is not None else nodata
            if nd is not None:
                arr[arr == nd] = np.nan
            arr = np.where(mask, arr, np.nan)
            series[i] = np.nanmean(arr)

    # Optional: debug info
    n_valid = np.isfinite(series).sum()
    print(f"    -> domain_mean_series: {n_valid} valid time steps (out of {len(series)})")

    return series


def rolling_z_index(values, window):
    """
    SPI/SPEI-like index:
      - rolling sum over 'window'
      - then z-score over all valid sums.

    If there are too few valid data or zero variance, returns an array of NaNs
    (same length as 'values') and prints a warning instead of raising an error.
    """
    s = pd.Series(values)

    # Check how many valid (non-NaN) points are in the original series
    n_valid = s.notna().sum()
    if n_valid < window:
        print(f"    [rolling_z_index WARNING] Only {n_valid} valid points "
              f"(window={window}). Returning all NaNs for this series.")
        return np.full(len(values), np.nan, dtype="float32")

    roll_sum = s.rolling(window=window, min_periods=window).sum()

    valid = roll_sum.dropna()
    if valid.empty:
        print("    [rolling_z_index WARNING] All rolling sums are NaN. "
              "Returning all NaNs for this series.")
        return np.full(len(values), np.nan, dtype="float32")

    mu = valid.mean()
    sigma = valid.std()

    if sigma == 0 or np.isnan(sigma):
        print("    [rolling_z_index WARNING] Zero or NaN standard deviation "
              "of rolling sums. Returning all NaNs for this series.")
        return np.full(len(values), np.nan, dtype="float32")

    z = (roll_sum - mu) / sigma
    return z.to_numpy(dtype="float32")


def plot_spi_spei_watersheds(dates, spi_list, spei_list, names,
                             out_png, out_pdf, y_lim=3, dpi_fig=1500):
    """
    Plot SPI-like and SPEI-like indices for all watersheds in a single figure.

    spi_list, spei_list: list of 1D arrays (length = time) per watershed
    names: list of watershed names (same order)
    """
    n_ws = len(spi_list)

    fig, axes = plt.subplots(
        n_ws, 2,
        figsize=(13, 2.0 * n_ws),  # height scales with number of watersheds
        sharex=True,
        sharey=True
    )

    # Ensure axes is 2D array even if n_ws == 1
    if n_ws == 1:
        axes = np.array([axes])

    def plot_panel(ax, idx_values):
        idx = np.array(idx_values, dtype="float32")
        pos = np.where(idx > 0, idx, 0)
        neg = np.where(idx < 0, idx, 0)

        ax.bar(dates, pos, width=20, color="blue", align="center")
        ax.bar(dates, neg, width=20, color="red",  align="center")

        ax.axhline(0, color="black", linewidth=0.6)
        ax.set_ylim(-y_lim, y_lim)

        # Ticks every 2 years for readability
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    for i in range(n_ws):
        ax_spi  = axes[i, 0]
        ax_spei = axes[i, 1]

        plot_panel(ax_spi,  spi_list[i])
        plot_panel(ax_spei, spei_list[i])

        # Left column: y-label on each row; right column: none
        ax_spi.set_ylabel("Index")

        # Watershed name on left subplot
        ax_spi.text(
            0.01, 0.85,
            names[i],
            transform=ax_spi.transAxes,
            fontsize=9,
            fontweight="bold",
            ha="left",
            va="center"
        )

        # Only bottom row gets x-labels
        if i == n_ws - 1:
            ax_spi.set_xlabel("Year")
            ax_spei.set_xlabel("Year")

    # Column titles
    axes[0, 0].set_title("STANDARDIZED PRECIPITATION INDEX (SPI-like)", fontsize=11)
    axes[0, 1].set_title("STANDARDIZED PRECIPITATION EVAPOTRANSPIRATION INDEX (SPEI-like)", fontsize=11)

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

    # Read watersheds shapefile
    watersheds = gpd.read_file(WATERSHED_SHP)
    n_ws = len(watersheds)
    print(f"Number of watershed polygons: {n_ws}")

    # Get names
    if WATERSHED_NAME_FIELD and (WATERSHED_NAME_FIELD in watersheds.columns):
        names = watersheds[WATERSHED_NAME_FIELD].astype(str).tolist()
    else:
        names = [f"WS {i+1}" for i in range(n_ws)]
        if WATERSHED_NAME_FIELD:
            print(f"Warning: field '{WATERSHED_NAME_FIELD}' not found in shapefile. "
                  f"Using generic names WS 1..WS {n_ws}.")

    spi_list  = []
    spei_list = []

    # Loop over watersheds
    for i in range(n_ws):
        print(f"\n=== Processing watershed {i+1}/{n_ws}: {names[i]} ===")
        ws_boundary = watersheds.iloc[[i]].copy()  # keep as GeoDataFrame

        # Domain-mean P and E for this watershed
        P_dom = domain_mean_series(P_files, ws_boundary)
        E_dom = domain_mean_series(E_files, ws_boundary)

        # SPI from P
        spi = rolling_z_index(P_dom, WINDOW)

        # SPEI from climate balance = P - E
        climate = P_dom - E_dom
        spei = rolling_z_index(climate, WINDOW)

        spi_list.append(spi)
        spei_list.append(spei)

    # Plot & save both formats
    plot_spi_spei_watersheds(dates, spi_list, spei_list, names,
                             OUT_PNG, OUT_PDF, y_lim=Y_LIM, dpi_fig=DPI_FIG)


if __name__ == "__main__":
    main()
