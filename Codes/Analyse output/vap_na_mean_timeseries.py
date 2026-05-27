#!/usr/bin/env python3
import os
import re
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import matplotlib.pyplot as plt

# ============================================================
# PATHS (YOUR PATHS)
# ============================================================
EVAP_DIR = "/media/mohammad/My Book/1800/evap/downscaled/tif/N_America"

# Boundary (NO Greenland)
BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

# Watersheds (NO Greenland) → use name1 column
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"

OUT_CSV = "/media/mohammad/My Book/1800/evap/downscaled/tif/N_America/evap_means2.csv"
OUT_PNG = "/media/mohammad/My Book/1800/evap/downscaled/tif/N_America/evap_na_mean_timeseries2.png"

# ============================================================
# YEAR RANGE (EDIT HERE)
# ============================================================
START_YEAR = 2000
END_YEAR   = 2015

# ============================================================
# HELPERS
# ============================================================
def year_from_filename(path: str) -> int:
    """
    Parse year from filenames like:
      N_America_001800_evaporation.tif
      N_America_002015_evaporation.tif
    """
    base = os.path.basename(path)
    m = re.search(r"N_America_(\d{6})_evaporation\.tif$", base)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {base}")
    return int(m.group(1))


def make_valid_mask(data: np.ndarray, nodata):
    if nodata is None:
        return np.isfinite(data)
    return np.isfinite(data) & (data != nodata)


def safe_colname(name: str) -> str:
    s = str(name).strip()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^0-9A-Za-z_]+", "", s)
    return s if s else "watershed"


def find_name1_column(gdf: gpd.GeoDataFrame) -> str:
    """
    Your shapefile might have name1 / Name1 / NAME1, etc.
    This finds it robustly.
    """
    cols_lower = {c.lower(): c for c in gdf.columns}
    if "name1" in cols_lower:
        return cols_lower["name1"]
    raise ValueError("Could not find a 'name1' column (case-insensitive) in watershed shapefile.")


# ============================================================
# MAIN
# ============================================================
def main():
    # Correct filename pattern for evaporation
    tif_list = sorted(glob.glob(os.path.join(EVAP_DIR, "N_America_*_evaporation.tif")))
    if not tif_list:
        raise FileNotFoundError(
            f"No GeoTIFFs found in {EVAP_DIR}\n"
            f"Expected filenames like: N_America_001800_evaporation.tif"
        )

    # Filter years
    files, years = [], []
    for fp in tif_list:
        y = year_from_filename(fp)
        if START_YEAR <= y <= END_YEAR:
            files.append(fp)
            years.append(y)

    if not files:
        raise FileNotFoundError(f"No files found in range {START_YEAR}–{END_YEAR}")

    # Raster grid info
    with rasterio.open(files[0]) as src0:
        raster_crs = src0.crs
        raster_transform = src0.transform
        raster_shape = (src0.height, src0.width)

    # Read shapefiles and reproject
    boundary = gpd.read_file(BOUNDARY_SHP).to_crs(raster_crs)
    watersheds = gpd.read_file(WATERSHED_SHP).to_crs(raster_crs)

    # Robust name1 column handling
    name1_col = find_name1_column(watersheds)

    # Boundary union + rasterize once
    boundary_geom = boundary.unary_union
    boundary_mask = rasterize(
        [(boundary_geom, 1)],
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    # Watersheds prep
    watersheds = watersheds[watersheds[name1_col].notna()].copy()
    watersheds[name1_col] = watersheds[name1_col].astype(str)

    watersheds = watersheds.reset_index(drop=True)
    watersheds["ws_id"] = np.arange(1, len(watersheds) + 1)

    # Rasterize watersheds once (label grid)
    ws_label = rasterize(
        [(geom, int(ws_id)) for geom, ws_id in zip(watersheds.geometry, watersheds["ws_id"])],
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    ws_ids = list(watersheds["ws_id"].values)
    ws_cols = [safe_colname(n) for n in watersheds[name1_col].values]

    rows = []

    # Loop through rasters by year
    for fp, y in sorted(zip(files, years), key=lambda t: t[1]):
        with rasterio.open(fp) as src:
            data = src.read(1)
            nodata = src.nodata

        valid = make_valid_mask(data, nodata)

        # NA mean (within NA boundary mask)
        na_valid = valid & boundary_mask
        na_mean = float(np.nan) if not np.any(na_valid) else float(np.mean(data[na_valid]))

        # Watershed means
        ws_means = []
        for ws_id in ws_ids:
            m = valid & (ws_label == ws_id)
            ws_means.append(float(np.nan) if not np.any(m) else float(np.mean(data[m])))

        row = {"year": y, "NA_mean": na_mean}

        # Add watershed columns (ensure unique names)
        used = set(row.keys())
        for col, val in zip(ws_cols, ws_means):
            c = col
            if c in used:
                k = 2
                while f"{c}_{k}" in used:
                    k += 1
                c = f"{c}_{k}"
            row[c] = val
            used.add(c)

        rows.append(row)
        print(f"Done {y}: NA_mean={na_mean:.6g}")

    df = pd.DataFrame(rows).sort_values("year")

    # Save CSV
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")

    # Plot NA_mean
    plt.figure(figsize=(12, 4))
    plt.plot(df["year"], df["NA_mean"])
    plt.xlabel("Year")
    plt.ylabel("Mean evaporation (NA boundary)")
    plt.title(f"Mean evaporation over North America (without Greenland), {START_YEAR}–{END_YEAR}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
