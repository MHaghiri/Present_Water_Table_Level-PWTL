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
WTD_DIR = "/home/mohammad/Desktop/1800-2015"

# Boundary (NO Greenland)
BOUNDARY_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_boundery_without_greenland.shp"

# Watersheds (NO Greenland) → use name1 column
WATERSHED_SHP = "/home/mohammad/Desktop/N_America_shapefile/N_America_level2_watershed_without_greenland.shp"

OUT_CSV = "/home/mohammad/Desktop/1800-2015/wtd_means.csv"
OUT_PNG = "/home/mohammad/Desktop/1800-2015/wtd_na_mean_timeseries.png"


# ============================================================
# YEAR RANGE (EDIT HERE)
# ============================================================
START_YEAR = 1875
END_YEAR   = 2015


# ============================================================
# HELPERS
# ============================================================
def year_from_filename(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"N_America_(\d{6})_petsc_", base)
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


# ============================================================
# MAIN
# ============================================================
def main():
    tif_list = sorted(glob.glob(os.path.join(WTD_DIR, "N_America_*_petsc_*.tif")))
    if not tif_list:
        raise FileNotFoundError(f"No GeoTIFFs found in {WTD_DIR}")

    files, years = [], []
    for fp in tif_list:
        y = year_from_filename(fp)
        if START_YEAR <= y <= END_YEAR:
            files.append(fp)
            years.append(y)

    if not files:
        raise FileNotFoundError(f"No files found in range {START_YEAR}–{END_YEAR}")

    with rasterio.open(files[0]) as src0:
        raster_crs = src0.crs
        raster_transform = src0.transform
        raster_shape = (src0.height, src0.width)

    boundary = gpd.read_file(BOUNDARY_SHP).to_crs(raster_crs)
    watersheds = gpd.read_file(WATERSHED_SHP).to_crs(raster_crs)

    # IMPORTANT: use 'name1'
    if "name1" not in watersheds.columns:
        raise ValueError("'name1' column not found in watershed shapefile")

    boundary_geom = boundary.unary_union

    boundary_mask = rasterize(
        [(boundary_geom, 1)],
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype="uint8",
        all_touched=False,
    ).astype(bool)

    watersheds = watersheds[watersheds["name1"].notna()].copy()
    watersheds["name1"] = watersheds["name1"].astype(str)

    watersheds = watersheds.reset_index(drop=True)
    watersheds["ws_id"] = np.arange(1, len(watersheds) + 1)

    ws_label = rasterize(
        [(geom, int(ws_id)) for geom, ws_id in zip(watersheds.geometry, watersheds["ws_id"])],
        out_shape=raster_shape,
        transform=raster_transform,
        fill=0,
        dtype="int32",
        all_touched=False,
    )

    ws_ids = list(watersheds["ws_id"].values)
    ws_cols = [safe_colname(n) for n in watersheds["name1"].values]

    rows = []

    for fp, y in sorted(zip(files, years), key=lambda t: t[1]):
        with rasterio.open(fp) as src:
            data = src.read(1)
            nodata = src.nodata

        valid = make_valid_mask(data, nodata)

        na_valid = valid & boundary_mask
        na_mean = float(np.nan) if not np.any(na_valid) else float(np.mean(data[na_valid]))

        ws_means = []
        for ws_id in ws_ids:
            m = valid & (ws_label == ws_id)
            ws_means.append(float(np.nan) if not np.any(m) else float(np.mean(data[m])))

        row = {"year": y, "NA_mean": na_mean}
        for col, val in zip(ws_cols, ws_means):
            if col in row:
                k = 2
                newcol = f"{col}_{k}"
                while newcol in row:
                    k += 1
                    newcol = f"{col}_{k}"
                col = newcol
            row[col] = val

        rows.append(row)
        print(f"Done {y}: NA_mean={na_mean:.4f}")

    df = pd.DataFrame(rows).sort_values("year")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved CSV: {OUT_CSV}")

    plt.figure(figsize=(12, 4))
    plt.plot(df["year"], df["NA_mean"])
    plt.xlabel("Year")
    plt.ylabel("Mean WTD (NA boundary)")
    plt.title(f"Mean WTD over North America (without Greenland), {START_YEAR}–{END_YEAR}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=300)
    plt.close()
    print(f"Saved plot: {OUT_PNG}")


if __name__ == "__main__":
    main()
