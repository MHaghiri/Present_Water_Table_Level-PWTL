#!/usr/bin/env python3

import grass.script as gs  # If you're not actually using GRASS functions, you can remove this import
import xarray as xr
import os

# -----------------------------------------------------------------------------
# 1) File paths and folder setup
# -----------------------------------------------------------------------------
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/Open_water_evap.nc'
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/annual'

os.makedirs(output_dir, exist_ok=True)

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found.")
    exit(1)

# -----------------------------------------------------------------------------
# 2) Open dataset WITHOUT decoding time
# -----------------------------------------------------------------------------
try:
    # decode_times=False means Xarray will keep 'time' as raw numeric values
    ds = xr.open_dataset(input_file, decode_times=False)
except Exception as e:
    print(f"Error opening the dataset: {e}")
    exit(1)

# Ensure 'time' dimension exists
if 'time' not in ds.dims:
    print("Error: No 'time' dimension found in the dataset.")
    exit(1)

total_time_steps = ds.dims['time']
if total_time_steps == 0:
    print("Error: 'time' dimension has length 0.")
    exit(1)

# -----------------------------------------------------------------------------
# 3) Split dataset by 1-year blocks (12 months per block)
# -----------------------------------------------------------------------------
years_per_block = 1
time_steps_per_block = 12  # 12 months = 1 year

# We'll assume the data starts in January 2015
start_year = 2015

for start_idx in range(0, total_time_steps, time_steps_per_block):
    end_idx = start_idx + time_steps_per_block

    # Extract a 1-year chunk: 12 time steps
    one_year_block = ds.isel(time=slice(start_idx, end_idx))

    # Compute the "end year" for the file name
    end_year = start_year + years_per_block - 1

    # Create file name (e.g., "open_water_evaporation_2015CE_2015CE.nc")
    file_name = f"open_water_evaporation_{start_year}CE_{end_year}CE.nc"
    output_file = os.path.join(output_dir, file_name)

    # -----------------------------------------------------------------------------
    # 4) Save the chunk
    # -----------------------------------------------------------------------------
    try:
        one_year_block.to_netcdf(output_file)
        print(f"Saved 1-year block from {start_year}CE to {end_year}CE in {output_file}")
    except Exception as e:
        print(f"Error saving '{output_file}': {e}")

    # Increment the year for the next block
    start_year = end_year + 1

print("All 1-year blocks have been processed and saved.")
