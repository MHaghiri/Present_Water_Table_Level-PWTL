#!/usr/bin/env python3

import grass.script as gs  # Remove if not needed
import xarray as xr
import os

# -----------------------------------------------------------------------------
# 1) File paths and folder setup
# -----------------------------------------------------------------------------
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/Open_water_evap.nc'
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/monthly'

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found.")
    exit(1)

# -----------------------------------------------------------------------------
# 2) Open dataset WITHOUT decoding time
#    This way, 'time' remains in "months since 2015-01-01" integer format.
# -----------------------------------------------------------------------------
try:
    ds = xr.open_dataset(input_file, decode_times=False)
except Exception as e:
    print(f"Error opening the dataset: {e}")
    exit(1)

# Ensure 'time' dimension exists and is non-empty
if 'time' not in ds.dims:
    print("Error: No 'time' dimension found in the dataset.")
    exit(1)

total_time_steps = ds.dims['time']
if total_time_steps == 0:
    print("Error: 'time' dimension has length 0.")
    exit(1)

# -----------------------------------------------------------------------------
# 3) Define the base year and a dict to gather (year, month) -> list of indices
# -----------------------------------------------------------------------------
base_year = 2015
monthly_data = {}  # key: (year, month), value: list of time indices

# -----------------------------------------------------------------------------
# 4) Loop over each time index, determine (year, month), and group
# -----------------------------------------------------------------------------
for i in range(total_time_steps):
    # Get raw "months since 2015-01-01" (make sure it's an integer)
    months_since_2015 = int(ds['time'].values[i])

    # Calculate the "month" (1..12) and "year"
    month = (months_since_2015 % 12) + 1
    year  = base_year + (months_since_2015 // 12)

    # Build the dictionary key
    dict_key = (year, month)
    if dict_key not in monthly_data:
        monthly_data[dict_key] = []
    monthly_data[dict_key].append(i)

# -----------------------------------------------------------------------------
# 5) For each (year, month), create a sub-dataset and save
# -----------------------------------------------------------------------------
for (year, month), indices in monthly_data.items():
    # Slice out only these time steps
    month_block = ds.isel(time=indices)

    # Construct output file name, e.g. open_water_evaporation_2015CE_month01.nc
    file_name = f"open_water_evaporation_{year}CE_month{month:02d}.nc"
    output_file = os.path.join(output_dir, file_name)

    try:
        month_block.to_netcdf(output_file)
        print(f"Saved month={month:02d} of year={year} => {output_file}")
    except Exception as e:
        print(f"Error saving '{output_file}': {e}")

print("All monthly blocks have been processed and saved.")
