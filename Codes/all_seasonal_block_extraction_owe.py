#!/usr/bin/env python3

import grass.script as gs  # Remove if not needed
import xarray as xr
import os

# -----------------------------------------------------------------------------
# 1) File paths and folder setup
# -----------------------------------------------------------------------------
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/Open_water_evap.nc'
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/seasonal'

os.makedirs(output_dir, exist_ok=True)

if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found.")
    exit(1)

# -----------------------------------------------------------------------------
# 2) Open dataset WITHOUT decoding time
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
# 3) Define the meteorological seasons
# -----------------------------------------------------------------------------
seasons = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall":   [9, 10, 11]
}

# For reference, the dataset's base year (assuming time=0 => Jan 2015)
base_year = 2015

# We'll group time-step indices by (year, season).
season_data = {}  # key: (year, season), value: list of time indices

# -----------------------------------------------------------------------------
# 4) Figure out which season each time index belongs to
# -----------------------------------------------------------------------------
for i in range(total_time_steps):
    # Raw "months since 2015-01-01" value:
    months_since_2015 = ds['time'].values[i]
    # If it's actually a float, ensure integer:
    months_since_2015 = int(months_since_2015)

    # Calculate the "month" and "year" from that
    # Example: if months_since_2015=0 => January 2015
    #          if months_since_2015=1 => February 2015
    #          if months_since_2015=11 => December 2015
    # month in [1..12]
    month = (months_since_2015 % 12) + 1
    # year
    year = base_year + (months_since_2015 // 12)

    # Figure out which season this month belongs to
    # (Could be Winter, Spring, Summer, or Fall)
    season_name = None
    for s_name, months in seasons.items():
        if month in months:
            season_name = s_name
            break

    if season_name is None:
        # If for some reason month wasn't 1..12, skip or raise an error
        print(f"Warning: Could not determine season for time index {i} (month={month})")
        continue

    # Store this index in our dictionary
    # We'll group by (year, season_name)
    dict_key = (year, season_name)
    if dict_key not in season_data:
        season_data[dict_key] = []
    season_data[dict_key].append(i)

# -----------------------------------------------------------------------------
# 5) For each (year, season), slice out all relevant time steps and save
# -----------------------------------------------------------------------------
for (year, season_name), indices in season_data.items():
    # Slice the dataset to only these time steps
    season_block = ds.isel(time=indices)

    # Output file name (e.g., "open_water_evaporation_2015CE_Winter.nc")
    file_name = f"open_water_evaporation_{year}CE_{season_name}.nc"
    output_file = os.path.join(output_dir, file_name)

    try:
        season_block.to_netcdf(output_file)
        print(f"Saved {season_name} of {year}CE => {output_file}")
    except Exception as e:
        print(f"Error saving '{output_file}': {e}")

print("All seasonal blocks have been processed and saved.")
