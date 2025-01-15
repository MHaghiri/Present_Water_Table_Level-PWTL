#!/usr/bin/env python3

import xarray as xr
import os

# File path for the input data
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/pr_Amon_CESM2_ssp126_r4i1p1f1_gn_20150115-21001215.nc'

# Directory to save the output files
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/pr/seasonal/'
os.makedirs(output_dir, exist_ok=True)

# Check if the input file exists
if not os.path.exists(input_file):
    print(f"Error: Input file '{input_file}' not found. Please check the file path.")
    exit(1)

# Open the NetCDF file
try:
    ds = xr.open_dataset(input_file)
except Exception as e:
    print(f"Error opening the dataset: {e}")
    exit(1)

# Get the total number of time steps in the dataset
total_time_steps = ds.dims.get('time', 0)

if total_time_steps == 0:
    print("Error: No time dimension found in the dataset. Please check the file structure.")
    exit(1)

# Define the mapping of months to seasons
seasons = {
    "Winter": [12, 1, 2],
    "Spring": [3, 4, 5],
    "Summer": [6, 7, 8],
    "Fall": [9, 10, 11]
}

# Define the starting year (2015 CE)
start_year = 2015

# Loop through each year and season
for year in range(start_year, start_year + total_time_steps // 12):
    for season, months in seasons.items():
        # Extract data for the specified months in the current year
        seasonal_data = ds.sel(time=ds['time'].dt.month.isin(months) & (ds['time'].dt.year == year))

        if seasonal_data.time.size == 0:
            print(f"No data found for {season} {year}. Skipping...")
            continue

        # Output file name for the current season
        file_name = f"precipitation_{year}_{season}.nc"
        output_file = os.path.join(output_dir, file_name)
        
        # Save the seasonal block as a new NetCDF file
        try:
            seasonal_data.to_netcdf(output_file)
            print(f"Saved {season} data for {year} to {output_file}")
        except Exception as e:
            print(f"Error saving the file '{output_file}': {e}")

print("All seasonal blocks have been processed and saved.")
