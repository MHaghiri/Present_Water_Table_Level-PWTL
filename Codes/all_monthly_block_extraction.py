#!/usr/bin/env python3

import xarray as xr
import os

# File path for the input data
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/evspsbl_Amon_CESM2_ssp126_r4i1p1f1_gn_20150115-21001215.nc'

# Directory to save the output files
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/pr/monthly/'
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

# Define the starting year (assume 2015 CE)
start_year = 2015

# Loop through each year and month
for year in range(start_year, start_year + total_time_steps // 12):
    for month in range(1, 13):  # Iterate through months 1 to 12
        # Extract data for the current month and year
        monthly_data = ds.sel(time=(ds['time'].dt.month == month) & (ds['time'].dt.year == year))

        if monthly_data.time.size == 0:
            print(f"No data found for {year}-{month:02d}. Skipping...")
            continue

        # Output file name for the current month
        file_name = f"evaporation_{year}_{month:02d}.nc"
        output_file = os.path.join(output_dir, file_name)
        
        # Save the monthly data as a new NetCDF file
        try:
            monthly_data.to_netcdf(output_file)
            print(f"Saved data for {year}-{month:02d} to {output_file}")
        except Exception as e:
            print(f"Error saving the file '{output_file}': {e}")

print("All monthly blocks have been processed and saved.")
