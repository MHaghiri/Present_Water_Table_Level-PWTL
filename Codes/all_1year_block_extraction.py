#!/usr/bin/env python3

import grass.script as gs
import xarray as xr
import os

# File path for the input data
input_file = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/pr/pr_Amon_CESM2_ssp126_r4i1p1f1_gn_20150115-21001215.nc'

# Directory to save the output files
output_dir = '/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/pr/'

# Create output directory if it doesn't exist
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

# Define the length of a 10-year period (assuming monthly data, 12 months per year)
years_per_block = 1  # Change as per your requirement (e.g., 10 for 10-year blocks)
time_steps_per_block = years_per_block * 12

# Define the starting year (assume 2015 CE)
start_year = 2015

# Loop through the dataset and save each 10-year block
for start in range(0, total_time_steps, time_steps_per_block):
    end = start + time_steps_per_block
    ten_year_block = ds.isel(time=slice(start, end))

    # Calculate the corresponding time period in CE
    end_year = start_year + years_per_block - 1
    
    # Output file name for the current block
    file_name = f"precipitation_{start_year}CE_{end_year}CE.nc"
    output_file = os.path.join(output_dir, file_name)
    
    # Save the selected block as a new NetCDF file
    try:
        ten_year_block.to_netcdf(output_file)
        print(f"Saved 1-year block from {start_year}CE to {end_year}CE to {output_file}")
    except Exception as e:
        print(f"Error saving the file '{output_file}': {e}")
    
    # Update the start year for the next block
    start_year = end_year + 1

print("All blocks have been processed and saved.")
