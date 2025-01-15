#!/usr/bin/env python3

import grass.script as gs
import os
import xarray as xr

# Path to the folder containing the NetCDF files
input_folder = "/home/mohammad/Desktop/importnc/2100/Teraclimate/evap"
output_folder = "/home/mohammad/Desktop/importnc/2100/Teraclimate/evap/converted"

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Conversion factor from mm to m
mm_to_m = 1 / 1000

# Process each NetCDF file in the folder
for file in os.listdir(input_folder):
    if file.endswith(".nc"):
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)

        # Open the NetCDF file
        with xr.open_dataset(input_path) as ds:
            # Identify the variable containing evaporation data
            # Replace 'aet' with the actual variable name if different
            evap_var = ds['aet']

            # Convert the unit from mm to m
            evap_converted = evap_var * mm_to_m

            # Update the dataset with the converted data
            ds['aet'] = evap_converted
            ds['aet'].attrs['units'] = 'm'

            # Save the modified dataset to the output folder
            ds.to_netcdf(output_path)

print("Conversion complete. Converted files are saved in:", output_folder)
