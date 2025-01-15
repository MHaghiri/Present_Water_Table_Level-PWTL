#!/usr/bin/env python3

import grass.script as gs
import numpy as np
import xarray as xr
from scipy.interpolate import griddata
import os

def read_nc_data(filepath, variable):
    data = xr.open_dataset(filepath)
    var_data = data[variable].values
    lats = data['lat'].values
    lons = data['lon'].values
    return var_data, lats, lons

def quantile_mapping(obs_data, model_data):
    obs_data_flat = obs_data.flatten()
    model_data_flat = model_data.flatten()
    
    # Sort observed and model data
    sorted_obs = np.sort(obs_data_flat)
    sorted_model = np.sort(model_data_flat)
    
    # Interpolate
    sorted_model = np.linspace(0, 1, len(sorted_model))
    sorted_obs = np.interp(sorted_model, np.linspace(0, 1, len(sorted_obs)), sorted_obs)
    
    # Apply quantile mapping
    model_data_cdf = np.argsort(np.argsort(model_data_flat)) / float(len(model_data_flat) - 1)
    mapped_data = np.interp(model_data_cdf, np.linspace(0, 1, len(sorted_obs)), sorted_obs)
    
    return mapped_data.reshape(model_data.shape)

def bilinear_interpolation(high_res_lats, high_res_lons, low_res_data, low_res_lats, low_res_lons):
    high_res_grid_lon, high_res_grid_lat = np.meshgrid(high_res_lons, high_res_lats)
    low_res_points = np.array([(lon, lat) for lat in low_res_lats for lon in low_res_lons])
    high_res_data = griddata(low_res_points, low_res_data.flatten(), (high_res_grid_lon, high_res_grid_lat), method='linear')
    return high_res_data

def save_nc_data(filepath, variable, data, lats, lons):
    # Subset longitude range (-180 to 180)
    lon_mask = (lons >= -180) & (lons <= 180)
    subset_data = data[:, lon_mask]
    subset_lons = lons[lon_mask]

    ds = xr.Dataset(
        {variable: (['lat', 'lon'], subset_data)},
        coords={'lat': lats, 'lon': subset_lons}
    )
    ds.to_netcdf(filepath)

def process_and_downscale(file_path, terraclimate_data, terraclimate_lats, terraclimate_lons, corrected_cmip2015_data, cmip2015_lats, cmip2015_lons, output_folder):
    # Read the CMIP data for the given file (e.g., 2100)
    cmip_data, cmip_lats, cmip_lons = read_nc_data(file_path, 'evspsbl')

    # Bias Correction (Quantile Mapping) using corrected CMIP6 2020 data
    corrected_cmip_data = quantile_mapping(corrected_cmip2015_data, cmip_data)

    # Spatial Downscaling (Bilinear Interpolation)
    downscaled_cmip_data = bilinear_interpolation(terraclimate_lats, terraclimate_lons, corrected_cmip_data, cmip_lats, cmip_lons)

    # Save the downscaled data to a new NetCDF file with the same name as the input file
    output_file = os.path.join(output_folder, os.path.basename(file_path))
    save_nc_data(output_file, 'evap', downscaled_cmip_data, terraclimate_lats, terraclimate_lons)

    print(f"Saved: {output_file}")

# Directory containing the input maps
input_folder = "/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/evap/annual/evap"
output_folder = "/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/evap/annual/evap/downscale"

# Create the output directory if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Read TerraClimate data for 2020 (high resolution)
terraclimate_data, terraclimate_lats, terraclimate_lons = read_nc_data("/home/mohammad/Desktop/importnc/2100/Teraclimate/evap/filled/Terraclimate_open_water_evaporation_2020_filled.nc", 'evaporation')

# Read CMIP6 data for 2020 (to use for bias correction)
cmip2015_data, cmip2015_lats, cmip2015_lons = read_nc_data("/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/evap/annual/evap/evaporation_2020CE_2020CE_annual.nc", 'evspsbl')

# Bias correction: Apply quantile mapping on CMIP6 2020 data using TerraClimate 2020 data
corrected_cmip2015_data = quantile_mapping(terraclimate_data, cmip2015_data)

# Loop through all files in the input folder and downscale each
for file_name in os.listdir(input_folder):
    if file_name.endswith(".nc"):  # Process only NetCDF files
        file_path = os.path.join(input_folder, file_name)
        process_and_downscale(file_path, terraclimate_data, terraclimate_lats, terraclimate_lons, corrected_cmip2015_data, cmip2015_lats, cmip2015_lons, output_folder)

print("All files processed, downscaled, and saved.")