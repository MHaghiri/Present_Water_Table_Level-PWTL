#!/usr/bin/env python3

import numpy as np
from netCDF4 import Dataset

# Function to calculate VPD for a given time range
def get_VPD(temperature_file, relhum_file, outfile):
    # Constants
    l_vap = 2.5 * 10**6  # Latent heat of vaporization, J/kg
    R_v = 461.  # Specific gas constant of water vapor, J/kg/K
    
    # Import temperature and relative humidity files
    rootgrp_T = Dataset(temperature_file, 'r', format='NETCDF4')
    rootgrp_RH = Dataset(relhum_file, 'r', format='NETCDF4')

    # Variable names in the NetCDF files
    temperature_var_name = 'tas'  # Variable for temperature
    relhum_var_name = 'hur'      # Variable for relative humidity

    # Get temperature and relative humidity
    temperature = rootgrp_T[temperature_var_name][:]  # Extract temperature (time, lat, lon)
    relhum = rootgrp_RH[relhum_var_name][:]           # Extract relative humidity (time, 1, lat, lon)

    # Squeeze the relative humidity array to remove the extra dimension
    relhum = np.squeeze(relhum)  # Shape becomes (time, lat, lon)

    # Debugging: Print shapes of temperature and relative humidity
    print(f"Temperature shape: {temperature.shape}")
    print(f"Relative Humidity shape after squeezing: {relhum.shape}")

    # Ensure temperature and humidity dimensions match
    assert temperature.shape == relhum.shape, "Temperature and Relative Humidity data do not match in shape"

    # Dimensions
    num_time_steps = temperature.shape[0]
    lat_dim = temperature.shape[1]
    lon_dim = temperature.shape[2]

    # Export to NetCDF
    rootgrp_out = Dataset(outfile, "w", format="NETCDF4")
    rootgrp_out.createDimension("lat", lat_dim)
    rootgrp_out.createDimension("lon", lon_dim)
    rootgrp_out.createDimension("time", num_time_steps)

    latitudes = rootgrp_out.createVariable("lat", "f4", ("lat",))
    longitudes = rootgrp_out.createVariable("lon", "f4", ("lon",))
    time_var = rootgrp_out.createVariable("time", "f4", ("time",))
    values = rootgrp_out.createVariable("VPD", "f4", ("time", "lat", "lon",))

    latitudes.units = "degrees north"
    longitudes.units = "degrees east"
    values.units = "kPa"

    # Add metadata
    rootgrp_out.description = "Vapour Pressure Deficit (VPD) calculated using relative humidity and temperature"

    # Generate latitude and longitude arrays
    latitudes[:] = rootgrp_T['lat'][:]
    longitudes[:] = rootgrp_T['lon'][:]
    time_var[:] = rootgrp_T['time'][:]

    # Calculate VPD
    for t in range(num_time_steps):
        temp = temperature[t, :, :]  # Temperature for current time step
        rh = relhum[t, :, :]         # Relative humidity for current time step

        # Calculate saturated vapor pressure (esat)
        esat = 611 * np.exp(l_vap / R_v * (1 / 273.15 - 1 / temp))

        # Calculate actual vapor pressure (ea)
        ea = (rh / 100) * esat

        # Calculate VPD in kPa
        VPD = (esat - ea) / 1000  # Convert to kPa

        # Ensure VPD values are non-negative
        VPD = np.maximum(VPD, 0)

        # Save VPD data for the current time step
        values[t, :, :] = VPD

    # Close the output file
    rootgrp_out.close()
    print(f"VPD saved to {outfile}")

# File paths
temperature_file = "/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/tas_Amon_CESM2_ssp126_r4i1p1f1_gn_20150115-21001215.nc"
relhum_file = "/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/hur_Amon_CESM2_ssp126_r4i1p1f1_gn_20150115-21001215.nc"
output_file = "/home/mohammad/Desktop/importnc/2100/2CESM2/ssp1/owe/VPD_2015-2100_kPa.nc"

# Run the function
get_VPD(temperature_file, relhum_file, output_file)
