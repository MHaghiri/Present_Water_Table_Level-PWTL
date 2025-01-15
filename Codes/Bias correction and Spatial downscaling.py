#!/usr/bin/env python3

import grass.script as gs

import xarray as xr
import numpy as np
import xesmf as xe

##############################################################################
# 1. Load the data
##############################################################################

# Paths to your NetCDF files
tera_file   = 'TeraClimate_2020.nc'  # Reference observational dataset
cmip_hist   = 'CMIP6_2020.nc'       # CMIP6 for year 2020 (historical / present)
cmip_future = 'CMIP6_2100.nc'       # CMIP6 for year 2100 (future projection)

# Open datasets
ds_ref       = xr.open_dataset(tera_file)   # TeraClimate (reference)
ds_mod_hist  = xr.open_dataset(cmip_hist)   # CMIP6 historical/present
ds_mod_fut   = xr.open_dataset(cmip_future) # CMIP6 future

# Example: Let's assume our variable of interest is called 'temperature' 
# in each dataset. Adjust to match your actual variable name(s).
var_name = 'temperature'

##############################################################################
# 2. Spatial Regridding (Downscaling) Using xESMF
##############################################################################
# We'll create a regridder that maps from the CMIP6 grid to the TeraClimate grid.

# 2.1. Prepare the regridder
regridder = xe.Regridder(
    ds_mod_hist,        # Source grid (CMIP6)
    ds_ref,             # Target grid (TeraClimate)
    method='bilinear',  # or 'nearest_s2d', 'conservative', etc.
    reuse_weights=False # Set True if re-using existing weights
)

# 2.2. Regrid the historical and future CMIP6 data to TeraClimate resolution
ds_mod_hist_down = regridder(ds_mod_hist[var_name])  # downscaled historical
ds_mod_fut_down  = regridder(ds_mod_fut[var_name])   # downscaled future

# Now both ds_mod_hist_down and ds_mod_fut_down have the same spatial
# resolution and coordinates as ds_ref[var_name].

##############################################################################
# 3. Bias Correction via Quantile Mapping
##############################################################################
# We'll implement a simple quantile-mapping function that operates along 
# time for each grid cell independently. For demonstration purposes, we’ll 
# assume we have daily or monthly data, and we correct each grid cell in 
# the same manner. Adjust as needed for your data volume and time resolution.

def quantile_mapping(
    ref_data: xr.DataArray, 
    hist_data: xr.DataArray, 
    fut_data: xr.DataArray
) -> xr.DataArray:
    """
    Perform empirical quantile mapping to bias-correct `fut_data` 
    based on the distribution of `ref_data` (observations) 
    and `hist_data` (model historical).
    
    Assumes all DataArrays are on the same spatial grid 
    and share a 'time' dimension.
    """
    # Make sure dimensions match (except time, which should match in length).
    # If not, you may need to align or reindex.
    
    # Flatten over time so we can work with distributions
    # (Alternatively, you can do a time-slice approach, e.g. monthly).
    # For each grid cell, we do the correction.
    
    corrected = xr.full_like(fut_data, np.nan)  # empty container

    # We'll iterate over each lat-lon. (For large data, consider vectorized or dask approaches)
    for i in range(ref_data.lat.size):
        for j in range(ref_data.lon.size):

            # 1. Extract reference (obs) and historical (model) at [i,j]
            ref_series = ref_data[:, i, j].values
            hist_series = hist_data[:, i, j].values
            fut_series = fut_data[:, i, j].values

            # Remove NaNs if present
            valid_ref = np.isfinite(ref_series)
            valid_hist = np.isfinite(hist_series)
            valid_fut = np.isfinite(fut_series)

            # If there's not enough valid data, skip
            if (valid_ref.sum() < 10) or (valid_hist.sum() < 10):
                # you can choose a threshold that makes sense for your data
                continue

            ref_series = ref_series[valid_ref]
            hist_series = hist_series[valid_hist]
            # We'll correct only the valid part of fut_series
            fut_valid_index = np.where(valid_fut)[0]
            fut_series_valid = fut_series[valid_fut]

            # 2. Sort the reference and historical series
            ref_sorted = np.sort(ref_series)
            hist_sorted = np.sort(hist_series)

            # 3. Compute fractional ranks for hist & ref
            ref_cdf = np.linspace(1./(len(ref_sorted)+1), 
                                  1-1./(len(ref_sorted)+1), 
                                  len(ref_sorted))
            hist_cdf = np.linspace(1./(len(hist_sorted)+1), 
                                   1-1./(len(hist_sorted)+1), 
                                   len(hist_sorted))

            # 4. For each future value, find its percentile in hist, 
            #    then map to ref
            #    a) For each fut_value, find approximate percentile in hist
            fut_percentiles = []
            for fv in fut_series_valid:
                # fraction of hist values less than fv
                cdf_val = (hist_sorted < fv).sum() / len(hist_sorted)
                fut_percentiles.append(cdf_val)

            fut_percentiles = np.array(fut_percentiles)

            #    b) Convert that percentile to the reference distribution
            # Use interpolation across ref_cdf -> ref_sorted
            # We do an inverse transform: percentile -> value in ref_sorted
            corrected_vals = np.interp(
                fut_percentiles,  # x
                ref_cdf,          # xp
                ref_sorted        # fp
            )

            # Put these corrected values back into the corrected array
            corrected[:, i, j][fut_valid_index] = corrected_vals

    return corrected


# 3.1. Apply Quantile Mapping
ds_bc_future = quantile_mapping(
    ref_data  = ds_ref[var_name],       # TeraClimate reference (2020)
    hist_data = ds_mod_hist_down,       # CMIP6 historical (2020) downscaled
    fut_data  = ds_mod_fut_down         # CMIP6 future (2100) downscaled
)

##############################################################################
# 4. Prepare Output
##############################################################################

# ds_bc_future is now an xarray.DataArray with the same grid as ds_ref
# and hopefully with corrected biases (relative to TeraClimate).
# You might want to package it back into a Dataset and save to NetCDF:

ds_out = xr.Dataset(
    {f"{var_name}_bias_corrected": ds_bc_future},
    coords={
        "time": ds_mod_fut_down.time,
        "lat": ds_ref.lat,
        "lon": ds_ref.lon
    }
)

# Save to a new NetCDF file
output_file = 'CMIP6_2100_bias_corrected_downscaled.nc'
ds_out.to_netcdf(output_file)

print(f"Bias-corrected and downscaled data saved to: {output_file}")
