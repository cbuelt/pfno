import apache_beam 
import weatherbench2
import xarray as xr
import gcsfs
import numpy as np
import pandas as pd
import os


if __name__ == "__main__":
    # Set parameters
    era5_path = "gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr"   
    date_range = pd.date_range(f"2011-01-01", f"2022-12-31T18", freq = "6h")
    # Europe grid
    lat_range = np.arange(35, 75, 0.25)
    lon_range = np.append(np.arange(347.5,360, 0.25),np.arange(0, 42.5,0.25))
    variable = "2m_temperature"
    output_dir = "data/era5/"
    filename = "era5_2m_temperature_2011-2022.nc"
    # Create output dir
    os.makedirs(output_dir, exist_ok = True)
    # Open file from weatherbench
    data = xr.open_zarr(era5_path)
    # Filter data and save to file
    data["2m_temperature"].sel(time = date_range, method = "nearest").sel(latitude = lat_range, longitude = lon_range, method = "nearest").to_netcdf(output_dir + filename)