import numpy as np
import xarray as xr
import os
from tqdm import tqdm
from torchvision.datasets.utils import download_url


def download_yearly_data(data_dir, year):
    base_url = "https://opendata.dwd.de/climate_environment/CDC/grids_germany/5_minutes/radolan/reproc/2017_002/netCDF/"
    # Year 2021 is in different folder
    if year == 2021:
        base_url += "supplement/"
        filename = "YW2017.002_2021_netcdf_supplement.tar.gz"
    else:
        base_url += f"{year}/"
        filename = f"YW2017.002_{year}_netcdf.tar.gz"
    full_url = base_url + filename
    download_url(full_url, data_dir, f"YW2017.002_{year}_netcdf.tar.gz")

def extract_data(data_dir_raw, data_dir_processed, year):
    tar_path = os.path.join(data_dir_raw, f"YW2017.002_{year}_netcdf.tar.gz")
    os.system(f"tar -xvf {tar_path} -C {data_dir_processed}")
    os.remove(tar_path)

def extract_months(data_dir, months):
    for i in range(12):
        if i+1 not in months:
            os.system(f"rm -r {data_dir}/{i+1:02d}")

if __name__ == "__main__":
    years = [2015, 2016, 2017, 2018, 2019, 2020, 2021]
    months = [5,6,7,8,9]

    # Create directories
    data_dir_raw = "data/RYDL/raw/"
    data_dir_processed = "data/RYDL/processed/"
    os.makedirs(data_dir_raw, exist_ok=True)
    os.makedirs(data_dir_processed, exist_ok=True)

    # Download and extract data
    for year in years:
        download_yearly_data(data_dir_raw, year)
        extract_data(data_dir_raw, data_dir_processed, year)
        extract_months(f"{data_dir_processed}/{year}", months)
    os.system(f"rm -r {data_dir_raw}")


