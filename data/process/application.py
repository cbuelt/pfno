import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
from netCDF4 import Dataset
import h5py
from matplotlib import pyplot as plt


def get_file_list(
    years: list, data_dir: str = "data/RYDL/processed/", months: list = [5, 6, 7, 8, 9]
) -> list:
    """Create a file list of all files in the given years and months

    Args:
        years (list): Years to include
        data_dir (str, optional): _description_. Defaults to "data/RYDL/processed/".
        months (list, optional): _description_. Defaults to [5,6,7,8,9].

    Returns:
        list: File list
    """
    files = []
    for year in years:
        path = f"{data_dir}{year}/"
        for month in months:
            monthly_files = os.listdir(path + f"{month:02d}/")
            for f in monthly_files:
                if f.endswith(".nc"):
                    files.append(path + f"{month:02d}/" + f)
    return files


def load_data(file_list: list) -> xr.Dataset:
    """Load data from a list of files

    Args:
        file_list (list): File list

    Returns:
        xr.Dataset: Dask/Xarray dataset
    """
    data = xr.open_mfdataset(file_list, chunks="auto")
    return data


def extract_samples(
    data: xr.Dataset,
    threshold: float = 0.00625,
    domain_treshold: float = 0.1,
    resample: int = 2,
) -> np.ndarray:
    """Extract samples from dataset where the threshold coverage is met.

    Args:
        data (xr.Dataset): _description_
        threshold (float, optional): Threshold of rain in mm/h. Defaults to 0.00625.
        domain_treshold (float, optional): Domain threshold that should be covered. Defaults to 0.1.
        resample (int, optional): Time resolution to resample times 5min. Defaults to 2.

    Returns:
        np.ndarray: _description_
    """
    y_subset = slice(410, -434)
    x_subset = slice(159, -485)

    precip = data.RR.fillna(0)[:, y_subset, x_subset]
    if resample is not None:
        precip = precip[::resample]
    if threshold is not None:
        filtered = xr.where(precip > threshold, 1, 0).mean(dim=["x", "y"])
        final = filtered[(filtered > domain_treshold).compute()]
    else:
        final = precip
    return final.time.to_numpy()


def save_index_list(
    times_raw: list,
    init_steps: int,
    prediction_steps: int,
    filename: str,
    output_dir: str = "data/RYDL/processed/",
) -> None:
    """Generate datetime list based on moving window for init_steps and prediction_steps

    Args:
        times_raw (list): Raw time list with available datetimes
        init_steps (int): Number of input timesteps
        prediction_steps (int): Number of prediction timesteps
        filename (str): _description_
        output_dir (str, optional): _description_. Defaults to "data/RYDL/".
    """
    length = len(times_raw) - prediction_steps - 1
    time_list = []
    for i in range(length):
        diff = np.timedelta64(
            times_raw[i + prediction_steps - 1] - times_raw[i - init_steps], "m"
        )
        if diff.astype("int") == (init_steps + prediction_steps - 1) * 10:
            time_list.append(times_raw[i - init_steps])
    np.save(output_dir + filename + "_index.npy", time_list)


def prepare_data(
    years: list,
    months: list,
    init_steps: int,
    prediction_steps: int,
    filename: str,
    threshold: float = 0.00625,
    domain_treshold: float = 0.1,
    resample: int = 2,
):
    """Prepare data and save index files to disk

    Args:
        years (list): Years to consider
        months (list): Months to consider
        init_steps (int): Number of input timesteps
        prediction_steps (int): Number of prediction timesteps
        filename (str): _description_
        threshold (float, optional): Threshold of rain in mm/h. Defaults to 0.00625.
        domain_treshold (float, optional): Domain threshold that should be covered. Defaults to 0.1.
        resample (int, optional): Time resolution to resample times 5min. Defaults to 2.
    """
    file_list = get_file_list(years, months=months)
    data = load_data(file_list)
    samples = extract_samples(
        data, threshold=threshold, domain_treshold=domain_treshold, resample=resample
    )
    save_index_list(samples, init_steps, prediction_steps, filename)


def load_data(
    idx,
    date_index,
    init_steps,
    prediction_steps,
    x_subset,
    y_subset,
    resample,
    data_dir = "data/RYDL/",
):
    t_init = date_index[idx]
    t_end = date_index[idx] + np.timedelta64(
        5 * resample * (init_steps + prediction_steps - 1), "m"
    )
    t_init = pd.to_datetime(str(t_init))
    t_end = pd.to_datetime(str(t_end))
    range = pd.date_range(t_init, t_end, freq=f"{5 *resample}min")

    # Compare strings for overlapping days
    s1 = t_init.strftime("%Y%m%d")
    s2 = t_end.strftime("%Y%m%d")

    # Load data
    if s1 == s2:
        data = xr.open_dataset(
            f"{data_dir}processed/{t_init.year}/{t_init.month:02d}/YW_2017.002_{s1}.nc"
        )
    else:
        data = xr.open_mfdataset(
            [
                f"{data_dir}processed/{t_init.year}/{t_init.month:02d}/YW_2017.002_{s1}.nc",
                f"{data_dir}processed/{t_end.year}/{t_end.month:02d}/YW_2017.002_{s2}.nc",
            ]
        )
    # Filter domain
    data = data.isel(y=y_subset, x=x_subset)
    # Precipitation data
    precip = data.RR.fillna(0)
    x = precip.coords["x"].values
    y = precip.coords["y"].values
    t = np.arange(0, 5 * resample * (prediction_steps), 5 * resample)

    train = precip.sel(time = range[0:init_steps])
    target = precip.sel(time = range[init_steps:])

    return train, target, x, y, t


def create_file(
    var,
    init_steps,
    prediction_steps,
    resample,
    index_dir="data/RYDL/",
):
    y_subset = slice(410, -434)
    x_subset = slice(159, -485)
    output_dir = f"{index_dir}processed/"
    os.makedirs(output_dir, exist_ok=True)

    date_index = np.load(f"data/RYDL/processed/{var}_index.npy")

    # Initialize
    train, target, x, y, t = load_data(
        0,
        date_index,
        init_steps,
        prediction_steps,
        x_subset,
        y_subset,
        resample,
        index_dir,
    )

    # Create nc file for forecast
    filename_out = os.path.join(output_dir, f"{var}.nc" )
    # Check if file already exists
    if os.path.exists(filename_out):
        os.remove(filename_out)
    # Create file
    f = Dataset(filename_out, "a")
    
    # Create dimensions
    f.createDimension("samples", None)
    f.createDimension("init_steps", init_steps)
    f.createDimension("prediction_steps", prediction_steps)
    f.createDimension("x", len(x))
    f.createDimension("y", len(y))
    f.createDimension("t", len(t))
    
    # Create variables
    x_dim = f.createVariable("x", "f4", ("x",))
    y_dim = f.createVariable("y", "f4", ("y",))
    t_dim = f.createVariable("t", "f4", ("t",))
    a = f.createVariable("a", "f4", ("samples", "init_steps", "y", "x"))
    u = f.createVariable("u", "f4", ("samples", "prediction_steps", "y", "x"))

    # Fill coordinate variables
    x_dim[:] = x
    y_dim[:] = y
    t_dim[:] = t

    for idx in range(len(date_index)):
        train, target, _,_,_ = load_data(
            idx,
            date_index,
            init_steps,
            prediction_steps,
            x_subset,
            y_subset,
            resample,
            index_dir,
        )
        a[idx] = train.values
        u[idx] = target.values

    f.close()
    print(f"Successfully created file {filename_out}")



if __name__ == "__main__":
    train_years = [2015, 2016, 2017, 2018, 2019]
    val_years = [2020]
    test_years = [2021]
    months = [5, 6, 7, 8, 9]  # [5,6,7,8,9]
    init_steps = 8
    prediction_steps = 8
    resample = 2

   # prepare_data(val_years, months, init_steps, prediction_steps, "val")
   # prepare_data(test_years, months, init_steps, prediction_steps, "test")
   # prepare_data(train_years, months, init_steps, prediction_steps, "train")

    # Create nc Files
    create_file("val", init_steps, prediction_steps, resample)
    create_file("test", init_steps, prediction_steps, resample)
    #create_file("train", init_steps, prediction_steps, resample)
