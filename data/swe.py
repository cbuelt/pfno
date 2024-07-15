import numpy as np
import pandas as pd
import xarray as xr
import os
from tqdm import tqdm
from torchvision.datasets.utils import download_url
import h5py


def download_data(
    data_dir, url_file="PDEBench/pdebench/data_download/pdebench_data_urls.csv"
):
    """Download SWE dataset from pdebench
    Args:
        data_dir (_type_): Directory to save the downloaded files.
        url_file (str, optional): Path to csv with urls from pdebench. Defaults to "PDEBench/pdebench/data_download/pdebench_data_urls.csv".
    """
    url_df = pd.read_csv(url_file)
    # Filter Darcy Flow
    data_df = url_df[url_df["PDE"].str.lower() == "swe"]
    # Iterate filtered dataframe and download the files
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        file_path = os.path.join(data_dir + "raw/", row["Path"])
        download_url(row["URL"], file_path, row["Filename"], md5=row["MD5"])


def load_dataset(file_path):
    data_raw = h5py.File(file_path, "r")
    # Get coordinates
    x = data_raw["0000"]["grid"]["x"]
    y = data_raw["0000"]["grid"]["y"]
    t = data_raw["0000"]["grid"]["t"]
    data_array = xr.concat(
        [
            xr.DataArray(
                data_raw[key]["data"][..., 0], dims=["time", "x", "y"], coords=[t, x, y]
            )
            for key in data_raw.keys()
        ],
        dim="samples",
    )
    dataset = xr.Dataset({"u": data_array})
    return dataset


def train_test_split(ds, seed, train_size, type = "random"):
    """Split darcy flow dataset into training and testing datasets

    Args:
        ds (_type_): Input dataset
        seed (int, optional): Random seed.
        train_size (float, optional): Size of training dataset.

    Returns:
        _type_: Output training and testing datasets
    """
    n_samples = ds.sizes["samples"]
    np.random.seed(seed)
    if type == "random":
        indices = np.random.permutation(n_samples)
        train_data = ds.isel(samples=indices[: int(n_samples * train_size)])
        test_data = ds.isel(samples=indices[int(n_samples * train_size) :])
    elif type == "ood":
        # Filter radius
        radius = ds.u[:,0,:,63]
        radius = (radius > 1.0).sum(dim = "x") / 128 / 2 * 5
        # Select lowest and highest discrete radius
        radius_filtered = ((radius < 0.34) | (radius > 0.68))
        ood_test_index = np.where(radius_filtered == True)[0]
        ood_train_index = np.where(radius_filtered == False)[0]
        train_data = ds.sel(samples = ood_train_index)
        test_data = ds.sel(samples = ood_test_index)
    return train_data, test_data


def main(data_dir, train_split, seed, download=True, remove=True):
    """Main function to download, process and split Darcy Flow datasets

    Args:
        data_dir (_type_): Data directory
        train_split (_type_): Train test split ratio
        seed (_type_): Random seed
        beta_values (_type_): Beta values to process
        download (bool, optional): Whether to download data. Defaults to True.
        remove (bool, optional): Whether to remove raw data. Defaults to True.
    """
    if download:
        download_data(data_dir)
    # Create output folder
    os.makedirs(data_dir + "processed/", exist_ok=True)
    # Load datasets and create train/test splits
    file_path = data_dir + "raw/2D/shallow-water/2D_rdb_NA_NA.h5"
    ds = load_dataset(file_path)
    # Random data split
    train_data, test_data = train_test_split(ds, seed, train_split, type = "random")
    train_data.to_netcdf(data_dir + "processed/swe_train.nc")
    test_data.to_netcdf(data_dir + "processed/swe_test.nc")

    # OOD data split
    train_data, test_data = train_test_split(ds, seed, train_split, type = "ood")
    train_data.to_netcdf(data_dir + "processed/swe_ood_train.nc")
    test_data.to_netcdf(data_dir + "processed/swe_ood_test.nc")


    # Remove raw data
    if remove:
        os.system(f"rm -r {data_dir}/raw")


if __name__ == "__main__":
    data_dir = "data/SWE/"
    train_split = 0.9
    seed = 42
    download = True
    main(data_dir, train_split, seed, download)
