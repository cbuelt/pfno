import numpy as np
import pandas as pd
import xarray as xr
import os
from tqdm import tqdm
from torchvision.datasets.utils import download_url


def download_data(
    data_dir, url_file="data/download/pdebench_data_urls.csv"
):
    """Download Darcy Flow datasets from pdebench
    Args:
        data_dir (_type_): Directory to save the downloaded files.
        url_file (str, optional): Path to csv with urls from pdebench. Defaults to "data/download/pdebench_data_urls.csv".
    """
    url_df = pd.read_csv(url_file)
    # Filter Darcy Flow
    data_df = url_df[url_df["PDE"].str.lower() == "darcy"]
    # Iterate filtered dataframe and download the files
    for index, row in tqdm(data_df.iterrows(), total=data_df.shape[0]):
        file_path = os.path.join(data_dir + "raw/", row["Path"])
        download_url(row["URL"], file_path, row["Filename"], md5=row["MD5"])


def rename(ds):
    """Rename darcy flow dataset

    Args:
        ds (_type_): Input dataset

    Returns:
        _type_: Output dataset with renamed dimensions and variables
    """
    return ds.rename(
        {
            "phony_dim_0": "samples",
            "phony_dim_1": "x",
            "phony_dim_2": "y",
            "phony_dim_3": "t",
        }
    ).rename_vars({"nu": "a", "tensor": "u"})


def train_test_split(ds, seed, train_size):
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
    indices = np.random.permutation(n_samples)
    train_data = ds.isel(samples=indices[: int(n_samples * train_size)])
    test_data = ds.isel(samples=indices[int(n_samples * train_size) :])
    return train_data, test_data


def main(data_dir, train_split, seed, beta_values, download=True, remove=True):
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
    for beta in beta_values:
        file_path = data_dir + "raw/2D/DarcyFlow/" f"2D_DarcyFlow_beta{beta}_Train.hdf5"
        ds = xr.load_dataset(file_path)
        ds = rename(ds)
        train_data, test_data = train_test_split(ds, seed, train_split)
        train_data.to_netcdf(data_dir + "processed/" + f"DarcyFlow_beta{beta}_train.nc")
        test_data.to_netcdf(data_dir + "processed/" + f"DarcyFlow_beta{beta}_test.nc")

    # Remove raw data
    if remove:
        os.system(f"rm -r {data_dir}/raw")


if __name__ == "__main__":
    data_dir = "data/DarcyFlow/"
    train_split = 0.9
    seed = 42
    beta_values = [1.0]
    download = True
    main(data_dir, train_split, seed, beta_values, download)
