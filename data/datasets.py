import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd


class DarcyFlowDataset(Dataset):
    """
    A class used to handle the Darcy Flow data.

    Attributes
    ----------
    filename : str
        the name of the file containing the data
    dataset : xarray.Dataset
        the dataset containing the data
    downscaling_factor : int
        the downscaling factor for the spatial resolution

    Methods
    -------
    __len__()
        Returns the length of the dataset
    __getitem__(idx)
        Returns the idx-th element of the dataset
    get_coordinates()
        Returns the x and y coordinates of the dataset
    get_domain_range()
        Returns the domain range of the dataset
    """

    def __init__(
        self,
        data_dir: str,
        test: bool = False,
        beta: float = 1.0,
        downscaling_factor: int = 1,
        normalize=True,
    ) -> None:
        """Initialize Darcy Flow Dataset

        Args:
            data_dir (str): Data directory.
            test (bool, optional): Whether to load train or test data. Defaults to False.
            beta (float, optional): Beta coefficient, determining the dataset. Defaults to 1.0.
            downscaling_factor (int, optional): Downscaling for spatial resolution. Defaults to 1.
        """
        if test:
            self.filename = f"DarcyFlow_beta{beta}_test.nc"
        else:
            self.filename = f"DarcyFlow_beta{beta}_train.nc"
        self.dataset = xr.open_dataset(data_dir + self.filename)
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.normalize = normalize

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.dataset["samples"])

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        a = self.dataset.a[
            idx, :: self.downscaling_factor, :: self.downscaling_factor
        ].to_numpy()
        u = self.dataset.u[
            idx, 0, :: self.downscaling_factor, :: self.downscaling_factor
        ].to_numpy()
        # Create grid
        x = self.dataset["x-coordinate"][:: self.downscaling_factor]
        y = self.dataset["y-coordinate"][:: self.downscaling_factor]
        # Min/max normalization
        if self.normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        grid = np.stack(np.meshgrid(x, y))
        # Stack input and grid
        tensor_a = torch.cat([torch.tensor(a).unsqueeze(0), torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u).unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self, normalize=True) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor]
        y = self.dataset["y-coordinate"][:: self.downscaling_factor]
        # Min/max normalization
        if normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
        return (x, y)

    def get_domain_range(self) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        x, y = self.get_coordinates()
        L_x = x[-1] - x[0]
        L_y = y[-1] - y[0]
        return [L_x, L_y]


class SWEDataset(Dataset):
    """
    A class used to handle the Shallow Water equation data.

    Attributes
    ----------
    filename : str
        the name of the file containing the data
    downscaling_factor : int
        the downscaling factor for the spatial resolution
    temporal_downscaling_factor : int
        the downscaling factor for the temporal resolution
    init_steps : int
        the number of time steps given as input
    t_start : int
        the starting time step for the input
    mode : str
        the mode of the dataset, either "single" or "autoregressive"
    pred_horizon : int
        the prediction horizon
    Methods
    -------
    __len__()
        Returns the length of the dataset
    __getitem__(idx)
        Returns the idx-th element of the dataset
    get_coordinates()
        Returns the x and y coordinates of the dataset
    get_domain_range()
        Returns the domain range of the dataset
    """

    def __init__(
        self,
        data_dir: str,
        test: bool = False,
        init_steps: int = 10,
        mode: str = "single",
        downscaling_factor: int = 1,
        temporal_downscaling_factor: int = 1,
        pred_horizon: int = 10,
        t_start: int = 0,
        ood: bool = False,
        normalize: bool = True,
    ) -> None:

        self.filename = "swe"
        if ood:
            self.filename += "_ood"
        if test:
            self.filename += "_test.nc"
        else:
            self.filename += "_train.nc"
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        assert isinstance(
            temporal_downscaling_factor, int
        ), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.temporal_downscaling_factor = temporal_downscaling_factor
        self.init_steps = init_steps
        self.t_start = t_start
        self.mode = mode
        self.pred_horizon = pred_horizon
        self.normalize = normalize

        # Downscaling
        self.dataset = xr.open_dataset(data_dir + self.filename).u[
            :,
            :: self.temporal_downscaling_factor,
            :: self.downscaling_factor,
            :: self.downscaling_factor,
        ]

        # Set timesteps for extraction
        self.a_start = self.t_start
        self.a_end = self.a_start + self.init_steps
        self.u_end = self.a_end + self.pred_horizon

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.dataset["samples"])

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        a = self.dataset[
            idx,
            self.a_start : self.a_end,
        ]
        if self.mode == "single":
            u = self.dataset[
                idx,
                self.u_end : self.u_end + 1,
            ]
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                self.a_end : self.u_end,
            ]
        # Create grid
        x = a.coords["x"]
        y = a.coords["y"]
        t = a.coords["time"]
        # Min/max normalization
        if self.normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            t = (t - np.min(t)) / (np.max(t) - np.min(t))
        grid = np.stack(np.meshgrid(x, t, y))
        # Stack input and grid
        tensor_a = torch.tensor(a.to_numpy()).float().unsqueeze(0)
        tensor_a = torch.cat([tensor_a, torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u.to_numpy()).float().unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self, normalize=True) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset.coords["x"].values
        y = self.dataset.coords["y"].values
        t = self.dataset.coords["time"].values
        # Min/max normalization
        if normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))
            t = (t - np.min(t)) / (np.max(t) - np.min(t))
        return (x, y, t)

    def get_domain_range(self) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        x, y, t = self.get_coordinates()
        L_x = x[-1] - x[0]
        L_y = y[-1] - y[0]
        if self.mode == "single":
            t = t[self.u_end : self.u_end + 1]
        elif self.mode == "autoregressive":
            t = t[self.a_end : self.u_end + 1]
        L_t = t[-1] - t[0]
        return [L_t, L_x, L_y]


class KSDataset(Dataset):
    """
    A class used to handle the Kuramoto-Sivashinsky equation data.

    Attributes
    ----------
    filename : str
        the name of the file containing the data
    downscaling_factor : int
        the downscaling factor for the spatial resolution
    temporal_downscaling_factor : int
        the downscaling factor for the temporal resolution
    init_steps : int
        the number of time steps given as input
    t_start : int
        the starting time step for the input
    mode : str
        the mode of the dataset, either "single" or "autoregressive"
    pred_horizon : int
        the prediction horizon
    Methods
    -------
    __len__()
        Returns the length of the dataset
    __getitem__(idx)
        Returns the idx-th element of the dataset
    get_coordinates()
        Returns the x and y coordinates of the dataset
    get_domain_range()
        Returns the domain range of the dataset
    """

    def __init__(
        self,
        data_dir: str,
        test: bool = False,
        init_steps: int = 10,
        mode: str = "single",
        downscaling_factor: int = 1,
        temporal_downscaling_factor: int = 1,
        pred_horizon: int = 10,
        t_start: int = 0,
        normalize: bool = True,
    ) -> None:

        if test:
            self.filename = "ks_test.nc"
        else:
            self.filename = "ks_train.nc"
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        assert isinstance(
            temporal_downscaling_factor, int
        ), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.temporal_downscaling_factor = temporal_downscaling_factor
        self.init_steps = init_steps
        self.t_start = t_start
        self.mode = mode
        self.pred_horizon = pred_horizon
        self.normalize = normalize

        # Downscaling
        self.dataset = xr.open_dataset(data_dir + self.filename).u[
            :,
            :: self.temporal_downscaling_factor,
            :: self.downscaling_factor,
        ]

        # Set timesteps for extraction
        self.a_start = self.t_start
        self.a_end = self.a_start + self.init_steps
        self.u_end = self.a_end + self.pred_horizon

    def __len__(self) -> int:
        """Returns the length of the dataset

        Returns:
            int: Length of the dataset
        """
        return len(self.dataset["samples"])

    def __getitem__(self, idx: int) -> tuple:
        """Returns the idx-th element of the dataset

        Args:
            idx (int): Index of the element to be returned

        Returns:
            tuple: Tuple containing the input and output tensors
        """
        a = self.dataset[
            idx,
            self.a_start : self.a_end,
        ]
        if self.mode == "single":
            u = self.dataset[
                idx,
                self.u_end : self.u_end + 1,
            ]
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                self.a_end : self.u_end,
            ]
        # Create grid
        x = a.coords["x"]
        t = a.coords["t"]
        # Min/max normalization
        if self.normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            t = (t - np.min(t)) / (np.max(t) - np.min(t))
        grid = np.stack(np.meshgrid(x, t))
        # Stack input and grid
        tensor_a = torch.tensor(a.to_numpy()).float().unsqueeze(0)
        tensor_a = torch.cat([tensor_a, torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u.to_numpy()).float().unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self, normalize=True) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset.coords["x"].values
        t = self.dataset.coords["t"].values
        # Min/max normalization
        if normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            t = (t - np.min(t)) / (np.max(t) - np.min(t))
        return (x, t)

    def get_domain_range(self) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        x, t = self.get_coordinates()
        L_x = x[-1] - x[0]
        if self.mode == "single":
            L_t = t[self.u_end + 1] - t[self.u_end]
        elif self.mode == "autoregressive":
            t = t[self.a_end : self.u_end + 1]
            L_t = t[-1] - t[0]
        return [L_t, L_x]


class ERA5Dataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        var: str = "train",
        normalize: bool = True,
        downscaling_factor: int = 1,
        init_steps: int = 10,
        prediction_steps: int = 10,
    ):
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.var = var
        self.data_dir = data_dir
        self.normalize = normalize
        self.downscaling_factor = downscaling_factor
        self.init_steps = init_steps
        self.prediction_steps = prediction_steps

        # Date splits
        self.dates = dict(
            {
                "train": pd.date_range("2011-01-01", "2020-12-31", freq="6h"),
                "val": pd.date_range("2021-01-01", "2021-12-31", freq="6h"),
                "test": pd.date_range("2022-01-01", "2022-12-31", freq="6h"),
            }
        )

        # Load dataset
        self.dataset = xr.open_dataset(
            self.data_dir + "era5_2m_temperature_2011-2022.nc"
        )["2m_temperature"].sel(time=self.dates[self.var])
        # Get coordinates
        self.x = self.dataset.coords["latitude"].values
        # Create y manually
        self.y = np.append(np.arange(-12.5, 0, 0.25), np.arange(0, 42.5, 0.25))
        # Create t manually
        self.t = np.arange(0, (init_steps + prediction_steps) * 6, 6)

        # Normalize grid
        self.x_norm = (self.x - np.min(self.x)) / (np.max(self.x) - np.min(self.x))
        self.y_norm = (self.y - np.min(self.y)) / (np.max(self.y) - np.min(self.y))
        self.t_norm = (self.t - np.min(self.t)) / (np.max(self.t) - np.min(self.t))

        # Get normalization
        if self.normalize:
            self.get_normalization()
    

    def get_normalization(self) -> Tuple:
        self.mean = self.dataset.mean().to_numpy()
        self.std = self.dataset.std().to_numpy()


    def __len__(self) -> int:
        length = len(self.dates[self.var]) - (self.init_steps + self.prediction_steps)
        return length
    

    def __getitem__(self, idx: int) -> tuple:
        # Get data
        train = self.dataset.isel(time=slice(idx, idx + self.init_steps)).to_numpy()
        target = self.dataset.isel(
            time=slice(
                idx + self.init_steps, idx + self.init_steps + self.prediction_steps
            )
        ).to_numpy()
        # Normalize
        if self.normalize:
            train = (train - self.mean) / self.std
            target = (target - self.mean) / self.std
        # Turn to tensor
        train = torch.tensor(train).float().unsqueeze(0)
        target = torch.tensor(target).float().unsqueeze(0)
        # Create and stack grid
        grid = np.stack(np.meshgrid(self.x_norm, self.t_norm[:self.init_steps], self.y_norm))
        train = torch.cat([train, torch.tensor(grid)], dim=0).float()
        return train, target
    

    def get_coordinates(self, normalize=True) -> Tuple:
        _ = self.__getitem__(0)
        if normalize:
            return self.x_norm, self.y_norm, self.t_norm[self.init_steps:]
        else:
            return self.x, self.y, self.t[self.init_steps:]
        

    def get_domain_range(self, normalize=True) -> List[float]:
        if normalize:
            x = self.x_norm
            y = self.y_norm
            t = self.t_norm[self.init_steps:]
        else:
            x = self.x
            y = self.y
            t = self.t[self.init_steps:]
        L_x = np.abs(x[-1] - x[0])
        L_y = y[-1] - y[0]
        L_t = t[-1] - t[0]
        return [L_t, L_y, L_x]


if __name__ == "__main__":
    data_dir = "data/era5/"
    dataset = ERA5Dataset(data_dir, var="test")
    print(len(dataset))
    train, target = dataset.__getitem__(10)
    print(train.shape)
    print(target.shape)
    x, y, t = dataset.get_coordinates(normalize=True)
    L = dataset.get_domain_range(normalize=False)
    print(L)
