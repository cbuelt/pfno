import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr
import pandas as pd
from torch_harmonics.quadrature import clenshaw_curtiss_weights


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
        normalize_grid=True,
        normalize=True,
    ) -> None:
        """Initialize Darcy Flow Dataset

        Args:
            data_dir (str): Data directory.
            test (bool, optional): Whether to load train or test data. Defaults to False.
            beta (float, optional): Beta coefficient, determining the dataset. Defaults to 1.0.
            downscaling_factor (int, optional): Downscaling for spatial resolution. Defaults to 1.
        """
        self.beta = beta
        if test:
            self.filename = f"DarcyFlow_beta{self.beta}_test.nc"
        else:
            self.filename = f"DarcyFlow_beta{self.beta}_train.nc"
        self.dataset = xr.open_dataset(data_dir + self.filename)
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.normalize_grid = normalize_grid
        self.normalize = normalize
        self.data_dir = data_dir

        # Get normalization
        if self.normalize:
            self.get_normalization()

    def get_normalization(self) -> Tuple:
        train_dataset = xr.open_dataset(
            self.data_dir + f"DarcyFlow_beta{self.beta}_train.nc"
        ).u
        self.mean = train_dataset.mean().to_numpy()
        self.std = train_dataset.std().to_numpy()

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

        # Normalize
        if self.normalize:
            u = (u - self.mean) / self.std

        # Min/max normalization
        if self.normalize_grid:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            y = (y - np.min(y)) / (np.max(y) - np.min(y))

        grid = np.stack(np.meshgrid(x, y))
        # Stack input and grid
        tensor_a = torch.cat(
            [torch.tensor(a).unsqueeze(0), torch.tensor(grid)], dim=0
        ).float()
        tensor_u = torch.tensor(u).unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self, normalize=True) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor].values
        y = self.dataset["y-coordinate"][:: self.downscaling_factor].values
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
        mode: str = "autoregressive",
        downscaling_factor: int = 1,
        temporal_downscaling_factor: int = 1,
        pred_horizon: int = 10,
        t_start: int = 0,
        normalize_grid: bool = True,
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
        self.normalize_grid = normalize_grid
        self.normalize = normalize
        self.data_dir = data_dir

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

        # Get normalization
        if self.normalize:
            self.get_normalization()

    def get_normalization(self) -> Tuple:
        train_dataset = xr.open_dataset(self.data_dir + "ks_train.nc").u
        self.mean = train_dataset.mean().to_numpy()
        self.std = train_dataset.std().to_numpy()

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

        # Normalize
        if self.normalize:
            a = (a - self.mean) / self.std
            u = (u - self.mean) / self.std

        # Min/max normalization of grid
        if self.normalize_grid:
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
        t = self.dataset.coords["t"].values[self.a_end : self.u_end]
        # Min/max normalization
        if normalize:
            x = (x - np.min(x)) / (np.max(x) - np.min(x))
            t = (t - np.min(t)) / (np.max(t) - np.min(t))
        return (x, t)

    def get_domain_range(self, normalize=True) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        x, t = self.get_coordinates(normalize=normalize)
        L_x = x[-1] - x[0]
        L_t = t[-1] - t[0]
        return [L_t, L_x]


class ERA5Dataset(Dataset):
    """
    A class used to handle the ERA5 data.


    Attributes
    ----------
    data_dir : str
        the directory containing the data
    var : str
        the variable to be loaded
    normalize : bool
        whether to normalize the data
    init_steps : int
        the number of time steps given as input
    prediction_steps : int
        the number of time steps to predict

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
        var: str = "train",
        normalize: bool = True,
        init_steps: int = 10,
        prediction_steps: int = 10,
    ):
        """Initialize ERA5 Dataset.

        Args:
            data_dir (str): Directory containing the data.
            var (str, optional): Which data type to use. Defaults to "train".
            normalize (bool, optional): Whether to normalize the data. Defaults to True.
            init_steps (int, optional): Initial steps as input to the model. Defaults to 10.
            prediction_steps (int, optional): Prediction steps as output. Defaults to 10.
        """
        self.var = var
        self.data_dir = data_dir
        self.normalize = normalize
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

    def get_normalization(self):
        """
        Get normalization statistics.
        """
        train_dataset = xr.open_dataset(
            self.data_dir + "era5_2m_temperature_2011-2022.nc"
        )["2m_temperature"].sel(time=self.dates["train"])
        self.mean = train_dataset.mean().to_numpy()
        self.std = train_dataset.std().to_numpy()

    def __len__(self) -> int:
        """Get length of the dataset.

        Returns:
            int: Length
        """
        length = len(self.dates[self.var]) - (self.init_steps + self.prediction_steps)
        return length

    def __getitem__(self, idx: int) -> tuple:
        """Get item from dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Returns the input and output tensors.
        """
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
        grid = np.stack(
            np.meshgrid(self.x_norm, self.t_norm[: self.init_steps], self.y_norm)
        )
        train = torch.cat([train, torch.tensor(grid)], dim=0).float()
        return train, target

    def get_coordinates(self, normalize=True) -> Tuple:
        """Get coordinates (x,y,t) of the dataset.

        Args:
            normalize (bool, optional): Whether to normalize the coordinates to range [0,1]. Defaults to True.

        Returns:
            Tuple: x,y,t coordinates.
        """
        _ = self.__getitem__(0)
        if normalize:
            return self.x_norm, self.y_norm, self.t_norm[self.init_steps :]
        else:
            return self.x, self.y, self.t[self.init_steps :]

    def get_domain_range(self, normalize=True) -> List[float]:
        """Get domain range of the dataset.

        Args:
            normalize (bool, optional): Whether to normalize grids to range [0,1]. Defaults to True.

        Returns:
            List[float]: Grid constants for each variable.
        """
        if normalize:
            x = self.x_norm
            y = self.y_norm
            t = self.t_norm[self.init_steps :]
        else:
            x = self.x
            y = self.y
            t = self.t[self.init_steps :]
        L_x = np.abs(x[-1] - x[0])
        L_y = y[-1] - y[0]
        L_t = t[-1] - t[0]
        return [L_t, L_y, L_x]


class SSWEDataset(Dataset):
    """
    A class used to handle the SSWE data.


    Attributes
    ----------
    data_dir : str
        the directory containing the data
    test : bool
        whether to load train or test data
    pred_horizon : int
        the number of time steps to predict
    return_all : bool
        whether to return all prediction steps

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
    get_train_weights()
        Returns the training quadrature weights
    get_eval_weights()
        Returns the evaluation cosine weights
    get_nlon()
        Returns the number of longitude points
    """

    def __init__(
        self,
        data_dir: str,
        test: bool = False,
        pred_horizon: int = 1,
        return_all: bool = False,
    ) -> None:

        if test:
            self.filename = "test.nc"
        else:
            self.filename = "train.nc"
        self.pred_horizon = pred_horizon
        self.return_all = return_all

        # Load dataset
        self.dataset = xr.open_dataset(data_dir + self.filename)

        # Assert pred horizon
        assert self.pred_horizon < len(
            self.dataset["time"]
        ), "Prediction horizon must be smaller than the dataset"

        # Generate training quadrature weights
        self.nlat = len(self.dataset["latitude"])
        self.nlon = len(self.dataset["longitude"])
        _, quad_weights = clenshaw_curtiss_weights(self.nlat, -1, 1)
        self.train_weights = torch.as_tensor(quad_weights).reshape(-1, 1)

        # Generate evaluation cosine weights
        lat_weight = np.cos(self.dataset["latitude"].values / 360 * 2 * np.pi)
        lat_weight = np.clip(lat_weight / lat_weight.sum(), 0, 1)
        self.eval_weights = torch.tensor(lat_weight).reshape(-1, 1)

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
        a = self.dataset.u[
            idx,
            0,
        ]
        if self.return_all:
            u = self.dataset.u[
                idx,
                1 : (self.pred_horizon + 1),
            ]
            tensor_u = torch.tensor(u.to_numpy()).float()
            tensor_u = torch.permute(tensor_u, (1, 0, 2, 3))

        else:
            u = self.dataset.u[
                idx,
                self.pred_horizon,
            ]
            tensor_u = torch.tensor(u.to_numpy()).float()
        tensor_a = torch.tensor(a.to_numpy()).float()
        return tensor_a, tensor_u

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        lat = self.dataset["latitude"].values
        lon = self.dataset["longitude"].values
        t = self.dataset["time"].values[1 : self.pred_horizon + 1]
        return (lat, lon, t)

    def get_domain_range(self) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        return [1, 1, 1]

    def get_train_weights(self) -> torch.Tensor:
        return self.train_weights

    def get_eval_weights(self) -> torch.Tensor:
        return self.eval_weights

    def get_nlon(self) -> int:
        return self.nlon


if __name__ == "__main__":
    data_dir = "data/SSWE/processed/"
    dataset = SSWEDataset(data_dir, test=False, pred_horizon=2, return_all=True)
    print(len(dataset))
    train, target = dataset.__getitem__(10)
    train_weights = dataset.get_train_weights()
    eval_weights = dataset.get_eval_weights()
    print(train_weights.shape, eval_weights.shape)
    print(train_weights)
    print(eval_weights)
