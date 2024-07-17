import numpy as np
from typing import List, Tuple
import torch
from torch.utils.data import Dataset
import xarray as xr


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
        grid = np.stack(np.meshgrid(x, y))
        # Stack input and grid
        tensor_a = torch.cat([torch.tensor(a).unsqueeze(0), torch.tensor(grid)], dim=0)
        tensor_u = torch.tensor(u).unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset["x-coordinate"][:: self.downscaling_factor]
        y = self.dataset["y-coordinate"][:: self.downscaling_factor]
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
    ) -> None:

        if test:
            self.filename = "swe_test.nc"
        else:
            self.filename = "swe_train.nc"
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
        grid = np.stack(np.meshgrid(x, t, y))
        # Stack input and grid
        tensor_a = torch.tensor(a.to_numpy()).float().unsqueeze(0)
        tensor_a = torch.cat([tensor_a, torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u.to_numpy()).float().unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset.coords["x"].values
        y = self.dataset.coords["y"].values
        t = self.dataset.coords["time"].values
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
        grid = np.stack(np.meshgrid(x, t))
        # Stack input and grid
        tensor_a = torch.tensor(a.to_numpy()).float().unsqueeze(0)
        tensor_a = torch.cat([tensor_a, torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u.to_numpy()).float().unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self) -> Tuple:
        """Returns the x and y coordinates of the dataset

        Returns:
            Tuple: Tuple containing the x and y coordinates
        """
        x = self.dataset.coords["x"].values
        t = self.dataset.coords["t"].values
        return (x, t)

    def get_domain_range(self) -> List[float]:
        """Returns the domain range of the dataset.

        Returns:
            List[float]: List containing the domain range.
        """
        x, t = self.get_coordinates()
        L_x = x[-1] - x[0]
        if self.mode == "single":
            t = t[self.u_end : self.u_end + 1]
        elif self.mode == "autoregressive":
            t = t[self.a_end : self.u_end + 1]
        L_t = t[-1] - t[0]
        return [L_t, L_x]


if __name__ == "__main__":
    data_dir = "data/KS/processed/"
    dataset = KSDataset(
        data_dir,
        test=True,
        downscaling_factor=2,
        temporal_downscaling_factor=2,
        mode="autoregressive",
        init_steps=10,
        pred_horizon=10,
    )
    print(dataset.__len__())
    print(dataset.get_domain_range())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for sample in train_loader:
        a, u = sample
        print(a.shape, u.shape)
        break
