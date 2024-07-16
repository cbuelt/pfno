import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from torch.utils.data import DataLoader
import xarray as xr


class DarcyFlowDataset(Dataset):
    """Darcy Flow Dataset."""

    def __init__(
        self,
        data_dir: str,
        test: bool = False,
        beta: float = 1.0,
        downscaling_factor: int = 1,
    ) -> None:
        """Initialize Darcy Flow Dataset

        Args:
            data_dir (str): Data directory
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

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, idx):
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

    def get_coordinates(self):
        x = self.dataset["x-coordinate"][:: self.downscaling_factor]
        y = self.dataset["y-coordinate"][:: self.downscaling_factor]
        return (x, y)

    def get_domain_range(self):
        return [1.0, 1.0]


class SWEDataset(Dataset):
    """Shallow Water Equation Dataset."""

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

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, idx):
        a_start = self.t_start
        a_end = a_start + self.init_steps
        u_end = a_end + self.pred_horizon

        a = self.dataset[
            idx,
            a_start:a_end,
        ].to_numpy()
        if self.mode == "single":
            u = self.dataset[
                idx,
                u_end : u_end + 1,
            ].to_numpy()
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                a_end:u_end,
            ].to_numpy()
        # Create grid
        x = self.dataset.coords["x"]
        y = self.dataset.coords["y"]
        grid = np.stack(np.meshgrid(x, y))
        # Stack input and grid
        tensor_a = torch.cat([torch.tensor(a), torch.tensor(grid)], dim=0)
        tensor_u = torch.tensor(u)
        return tensor_a, tensor_u

    def get_coordinates(self):
        x = self.dataset.coords["x"].values
        y = self.dataset.coords["y"].values
        t = self.dataset.coords["time"].values
        return (x, y, t)

    def get_domain_range(self):
        return [5.0, 5.0]
    



class KSDataset(Dataset):
    """Kuramoto Sivashinsky Equation Dataset."""

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

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, idx):
        a_start = self.t_start
        a_end = a_start + self.init_steps
        u_end = a_end + self.pred_horizon

        a = self.dataset[
            idx,
            a_start:a_end,
        ].to_numpy()
        if self.mode == "single":
            u = self.dataset[
                idx,
                u_end : u_end + 1,
            ].to_numpy()
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                a_end:u_end,
            ].to_numpy()
        # Create grid
        x = np.stack(np.meshgrid(self.dataset.coords["x"]))
        # Stack input and grid
        tensor_a = torch.cat([torch.tensor(a), torch.tensor(x)], dim=0)
        tensor_u = torch.tensor(u)
        return tensor_a, tensor_u

    def get_coordinates(self):
        x = self.dataset.coords["x"].values
        t = self.dataset.coords["t"].values
        return (x, t)

    def get_domain_range(self):
        return [100.0]


if __name__ == "__main__":
    data_dir = "data/KS/processed/"
    dataset = KSDataset(
        data_dir,
        test=False,
        downscaling_factor=1,
        temporal_downscaling_factor=4,
        mode="autoregressive",
        init_steps=10,
        pred_horizon=5,
    )
    print(dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for sample in train_loader:
        a, u = sample
        print(a.shape, u.shape)
        print(a.dtype, u.dtype)
        break
