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
        x,y = self.get_coordinates()
        L_x = x[-1]-x[0]
        L_y = y[-1] - y[0]
        return [L_x, L_y]


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

        # Set timesteps for extraction
        self.a_start = self.t_start
        self.a_end = self.a_start + self.init_steps
        self.u_end = self.a_end + self.pred_horizon

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, idx):
        a = self.dataset[
            idx,
            self.a_start:self.a_end,
        ]
        if self.mode == "single":
            u = self.dataset[
                idx,
                self.u_end : self.u_end + 1,
            ]
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                self.a_end:self.u_end,
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

    def get_coordinates(self):
        x = self.dataset.coords["x"].values
        y = self.dataset.coords["y"].values
        t = self.dataset.coords["time"].values
        return (x, y, t)

    def get_domain_range(self):
        x, y, t = self.get_coordinates()
        L_x = x[-1] - x[0]
        if self.mode == "single":
            t = t[self.u_end:self.u_end+1]
        elif self.mode == "autoregressive":
            t = t[self.a_end:self.u_end+1]
        L_t = t[-1]-t[0]
        return [L_x, L_t]
    



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

        # Set timesteps for extraction
        self.a_start = self.t_start
        self.a_end = self.a_start + self.init_steps
        self.u_end = self.a_end + self.pred_horizon

    def __len__(self):
        return len(self.dataset["samples"])

    def __getitem__(self, idx):
        a = self.dataset[
            idx,
            self.a_start:self.a_end,
        ]
        if self.mode == "single":
            u = self.dataset[
                idx,
                self.u_end : self.u_end + 1,
            ]
        elif self.mode == "autoregressive":
            u = self.dataset[
                idx,
                self.a_end:self.u_end,
            ]
        # Create grid
        x = a.coords["x"]
        t = a.coords["t"]
        grid = np.stack(np.meshgrid(x,t))
        # Stack input and grid
        tensor_a = torch.tensor(a.to_numpy()).float().unsqueeze(0)
        tensor_a = torch.cat([tensor_a, torch.tensor(grid)], dim=0).float()
        tensor_u = torch.tensor(u.to_numpy()).float().unsqueeze(0)
        return tensor_a, tensor_u

    def get_coordinates(self):
        x = self.dataset.coords["x"].values
        t = self.dataset.coords["t"].values
        return (x, t)

    def get_domain_range(self):
        x, t = self.get_coordinates()
        L_x = x[-1] - x[0]
        if self.mode == "single":
            t = t[self.u_end:self.u_end+1]
        elif self.mode == "autoregressive":
            t = t[self.a_end:self.u_end+1]
        L_t = t[-1]-t[0]
        return [L_x, L_t]


if __name__ == "__main__":
    data_dir = "data/SWE/processed/"
    dataset = SWEDataset(
        data_dir,
        test=False,
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
