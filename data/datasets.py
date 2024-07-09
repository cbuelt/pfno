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
        tensor_a = torch.cat([torch.tensor(a).unsqueeze(0), torch.tensor(grid)], dim = 0)
        tensor_u = torch.tensor(u).unsqueeze(0)
        return tensor_a, tensor_u
    

class SWEDataset(Dataset):
    """Shallow Water Equation Dataset."""

    def __init__(
            self,
            data_dir: str,
            test:bool = False,
            init_steps: int = 10,
            mode:str = "single",
            downscaling_factor: int = 1,
            max_steps: int = 10,
            ) -> None:
        
        if test:
            self.filename = "swe_test.nc"
        else:
            self.filename = "swe_train.nc"
        self.dataset = xr.open_dataset(data_dir + self.filename)
        assert isinstance(downscaling_factor, int), "Scaling factor must be Integer"
        self.downscaling_factor = downscaling_factor
        self.init_steps = init_steps
        self.mode = mode
        self.max_steps = max_steps


    def __len__(self):
        return len(self.dataset["samples"])
    

    def __getitem__(self, idx):
        a = self.dataset.data[idx, 0:self.init_steps, ::self.downscaling_factor, ::self.downscaling_factor].to_numpy()
        if self.mode == "single":
            u = self.dataset.data[idx, self.init_steps:self.init_steps+1, ::self.downscaling_factor, ::self.downscaling_factor].to_numpy()
        elif self.mode == "autoregressive":
            u = self.dataset.data[idx, self.init_steps:self.init_steps+self.max_steps, ::self.downscaling_factor, ::self.downscaling_factor].to_numpy()
        # Create grid
        x = self.dataset.coords["x"][::self.downscaling_factor]
        y = self.dataset.coords["y"][::self.downscaling_factor]
        grid = np.stack(np.meshgrid(x, y))
        # Stack input and grid
        tensor_a = torch.cat([torch.tensor(a), torch.tensor(grid)], dim = 0)
        tensor_u = torch.tensor(u)
        return tensor_a, tensor_u


if __name__ == "__main__":
    data_dir = "data/SWE/processed/"
    dataset = SWEDataset(data_dir, test=False, downscaling_factor=2, mode = "autoregressive", init_steps = 1, max_steps = 10)
    print(dataset.__len__())
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    for sample in train_loader:
        a, u = sample
        print(a.shape, u.shape)
        print(a.dtype, u.dtype)
        break
