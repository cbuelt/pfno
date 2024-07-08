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
    ):
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
        x = self.dataset["x-coordinate"][:: self.downscaling_factor]
        y = self.dataset["y-coordinate"][:: self.downscaling_factor]
        grid = np.stack(np.meshgrid(x, y))
        # Stack grid and input
        tensor_a = torch.cat(
            (torch.tensor(a).float().unsqueeze(0), torch.tensor(grid).float()), dim=0
        )
        tensor_u = torch.tensor(u).float().unsqueeze(0)
        return tensor_a, tensor_u


if __name__ == "__main__":
    data_dir = "data/DarcyFlow/processed/"
    dataset = DarcyFlowDataset(data_dir, test=True, beta=1.0, downscaling_factor=1)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    for sample in train_loader:
        a, u = sample
        print(a.shape, u.shape)
        break
