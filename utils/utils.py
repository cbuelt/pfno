import scipy
import torch
import h5py
import numpy as np
from torch.utils.data.dataset import Dataset

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.to(device)

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class TensorDataset(Dataset):
    def __init__(self, x, y, transform_x=None, transform_y=None):
        assert x.size(0) == y.size(0), "Size mismatch between tensors"
        self.x = x
        self.y = y
        self.transform_x = transform_x
        self.transform_y = transform_y

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]

        if self.transform_x is not None:
            x = self.transform_x(x)

        if self.transform_y is not None:
            y = self.transform_y(y)

        return x,y

    def __len__(self):
        return self.x.size(0)


def get_dataloaders(path, ntrain, ntest, batch_size, sampling_rate, grid_range=1):
    sub = 2**sampling_rate
    reader = MatReader(path)
    x_data = reader.read_field("a")[:, ::sub]
    y_data = reader.read_field("u")[:, ::sub]
    size_x = x_data.shape[1]

    # Add channel dimension
    x_data = x_data[:, None, :]
    y_data = y_data[:, None, :]

    x_train, y_train = x_data[:ntrain], y_data[:ntrain]
    x_test, y_test = x_data[-ntest:], y_data[-ntest:]

    # Add grid
    gridx = torch.tensor(np.linspace(0, grid_range, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, 1, size_x)

    x_train = torch.cat((x_train, gridx.repeat([ntrain, 1, 1])), dim=1)
    x_test = torch.cat((x_test, gridx.repeat([ntest, 1, 1])), dim=1)

    train_loader = torch.utils.data.DataLoader(
        TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False
    )
    test_data = [x_test, y_test]
    return train_loader, test_loader, test_data
