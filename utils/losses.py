import math
import numpy as np
from typing import List
import os
import sys
sys.path.append(os.getcwd())
from data.datasets import SSWEDataset

import torch


# loss function with abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2 * math.pi, reduce_dims=[0, 1], reductions="mean", rel = True):
        super().__init__()

        self.d = d
        self.p = p
        self.rel = rel

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)

        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x, y, h=None):
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        const = torch.tensor(np.array(math.prod(h) ** (1.0 / self.p)), device=x.device)
        diff = const * torch.norm(
            torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()

        return diff

    def relative(self, x, y):
        diff = torch.norm(
            torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=-self.d), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
        return diff

    def __call__(self, y_pred, y, **kwargs):
        if self.rel:
            return self.relative(y_pred, y)
        else:
            return self.abs(y_pred,y)
        


class SphericalL2Loss(object):
    def __init__(self, nlon, weights, reduce_dims=[0], reductions="mean", rel = True):
        super().__init__()

        self.dlon = 2* torch.pi/nlon
        self.weights = weights
        self.rel = rel

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == "sum" or reductions == "mean"
                self.reductions = [reductions] * len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == "sum" or reductions[j] == "mean"
                self.reductions = reductions


    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x, y):
        sq_diff = torch.pow(x - y, 2)
        loss = torch.sqrt(torch.sum(sq_diff*self.weights*self.dlon, dim=(-3,-2,-1)))
        if self.reduce_dims is not None:
            loss = self.reduce_all(loss).squeeze()
        return loss

    def relative(self, x, y):
        loss = self.abs(x,y)
        loss = loss / torch.sqrt(torch.sum(torch.pow(y,2)*self.weights*self.dlon, dim=(-3,-2,-1)))

        if self.reduce_dims is not None:
            loss = self.reduce_all(loss).squeeze()
        return loss

    def __call__(self, y_pred, y, **kwargs):
        if self.rel:
            return self.relative(y_pred, y)
        else:
            return self.abs(y_pred,y)
        

def lp_norm(x, y, const, p=2):
    """Calculates the Lp norm between two tensors with different sample sizes

    Args:
        x (Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).
        const (int, optional): Grid spacing.
        p (int, optional): Order of the norm. Defaults to 2.

    Returns:
        Tensor: Lp norm
    """
    norm = const * torch.cdist(x, y, p=p)
    return norm


def complex_norm(x, y, **kwargs):
    """Calculates the complex norm between two tensors with different sample sizes

    Args:
        x (Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).

    Returns:
        Tensor: Complex norm norm
    """
    # Calculate dot product
    x_expand = torch.unsqueeze(x, dim=2)
    y_expand = torch.unsqueeze(y, dim=1)
    diff = x_expand - y_expand
    norm = torch.sqrt(torch.sum(diff * torch.conj(diff), dim=-1))

    return torch.real(norm)


class EnergyScore(object):
    def __init__(self, d=1, type="lp", reduction="mean", reduce_dims=True, **kwargs):
        super().__init__()

        self.d = d
        self.type = type
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.rel = kwargs.get("rel", True)
        self.p = kwargs.get("p", 2)
        self.L = kwargs.get("L", 2 * math.pi)
        # Arguments for spherical score
        if self.type == "spherical":
            self.nlon = kwargs.get("nlon", 256)
            self.weights = kwargs.get("weights", 1)
            self.dlon = 2* torch.pi/nlon
            self.p = 2
            self.d = 2

        if isinstance(self.L, float):
            self.L = [self.L] * self.d

        if self.type == "complex":
            self.norm = complex_norm
        else:
            self.norm = lp_norm

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j - 1)
        return h

    def calculate_score(self, x, y):
        """Calculates the energy score for different metrics

        Args:
            x (_type_): Model prediction (Batch size, ..., n_samples)
            y (_type_): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = y.unsqueeze(-1)

        if self.type == "spherical":
            weights = self.weights.unsqueeze(-1)
            x = x * torch.sqrt(weights*self.dlon)
            y = y * torch.sqrt(weights*self.dlon)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Assume uniform mesh
        if self.type == "lp":
            const = torch.tensor(
                np.array(math.prod(self.L) ** (1.0 / self.p)), device=x.device
            )
        else:
            const = 1.0

        # Calculate energy score
        term_1 = torch.mean(
            self.norm(x_flat, y_flat, const=const, p=self.p), dim=(1, 2)
        )
        term_2 = torch.sum(self.norm(x_flat, x_flat, const=const, p=self.p), dim=(1, 2))

        if self.rel:
            ynorm = const * torch.norm(
                torch.flatten(y_flat, start_dim=-1), p=self.p, dim=-1, keepdim=False
            ).squeeze()
            term_1 = term_1/ynorm
            term_2 = term_2/ynorm

        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class KernelScore(object):
    def __init__(
        self,
        d=1,
        type="lp",
        kernel="gauss",
        reduction="mean",
        reduce_dims=True,
        **kwargs
    ):
        super().__init__()

        self.d = d
        self.type = type
        self.kernel = kernel
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.p = kwargs.get("p", 2)
        self.L = kwargs.get("L", 2 * math.pi)
        self.rel = kwargs.get("rel", True)
        self.gamma = kwargs.get("gamma", 1.0)

        if isinstance(self.L, float):
            self.L = [self.L] * self.d

        if self.type == "complex":
            self.norm = complex_norm
        else:
            self.norm = lp_norm

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j - 1)
        return h

    def calculate_score(self, x, y):
        """Calculates the energy score for different metrics

        Args:
            x (_type_): Model prediction (Batch size, ..., n_samples)
            y (_type_): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Assume uniform mesh
        if self.type == "lp":
            const = torch.tensor(
                np.array(math.prod(self.L) ** (1.0 / self.p)), device=x.device
            )
        else:
            const = 1.0

        # Calculate Gram matrix
        K1 = self.norm(x_flat, y_flat, const=const, p=self.p)
        K2 = self.norm(x_flat, x_flat, const=const, p=self.p)

        print(K1.shape)
        
        if self.rel:
            ynorm = const * torch.norm(
                torch.flatten(y_flat, start_dim=-1), p=self.p, dim=-1, keepdim=True
            )
            K1 = K1/ynorm
            K2 = K2/ynorm  

        if self.kernel == "laplace":
            K1 = torch.exp(-K1 / (2 * self.gamma**2))
            K2 = torch.exp(-K2 / (2 * self.gamma**2))
        elif self.kernel == "gauss":
            K1 = torch.exp(-torch.pow(K1 / (2 * self.gamma), 2))
            K2 = torch.exp(-torch.pow(K2 / (2 * self.gamma), 2))

        # Calculate kernel score
        # Remove diag
        K2 = K2 - torch.eye(n_samples).to(x.device)
        score = (
            torch.sum(K2, dim=(1, 2)) / (2 * n_samples * (n_samples - 1))
            - torch.mean(K1, dim=(1, 2))
            + 1
        )

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class VariogramScore(object):
    def __init__(self, p=1, reduction="mean", reduce_dims=True, **kwargs):
        super().__init__()

        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.p = kwargs.get("p", 1)

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x, y, h=None):
        """Calculates the energy score for different metrics

        Args:
            x (_type_): Model prediction (Batch size, ..., n_samples)
            y (_type_): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        d = x_flat.size(-1)

        # Calculate variogram score
        weights = (1 / d**2) * torch.ones(d, d).to(x.device)
        term_1 = torch.pow(
            torch.abs(
                torch.unsqueeze(y_flat, dim=-1) - torch.unsqueeze(y_flat, dim=-2)
            ),
            self.p,
        )
        term_2 = torch.pow(
            torch.abs(
                torch.unsqueeze(x_flat, dim=-1) - torch.unsqueeze(x_flat, dim=-2)
            ),
            self.p,
        )

        score = torch.sum(
            weights * torch.pow(term_1 - torch.mean(term_2, dim=1, keepdim=True), 2),
            dim=(-2, -1),
        )

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)
    

class CRPS(object):
    def __init__(self, reduction="mean", reduce_dims=True, **kwargs):
        super().__init__()

        self.reduction = reduction
        self.reduce_dims = reduce_dims

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x, y):
        """Calculates the energy score for different metrics

        Args:
            x (_type_): Model prediction (Batch size, ..., n_samples)
            y (_type_): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Flatten tensors
        x_flat = torch.flatten(x, start_dim=1, end_dim=-2)
        y_flat = torch.flatten(y, start_dim=1, end_dim=-2)

        # Calculate CRPS
        term_1 = torch.abs(x_flat - y_flat).mean(dim = -1)
        term_2 = torch.abs(
            torch.unsqueeze(x_flat, dim=-1) - torch.unsqueeze(x_flat, dim=-2)
        ).mean(dim = (-2,-1))
        score = term_1 - 0.5*term_2 
        if self.reduce_dims:
            # Aggregate CRPS over spatial dimensions
            score = score.mean(dim = -1)

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


if __name__ == "__main__":
    # set torch seed
    torch.manual_seed(0)
    input = torch.rand(4, 1, 2, 3)
    truth = torch.ones(4, 1, 2, 3)

    data_dir = "data/SSWE/processed/"
    train_data = SSWEDataset(data_dir, test = False)
    val_data = SSWEDataset(data_dir, test = True, pred_horizon = 1)

    x,y,t = train_data.get_coordinates()
    L = train_data.get_domain_range()
    weights = train_data.weights
    nlon = train_data.nlon
    input, truth = train_data[0]
    input = input.unsqueeze(0)
    truth = truth.unsqueeze(0)

    score_func = SphericalL2Loss(nlon, weights, rel = True)

    score = score_func(input, truth)

    print(score.shape)
    print(score.item())

    energy_score = EnergyScore(type="spherical", rel = True, nlon = nlon, weights = weights)
    e_score = energy_score(input.unsqueeze(-1).repeat(1,1,1,1,2), truth)
    print(e_score.item())
