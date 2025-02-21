# Implements the different losses used in the training of the models.

import math
import numpy as np
from typing import List
import torch


class LpLoss(object):
    """
    A class for calculating the Lp loss between two tensors.
    """

    def __init__(
        self,
        d: int = 1,
        p: float = 2,
        L: float = 1.0,
        reduce_dims: List = [0],
        reduction: str = "mean",
        rel: bool = False,
    ):
        """Initializes the Lp loss class.

        Args:
            d (int, optional): Dimension of the domain. Defaults to 1.
            p (float, optional): Parameter for the Lp loss. Defaults to 2.
            L (float, optional): Grid spacing constant. Can be supplied as a list. Defaults to 1.0.
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()
        self.d = d
        self.p = p
        self.rel = rel

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reduction, str):
                assert reduction == "sum" or reduction == "mean"
                self.reductions = [reduction] * len(self.reduce_dims)
            else:
                for j in range(len(reduction)):
                    assert reduction[j] == "sum" or reduction[j] == "mean"
                self.reductions = reduction

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x: torch.Tensor) -> float:
        """Calculates the integration constant for uniform mesh.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Integration constant
        """
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j)
        return h

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)

        return x

    def abs(self, x: torch.Tensor, y: torch.tensor, h=None) -> torch.Tensor:
        """Calculates the absolute Lp loss between two tensors.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.tensor): Input tensor y
            h (_type_, optional): Integration constant. Defaults to None.

        Returns:
            torch.Tensor: Lp loss between two tensors
        """
        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(y)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        const = torch.tensor(np.array(math.prod(h) ** (1.0 / self.p)), device=x.device)
        diff = const * torch.norm(
            torch.flatten(x, start_dim=1) - torch.flatten(y, start_dim=1),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        if self.reduce_dims is not None:
            diff = self.reduce(diff).squeeze()

        return diff

    def relative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the relative Lp loss between two tensors.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.tensor): Input tensor y

        Returns:
            torch.Tensor: Relative Lp loss between two tensors
        """
        diff = torch.norm(
            torch.flatten(x, start_dim=1) - torch.flatten(y, start_dim=1),
            p=self.p,
            dim=-1,
            keepdim=False,
        )
        ynorm = torch.norm(
            torch.flatten(y, start_dim=1), p=self.p, dim=-1, keepdim=False
        )

        diff = diff / ynorm

        if self.reduce_dims is not None:
            diff = self.reduce(diff).squeeze()
        return diff

    def __call__(self, y_pred, y, **kwargs):
        if self.rel:
            return self.relative(y_pred, y)
        else:
            return self.abs(y_pred, y)


class SphericalL2Loss(object):
    """
    Calculates the spherical L2 loss between two tensors. The loss is weightet according to the spherical grid.
    """

    def __init__(
        self,
        nlon: int,
        weights: torch.Tensor,
        reduce_dims: List = [0],
        reduction: str = "mean",
        rel: bool = False,
    ):
        """Initializes the Spherical Lp loss class. The loss is calculated in a unit-free manner with weights summed to one.

        Args:
            nlon (int,): Dimension of longitude.
            weights (torch.Tensor): Weight tensor for the latitude spacing.
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()
        self.dlon = 1 / nlon
        self.weights = weights / weights.sum()
        self.rel = rel

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims

        if self.reduce_dims is not None:
            if isinstance(reduction, str):
                assert reduction == "sum" or reduction == "mean"
                self.reductions = [reduction] * len(self.reduce_dims)
            else:
                for j in range(len(reduction)):
                    assert reduction[j] == "sum" or reduction[j] == "mean"
                self.reductions = reduction

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == "sum":
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        return x

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the absolute L2 loss between two tensors weighted by the spherical grid.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.Tensor): Input tensor y

        Returns:
            torch.Tensor: Absolute weighted L2 loss.
        """
        sq_diff = torch.pow(x - y, 2)
        loss = torch.sqrt(
            torch.sum(sq_diff * self.weights * self.dlon, dim=(-3, -2, -1))
        )
        if self.reduce_dims is not None:
            loss = self.reduce(loss).squeeze()
        return loss

    def relative(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the relative L2 loss between two tensors weighted by the spherical grid.

        Args:
            x (torch.Tensor): Input tensor x
            y (torch.Tensor): Input tensor y

        Returns:
            torch.Tensor: Relative weighted L2 loss.
        """
        loss = self.abs(x, y)
        loss = loss / torch.sqrt(
            torch.sum(torch.pow(y, 2) * self.weights * self.dlon, dim=(-3, -2, -1))
        )
        if self.reduce_dims is not None:
            loss = self.reduce(loss).squeeze()
        return loss

    def __call__(self, y_pred, y, **kwargs):
        if self.rel:
            return self.relative(y_pred, y)
        else:
            return self.abs(y_pred, y)


def lp_norm(
    x: torch.Tensor, y: torch.Tensor, const: float, p: float = 2
) -> torch.Tensor:
    """Calculates the Lp norm between two tensors with different sample sizes

    Args:
        x (torch.Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (torch.Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).
        const (float): Integration constant.
        p (float, optional): Order of the norm. Defaults to 2.

    Returns:
        Tensor: Lp norm
    """
    norm = const * torch.cdist(x, y, p=p)
    return norm


class EnergyScore(object):
    """
    A class for calculating the energy score between two tensors.
    """

    def __init__(
        self,
        d: int = 1,
        type: str = "lp",
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict
    ):
        """Initializes the Energy score class.

        Args:
            d (int, optional): Dimension of the domain. Defaults to 1.
            type (str, optional): Type of metric to apply. Defaults to "lp".
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()
        self.d = d
        self.type = type
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.rel = kwargs.get("rel", False)
        self.p = kwargs.get("p", 2)
        self.L = kwargs.get("L", 1.0)
        # Arguments for spherical score
        if self.type == "spherical":
            self.nlon = kwargs.get("nlon", 256)
            self.weights = kwargs.get("weights", 1)
            self.dlon = 1 / self.nlon
            self.p = 2
            self.d = 2

        if isinstance(self.L, float):
            self.L = [self.L] * self.d

        self.norm = lp_norm

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def uniform_h(self, x: torch.Tensor) -> float:
        """Calculates the integration constant for uniform mesh.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Integration constant
        """
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j - 1)
        return h

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor, h=None) -> torch.Tensor:
        """Calculates the energy score between two tensors for different metrics.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)
            h (_type_, optional): Integration constant. Defaults to None.

        Returns:
            torch.Tensor: Energy score
        """
        n_samples = x.size()[-1]

        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = y.unsqueeze(-1)

        if self.type == "spherical":
            weights = self.weights.unsqueeze(-1) / self.weights.sum()
            x = x * torch.sqrt(weights * self.dlon)
            y = y * torch.sqrt(weights * self.dlon)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Assume uniform mesh
        if self.type == "lp":
            const = torch.tensor(
                np.array(math.prod(h) ** (1.0 / self.p)), device=x.device
            )
        else:
            const = 1.0

        # Calculate energy score
        term_1 = torch.mean(
            self.norm(x_flat, y_flat, const=const, p=self.p), dim=(1, 2)
        )
        term_2 = torch.sum(self.norm(x_flat, x_flat, const=const, p=self.p), dim=(1, 2))

        if self.rel:
            ynorm = (
                const
                * torch.norm(
                    torch.flatten(y_flat, start_dim=-1), p=self.p, dim=-1, keepdim=False
                ).squeeze()
            )
            term_1 = term_1 / ynorm
            term_2 = term_2 / ynorm

        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class KernelScore(object):
    """
    A class for calculating the kernel score between two tensors.
    """

    def __init__(
        self,
        d: int = 1,
        type: str = "lp",
        kernel: str = "gauss",
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict
    ):
        """Initializes the Energy score class.

        Args:
            d (int, optional): Dimension of the domain. Defaults to 1.
            type (str, optional): Type of metric to apply. Defaults to "lp".
            kernel (str, optional): Type of kernel to apply. Defaults to "gauss".
            reduction (str, optional): Which reduction should be applied. Defaults to "mean".
            reduce_dims (List, optional): Which dimensions to reduce loss across. Defaults to [0].
            rel (bool, optional): Whether to calculate relative or absolute loss. Defaults to False.
        """
        super().__init__()

        self.d = d
        self.type = type
        self.kernel = kernel
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.p = kwargs.get("p", 2)
        self.L = kwargs.get("L", 1.0)
        self.rel = kwargs.get("rel", False)
        self.gamma = kwargs.get("gamma", 1.0)

        if isinstance(self.L, float):
            self.L = [self.L] * self.d

        self.norm = lp_norm

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def uniform_h(self, x: torch.Tensor) -> float:
        """Calculates the integration constant for uniform mesh.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            float: Integration constant
        """
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j - 1)
        return h

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor, h=None) -> torch.Tensor:
        """Calculates the kernel score between two tensors for different metrics.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            torch.Tensor: Kernel score
        """
        n_samples = x.size()[-1]

        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Assume uniform mesh
        if self.type == "lp":
            const = torch.tensor(
                np.array(math.prod(h) ** (1.0 / self.p)), device=x.device
            )
        else:
            const = 1.0

        # Calculate Gram matrix
        K1 = self.norm(x_flat, y_flat, const=const, p=self.p)
        K2 = self.norm(x_flat, x_flat, const=const, p=self.p)

        if self.rel:
            ynorm = const * torch.norm(
                torch.flatten(y_flat, start_dim=-1), p=self.p, dim=-1, keepdim=True
            )
            K1 = K1 / ynorm
            K2 = K2 / ynorm

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


class CRPS(object):
    """
    A class for calculating the Continuous Ranked Probability Score (CRPS) between two tensors.
    """

    def __init__(
        self, reduction: str = "mean", reduce_dims: bool = True, **kwargs: dict
    ):
        """Initializes the CRPS class.

        Args:
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.nlon = kwargs.get("nlon", 256)
        self.nlat = self.nlon / 2
        self.weights = kwargs.get("weights", None)
        self.dlon = 1 / self.nlon

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the CRPS for two tensors.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: CRPS
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Dimensions of the input
        d = torch.prod(torch.tensor(x.shape[1:-1]))
        # Adjust weights for spherical grid
        if self.weights is not None:
            weights = self.weights / self.weights.sum()
            weights = weights.unsqueeze(-1) * self.dlon / (d / (self.nlon * self.nlat))
        else:
            weights = 1 / d
        x = x * weights
        y = y * weights

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Calculate energy score
        term_1 = torch.mean(torch.cdist(x_flat, y_flat, p=1), dim=(1, 2))  # /d
        term_2 = torch.sum(torch.cdist(x_flat, x_flat, p=1), dim=(1, 2))  # /d

        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class GaussianNLL(object):
    """
    A class for calculating the Gaussian negative log likelihood between two tensors.
    The loss assumes that the input consists of several samples from which the mean and standard deviation can be calculated.
    """

    def __init__(
        self, reduction: str = "mean", reduce_dims: bool = True, **kwargs: dict
    ):
        """Initializes the Gaussian negative log likelihood class.

        Args:
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.weights = kwargs.get("weights", None)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Gaussian negative log likelihood between two Tensors.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Gaussian negative log likelihood
        """
        n_dims = len(x.shape) - 2

        # Calculate sample mean and standard deviation
        mu = torch.mean(x, dim=-1)
        sigma = torch.clamp(torch.std(x, dim=-1), min=1e-6, max=1e6)

        # Assert dimension
        assert mu.size() == y.size()

        # Calculate Gaussian NLL
        gaussian = torch.distributions.Normal(mu, sigma)
        score = -gaussian.log_prob(y)
        # Weighting
        if self.weights is not None:
            weights = (self.weights / self.weights.sum()) * self.weights.size(0)
            score = score * weights

        if self.reduce_dims:
            # Aggregate CRPS over spatial dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class Coverage(object):
    """
    A class for calculating the Coverage probability between two tensors. The corresponding quantiles are calculated from the model predictions.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict
    ):
        """Initializes the Coverage probability class.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.weights = kwargs.get("weights", None)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Coverage probability between two tensors with respect to the significance level alpha.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Coverage probability.
        """
        n_dims = len(x.shape) - 2

        # Calculate quantiles
        q_lower = torch.quantile(x, self.alpha / 2, dim=-1)
        q_upper = torch.quantile(x, 1 - self.alpha / 2, dim=-1)

        # Assert dimension and alpha
        assert q_lower.size() == y.size()
        assert 0 < self.alpha < 1

        # Calculate coverage probability
        score = ((y > q_lower) & (y < q_upper)).float()
        # Weighting
        if self.weights is not None:
            weights = (self.weights / self.weights.sum()) * self.weights.size(0)
            score = score * weights

        if self.reduce_dims:
            # Aggregate over spatial and channel dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])
        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


class IntervalWidth(object):
    """
    A class for calculating the Intervalwidth between two tensors. The corresponding quantiles are calculated from the model predictions.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        reduction: str = "mean",
        reduce_dims: bool = True,
        **kwargs: dict
    ):
        """Initializes the Intervalwidth class.

        Args:
            alpha (float, optional): Significance level. Defaults to 0.05.
            reduction (str, optional): Which reduction to apply. Defaults to "mean".
            reduce_dims (bool, optional): Which dimensions to reduce. Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        # Kwargs for spherical loss
        self.weights = kwargs.get("weights", None)

    def reduce(self, x: torch.Tensor) -> torch.Tensor:
        """Reduces the tensor across all dimensions specified in reduce_dims.

        Args:
            x (torch.Tensor): Input tensor

        Returns:
            torch.Tensor: Reduced tensor
        """
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the Intervalwidth of a specified quantile.

        Args:
            x (torch.Tensor): Model prediction (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: Intervalwidth
        """
        n_dims = len(x.shape) - 2

        # Calculate quantiles
        q_lower = torch.quantile(x, self.alpha / 2, dim=-1)
        q_upper = torch.quantile(x, 1 - self.alpha / 2, dim=-1)

        # Assert dimension and alpha
        assert q_lower.size() == y.size()
        assert 0 < self.alpha < 1

        # Calculate interval width
        score = torch.abs(q_upper - q_lower)
        # Weighting
        if self.weights is not None:
            weights = self.weights / self.weights.sum() * self.weights.size(0)
            score = score * weights

        if self.reduce_dims:
            # Aggregate CRPS over spatial dimensions
            score = score.mean(dim=[d for d in range(1, n_dims + 1)])

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)
