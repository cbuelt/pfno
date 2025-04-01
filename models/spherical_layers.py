# This file provides the implementation of the SphericalConv classes as layers of the neuraloperator.
# The code is adapted from https://github.com/neuraloperator/neuraloperator.

# MIT License

# Copyright (c) 2023 NeuralOperator developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from neuralop.utils import validate_scaling_factor
import torch
from torch import nn
from tltorch.factorized_tensors.core import FactorizedTensor
from neuralop.layers.base_spectral_conv import BaseSpectralConv
from neuralop.layers.spherical_convolution import get_contract_fun
from neuralop.layers.spectral_convolution import SubConv
from neuralop.layers.spherical_convolution import SHT
Number = Union[int, float]

class SphericalConv(BaseSpectralConv):
    """Spherical Convolution, base class for the SFNO [1]_
    
    Parameters
    ----------
    sht_norm : str, {'ortho'}
    sht_grids : str or str list, default is "equiangular", {"equiangular", "legendre-gauss"}
                * If str, the same grid is used for all layers
                * If list, should have n_layers + 1 values, corresponding to the input and output grid of each layer
                  e.g. for 1 layer, ["input_grid", "output_grid"]

    See SpectralConv for full list of other parameters

    References
    ----------
    .. [1] Spherical Fourier Neural Operators: Learning Stable Dynamics on the Sphere,
           Boris Bonev, Thorsten Kurth, Christian Hundt, Jaideep Pathak, Maximilian Baust, Karthik Kashinath, Anima Anandkumar,
           ICML 2023.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        dropout_rate=None,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        # fno_block_precision="full",
        rank=0.5,
        factorization="cp",
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs=dict(),
        init_std="auto",
        sht_norm="ortho",
        sht_grids="equiangular",
        device=None,
        dtype=torch.float32,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization
        #Dropout
        self.dropout_rate = dropout_rate

        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self._n_modes = n_modes
        self.order = len(n_modes)

        if max_n_modes is None:
            max_n_modes = self.n_modes
        elif isinstance(max_n_modes, int):
            max_n_modes = [max_n_modes]
        self.max_n_modes = max_n_modes

        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.order, n_layers)

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} "
                    f"and out_channels={out_channels}",
                )
            weight_shape = (in_channels, *self.n_modes[:-1])
        else:
            weight_shape = (in_channels, out_channels, *self.n_modes[:-1])
        self.separable = separable

        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **decomposition_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **decomposition_kwargs,
                    )
                    for _ in range(n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation, separable=separable
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

        self.sht_norm = sht_norm
        if isinstance(sht_grids, str):
            sht_grids = [sht_grids]*(self.n_layers + 1)
        self.sht_grids = sht_grids
        self.sht_handle = SHT(dtype=self.dtype, device=self.device)

    def _get_weight(self, index):
        return self.weight[index]
    
    def transform(self, x, layer_index=0, output_shape=None):
        *_, in_height, in_width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            height = round(in_height * self.output_scaling_factor[layer_index][0])
            width = round(in_width * self.output_scaling_factor[layer_index][1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]
        else:
            height, width = in_height, in_width

        # Return the identity if the resolution and grid of the input and output are the same
        if ((in_height, in_width) == (height, width)) and (self.sht_grids[layer_index] == self.sht_grids[layer_index+1]):
            return x
        else:
            coefs = self.sht_handle.sht(x, s=self.n_modes, norm=self.sht_norm, grid=self.sht_grids[layer_index])
            return self.sht_handle.isht(coefs, s=(height, width), norm=self.sht_norm, grid=self.sht_grids[layer_index + 1])
        

    # Dropout mask
    def get_dropout_mask(self, weights):
        mask = torch.ones_like(weights.real)
        if self.dropout_rate is not None:
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout_rate, training=self.training
            )
        return mask

    def forward(self, x, indices=0, output_shape=None):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, height, width = x.shape

        if self.output_scaling_factor is not None and output_shape is None:
            scaling_factors = self.output_scaling_factor[indices]
            height = round(height * scaling_factors[0])
            width = round(width * scaling_factors[1])
        elif output_shape is not None:
            height, width = output_shape[0], output_shape[1]

        out_fft = self.sht_handle.sht(x, s=(self.n_modes[0], self.n_modes[1]//2),
                                      norm=self.sht_norm, grid=self.sht_grids[indices])

        out_fft = self._contract(
            out_fft[:, :, :self.n_modes[0], :self.n_modes[1]//2],
            self._get_weight(indices)[:, :, :self.n_modes[0]],
            separable=self.separable,
            dhconv=True,
        )

        # Apply Dropout in Frequency space
        out_fft = out_fft * self.get_dropout_mask(out_fft)

        x = self.sht_handle.isht(out_fft, s=(height, width), norm=self.sht_norm,
                                 grid=self.sht_grids[indices+1])

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x

    @property
    def n_modes(self):
        return self._n_modes
    
    @n_modes.setter
    def n_modes(self, n_modes):
        if isinstance(n_modes, int): # Should happen for 1D FNO only
            n_modes = [n_modes]
        else:
            n_modes = list(n_modes)
        self._n_modes = n_modes

    def get_conv(self, indices):
        """Returns a sub-convolutional layer from the joint parametrize main-convolution

        The parametrization of sub-convolutional layers is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single convolution is parametrized, directly use the main class."
            )

        return SubConv(self, indices)

    def __getitem__(self, indices):
        return self.get_conv(indices)