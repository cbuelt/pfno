from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP

Number = Union[int, float]

class SpectralConv1d(SpectralConv):
    """1D Spectral Conv

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            dropout_rate = None,
            output_scaling_factor = None
    ):
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            n_modes = n_modes,
            output_scaling_factor=output_scaling_factor
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

    # Dropout mask
    def get_dropout_mask(self, weights):
        mask = torch.ones_like(weights.real)
        if self.dropout_rate is not None:
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout_rate, training=self.training
            )
        return mask

    def forward(self, x, indices=0):
        batchsize, channels, width = x.shape

        x = torch.fft.rfft(x, norm=self.fft_norm)

        out_fft = torch.zeros(
            [batchsize, self.out_channels, width // 2 + 1],
            device=x.device,
            dtype=torch.cfloat,
        )
        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None, self.n_modes[0]), # :half_n_modes[0]]
        )
        out_fft[slices] = self._contract(
            x[slices], self._get_weight(indices)[slices], separable=self.separable
        )

        # Apply Dropout in Fourier space
        out_fft = out_fft * self.get_dropout_mask(out_fft)

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])

        x = torch.fft.irfft(out_fft, n=width, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SpectralConv2d(SpectralConv):
    """2D Spectral Conv, see :class:`neuralop.layers.SpectraConv` for the general case

    This is provided for reference only,
    see :class:`neuralop.layers.SpectraConv` for the preferred, general implementation
    """
    def __init__(
            self,
            in_channels,
            out_channels,
            n_modes,
            dropout_rate = None,
            output_scaling_factor = None

    ):
        super().__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            n_modes = n_modes,
            output_scaling_factor=output_scaling_factor
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

    # Dropout mask
    def get_dropout_mask(self, weights):
        mask = torch.ones_like(weights.real)
        if self.dropout_rate is not None:
            mask = torch.nn.functional.dropout(
                mask, p=self.dropout_rate, training=self.training
            )
        return mask


    def forward(self, x, indices=0):
        batchsize, channels, height, width = x.shape

        x = torch.fft.rfft2(x.float(), norm=self.fft_norm, dim=(-2, -1))

        # The output will be of size (batch_size, self.out_channels,
        # x.size(-2), x.size(-1)//2 + 1)
        out_fft = torch.zeros(
            [batchsize, self.out_channels, height, width // 2 + 1],
            dtype=x.dtype,
            device=x.device,
        )

        slices0 = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.n_modes[0] // 2),  # :half_n_modes[0],
            slice(self.n_modes[1]),  #      :half_n_modes[1]]
        )
        slices1 = (
            slice(None),  # Equivalent to:        [:,
            slice(None),  # ...................... :,
            slice(-self.n_modes[0] // 2, None),  # -half_n_modes[0]:,
            slice(self.n_modes[1]),  # ......      :half_n_modes[1]]
        )

        """Upper block (truncate high frequencies)."""
        out_fft[slices0] = self._contract(
            x[slices0], self._get_weight(indices)[slices1], separable=self.separable
        )

        """Lower block"""
        out_fft[slices1] = self._contract(
            x[slices1], self._get_weight(indices)[slices0], separable=self.separable
        )

        # Apply Dropout in Fourier space
        out_fft = out_fft * self.get_dropout_mask(out_fft)

        if self.output_scaling_factor is not None:
            width = round(width * self.output_scaling_factor[0])

        x = torch.fft.irfft2(
            out_fft, s=(height, width), dim=(-2, -1), norm=self.fft_norm
        )

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


# Test spectral conv in main method
if __name__ == "__main__":
    # Create a spectral conv layer
    layer = SpectralConv2d(in_channels=32, out_channels=32, modes=(16,16), dropout_rate=None)
    x = torch.rand(5, 32, 64, 64)
    y = layer(x)
    print(y.shape)
    print(y.dtype)

    from neuralop.utils import count_model_params
    n_params = count_model_params(layer)
    print(f'\nOur model has {n_params} parameters.')

    from neuralop.layers.spectral_convolution import SpectralConv2d
    layer = SpectralConv2d(in_channels=32, out_channels=32, n_modes=(16,16), factorization = "tucker")
    n_params = count_model_params(layer)
    print(f'\nOur model has {n_params} parameters.')
