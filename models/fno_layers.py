from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP
# from complexPyTorch.complexLayers import ComplexLinear
# from complexPyTorch.complexFunctions import complex_relu

Number = Union[int, float]


# Reimplementation of the MLP class using Linear instead of Conv
class MLPLinear(torch.nn.Module):
    # Obtain input of shape [Batch, channels, d1, ..., dn]
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        non_linearity=F.gelu,
        dropout_rate=None,
        n_layers=2,
    ):
        super().__init__()
        self.n_layers = n_layers
        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        if dropout_rate is not None:
            self.dropout_rate = nn.ModuleList(
                [nn.Dropout(dropout_rate) for _ in range(self.n_layers - 1)]
            )
        else:
            self.dropout_rate = None

        # Input layer
        self.fcs.append(torch.nn.Linear(in_channels, hidden_channels))
        # Hidden layers
        for j in range(self.n_layers - 2):
            self.fcs.append(torch.nn.Linear(hidden_channels, hidden_channels))
        # Output layer
        self.fcs.append(torch.nn.Linear(hidden_channels, out_channels))

    def forward(self, x):
        # Reorder channel dim to last dim
        x = torch.movedim(x, 1, -1)
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                if self.dropout_rate is not None:
                    x = self.dropout_rate[i](x)
                x = self.non_linearity(x)

        # Return channel dim
        x = torch.movedim(x, -1, 1)                
        return x
    

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


    
    
# class SpectralConv2d(nn.Module):
#     def __init__(self, in_channels, out_channels, modes, dropout_rate=None):
#         super(SpectralConv2d, self).__init__()

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.mode1, self.mode2 = modes  # Number of Fourier modes to keep

#         self.scale = (1 / (in_channels * out_channels))
#         self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.mode1, self.mode2, dtype=torch.cfloat))
#         self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.mode1, self.mode2, dtype=torch.cfloat))

#         # Create dropout layer
#         self.dropout_rate = dropout_rate

#     # Complex multiplication
#     def compl_mul2d(self, input, weights):
#         return torch.einsum("bixy,ioxy->boxy", input, weights)
    
#     # Dropout mask
#     def get_dropout_mask(self, weights):
#         mask = torch.ones_like(weights.real)
#         if self.dropout_rate is not None:
#             mask = torch.nn.functional.dropout(
#                 mask, p=self.dropout_rate, training=self.training
#             )
#         return mask

#     def forward(self, x):
#         batchsize = x.shape[0]
#         x_ft = torch.fft.rfft2(x)

#         # Multiply relevant Fourier modes
#         out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
#         out_ft[:, :, :self.mode1, :self.mode2] = \
#             self.compl_mul2d(x_ft[:, :, :self.mode1, :self.mode2], self.weights1)
#         out_ft[:, :, -self.mode1:, :self.mode2] = \
#             self.compl_mul2d(x_ft[:, :, -self.mode1:, :self.mode2], self.weights2)
        
#         # Apply dropout        
#         out_ft = out_ft * self.get_dropout_mask(out_ft)


#         #Return to physical space
#         x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
#         return x

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
