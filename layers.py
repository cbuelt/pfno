from functools import partialmethod
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu

Number = Union[int, float]

class SpectralConv_complex(SpectralConv):

    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        max_n_modes=None,
        bias=True,
        n_layers=1,
        separable=False,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="backward",
        device=None,
        dtype=None,
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            n_modes=n_modes,
            max_n_modes=max_n_modes,
            bias=bias,
            n_layers=n_layers,
            separable=separable,
            output_scaling_factor=output_scaling_factor,
            fno_block_precision=fno_block_precision,
            rank=rank,
            factorization=factorization,
            implementation=implementation,
            fixed_rank_modes=fixed_rank_modes,
            joint_factorization=joint_factorization,
            decomposition_kwargs=decomposition_kwargs,
            init_std=init_std,
            fft_norm=fft_norm,
            device=device,
            dtype=dtype,
        )

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
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
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)
        if self.order > 1:
            x = torch.fft.fftshift(x, dim=fft_dims[:-1])

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_dtype = torch.chalf
        else:
            out_dtype = torch.cfloat
        out_fft = torch.zeros([batchsize, self.out_channels, *fft_size],
                              device=x.device, dtype=out_dtype)
        starts = [(max_modes - min(size, n_mode)) for (size, n_mode, max_modes) in zip(fft_size, self.n_modes, self.max_n_modes)]
        slices_w =  [slice(None), slice(None)] # Batch_size, channels
        slices_w += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_w += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        weight = self._get_weight(indices)[slices_w]

        starts = [(size - min(size, n_mode)) for (size, n_mode) in zip(list(x.shape[2:]), list(weight.shape[2:]))]
        slices_x =  [slice(None), slice(None)] # Batch_size, channels
        slices_x += [slice(start//2, -start//2) if start else slice(start, None) for start in starts[:-1]]
        slices_x += [slice(None, -starts[-1]) if starts[-1] else slice(None)] # The last mode already has redundant half removed
        out_fft[slices_x] = self._contract(x[slices_x], weight, separable=False)

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple([round(s * r) for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])])

        if output_shape is not None:
            mode_sizes = output_shape

        if self.order > 1:
            out_fft = torch.fft.fftshift(out_fft, dim=fft_dims[:-1])
     #   x = torch.fft.irfftn(out_fft, s=mode_sizes, dim=fft_dims, norm=self.fft_norm)

     
        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x
    

    # Reimplementation of the MLP class using Linear instead of Conv
class MLP_complex(torch.nn.Module):
    # Obtain input of shape [Batch, channels, d1, ..., dn]
    def __init__(self, in_channels, out_channels, hidden_channels, non_linearity=complex_relu, dropout=0.0,
                 n_layers = 2):
        super().__init__()

        self.n_layers = n_layers

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = non_linearity
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        # Input layer
        self.fcs.append(ComplexLinear(in_channels, hidden_channels))
        #Hidden layers
        for j in range(self.n_layers-2):
            self.fcs.append(ComplexLinear(hidden_channels, hidden_channels))
        # Output layer
        self.fcs.append(ComplexLinear(hidden_channels, out_channels))

    def forward(self, x):
        # Reorder channel dim to last dim
        x = torch.movedim(x, 1, -1)

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)
        # Return channel dim
        x = torch.movedim(x, -1, 1)

        return x
    



# Test spectral conv in main method
if __name__ == '__main__':
    # Create a spectral conv layer
    layer = MLP_complex(32, 1, 128)
    x = torch.rand(5, 32, 64, 64, dtype = torch.complex64)
    y = layer(x)
    print(y.shape)
    print(y.dtype)
