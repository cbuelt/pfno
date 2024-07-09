from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import sys
sys.path.append(os.getcwd())
from models.fno_layers import SpectralConv1d, SpectralConv2d, MLPLinear


class FNO1d(nn.Module):
    def __init__(     
            self, 
            n_modes,
            hidden_channels,
            in_channels = 3,
            out_channels = 1,
            lifting_channels = 256,
            projection_channels = 256,
            n_layers = 4,
            non_linearity = F.gelu,
            dropout_rate = None,
            fourier_dropout_rate = None,
            type = "complex",
            ):
        """A one-dimensional FNO with n_layers layers of spectral convolutions.

        Args:
            n_modes (_type_): _description_
            hidden_channels (_type_): _description_
            in_channels (int, optional): _description_. Defaults to 3.
            out_channels (int, optional): _description_. Defaults to 1.
            lifting_channels (int, optional): _description_. Defaults to 256.
            projection_channels (int, optional): _description_. Defaults to 256.
            n_layers (int, optional): _description_. Defaults to 4.
            non_linearity (_type_, optional): _description_. Defaults to F.gelu.
            dropout_rate (_type_, optional): _description_. Defaults to None.
        """
        super(FNO1d, self).__init__()

        # Initialize the network
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.dropout_rate = dropout_rate
        self.fourier_dropout_rate = fourier_dropout_rate
        self.type = type
        # Different dropout ratio in Fourier space
        if self.fourier_dropout_rate is None:
            self.fourier_dropout_rate = dropout_rate


        self.lifting = MLPLinear(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            hidden_channels = self.lifting_channels,
            n_layers = 2,
            non_linearity = self.non_linearity,
            dropout_rate = self.dropout_rate,
        )

        # Spectral convolutions
        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv1d(
                    in_channels = self.hidden_channels,
                    out_channels = self.hidden_channels,
                    modes = self.n_modes,
                    dropout_rate=self.fourier_dropout_rate,
                    type = self.type,
                )
                for _ in range(n_layers)
            ]
        )

        # Local convolutions
        self.convs =  nn.ModuleList(
            [
                nn.Conv1d(self.hidden_channels, self.hidden_channels, 1)
                for _ in range(n_layers)
            ]
        )

        # Projection layer
        self.projection = MLPLinear(
            in_channels = self.hidden_channels,
            out_channels = self.out_channels,
            hidden_channels = self.projection_channels,
            n_layers = 2,
            non_linearity = self.non_linearity,
            dropout_rate = self.dropout_rate,
        )

    def forward(self, x):
        """Forward function

        Args:
            x (_type_): Size of x is [B, C, d1]

        Returns:
            _type_: _description_
        """

        # Lift the input to the desired channel dimension
        x = self.lifting(x)

        # Apply the spectral convolutions
        for layer_idx in range(self.n_layers):
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.convs[layer_idx](x)
            x = x1 + x2
            if layer_idx < self.n_layers - 1:
                x = self.non_linearity(x)

        # Project from the channel space to the output space
        x = self.projection(x)
        return x
    

class FNO2d(nn.Module):
    def __init__(     
            self, 
            n_modes,
            hidden_channels,
            in_channels = 3,
            out_channels = 1,
            lifting_channels = 256,
            projection_channels = 256,
            n_layers = 4,
            non_linearity = F.gelu,
            dropout_rate = None,
            fourier_dropout_rate = None,
            type = "complex",
            ):
        """A one-dimensional FNO with n_layers layers of spectral convolutions.

        Args:
            n_modes (_type_): _description_
            hidden_channels (_type_): _description_
            in_channels (int, optional): _description_. Defaults to 3.
            out_channels (int, optional): _description_. Defaults to 1.
            lifting_channels (int, optional): _description_. Defaults to 256.
            projection_channels (int, optional): _description_. Defaults to 256.
            n_layers (int, optional): _description_. Defaults to 4.
            non_linearity (_type_, optional): _description_. Defaults to F.gelu.
            dropout_rate (_type_, optional): _description_. Defaults to None.
        """
        super(FNO2d, self).__init__()

        # Initialize the network
        self.n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.n_layers = n_layers
        self.non_linearity = non_linearity
        self.dropout_rate = dropout_rate
        self.type = type
        self.fourier_dropout_rate = fourier_dropout_rate
        # Different dropout ratio in Fourier space
        if self.fourier_dropout_rate is None:
            self.fourier_dropout_rate = dropout_rate


        self.lifting = MLPLinear(
            in_channels = self.in_channels,
            out_channels = self.hidden_channels,
            hidden_channels = self.lifting_channels,
            n_layers = 2,
            non_linearity = self.non_linearity,
            dropout_rate = self.dropout_rate,
        )

        # Spectral convolutions
        self.spectral_convs = nn.ModuleList(
            [
                SpectralConv2d(
                    in_channels = self.hidden_channels,
                    out_channels = self.hidden_channels,
                    modes = self.n_modes,
                    dropout_rate=self.fourier_dropout_rate,
                    type = self.type,
                )
                for _ in range(n_layers)
            ]
        )

        # Local convolutions
        self.convs =  nn.ModuleList(
            [
                nn.Conv2d(self.hidden_channels, self.hidden_channels, 1)
                for _ in range(n_layers)
            ]
        )

        # Projection layer
        self.projection = MLPLinear(
            in_channels = self.hidden_channels,
            out_channels = self.out_channels,
            hidden_channels = self.projection_channels,
            n_layers = 2,
            non_linearity = self.non_linearity,
            dropout_rate = self.dropout_rate,
        )

    def forward(self, x):
        """Forward function

        Args:
            x (_type_): Size of x is [B, C, d1, d2]

        Returns:
            _type_: _description_
        """

        # Lift the input to the desired channel dimension
        x = self.lifting(x)

        # Apply the spectral convolutions
        for layer_idx in range(self.n_layers):
            x1 = self.spectral_convs[layer_idx](x)
            x2 = self.convs[layer_idx](x)
            x = x1 + x2
            if layer_idx < self.n_layers - 1:
                x = self.non_linearity(x)

        # Project from the channel space to the output space
        x = self.projection(x)
        return x


# Main method
if __name__ == "__main__":
    # Create a model
    model = FNO2d(n_modes = (12,12), hidden_channels = 64, in_channels = 2,
                  out_channels = 2, n_layers = 4, dropout_rate=0.6)
    x = torch.randn(5, 2, 128, 128)

    out = model(x)
    print(out.shape)
    print(out.dtype)
