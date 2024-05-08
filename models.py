from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.spherical_convolution import SphericalConv
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP
from neuralop.models.base_model import BaseModel
from neuralop.models.fno import FNO


class FNO_reparam(FNO, name = "FNO_reparam"):
    def __init__(
    self,
    n_modes,
    hidden_channels,
    n_samples,
    in_channels=3,
    out_channels=1,
    lifting_channels=256,
    projection_channels=256,
    n_layers=4
):
        super().__init__(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers
        )
        self.n_samples = n_samples

        # Add a reparameterization layer
        self.mu = MLP(
        in_channels=self.hidden_channels,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=self.non_linearity,
    )
        
        self.sigma = MLP(
        in_channels=self.hidden_channels,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=nn.Softplus(),
    )
        
    def forward(self, x, n_samples = None, output_shape = None, **kwargs):
        """TFNO's forward pass

        Parameters
        ----------
        x : tensor
            input tensor
        output_shape : {tuple, tuple list, None}, default is None
            Gives the option of specifying the exact output shape for odd shaped inputs.
            * If None, don't specify an output shape
            * If tuple, specifies the output-shape of the **last** FNO Block
            * If tuple list, specifies the exact output-shape of each FNO Block
        """

        if output_shape is None:
            output_shape = [None]*self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None]*(self.n_layers - 1) + [output_shape]

        if n_samples is None:
            n_samples = self.n_samples

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        mu = self.mu(x)
        sigma = self.sigma(x)

        # Reparameterization trick
        x = mu + sigma * torch.randn(n_samples, *mu.shape)
        x = x.permute(1,2,3,0)

        return x



# Main method
if __name__ == '__main__':
    # Create a model
    model = FNO_reparam(n_modes=(64,), hidden_channels=64, in_channels=2, out_channels=1, n_samples = 10)
    x = torch.randn(5, 2, 128)

    out = model(x)
    print(out.shape)

