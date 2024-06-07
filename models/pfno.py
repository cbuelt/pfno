from functools import partialmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralop.layers.spectral_convolution import SpectralConv
from neuralop.layers.padding import DomainPadding
from neuralop.layers.fno_block import FNOBlocks
from neuralop.layers.mlp import MLP
from neuralop.layers.resample import resample
from neuralop.layers.skip_connections import skip_connection
from neuralop.models.base_model import BaseModel
from neuralop.models.fno import FNO, partialclass
from neuralop.models.uno import UNO
from models.fno_layers import SpectralConv_complex, MLP_complex, MLPLinear
from complexPyTorch.complexFunctions import complex_relu



class PFNO_Wrapper(nn.Module):
    def __init__(self, model: nn.Module, n_samples: int = 3):
        """ Takes a deterministic model and wraps it to simulate n_samples

        Args:
            model (nn.Module): Neural network.
            n_samples (int, optional): Number of output samples. Defaults to 3.
        """
        super(PFNO_Wrapper, self).__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, input, n_samples: int = None):
        """
        Forward pass through the network self.n_samples times

        Args:
            _input: torch Tensor input to the network

        Returns:
            Outputs of the network, stacked along the last dimension
            Shape is therefore [batch, out_size, n_samples].
        """

        if n_samples is None:
            n_samples = self.n_samples

        outputs = [self.model(input) for _ in range(n_samples)]

        # stack along the second dimension and add a last dimension if missing.
        return torch.atleast_3d(torch.stack(outputs, dim=-1))




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
    n_layers=4,
    output_scaling_factor=None,
    max_n_modes=None,
    fno_block_precision="full",
    use_mlp=False,
    mlp_dropout=0,
    mlp_expansion=0.5,
    non_linearity=F.gelu,
    stabilizer=None,
    norm=None,
    preactivation=False,
    fno_skip="linear",
    mlp_skip="soft-gating",
    separable=False,
    factorization=None,
    rank=1.0,
    joint_factorization=False,
    fixed_rank_modes=False,
    implementation="factorized",
    decomposition_kwargs=dict(),
    domain_padding=None,
    domain_padding_mode="one-sided",
    fft_norm="forward",
    SpectralConv=SpectralConv,
    **kwargs
    ):
        super().__init__(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=output_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
            SpectralConv=SpectralConv,
        )
        self.n_samples = n_samples

        # Add a reparameterization layer
    #     self.mu = MLP(
    #     in_channels=self.hidden_channels,
    #     out_channels=out_channels,
    #     hidden_channels=self.projection_channels,
    #     n_layers=2,
    #     n_dim=self.n_dim,
    #     non_linearity=self.non_linearity,
    # )
        
    #     self.sigma = MLP(
    #     in_channels=self.hidden_channels,
    #     out_channels=out_channels,
    #     hidden_channels=self.projection_channels,
    #     n_layers=2,
    #     n_dim=self.n_dim,
    #     non_linearity=self.non_linearity,
    # )
        
        self.projection = MLPLinear(
        in_channels=self.hidden_channels,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        non_linearity=self.non_linearity,
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

      #  mu = self.mu(x).unsqueeze(-1)
       # sigma = nn.functional.softplus(self.sigma(x)).unsqueeze(-1)

        # Reparameterization trick
        #x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples).to(x.device)

        # Noise before transformation
        x = x.unsqueeze(-1) + torch.randn(*x.shape, n_samples).to(x.device)
        x = self.projection(x)
        return x


class FNO_complex(FNO, name = "FNO_complex"):
    def __init__(
    self,
    n_modes,
    hidden_channels,
    n_samples,
    output_type = "real",
    in_channels=3,
    out_channels=1,
    lifting_channels=256,
    projection_channels=256,
    n_layers=4,
    output_scaling_factor=None,
    max_n_modes=None,
    fno_block_precision="full",
    use_mlp=False,
    mlp_dropout=0,
    mlp_expansion=0.5,
    non_linearity=F.gelu,
    stabilizer=None,
    norm=None,
    preactivation=False,
    fno_skip="linear",
    mlp_skip="soft-gating",
    separable=False,
    factorization=None,
    rank=1.0,
    joint_factorization=False,
    fixed_rank_modes=False,
    implementation="factorized",
    decomposition_kwargs=dict(),
    domain_padding=None,
    domain_padding_mode="one-sided",
    fft_norm="forward",
    SpectralConv=SpectralConv,
    **kwargs
    ):
        super().__init__(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=in_channels,
            out_channels=out_channels,
            lifting_channels=lifting_channels,
            projection_channels=projection_channels,
            n_layers=n_layers,
            output_scaling_factor=output_scaling_factor,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            norm=norm,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            separable=separable,
            preactivation=preactivation,
            factorization=factorization,
            rank=rank,
            joint_factorization=joint_factorization,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            decomposition_kwargs=decomposition_kwargs,
            domain_padding=domain_padding,
            domain_padding_mode=domain_padding_mode,
            fft_norm=fft_norm,
            SpectralConv=SpectralConv,
        )
        self.n_samples = n_samples
        self.output_type = output_type

        # Add complex convolution layer
        self.final_conv = SpectralConv_complex(
            self.hidden_channels,
            self.hidden_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            max_n_modes=max_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        # Add a reparameterization layer
        self.mu = MLP(
        in_channels=hidden_channels,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=self.non_linearity,
    )

        
        self.sigma = MLP(
        in_channels=hidden_channels,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=self.non_linearity,
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

        batchsize, channels, *mode_sizes = x.shape
        order = len(self.n_modes)
        fft_dims_in = list(range(-order, 0))
        fft_dims_out = list(range(-order-1,-1))

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

        # Final layer Fourier transform
        mu = self.mu(x).unsqueeze(-1)
        sigma = self.sigma(x).unsqueeze(-1) + 0.5
        x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples, ).to(x.device) 

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims_out)


        # Return to physical
        if self.output_type == "real":
            x = torch.fft.irfftn(x, dim=fft_dims_out, norm=self.fft_norm)
            return x, None
        
        elif self.output_type == "complex":
            return torch.real(x), torch.imag(x)



# Main method
if __name__ == '__main__':
    # Create a model
    model = FNO_reparam(n_modes=(64,64), hidden_channels=64, in_channels=2, out_channels=1, n_samples = 10, output_type = "real")
    x = torch.randn(5, 2, 128, 128)

    out = model(x)
    print(out.shape)
    print(out.dtype)



