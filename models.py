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
from layers import SpectralConv_complex, MLP_complex
from complexPyTorch.complexFunctions import complex_relu



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

        mu = self.mu(x).unsqueeze(-1)
        sigma = nn.functional.softplus(self.sigma(x)).unsqueeze(-1)

        # Reparameterization trick
        x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples).to(x.device)
        return x

# Factorized FNO
TFNO_reparam = partialclass("TFNO_reparam", FNO_reparam, factorization="Tucker")

# UNO reparam
class UNO_reparam(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_samples,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        uno_out_channels=None,
        uno_n_modes=None,
        uno_scalings=None,
        horizontal_skips_map=None,
        incremental_n_modes=None,
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        norm=None,
        preactivation=False,
        fno_skip="linear",
        horizontal_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        integral_operator=SpectralConv,
        operator_block=FNOBlocks,
        implementation="factorized",
        decomposition_kwargs=dict(),
        domain_padding=None,
        domain_padding_mode="one-sided",
        fft_norm="forward",
        normalizer=None,
        verbose=False,
        **kwargs
    ):
        super().__init__()
        self.n_layers = n_layers
        assert uno_out_channels is not None, "uno_out_channels can not be None"
        assert uno_n_modes is not None, "uno_n_modes can not be None"
        assert uno_scalings is not None, "uno_scalings can not be None"
        assert (
            len(uno_out_channels) == n_layers
        ), "Output channels for all layers are not given"
        assert (
            len(uno_n_modes) == n_layers
        ), "number of modes for all layers are not given"
        assert (
            len(uno_scalings) == n_layers
        ), "Scaling factor for all layers are not given"

        self.n_samples = n_samples
        self.n_dim = len(uno_n_modes[0])
        self.uno_out_channels = uno_out_channels
        self.uno_n_modes = uno_n_modes
        self.uno_scalings = uno_scalings
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.horizontal_skips_map = horizontal_skips_map
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = (fno_skip,)
        self.mlp_skip = (mlp_skip,)
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self._incremental_n_modes = incremental_n_modes
        self.operator_block = operator_block
        self.integral_operator = integral_operator

        # constructing default skip maps
        if self.horizontal_skips_map is None:
            self.horizontal_skips_map = {}
            for i in range(
                0,
                n_layers // 2,
            ):  
                # example, if n_layers = 5, then 4:0, 3:1
                self.horizontal_skips_map[n_layers - i - 1] = i
        # self.uno_scalings may be a 1d list specifying uniform scaling factor at each layer
        # or a 2d list, where each row specifies scaling factors along each dimention.
        # To get the final (end to end) scaling factors we need to multiply
        # the scaling factors (a list) of all layer.

        self.end_to_end_scaling_factor = [1] * len(self.uno_scalings[0])
        # multiplying scaling factors
        for k in self.uno_scalings:
            self.end_to_end_scaling_factor = [
                i * j for (i, j) in zip(self.end_to_end_scaling_factor, k)
            ]

        # list with a single element is replaced by the scaler.
        if len(self.end_to_end_scaling_factor) == 1:
            self.end_to_end_scaling_factor = self.end_to_end_scaling_factor[0]

        if isinstance(self.end_to_end_scaling_factor, (float, int)):
            self.end_to_end_scaling_factor = [
                self.end_to_end_scaling_factor
            ] * self.n_dim

        if verbose:
            print("calculated out factor", self.end_to_end_scaling_factor)

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=self.end_to_end_scaling_factor,
            )
        else:
            self.domain_padding = None
        self.domain_padding_mode = domain_padding_mode

        self.lifting = MLP(
            in_channels=in_channels,
            out_channels=self.hidden_channels,
            hidden_channels=self.lifting_channels,
            n_layers=2,
            n_dim=self.n_dim,
        )
        self.fno_blocks = nn.ModuleList([])
        self.horizontal_skips = torch.nn.ModuleDict({})
        prev_out = self.hidden_channels

        for i in range(self.n_layers):
            if i in self.horizontal_skips_map.keys():
                prev_out = (
                    prev_out + self.uno_out_channels[self.horizontal_skips_map[i]]
                )

            self.fno_blocks.append(
                self.operator_block(
                    in_channels=prev_out,
                    out_channels=self.uno_out_channels[i],
                    n_modes=self.uno_n_modes[i],
                    use_mlp=use_mlp,
                    mlp_dropout=mlp_dropout,
                    mlp_expansion=mlp_expansion,
                    output_scaling_factor=[self.uno_scalings[i]],
                    non_linearity=non_linearity,
                    norm=norm,
                    preactivation=preactivation,
                    fno_skip=fno_skip,
                    mlp_skip=mlp_skip,
                    incremental_n_modes=incremental_n_modes,
                    rank=rank,
                    SpectralConv=self.integral_operator,
                    fft_norm=fft_norm,
                    fixed_rank_modes=fixed_rank_modes,
                    implementation=implementation,
                    separable=separable,
                    factorization=factorization,
                    decomposition_kwargs=decomposition_kwargs,
                    joint_factorization=joint_factorization,
                    normalizer=normalizer,
                )
            )

            if i in self.horizontal_skips_map.values():
                self.horizontal_skips[str(i)] = skip_connection(
                    self.uno_out_channels[i],
                    self.uno_out_channels[i],
                    skip_type=horizontal_skip,
                    n_dim=self.n_dim,
                )

            prev_out = self.uno_out_channels[i]

        # Add a reparameterization layer
        self.mu = MLP(
        in_channels=prev_out,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=self.non_linearity,
    )
        
        self.sigma = MLP(
        in_channels=prev_out,
        out_channels=out_channels,
        hidden_channels=self.projection_channels,
        n_layers=2,
        n_dim=self.n_dim,
        non_linearity=self.non_linearity,
    )

    def forward(self, x, n_samples = None, **kwargs):

        if n_samples is None:
            n_samples = self.n_samples
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)
        output_shape = [
            int(round(i * j))
            for (i, j) in zip(x.shape[-self.n_dim :], self.end_to_end_scaling_factor)
        ]

        skip_outputs = {}
        cur_output = None
        for layer_idx in range(self.n_layers):
            
            if layer_idx in self.horizontal_skips_map.keys():
                skip_val = skip_outputs[self.horizontal_skips_map[layer_idx]]
                output_scaling_factors = [
                    m / n for (m, n) in zip(x.shape, skip_val.shape)
                ]
                output_scaling_factors = output_scaling_factors[-1 * self.n_dim :]
                t = resample(
                    skip_val, output_scaling_factors, list(range(-self.n_dim, 0))
                )
                x = torch.cat([x, t], dim=1)

            if layer_idx == self.n_layers - 1:
                cur_output = output_shape
            x = self.fno_blocks[layer_idx](x, output_shape=cur_output)

            if layer_idx in self.horizontal_skips_map.values():
                skip_outputs[layer_idx] = self.horizontal_skips[str(layer_idx)](x)


        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)
        

        mu = self.mu(x).unsqueeze(-1)
        sigma = nn.functional.softplus(self.sigma(x)).unsqueeze(-1)

        # Reparameterization trick
        x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples).to(x.device)
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
    model = FNO_complex(n_modes=(64,64), hidden_channels=64, in_channels=2, out_channels=1, n_samples = 10, output_type = "real")
    x = torch.randn(5, 2, 128, 128)

    out, out2 = model(x)
    print(out.shape)
    print(out.dtype)
    print(out2.dtype)


