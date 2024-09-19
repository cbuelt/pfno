# This file provides the implementation of the probabilistic neural operator.
# Parts of this code are adapted from https://github.com/neuraloperator/neuraloperator.

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

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys

sys.path.append(os.getcwd())
from models.layers import FNOBlocks, SpectralConv, MLP
from models.spherical_layers import SphericalConv
from neuralop.models.base_model import BaseModel
from neuralop.layers.padding import DomainPadding
from neuralop.layers.skip_connections import skip_connection
from neuralop.layers.resample import resample
from neuralop.models.fno import partialclass


class PNO_Wrapper(nn.Module):
    """
    Takes a model and wraps it to generate n_samples from the predictive distribution.
    The model should be inherently stochastic (e.g. via dropout), otherwise all created samples will be identical.
    """

    def __init__(self, model: nn.Module, n_samples: int = 3):
        """Initiliaze the PNO_Wrapper.

        Args:
            model (nn.Module): Neural network.
            n_samples (int, optional): Number of output samples. Defaults to 3.
        """
        super(PNO_Wrapper, self).__init__()
        self.model = model
        self.n_samples = n_samples

    def forward(self, input: torch.Tensor, n_samples: int = None) -> torch.Tensor:
        """_summary_

        Args:
            input (torch.Tensor): Input to the network.
            n_samples (int, optional): Number of samples to generate. Defaults to None.

        Returns:
            torch.Tensor: Samples from the prdictive distribution.
        """
        if n_samples is None:
            n_samples = self.n_samples

        outputs = [self.model(input) for _ in range(n_samples)]

        # stack along the second dimension and add a last dimension if missing.
        return torch.atleast_3d(torch.stack(outputs, dim=-1))


class PFNO(BaseModel, name="PFNO"):
    """
    Probabilistic Fourier Neural Operator (PFNO) model.
    The model is a probabilistic version of the Fourier Neural Operator (FNO) model that utilizes the repametrization
    in the projection layers to generate samples from the predictive distribution.
    """
    def __init__(
        self,
        n_modes,
        hidden_channels,
        n_samples=3,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        dropout=None,
        fourier_dropout=None,
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
        conv_module=SpectralConv,
        **kwargs
    ):
        super().__init__()
        self.n_dim = len(n_modes)

        # See the class' property for underlying mechanism
        # When updated, change should be reflected in fno blocks
        self._n_modes = n_modes
        self.hidden_channels = hidden_channels
        self.lifting_channels = lifting_channels
        self.projection_channels = projection_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
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
        self.fno_block_precision = fno_block_precision
        self.n_samples = n_samples

        # Define dropout
        self.dropout = dropout
        if fourier_dropout is None:
            self.fourier_dropout = self.dropout
        else:
            self.fourier_dropout = fourier_dropout

        if domain_padding is not None and (
            (isinstance(domain_padding, list) and sum(domain_padding) > 0)
            or (isinstance(domain_padding, (float, int)) and domain_padding > 0)
        ):
            self.domain_padding = DomainPadding(
                domain_padding=domain_padding,
                padding_mode=domain_padding_mode,
                output_scaling_factor=output_scaling_factor,
            )
        else:
            self.domain_padding = None

        self.domain_padding_mode = domain_padding_mode

        if output_scaling_factor is not None and not joint_factorization:
            if isinstance(output_scaling_factor, (float, int)):
                output_scaling_factor = [output_scaling_factor] * self.n_layers
        self.output_scaling_factor = output_scaling_factor

        self.fno_blocks = FNOBlocks(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            n_modes=self.n_modes,
            dropout=self.fourier_dropout,
            output_scaling_factor=output_scaling_factor,
            use_mlp=use_mlp,
            mlp_dropout=mlp_dropout,
            mlp_expansion=mlp_expansion,
            non_linearity=non_linearity,
            stabilizer=stabilizer,
            norm=norm,
            preactivation=preactivation,
            fno_skip=fno_skip,
            mlp_skip=mlp_skip,
            max_n_modes=max_n_modes,
            fno_block_precision=fno_block_precision,
            rank=rank,
            fft_norm=fft_norm,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            conv_module=conv_module,
            n_layers=n_layers,
            **kwargs
        )

        # if lifting_channels is passed, make lifting an MLP
        # with a hidden layer of size lifting_channels
        if self.lifting_channels:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.lifting_channels,
                n_layers=2,
                dropout_rate=self.dropout,
            )
        # otherwise, make it a linear layer
        else:
            self.lifting = MLP(
                in_channels=in_channels,
                out_channels=self.hidden_channels,
                hidden_channels=self.hidden_channels,
                n_layers=1,
                dropout_rate=self.dropout,
            )

        # Define mu and sigma for reparametrization trick
        self.mu = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

        self.sigma = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, n_samples=None, **kwargs):
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

        if n_samples is None:
            n_samples = self.n_samples

        if output_shape is None:
            output_shape = [None] * self.n_layers
        elif isinstance(output_shape, tuple):
            output_shape = [None] * (self.n_layers - 1) + [output_shape]

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        # Reparametrization trick
        mu = self.mu(x).unsqueeze(-1)
        sigma = F.softplus(self.sigma(x)).unsqueeze(-1) + 1e-6
        x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples).to(x.device)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes


class PUNO(nn.Module):
    """
    Probabilistic U-shaped Neural Operator (PUNO) model.
    The model is a probabilistic version of the U-shaped Neural Operator (UNO) model that utilizes the repametrization
    in the projection layers to generate samples from the predictive distribution.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        n_samples=3,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        dropout=None,
        fourier_dropout=None,
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
        self.n_layers = len(uno_out_channels)
        assert uno_out_channels is not None, "uno_out_channels can not be None"
        assert uno_n_modes is not None, "uno_n_modes can not be None"
        assert uno_scalings is not None, "uno_scalings can not be None"
        assert (
            len(uno_out_channels) == self.n_layers
        ), "Output channels for all layers are not given"
        assert (
            len(uno_n_modes) == self.n_layers
        ), "number of modes for all layers are not given"
        assert (
            len(uno_scalings) == self.n_layers
        ), "Scaling factor for all layers are not given"

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
        self.n_samples = n_samples

        # Define dropout
        self.dropout = dropout
        if fourier_dropout is None:
            self.fourier_dropout = self.dropout
        else:
            self.fourier_dropout = fourier_dropout

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
            dropout_rate=self.dropout,
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
                    dropout=self.fourier_dropout,
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
                    conv_module=self.integral_operator,
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

        # Reparametrization layers
        self.mu = MLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

        self.sigma = MLP(
            in_channels=prev_out,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

    def forward(self, x, n_samples=None, **kwargs):
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

        # Reparametrization trick
        mu = self.mu(x).unsqueeze(-1)
        sigma = F.softplus(self.sigma(x)).unsqueeze(-1) + 1e-6
        x = mu + sigma * torch.randn(*mu.shape[:-1], n_samples).to(x.device)

        return x


# SFNO
PSFNO = partialclass("PSFNO", PFNO, conv_module=SphericalConv)
