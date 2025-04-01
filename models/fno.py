# This file provides the implementation of the Fourier neural operator.
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

import torch
import torch.nn.functional as F
import os
import sys
sys.path.append(os.getcwd())
from models.layers import FNOBlocks, SpectralConv, MLP
from models.spherical_layers import SphericalConv
from neuralop.models.base_model import BaseModel
from neuralop.layers.padding import DomainPadding
from neuralop.models.fno import partialclass

class FNO(BaseModel, name='FNO'):
    def __init__(
        self,
        n_modes,
        hidden_channels,
        in_channels=3,
        out_channels=1,
        lifting_channels=256,
        projection_channels=256,
        n_layers=4,
        dropout = None,
        fourier_dropout = None,
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
        self.projection = MLP(
            in_channels=self.hidden_channels,
            out_channels=out_channels,
            hidden_channels=self.projection_channels,
            n_layers=2,
            dropout_rate=self.dropout,
            non_linearity=non_linearity,
        )

    def forward(self, x, output_shape=None, **kwargs):
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

        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx, output_shape=output_shape[layer_idx])

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x

    @property
    def n_modes(self):
        return self._n_modes

    @n_modes.setter
    def n_modes(self, n_modes):
        self.fno_blocks.n_modes = n_modes
        self._n_modes = n_modes

# SFNO
SFNO = partialclass("SFNO", FNO, factorization="dense", conv_module=SphericalConv)

