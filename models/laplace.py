# This file provides the implementation of the Laplace approximation for the neural network.
# The code is adapted from https://github.com/aleximmer/Laplace.

# MIT License

# Copyright (c) 2021 Alex Immer

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

from torch.nn.utils import vector_to_parameters
from laplace import Laplace
import torch

class LA_Wrapper(torch.nn.Module):
    """
    A class used to handle the Laplace approximation.

    Attributes
    ----------
    n_samples : int
        number of samples to generate
    method : str
        the method to use for the Laplace approximation
    hessian_structure : str
        the structure of the Hessian matrix
    optimize : bool
        whether to optimize the prior

    Methods
    -------
    fit(train_loader)
        Fits the Laplace approximation to the training data
    optimize()
        Optimizes the prior precision and prior noise sigma
    parameter_samples()
        Generate samples from the weight distribution
    predictive_samples(x)
        Generate samples from the predictive distribution
    """

    def __init__(
        self,
        model: torch.nn.Module,
        n_samples: int = 100,
        method: str = "last_layer",
        hessian_structure: str = "full",
        optimize: bool = True,
    ):
        """Initializes the Laplace approximation wrapper.

        Args:
            model (torch.nn.Module): The underlying model.
            n_samples (int, optional): The number of samples to generate. Defaults to 100.
            method (str, optional): The type of the approximation. Defaults to "last_layer".
            hessian_structure (str, optional): The structure of the Hessian. Defaults to "full".
            optimize (bool, optional): Whether to optimize prior precision. Defaults to True.
        """
        super().__init__()
        self.n_samples = n_samples
        self.method = method
        self.hessian_structure = hessian_structure
        self.optimize= optimize
        model.eval()
        self.la = Laplace(
            model,
            "regression",
            subset_of_weights=self.method,
            hessian_structure=self.hessian_structure,
        )

    def fit(self, train_loader: torch.utils.data.DataLoader):
        """Fits the Laplace approximation to the training data.

        Args:
            train_loader (torch.utils.data.DataLoader): The train loader.
        """
        self.la.fit(train_loader)
        if self.optimize:
            self.optimize_parameters()

    def save_state_dict(self, path: str):
        """Saves the state dictionary.

        Args:
            path (str): The path to save the state dictionary.
        """
        torch.save(self.la.state_dict(), path)

    def load_state_dict(self, state_dict):
        """Loads the model state dictionary.

        Args:
            state_dict (str): The path to load the state dictionary.
        """
        self.la.model.load_state_dict(state_dict)

    def load_la_state_dict(self, state_dict: dict):
        """Loads the laplace state dictionary.

        Args:
            state_dict (str): The path to load the state dictionary.
        """
        self.la.load_state_dict(state_dict)



    def optimize_parameters(self):
        """Optimize prior precision of the laplace approximation."""

        log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(
            1, requires_grad=True
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=5e-2)
        for _ in range(500):
            hyper_optimizer.zero_grad()
            neg_marglik = -self.la.log_marginal_likelihood(
                log_prior.exp(), log_sigma.exp()
            )
            neg_marglik.backward()
            hyper_optimizer.step()

    def parameter_samples(self) -> torch.Tensor:
        """Generate samples from the weight distribution.

        Returns:
            torch.Tensor: Output samples.
        """
        return self.la.sample()

    def predictive_samples(self, x: torch.Tensor, n_samples = None) -> torch.Tensor:
        """Generate samples from the predictive distribution.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        if not n_samples:
            n_samples = self.n_samples
        
        prediction = list()
        feats = None
        for sample in self.la.sample(n_samples):
            vector_to_parameters(sample, self.la.model.last_layer.parameters())

            if feats is None:
                # Cache features at the first iteration
                f, feats = self.la.model.forward_with_features(
                    x.to(self.la._device),
                )
                # f-shape: (B, C, d1, ..., dn)

            else:
                # Used the cached features for the rest iterations
                f = self.la.model.last_layer(feats)
                # f-shape: (B, d1, ..., dn, C)
                f = torch.movedim(f, -1, 1)
            prediction.append(f.detach() if not self.la.enable_backprop else f)

        vector_to_parameters(self.la.mean, self.la.model.last_layer.parameters())
        prediction = torch.stack(prediction, dim=-1)
        return prediction
