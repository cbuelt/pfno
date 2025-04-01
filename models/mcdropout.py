# This file provides the implementation of the wrapper function to generate samples from a model using Monte Carlo Dropout.

import torch

def generate_mcd_samples(
    model: torch.nn.Module, x: torch.Tensor, shape: tuple, n_samples: int = 100
) -> torch.Tensor:
    """Generate samples from a model using Monte Carlo Dropout.

    Args:
        model (torch.nn.Module): The underlying model.
        x (torch.Tensor): The input tensor.
        shape (tuple): The output shape, excluding the sample dimension.
        n_samples (int, optional): Number of samples to generate. Defaults to 100.

    Returns:
        torch.Tensor: Output samples.
    """
    samples = torch.zeros(*shape, n_samples).to(x.device)
    for i in range(n_samples):
        with torch.no_grad():
            samples[..., i] = model(x).detach()
    return samples
