import torch

def generate_deterministic_samples(
    model: torch.nn.Module, x: torch.Tensor, shape: tuple, n_samples: int = 100
) -> torch.Tensor:
    """Generate samples from a deterministic model (as a simple baseline).

    Args:
        model (torch.nn.Module): The underlying model.
        x (torch.Tensor): The input tensor.
        shape (tuple): The output shape, excluding the sample dimension.
        n_samples (int, optional): Number of samples to generate. Defaults to 100.

    Returns:
        torch.Tensor: Output samples.
    """
    model.eval()
    with torch.no_grad():
        samples = model(x).detach()
        samples = samples.repeat_interleave(n_samples).reshape(*samples.shape, n_samples)
    return samples
