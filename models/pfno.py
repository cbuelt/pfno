import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
sys.path.append(os.getcwd())
#from models.layers import FNOBlocks, SpectralConv, MLP

class PFNO_Wrapper(nn.Module):
    def __init__(self, model: nn.Module, n_samples: int = 3):
        """ Takes a deterministic model and wraps it to generate n_samples from the predictive distribution.

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


# Main method
if __name__ == '__main__':
    print("test")


