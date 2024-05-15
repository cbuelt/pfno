import math
from typing import List

import torch
      
  
#loss function with abs Lp loss
class LpLoss(object):
    def __init__(self, d=1, p=2, L=2*math.pi, reduce_dims=0, reductions='mean'):
        super().__init__()

        self.d = d
        self.p = p

        if isinstance(reduce_dims, int):
            self.reduce_dims = [reduce_dims]
        else:
            self.reduce_dims = reduce_dims
        
        if self.reduce_dims is not None:
            if isinstance(reductions, str):
                assert reductions == 'sum' or reductions == 'mean'
                self.reductions = [reductions]*len(self.reduce_dims)
            else:
                for j in range(len(reductions)):
                    assert reductions[j] == 'sum' or reductions[j] == 'mean'
                self.reductions = reductions

        if isinstance(L, float):
            self.L = [L]*self.d
        else:
            self.L = L
    
    def uniform_h(self, x):
        h = [0.0]*self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j]/x.size(-j)
        
        return h

    def reduce_all(self, x):
        for j in range(len(self.reduce_dims)):
            if self.reductions[j] == 'sum':
                x = torch.sum(x, dim=self.reduce_dims[j], keepdim=True)
            else:
                x = torch.mean(x, dim=self.reduce_dims[j], keepdim=True)
        
        return x

    def abs(self, x, y, h=None):
        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        
        const = math.prod(h)**(1.0/self.p)
        diff = const*torch.norm(torch.flatten(x, start_dim=-self.d) - torch.flatten(y, start_dim=-self.d), \
                                              p=self.p, dim=-1, keepdim=False)

        if self.reduce_dims is not None:
            diff = self.reduce_all(diff).squeeze()
            
        return diff

    def __call__(self, y_pred, y, **kwargs):
        return self.abs(y_pred, y)



  
class LpLoss_energy(object):
    """Calculates the LPLoss as EnergyScore

    Args:
        prediction (Tensor): Samples from predictive distribution. Shape = [batch_size, ..., n_samples]
        y (Tensor): Observed outcome. Shape = [batch_size, ..., 1]
    """
    def __init__(self, d=1, p=2, L=2 * math.pi, reduction="mean"):
        super().__init__()

        self.d = d
        self.p = p
        self.reduction = reduction

        if isinstance(L, float):
            self.L = [L] * self.d
        else:
            self.L = L

    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j-1)

        return h

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)

        return x

    def es(self, x, y, h=None):
        """_summary_

        Args:
            x (_type_): Model prediction (B, ..., n_samples)
            y (_type_): Target (B, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        # Number of generative samples
        n_samples = x.size()[-1]

        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h] * self.d

        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        # Calculate constant
        const = math.prod(h) ** (1.0 / self.p)

        # Calculate parts of energy score

        # Term 1
        es1 = const * torch.cdist(x_flat, y_flat, p = self.p)        
        es1 = torch.mean(torch.squeeze(es1, dim = 2), dim=1)
    

        # Term 2
        es2 = const*torch.cdist(x_flat, x_flat, p = self.p)
        es2 = torch.sum(es2, dim=(1,2))

        # Full energy score
        es = es1 - es2 / (2* n_samples * (n_samples-1))

        # Reduce
        es = self.reduce(es).squeeze()

        return es

    def __call__(self, y_pred, y, **kwargs):
        return self.es(y_pred, y)
    


def lp_norm(x, y, const, p=2):
    """Calculates the Lp norm between two tensors

    Args:
        x (Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).
        const (int, optional): Grid spacing.
        p (int, optional): Order of the norm. Defaults to 2.

    Returns:
        Tensor: Lp norm
    """
    norm = const*torch.cdist(x, y, p = p)
    return norm


def complex_norm(x, y, **kwargs):
    """Calculates the complex norm between two tensors

    Args:
        x (Tensor): First tensor of shape (B, n_samples_x, flatted_dims).
        y (Tensor): Second tensor of shape (B, n_samples_y, flatted_dims).

    Returns:
        Tensor: Complex norm norm
    """
    # Calculate dot product
    x_expand = torch.unsqueeze(x, dim=2)
    y_expand = torch.unsqueeze(y, dim=1)
    diff = x_expand - y_expand
    norm = torch.sqrt(torch.sum(diff * torch.conj(diff), dim = -1))
    
    return torch.real(norm)


class EnergyScore(object):
    def __init__(self, d = 1, type = "p", reduction = 'mean', reduce_dims = True, **kwargs):
        super().__init__()

        self.d = d
        self.type = type
        self.reduction = reduction
        self.reduce_dims = reduce_dims
        self.p = kwargs.get('p', 2)
        self.L = kwargs.get('L', 2*math.pi)

        if isinstance(self.L, float):
            self.L = [self.L]*self.d

        if self.type == "complex":
            self.norm = complex_norm
        else:
            self.norm = lp_norm

    def reduce(self, x):
        if self.reduction == "sum":
            x = torch.sum(x, dim=0, keepdim=True)
        else:
            x = torch.mean(x, dim=0, keepdim=True)
        return x
    
    def uniform_h(self, x):
        h = [0.0] * self.d
        for j in range(self.d, 0, -1):
            h[-j] = self.L[-j] / x.size(-j-1)
        return h

    def calculate_score(self, x, y, h = None):
        """Calculates the energy score for different metrics

        Args:
            x (_type_): Model prediction (Batch size, ..., n_samples)
            y (_type_): Target (Batch size, ..., 1)
            h (_type_, optional): _description_. Defaults to None.

        Returns:
            _type_: Energy score
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)


        # Restructure tensors to shape (Batch size, n_samples, flatted dims)
        x_flat = torch.swapaxes(torch.flatten(x, start_dim=1, end_dim=-2), 1, 2)
        y_flat = torch.swapaxes(torch.flatten(y, start_dim=1, end_dim=-2), 1, 2)

        #Assume uniform mesh
        if h is None:
            h = self.uniform_h(x)
        else:
            if isinstance(h, float):
                h = [h]*self.d
        if self.type == "lp":
            const = math.prod(h)**(1.0/self.p)
        else:
            const = 1.0

        # Calculate energy score
        term_1 = torch.mean(self.norm(x_flat, y_flat, const = const, p = self.p), dim=(1,2))
        term_2 = torch.sum(self.norm(x_flat, x_flat, const = const, p = self.p), dim=(1,2))
        score = term_1 - term_2 / (2 * n_samples * (n_samples - 1))

        # Reduce
        return self.reduce(score).squeeze() if self.reduce_dims else score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)
    
    #test with main method
if __name__ == '__main__':
    # set torch seed
    torch.manual_seed(0)
    input = torch.rand(10, 10, 5)
    truth = torch.ones(10, 10, 1)
    es = EnergyScore(d = 1, p = 4, type = "lp")
    lp_es = LpLoss_energy(d = 1, p = 4)

    score = es(input, truth)
    print(score.shape)
    print(score)

    score2 = lp_es(input, truth)
    print(score2.shape)
    print(score2)
  