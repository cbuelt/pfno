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

  