from torch.nn.utils import vector_to_parameters
from laplace import Laplace
import torch

class LA_Wrapper():
    def __init__(self, model: torch.nn.Module, n_samples: int = 100, 
                 method = "last_layer", hessian_structure = "full"):

        super().__init__()
        self.model = model
        self.n_samples = n_samples
        self.method = method
        self.hessian_structure = hessian_structure
        self.model.eval()

    def fit(self,train_loader):
        self.la = Laplace(self.model, "regression",
             subset_of_weights=self.method,
             hessian_structure=self.hessian_structure)
        self.la.fit(train_loader)

    def optimize(self):
        raise NotImplementedError    
    
    def parameter_samples(self):
        return self.la.sample()
        
    def predictive_samples(self, x):
        prediction = list()
        feats = None
        for sample in self.la.sample(self.n_samples):
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
                #f-shape: (B, d1, ..., dn, C)
                f = torch.movedim(f, -1, 1)
            prediction.append(f.detach() if not self.la.enable_backprop else f)

        vector_to_parameters(self.la.mean, self.la.model.last_layer.parameters())
        prediction = torch.stack(prediction, dim = -1)
        return prediction