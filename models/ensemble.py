import torch
from torch import nn
from utils.train_utils import setup_model, resume

class Ensemble(nn.Module):
    def __init__(self, checkpoints, training_parameters, in_channels, out_channels, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.training_parameters = training_parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.device = device
        
        self.models = []
        for checkpoint in checkpoints:
            self.add_model(checkpoint)
        
    def add_model(self, checkpoint):
        model = setup_model(self.training_parameters, self.device, self.in_channels, self.out_channels)
        resume(model, checkpoint)
        self.models.append(model)
    
    def forward(self, x):
        y = []
        for model in self.models:
            y.append(model(x).unsqueeze(-1))
            
        return torch.cat(y, dim=-1)