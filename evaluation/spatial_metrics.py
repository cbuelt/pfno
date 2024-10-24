import torch
import sys
import os
import time
sys.path.append(os.getcwd())
import numpy as np
from models.fno import FNO
from models.pfno import PNO_Wrapper, PFNO
from models.laplace import LA_Wrapper
from data.datasets import ERA5Dataset
from evaluate import generate_samples
from utils.losses import CRPS, Coverage

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()



def get_model(name, checkpoint_path):
    sr_dropout_path = checkpoint_path + "20240925_115247_era5_fno_sr_dropout/Datetime_20240925_115259_Loss_era5_FNO_scoring-rule-dropout_dropout_0.2.pt"
    laplace_path = checkpoint_path + "20241008_125404_era5_fno_laplace/Datetime_20241008_125429_Loss_era5_FNO_laplace_dropout_0.01.pt"
    dropout_path = checkpoint_path + "20240922_105401_era5_fno_dropout/Datetime_20240922_105416_Loss_era5_FNO_dropout_dropout_0.05.pt"
    sr_reparam_path = checkpoint_path + "20240920_094424_era5_fno_sr_reparam/Datetime_20240920_094438_Loss_era5_FNO_scoring-rule-reparam_dropout_0.1.pt"
    
    if name == "dropout":
        model = FNO(n_modes=(10,12, 12), hidden_channels=20, in_channels = 4, dropout=0.05, lifting_channels = 128, projection_channels=128).to(device)
        dropout_cp = torch.load(dropout_path, map_location=torch.device(device))
        model.load_state_dict(dropout_cp)
    elif name == "laplace":
        la_model = FNO(n_modes=(10,12, 12), hidden_channels=20, in_channels = 4, dropout=0.01, lifting_channels = 128, projection_channels=128).to(device)
        model = LA_Wrapper(la_model)
        model.load_state_dict(torch.load(laplace_path))
        model.load_la_state_dict(torch.load(laplace_path[:-3] + "_la_state.pt"))
    elif name == "scoring-rule-dropout":
        sr_model = FNO(n_modes=(10,12, 12), hidden_channels=20, in_channels = 4, dropout=0.2, lifting_channels = 128, projection_channels=128).to(device)
        sr_dropout_cp = torch.load(sr_dropout_path, map_location=torch.device(device))
        model = PNO_Wrapper(sr_model, n_samples = n_samples)
        model.load_state_dict(sr_dropout_cp)
    elif name == "scoring-rule-reparam":
        model = PFNO(n_modes=(10,12, 12), hidden_channels=20, in_channels = 4, dropout=0.1, lifting_channels = 128, projection_channels=128, n_samples = n_samples).to(device)
        pfno_cp = torch.load(sr_reparam_path, map_location=torch.device(device))
        model.load_state_dict(pfno_cp)
    return model


class Spatial_CRPS(object):
    """
    A class for calculating the Continuous Ranked Probability Score (CRPS) between two tensors.
    """
    def __init__(self):
        super().__init__()

    def calculate_score(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates the CRPS for two tensors, without aggregating.

        Args:
            x (torch.Tensor): Model x (Batch size, ..., n_samples)
            y (torch.Tensor): Target (Batch size, ..., 1)

        Returns:
            torch.Tensor: CRPS
        """
        n_samples = x.size()[-1]

        # Add additional dimension if necessary
        if len(x.size()) != len(y.size()):
            y = torch.unsqueeze(y, dim=-1)

        # Calculate terms
        es_12 = torch.abs(x-y).mean(axis = -1)

        diff = torch.unsqueeze(x, dim=-2) - torch.unsqueeze(x, dim=-1)
        es_22 = torch.abs(diff).sum(axis = (-1,-2))

        score = es_12 - es_22 / (2 * n_samples * (n_samples - 1))
        return score

    def __call__(self, y_pred, y, **kwargs):
        return self.calculate_score(y_pred, y, **kwargs)


def get_spatial_metrics(out, u, alpha = 0.05):
    """
    Calculate spatial metrics that are averaged across the samples

    Args:
        u (_type_): Ground truth
        out (_type_): Model output
        alpha (float, optional): Significance level. Defaults to 0.05.

    Returns:
        _type_: _description_
    """
    crps_loss = Spatial_CRPS()
    coverage_loss = Coverage(alpha, reduce_dims = False)

    coverage = coverage_loss(out, u)
    coverage = coverage.sum(axis = 0).squeeze().cpu().numpy()
    crps = crps_loss(out, u)
    crps = crps.sum(axis = 0).squeeze().cpu().numpy()
    return crps, coverage


if __name__ == "__main__":
    checkpoint_path = "/home/groups/ai/scholl/pfno/weights/era5/fno/"
    output_path = "evaluation/spatial_metrics/"

    # Test data
    batch_size = 12
    n_samples = 100 # Samples to create from predictive distributions
    alpha = 0.05 # Parameter for confidence interval
    data_dir = "data/era5/"
    test_data = ERA5Dataset(data_dir, var = "test")
    lat,lon,t = test_data.get_coordinates(normalize = False)
    L = test_data.get_domain_range()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)
    data_mean = torch.tensor(test_data.mean, device = device)
    data_std = torch.tensor(test_data.std, device = device)
    n_test = len(test_data)

    # Lead time
    t = 3 # 24h

    uq_methods = ["dropout", "scoring-rule-dropout", "scoring-rule-reparam"]

    # Iterate across uq_methods
    for uq_method in uq_methods:
        torch.cuda.empty_cache()
        # Get model
        model = get_model(uq_method, checkpoint_path)

        # Create results arrays
        coverage_results = np.zeros((len(lat), len(lon)))
        crps_results = np.zeros((len(lat), len(lon)))

        with torch.no_grad():
            for sample in test_loader:
                a, u = sample
                a = a.to(device)
                u = u.to(device)
                batch_size = a.shape[0]
                out = generate_samples(
                    uq_method,
                    model,
                    a,
                    u,
                    n_samples,
                )
                # Renormalize and extract time
                out = out[:,0,t] * data_std + data_mean -273.15 # Transform to Celsius
                u = u[:,0,t] * data_std + data_mean -273.15 # Transform to Celsius

                crps, coverage = get_spatial_metrics(out, u)
                coverage_results += coverage
                crps_results += crps


        # Save results
        coverage_results = coverage_results / n_test
        crps_results = crps_results / n_test
        np.save(output_path + f"era5_{uq_method}_{(t+1)*6}h_coverage.npy", coverage_results)
        np.save(output_path + f"era5_{uq_method}_{(t+1)*6}h_crps.npy", crps_results)


