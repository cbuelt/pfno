from time import time
from itertools import product

import torch

from torch.distributed import destroy_process_group
import torch.distributed as dist
import numpy as np

import utils.losses as losses
from models import FNO, PNO_Wrapper, UNO, PFNO, PUNO, SFNO, PSFNO
from models import LA_Wrapper
import random


def autoregressive_step(uncertainty_quantification, model, a): 
    if uncertainty_quantification == 'laplace':
        out = model.model(a)
    elif uncertainty_quantification == "dropout":
        model.eval()
        out = model(a)
    elif uncertainty_quantification == "scoring-rule-reparam":
        model.eval()
        out = model(a).mean(axis = -1)
    else:
        model.train()
        out = model(a).mean(axis = -1)
    return out

def log_and_save_evaluation(value, key, results_dict, logging):
    value = np.round(value, decimals=5)
    logging.info(f"{key}: {value}")
    if not key in results_dict.keys():
        results_dict[key] = []
    results_dict[key].append(value)


def checkpoint(model, filename):
    torch.save(model.state_dict(), filename)
    if isinstance(model, LA_Wrapper):
        model.save_state_dict(filename[:-3] + "_la_state.pt")


def resume(model, filename):
    model.load_state_dict(torch.load(filename))
    if isinstance(model, LA_Wrapper):
        model.load_state_dict(torch.load(filename[:-3] + "_la_state.pt"))


def get_criterion(training_parameters, domain_range, d, device):
    """ Get the criterion for the training.

    Args:
        training_parameters (_type_): Dictionary of training parameters
        domain_range (_type_): Either list of domain range for LP based loss or nlon and weights for spherical loss
        d (_type_): Dimension of data

    Returns:
        _type_: Loss criterion
    """
    if training_parameters["model"] == "SFNO":
        # If Spherical model, nlon and weights are passed
        nlon, train_weights, _, _ = domain_range
        if training_parameters["uncertainty_quantification"].startswith("scoring-rule"):
            criterion = losses.EnergyScore(type="spherical", nlon = nlon, weights = train_weights.to(device))
        else:
            criterion = losses.SphericalL2Loss(nlon = nlon, weights = train_weights.to(device))
    else:
        if training_parameters["uncertainty_quantification"].startswith("scoring-rule"):
            criterion = losses.EnergyScore(d=d, p=2, type="lp", L=domain_range)
        else:
            criterion = losses.LpLoss(d=d, p=2, L=domain_range)
    return criterion


def initialize_weights(model, init):
    for name, param in model.named_parameters():
        if "weight" in name and param.data.dim() == 2:
            if init == "xavier":
                nn.init.xavier_uniform(param)
            elif init == "he":
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")
            else:
                raise NotImplementedError(
                    f'Please choose init as "default", "xavier" or "he". You chose: {init}.'
                )


def setup_model(training_parameters, device, in_channels, out_channels):
    if training_parameters["model"] == "FNO":
        if training_parameters["uncertainty_quantification"] == "scoring-rule-reparam":
            hidden_model = PFNO(
                n_modes=training_parameters["n_modes"],
                hidden_channels=training_parameters["hidden_channels"],
                in_channels=in_channels,
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
                fourier_dropout=training_parameters["fourier_dropout"],
                out_channels=out_channels,
                projection_channels=training_parameters["projection_channels"],
                n_samples=training_parameters["n_samples"],
            )
        else:
            hidden_model = FNO(
                n_modes=training_parameters["n_modes"],
                hidden_channels=training_parameters["hidden_channels"],
                in_channels=in_channels,
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
                fourier_dropout=training_parameters["fourier_dropout"],
                out_channels=out_channels,
                projection_channels=training_parameters["projection_channels"],
            )
    elif training_parameters["model"] == "UNO":
        if training_parameters["uncertainty_quantification"] == "scoring-rule-reparam":
            hidden_model = PUNO(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=training_parameters["hidden_channels"],
                projection_channels=training_parameters["projection_channels"],
                uno_out_channels=training_parameters["uno_out_channels"],
                uno_n_modes=training_parameters["uno_n_modes"],
                uno_scalings=training_parameters["uno_scalings"],
                fourier_dropout=training_parameters["fourier_dropout"],
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
                n_samples=training_parameters["n_samples"],
            )
        else:
            hidden_model = UNO(
                in_channels=in_channels,
                out_channels=out_channels,
                hidden_channels=training_parameters["hidden_channels"],
                projection_channels=training_parameters["projection_channels"],
                uno_out_channels=training_parameters["uno_out_channels"],
                uno_n_modes=training_parameters["uno_n_modes"],
                uno_scalings=training_parameters["uno_scalings"],
                fourier_dropout=training_parameters["fourier_dropout"],
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
            )
    elif training_parameters["model"] == "SFNO":
        if training_parameters["uncertainty_quantification"] == "scoring-rule-reparam":
            hidden_model = PSFNO(
                n_modes=training_parameters["n_modes"],
                hidden_channels=training_parameters["hidden_channels"],
                in_channels=in_channels,
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
                fourier_dropout=training_parameters["fourier_dropout"],
                out_channels=out_channels,
                projection_channels=training_parameters["projection_channels"],
                n_samples=training_parameters["n_samples"],
            )
        else:
            hidden_model = SFNO(
                n_modes=training_parameters["n_modes"],
                hidden_channels=training_parameters["hidden_channels"],
                in_channels=in_channels,
                dropout=training_parameters["dropout"],
                lifting_channels=training_parameters["lifting_channels"],
                fourier_dropout=training_parameters["fourier_dropout"],
                out_channels=out_channels,
                projection_channels=training_parameters["projection_channels"],
            )

    if training_parameters["uncertainty_quantification"] == "scoring-rule-dropout":
        return PNO_Wrapper(hidden_model, n_samples=training_parameters["n_samples"]).to(
            device
        )
    else:
        return hidden_model.to(device)


def scheduler_step(scheduler, optimizer, epoch):
    before_lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    after_lr = optimizer.param_groups[0]["lr"]
    if before_lr != after_lr:
        print(
            "[time: %d] Epoch %d: SGD lr %.8f -> %.8f"
            % (time(), epoch, before_lr, after_lr)
        )


def predict_data_loader(net, dataloader, num_samples_min, device):
    prediction_list = []
    target_list = []
    with torch.no_grad():
        for input, target in dataloader:
            input = input.to(device)
            target = target.to(device)
            prediction_list.append(net(input))
            target_list.append(target)
    return torch.cat(prediction_list, dim=0), torch.cat(target_list, dim=0)


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def ddp_setup(rank: int, world_size: int):
    """
    Args:
        rank: Unique identifier of each process
        world_size: Total number of processes
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)


# Gets all combinations of hyperparameters given in hp_dict and return a list,
# leaves the list of the keys of except_keys untouched though
def get_hyperparameters_combination(hp_dict, except_keys=[]):
    except_dict = {}
    for except_key in except_keys:
        if except_key in hp_dict.keys():
            except_dict[except_key] = hp_dict.pop(except_key)
    iterables = [
        value if isinstance(value, list) else [value] for value in hp_dict.values()
    ]
    all_combinations = list(product(*iterables))
    # Create a list of dictionaries for each combination
    combination_dicts = [
        {
            **{param: value for param, value in zip(hp_dict.keys(), combination)},
            **except_dict,
        }
        for combination in all_combinations
    ]
    return combination_dicts


def subsample(data, max_size):
    if max_size < len(data):
        return random.sample(data, max_size)
    else:
        return data
