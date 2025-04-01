import os
import configparser
import ast
from models import LA_Wrapper
from utils import train_utils
from torch.utils.data import DataLoader, random_split
from data.datasets import DarcyFlowDataset, SWEDataset, KSDataset, ERA5Dataset, SSWEDataset
import torch
from copy import copy 
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')

def get_weigth_filenames_directory(directory):
    filenames = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            filenames.append(os.path.join(path, name))
    
    filenames = [filename for filename in filenames if 'laplace' in filename]
    filenames = [filename for filename in filenames if filename.endswith('.pt')]
    filenames = [filename for filename in filenames if (not filename[:-3].endswith('la_state'))]
    return filenames

def get_ini_file(filename):
    directory = os.path.dirname(filename)
    ini_file = [file for file in os.listdir(directory) if file.endswith('.ini')][0]
    return os.path.join(directory, ini_file)

def get_dataset(data_parameters):
    data_dir = f"data/{data_parameters['dataset_name']}/processed/"
    if data_parameters['dataset_name'] == 'DarcyFlow':
        train_data = DarcyFlowDataset(data_dir, test = False, downscaling_factor=int(data_parameters['downscaling_factor']))
        train_data_full_res = DarcyFlowDataset(data_dir, test = False)
        test_data = DarcyFlowDataset(data_dir, test = True)
    elif data_parameters['dataset_name'] == 'SWE':
        downscaling_factor = int(data_parameters['downscaling_factor'])
        temporal_downscaling_factor = int(data_parameters['temporal_downscaling'])
        pred_horizon = data_parameters['pred_horizon']
        t_start = data_parameters['t_start']
        init_steps = data_parameters['init_steps']
        ood = data_parameters["ood"]
        
        assert 100 > temporal_downscaling_factor * (pred_horizon + t_start + init_steps)
        
        train_data = SWEDataset(data_dir, test = False, downscaling_factor=downscaling_factor, mode = "autoregressive",
                    pred_horizon=pred_horizon, t_start=t_start, init_steps=init_steps,
                    temporal_downscaling_factor=temporal_downscaling_factor, ood = ood)
        test_data = SWEDataset(data_dir, test = True, mode = "autoregressive",
                    pred_horizon=pred_horizon, t_start=t_start, init_steps=init_steps,
                    temporal_downscaling_factor=temporal_downscaling_factor, ood = ood)
        
    elif data_parameters["dataset_name"] == "KS":
        downscaling_factor = int(data_parameters['downscaling_factor'])
        temporal_downscaling_factor = int(data_parameters['temporal_downscaling'])
        pred_horizon = data_parameters['pred_horizon']
        t_start = data_parameters['t_start']
        init_steps = data_parameters['init_steps']

        assert 300 > temporal_downscaling_factor * (pred_horizon + t_start + init_steps)

        train_data = KSDataset(data_dir, test = False, downscaling_factor=downscaling_factor, mode = "autoregressive",
                    pred_horizon=pred_horizon, t_start=t_start, init_steps=init_steps,
                    temporal_downscaling_factor=temporal_downscaling_factor)
        test_data = KSDataset(data_dir, test = True, mode = "autoregressive",
                    pred_horizon=pred_horizon, t_start=t_start, init_steps=init_steps,
                    temporal_downscaling_factor=temporal_downscaling_factor)
        
    elif data_parameters["dataset_name"] == "era5":
        data_dir = f"data/{data_parameters['dataset_name']}/"
        pred_horizon = data_parameters['pred_horizon']
        init_steps = data_parameters['init_steps']
        train_data = ERA5Dataset(data_dir, var = "train", init_steps = init_steps, prediction_steps = pred_horizon)
        val_data = ERA5Dataset(data_dir, var = "val", init_steps = init_steps, prediction_steps = pred_horizon)
        test_data = ERA5Dataset(data_dir, var = "test", init_steps = init_steps, prediction_steps = pred_horizon)
    
    elif data_parameters["dataset_name"] == "SSWE":
        data_dir = f"data/{data_parameters['dataset_name']}/processed/"
        pred_horizon = data_parameters['pred_horizon']
        train_data = SSWEDataset(data_dir, test = False, pred_horizon = data_parameters["train_horizon"], return_all = True)
        test_data = SSWEDataset(data_dir, test = True, pred_horizon = pred_horizon, return_all = True)
    return train_data

def load_model(model, filename):
    state_dict_old = torch.load(filename)
    state_dict = {}
    for key in state_dict_old.keys():
        if key.startswith('model.'):
            state_dict[key[6:]] = state_dict_old[key]  # remove .model
        else:
            state_dict[key] = state_dict_old[key]
    model.la.model.model.load_state_dict(state_dict)
    model.la.load_state_dict(torch.load(filename[:-3] + "_la_state.pt"))

if __name__=='__main__':
    filenames = get_weigth_filenames_directory('results/optimal_hp_multiple_seeds')
    for filename in filenames:
        print(f'Transforming: {filename}')
        ini_file = get_ini_file(filename)
        config = configparser.ConfigParser()
        config.read(ini_file)        
        training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
        training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                    training_parameters_dict.keys()}
        training_parameters = train_utils.get_hyperparameters_combination(training_parameters_dict, 
                                                                           except_keys=['uno_out_channels', 'uno_scalings', 'uno_n_modes'])[0]
            
        data_parameters_dict = dict(config.items("DATAPARAMETERS"))
        data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                                    data_parameters_dict.keys()}
        data_parameters = train_utils.get_hyperparameters_combination(data_parameters_dict)[0] # except_keys for keys that are coming as a list for each training process
    
        batch_size = training_parameters['batch_size']
        
        train_data = get_dataset(data_parameters)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        in_channels = next(iter(train_loader))[0].shape[1]
        out_channels = next(iter(train_loader))[1].shape[1]
        
        
        model = train_utils.setup_model(training_parameters, device, in_channels, out_channels)
        model = LA_Wrapper(
            model,
            n_samples=training_parameters["n_samples_uq"],
            method="last_layer",
            hessian_structure="full",
            optimize=True,
        )
        
        load_model(model, filename)
        train_utils.checkpoint(model, filename)
                
    