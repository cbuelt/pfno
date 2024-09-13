import numpy as np

import torch
import gc
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, random_split

import pandas as pd

from time import time
import os
import sys
import datetime
import pathlib
import logging
import argparse
import configparser
import ast
import shutil

from data.datasets import DarcyFlowDataset, SWEDataset, KSDataset, ERA5Dataset, SSWEDataset
from train import trainer
from utils import train_utils
from evaluate import start_evaluation

print(os.getcwd())
sys.path[0] = os.getcwd()
import utils

torch.autograd.set_detect_anomaly(False)

msg = 'Start main'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
default_config = 'debug.ini'
default_config = 'sswe/sfno.ini'

parser.add_argument('-c', '--config', help='Name of the config file:', default=default_config)
parser.add_argument('-f', '--results_folder', help='Name of the results folder (only use if you only want to evaluate the models):', default=None)

args = parser.parse_args()

config_name = args.config
config = configparser.ConfigParser()
config.read(os.path.join('config', config_name))
results_path = config['META']['results_path']
experiment_name = config['META']['experiment_name']

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(f'Using {device}.')


def construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict):
    results_dict = {**{key: [] for key in data_parameters_dict[0].keys()},
                    **{key: [] for key in training_parameters_dict[0].keys()}}
    for entry_name in entry_names:
        results_dict[entry_name] = []
    return results_dict

def append_results_dict(results_dict, data_parameters, training_parameters, t_training):
    for key in data_parameters.keys():
        results_dict[key].append(data_parameters[key])
    for key in training_parameters.keys():
        results_dict[key].append(training_parameters[key])
    results_dict['t_training'].append(t_training)
    
    
if __name__ == '__main__':
    d_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_')
    directory = os.path.join(results_path, d_time + experiment_name)
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    shutil.copy(os.path.join('config', config_name), directory)
    print(f'Created directory {directory}')

    logging.basicConfig(filename=os.path.join(directory, 'experiment.log'), level=logging.INFO)
    logging.info('Starting the logger.')
    logging.debug(f'Directory: {directory}')
    logging.debug(f'File: {__file__}')

    logging.info(f'Using {device}.')
    
    logging.info(using(''))

    logging.info(f'############### Starting experiment with config file {config_name} ###############')

    training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
    training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                training_parameters_dict.keys()}
    # except_keys for keys that are coming as a list for each training process
    training_parameters_dict = train_utils.get_hyperparameters_combination(training_parameters_dict, 
                                                                           except_keys=['uno_out_channels', 'uno_scalings', 'uno_n_modes'])
    
    data_parameters_dict = dict(config.items("DATAPARAMETERS"))
    data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                            data_parameters_dict.keys()}
    data_parameters_dict = train_utils.get_hyperparameters_combination(data_parameters_dict) # except_keys for keys that are coming as a list for each training process
    
    entry_names = ['t_training'] 
    
    results_dict = construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict)

    for i, data_parameters in enumerate(data_parameters_dict):
        logging.info(f"###{i + 1} out of {len(data_parameters_dict)} data set parameter combinations ###")
        print(f'Data parameters: {data_parameters}')
        logging.info(f'Data parameters: {data_parameters}')
        
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

        logging.info(using('After loading the datasets'))

        if data_parameters["dataset_name"] != "SSWE":
            domain_range = train_data.get_domain_range()
        else:
            # Requires Longitude and quadrature weights instead of domain range
            domain_range = (train_data.get_nlon(), train_data.get_train_weights(), test_data.get_nlon(), test_data.get_eval_weights())    

        if data_parameters['dataset_name'] == 'DarcyFlow':
            # Validation data on full resolution
            train_data, _ = random_split(train_data, lengths = [0.8,0.2], generator = torch.Generator().manual_seed(42))
            _, val_data = random_split(train_data_full_res, lengths = [0.8,0.2], generator = torch.Generator().manual_seed(42))
        elif data_parameters['dataset_name'] != 'ERA5':
            train_data, val_data = random_split(train_data, lengths = [0.8,0.2], generator = torch.Generator().manual_seed(42))

        for i, training_parameters in enumerate(training_parameters_dict):
            logging.info(f"###{i + 1} out of {len(training_parameters_dict)} training parameter combinations ###")
            print(f'Training parameters: {training_parameters}')
            logging.info(f'Training parameters: {training_parameters}')
            
            seed = training_parameters['seed']
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            filename = f"{data_parameters['dataset_name']}_{training_parameters['model']}_{training_parameters['uncertainty_quantification']}_" + \
                       f"dropout_{training_parameters['dropout']}"
            
            batch_size = training_parameters['batch_size']
            
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
            
            # Additional loader for autoregressive laplace
            if training_parameters["uncertainty_quantification"] == "laplace" and data_parameters["dataset_name"] == "SSWE":
                laplace_train = SSWEDataset(data_dir, test = False, pred_horizon = 1, return_all = False)
                laplace_train_loader = DataLoader(laplace_train, batch_size=batch_size, shuffle=True)
            else:
                laplace_train_loader = None
                        
            logging.info(using('After creating the dataloaders'))

            t_0 = time()
            d_time_train = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            model, filename = trainer(train_loader, val_loader, directory=directory, training_parameters=training_parameters,
                                        data_parameters = data_parameters,logging=logging, filename_ending=filename, d_time=d_time_train,
                                        domain_range=domain_range, results_dict=results_dict)
                        
            t_1 = time()
            t_training = np.round(t_1 - t_0, 3)
            logging.info(f'Training the model took {t_training}s.')
            t_0 = time()
            torch.cuda.empty_cache()
            t_1 = time()
            logging.info(f'Emptying the cuda cache took {np.round(t_1 - t_0, 3)}s.')
            
            eval_batch_size = max(batch_size // 4, 1)
            
            train_loader = DataLoader(train_data, batch_size=eval_batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=eval_batch_size, shuffle=True)
            test_loader = DataLoader(test_data, batch_size=eval_batch_size, shuffle=True)
            
            start_evaluation(model, training_parameters, data_parameters, train_loader, val_loader, 
                            test_loader, results_dict, device, domain_range, logging, filename, laplace_train_loader=laplace_train_loader)
            append_results_dict(results_dict, data_parameters, training_parameters, t_training)
            results_pd = pd.DataFrame(results_dict)
            results_pd.T.to_csv(os.path.join(directory, 'test.csv'))
            
            logging.info(using('After validation'))
            
            del model
            torch.cuda.empty_cache()
            gc.collect()
