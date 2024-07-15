import numpy as np

import torch
import gc
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

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

from data.datasets import DarcyFlowDataset
from train import trainer


print(os.getcwd())
sys.path[0] = os.getcwd()
import utils

torch.autograd.set_detect_anomaly(False)

msg = 'Start main'

# initialize parser
parser = argparse.ArgumentParser(description=msg)
default_config = 'debug.ini'

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

def append_results_dict(results_dict, data_parameters, training_parameters, t_training,
                        t_data_creation):
    for key in data_parameters.keys():
        results_dict[key].append(data_parameters[key])
    for key in training_parameters.keys():
        results_dict[key].append(training_parameters[key])
    results_dict['t_training'].append(t_training)
    results_dict['t_data_creation'].append(t_data_creation)
    
    
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

    logging.info(f'############### Starting experiment with config file {config_name} ###############')

    training_parameters_dict = dict(config.items("TRAININGPARAMETERS"))
    training_parameters_dict = {key: ast.literal_eval(training_parameters_dict[key]) for key in
                                training_parameters_dict.keys()}
    training_parameters_dict = utils.get_hyperparameters_combination(training_parameters_dict) # except_keys for keys that are coming as a list for each training process

    data_parameters_dict = dict(config.items("DATAPARAMETERS"))
    data_parameters_dict = {key: ast.literal_eval(data_parameters_dict[key]) for key in
                            data_parameters_dict.keys()}
    data_parameters_dict = utils.get_hyperparameters_combination(data_parameters_dict) # except_keys for keys that are coming as a list for each training process
    
    objective = config['META']['objective']

    entry_names = ['t_training', 'mse_test'] 
    
    results_dict = construct_result_dict(entry_names, data_parameters_dict, training_parameters_dict)

    for i, data_parameters in enumerate(data_parameters_dict):
        logging.info(f"###{i + 1} out of {len(data_parameters_dict)} data set parameter combinations ###")
        print(f'Data parameters: {data_parameters}')
        logging.info(f'Data parameters: {data_parameters}')
        
        data_dir = f"data/{data_parameters['dataset_name']}/processed/"
        if data_parameters['dataset_name'] == 'DarcyFlow':
            train_data = DarcyFlowDataset(data_dir, test = False, downscaling_factor=2)
            test_data = DarcyFlowDataset(data_dir, test = True)
        
        train_data, val_data = train_test_split(train_data, test_size=0.20, random_state=42)
        

        for i, training_parameters in enumerate(training_parameters_dict):
            logging.info(f"###{i + 1} out of {len(training_parameters_dict)} training parameter combinations ###")
            print(f'Training parameters: {training_parameters}')
            logging.info(f'Training parameters: {training_parameters}')
            
            filename = f"{data_parameters['dataset_name']}_{training_parameters['model']}_dropout_{training_parameters['dropout']}"
            
            train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=True)
            test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)
                        
            t_0 = time()
            d_time_train = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            if not training_parameters['distributed_training']:
                net = trainer(0, train_loader, val_loader, directory=directory, training_parameters=training_parameters, logging=logging,
                              filename=filename, d_time=d_time_train)
            else:
                world_size = torch.cuda.device_count()
                mp.spawn(trainer, args=(input_training, target_training, target_validation,
                            input_validation, training_parameters, data_parameters,
                            data_parameters['num_samples_min'], training_parameters['lr_schedule'], objective, 
                            directory, d_time_train, world_size), nprocs=world_size)

                net = torch.load(os.path.join(directory,
                                f'Datetime_{d_time_train}_parameters_{filename}.pt'), map_location=device)
                        
            t_1 = time()
            t_training = np.round(t_1 - t_0, 3)
            logging.info(f'Training the model took {t_training}s.')
            t_0 = time()
            torch.cuda.empty_cache()
            t_1 = time()
            logging.info(f'Emptying the cuda cache took {np.round(t_1 - t_0, 3)}s.')
            evaluate(net=net, input_training=input_training, target_training=target_training, target_test=target_test, input_test=input_test,
                     results_dict=results_dict, batch_size=training_parameters['batch_size'], objective=objective, 
                     num_samples_min=data_parameters['num_samples_min'], function_names_str=data_parameters['function_names_str'],
                     one_hot=training_parameters['one_hot'], data_parameters=data_parameters)
            append_results_dict(results_dict, data_parameters, training_parameters, t_training,
                                t_data_creation)
            results_pd = pd.DataFrame(results_dict)
            results_pd.T.to_csv(os.path.join(directory, 'test.csv'))

            del net
            torch.cuda.empty_cache()
            gc.collect()
