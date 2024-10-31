import os
import ast 
import configparser

import pandas as pd

import sys
# print(f'Current working directory: {os.getcwd()}')
# print(f'System path: {sys.path}')
# sys.path[0] = os.getcwd()
# print(f'System path: {sys.path}')
from utils import train_utils


def get_log_filenames_directory(directory):
    filenames = []
    for path, subdirs, files in os.walk(directory):
        for name in files:
            filenames.append(os.path.join(path, name))
    
    filenames = [filename for filename in filenames if not ('fno_10' in filename)]  # Ignore fno_10 folder, since they were just for evaluation
    filenames = [filename for filename in filenames if filename.endswith('.log')]
    return filenames

def get_number_training_runs(log):
    start = 'INFO:root:Training starts now.\n'
    return len([line for line in log if line==start])

def get_number_epochs(log):
    string = 'INFO:root:['
    return len([line for line in log if line.startswith(string)])

def get_ini_file(filename):
    directory = os.path.dirname(filename)
    ini_file = [file for file in os.listdir(directory) if file.endswith('.ini')][0]
    return os.path.join(directory, ini_file)

def get_time_per_epoch(path):
    result_list = []
    directories = [os.path.join(path, directory) for directory in os.listdir(path)]
    for directory in directories:
        result_list.append(pd.read_csv(os.path.join(directory, 'test.csv'), index_col=0).T)
    return pd.concat(result_list)

if __name__=='__main__':
    standard_experiments = False
    if standard_experiments:
        results = get_time_per_epoch('results/timing/new')
    else:
        results = get_time_per_epoch('results/influence_of_n_samples/timing')
    
    results = results[['dataset_name', 'uncertainty_quantification', 'model', 't_training', 'n_samples', 'n_epochs']]
    
    if standard_experiments:
        filenames = get_log_filenames_directory('results/optimal_hp_multiple_seeds')
    else:
        filenames = get_log_filenames_directory('results/influence_of_n_samples/performance')
    
    for filename in filenames:
        # print(filename)
        if filename == 'results/optimal_hp_multiple_seeds/ks/uno/20240919_134024_ks_uno_dropout/experiment.log':
            print('This data is corrupted (the log file seems to have broken down.)')   
        
        with open(filename, 'r') as file:
            log = file.readlines()
        number_training_runs = get_number_training_runs(log)
        number_epochs = get_number_epochs(log)
        
        # Get the current model, uncertainty_quantification and dataset
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

        model = training_parameters['model']
        n_samples = str(training_parameters['n_samples'])
        uncertainty_quantification = training_parameters['uncertainty_quantification']
        dataset_name = data_parameters['dataset_name']
        try:
            entry = results[(results['model'] == model) & (results['uncertainty_quantification'] == uncertainty_quantification) & 
                            (results['dataset_name'] == dataset_name) & (results['n_samples'] == n_samples)]
            if dataset_name == 'SSWE':
                train_horizon = data_parameters['train_horizon']
                entry = entry[entry['train_horizon']==str(train_horizon)]
            time_per_epoch = float(entry['t_training'].iloc[0]) / int(entry['n_epochs'])
        except Exception as e:
            print(f'Caught exception {e}')
            continue
        
        if standard_experiments:
            if dataset_name != 'SSWE':
                # continue
                print(f'{model} + {uncertainty_quantification} on {dataset_name} took on average:'
                    f'{number_epochs / number_training_runs:.2f} epochs, {time_per_epoch}s per epoch, and {number_epochs * time_per_epoch / number_training_runs:.0f}s in total.')
            else:
                print(f'{model} + {uncertainty_quantification} on {dataset_name} (train_horizon={train_horizon}) took on average:'
                    f'{number_epochs / number_training_runs:.2f} epochs, {time_per_epoch}s per epoch, and {number_epochs * time_per_epoch / number_training_runs:.0f}s in total.')
        else:
                print(f'{model} + {uncertainty_quantification} using {n_samples} samples took on average:'
                    f'{number_epochs / number_training_runs:.2f} epochs, {time_per_epoch}s per epoch, and {number_epochs * time_per_epoch / number_training_runs:.0f}s in total.')

        
        