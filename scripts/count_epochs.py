import os

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

if __name__=='__main__':
    filenames = get_log_filenames_directory('results/optimal_hp_multiple_seeds')
    for filename in filenames:
        print(filename)
        if filename == 'results/optimal_hp_multiple_seeds/ks/uno/20240919_134024_ks_uno_dropout/experiment.log':
            print('This data is corrupted (the log file seems to have broken down.)')   
        
        with open(filename, 'r') as file:
            log = file.readlines()
        number_training_runs = get_number_training_runs(log)
        number_epochs = get_number_epochs(log)
        print(f'Average number of epochs: {number_epochs / number_training_runs:.2f}')