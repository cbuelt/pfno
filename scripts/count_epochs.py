import os

def get_log_filenames_directory(directory):
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

if __name__=='__main__':
    filenames = get_log_filenames_directory('results/optimal_hp_multiple_seeds')
    for filename in filenames:
        with open(filename, 'br') as file:
            log = file.readlines()
        