# PNO

This repository contains the code accompanying the paper
> Bülte, Scholl, Kutyniok
     
    
The code includes methods for reproducing the results, as well as general implementation of our parameter and functional estimation approaches. For an overview of the code and the methodology, see below.

# Abstract
- 
 
# Repository structure

| Folder | Description |
| ---- | ----------- | 
| `config` | Contains the config files for the different experiments. |
| `data` | Contains the scripts for downloading and generating data. |
| `evaluation` | Contains code for generating the plots. |
| `models` | Includes the different models and methods. |
| `scripts` | Contains utility scripts. |
| `utils` | Contains utility functions. |
| `main.py` | Main file to rund and evaluate experiments. |

# Setup

1. Clone project
1. In your shell run ```install.sh```. This will create the environment and install all relevant packages.

# Data

Scripts for downloading or generating the data automatically can be found in the data directory.

# Reproducing results

You can reproduce experiments, by adjusting the config files for each experiment and running the script ```main.py```. Furthermore, you can set the ```only_validate``` flag in the config to evaluate an experiment using an already trained model. As the evaluation process is bound to a specific seed, this will reproduce the exact results from the main paper.


# Methodology
-


## Some results
- 

# License
This work is published under the MIT license.









