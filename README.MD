# Probabilistic neural operators for functional uncertainty quantification

This repository contains the code accompanying the paper
> Bülte, C., Scholl, P., Kutyniok, G. (2025). Probabilistic neural operators for functional uncertainty quantification.     
    
The code includes methods for reproducing the results, as well as general implementation of our parameter and functional estimation approaches. For an overview of the code and the methodology, see below.

# Abstract
Neural operators aim to approximate the solution operator of a system of differential equations purely from data. They have shown immense success in modeling complex dynamical systems across various domains. However, the occurrence of uncertainties inherent in both model and data has so far rarely been taken into account\textemdash{}a critical limitation in complex, chaotic systems such as weather forecasting. In this paper, we introduce the probabilistic neural operator (PNO), a framework for learning probability distributions over the output function space of neural operators. PNO extends neural operators with generative modeling based on strictly proper scoring rules, integrating uncertainty information directly into the training process. We provide a theoretical justification for the approach and demonstrate improved performance in quantifying uncertainty across different domains and with respect to different baselines. Furthermore, PNO requires minimal adjustment to existing architectures, shows improved performance for most probabilistic prediction tasks, and leads to well-calibrated predictive distributions and adequate uncertainty representations even for long dynamical trajectories. Implementing our approach into large-scale models for physical applications can lead to improvements in corresponding uncertainty quantification and extreme event identification, ultimately leading to a deeper understanding of the prediction of such surrogate models.
 
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

# Running the code

You can run a specific model by calling the script ```main.py```. You can use the existing config files or adjust them to your needs and then call ```main.py --config "config.ini"```. You can adjust the hyperparameters, training parameters, models and datasets. Calling the script will train the model and create an output folder with the performance metrics and other logging information.

## Reproduce results

You can reproduce experiments, by using the existing config files. Furthermore, you can set the ```only_validate``` flag in the config to evaluate an experiment using an already trained model. As the evaluation process is bound to a specific seed, this will reproduce the exact results from the main paper.


# License
This work is published under the MIT license.









