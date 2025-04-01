# Data
This directory contains the scripts for generating or downloading the data required for the different experiments.

## Code


| File | Description |
| ---- | ----------- | 
| `download/darcy_flow.py` | Downloads the specified Darcy-Flow dataset from the PDEBench repository. |
| `download/era5.py` | Downloads the specified ERA5 data from the WeatherBench2 service. |
| `generation/ks_generation.py` | Script for generating data for the KS-equation for the specified parameters. |
| `generation/ks_config.yaml` | Config file including the parameters for the generation of the KS-equation. |
| `generation/sswe_generation.py` | Script for generating data for the SSW-equation for the specified parameters. |
| `generation/sswe_config.yaml` | Config file including the parameters for the generation of the SSW-equation. |
| `dataset.py` | File includes the datasets and dataloaders for the different experiments. |

## WeatherBench2
In order to download the ERA5 data from the WeatherBench2 repository, a new environment is required. For installation, follow the instructions in https://weatherbench2.readthedocs.io/en/latest/. Afterwards, the download script can be used as intended.