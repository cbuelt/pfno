conda create -n pfno python=3.12
conda activate pfno
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn matplotlib 
conda install wandb -c conda-forge
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda install h5py
pip install neuraloperator
pip install laplace-torch
pip install torch-harmonics
conda install -c conda-forge py-pde