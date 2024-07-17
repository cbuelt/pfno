conda env create -n pfno python=3.12
conda activate pfno
git clone git@github.com:pdebench/PDEBench.git
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install numpy pandas scikit-learn matplotlib 
conda install wandb -c conda-forge
conda install -c conda-forge xarray dask netCDF4 bottleneck
conda install h5py
pip install neuraloperator
pip install git+https://github.com/aleximmer/laplace.git@0.2
pip install torch-harmonics