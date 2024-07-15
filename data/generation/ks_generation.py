import pde
import numpy as np
import yaml
import multiprocessing as mp
import os
from tqdm import tqdm
import xarray as xr

def get_sim(grid, n_steps, t_range = 150, dt = 0.001):
    eq = pde.PDE({"u": "-u * d_dx(u) / 2 - laplace(u + laplace(u))"})
    state = pde.ScalarField.random_uniform(grid, vmin = -1, vmax = 1)  # generate initial condition
    # solve the system
    storage = pde.MemoryStorage()
    result = eq.solve(
    state,
    t_range=t_range,
    dt=dt,
    adaptive=True,
    tracker=[storage.tracker(n_steps)],
    )
    res = np.array(storage.data)
    return res

def load_config(path = "data/generation/ks_config.yaml"):
    with open(path, 'r') as f:
        config = yaml.load(f, yaml.FullLoader)
    return config

def simulate(config):
    # Check for number of cores
    assert(config["sim"]["n_cores"] <= mp.cpu_count())
    n_cores = config["sim"]["n_cores"]
    #Check for and create data directory
    if not os.path.exists("data/KS/raw"):
        os.makedirs("data/KS/raw")

    # Create grid
    grid = pde.CartesianGrid([(0, config["pde"]["L"])], [config["pde"]["res"]], periodic=True)
    # Number of subiterations
    n_sub = int(config["sim"]["num_samples"]/config["sim"]["subfile_size"])
    # Number of timesteps to record
    n_steps = np.arange(0, config["pde"]["t_max"], config["pde"]["dt"])

    # Iterate over subiterations
    for i in tqdm(range(0,n_sub)):
        pool = mp.Pool(n_cores)
        results = []        
        result_objects = [pool.apply_async(get_sim, args = (grid,  n_steps, config["pde"]["t_max"])) for i in range(config["sim"]["subfile_size"])]
        results = [r.get() for r in result_objects]
        pool.close()
        pool.join()
        result_array = np.array(results)
        np.save(f"data/KS/raw/ks_data_{i+1}.npy", result_array)

def aggregate(config):
    #Check for and create data directory
    if not os.path.exists("data/KS/processed"):
        os.makedirs("data/KS/processed")

    full_data = np.concatenate([np.load(f"data/KS/raw/{name}") for name in os.listdir("data/KS/raw/")], axis = 0)
    t = np.arange(0, config["pde"]["t_max"], config["pde"]["dt"])
    x = np.linspace(0, config["pde"]["L"], config["pde"]["res"])
    data_array = xr.DataArray(full_data, dims = ["samples", "t", "x"], coords = {"t": t, "x": x})
    ds = xr.Dataset({"u": data_array})
    # Train test split
    n_samples = config["sim"]["num_samples"]
    indices = np.random.permutation(n_samples)
    train_data = ds.isel(samples=indices[: int(n_samples * config["sim"]["train_split"])])
    test_data = ds.isel(samples=indices[int(n_samples * config["sim"]["train_split"]) :])
    train_data.to_netcdf("data/KS/processed/ks_train.nc")
    test_data.to_netcdf("data/KS/processed/ks_test.nc")

def main(config, sim = True):
    # Simulate data
    if sim:
        simulate(config)
    # Aggregate data
    aggregate(config)
 

if __name__== "__main__":
    config = load_config()
    main(config)