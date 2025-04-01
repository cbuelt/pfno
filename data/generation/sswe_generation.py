# Implementation of the SSWE data generation using the torch-harmonics library.

# coding=utf-8

# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

import torch
from torch_harmonics.examples.shallow_water_equations import ShallowWaterSolver
import os
from math import ceil, sqrt
from netCDF4 import Dataset
import numpy as np
import yaml


class SSWESolver(ShallowWaterSolver):
    def __init__(self, nlat, nlon, dt, lmax=None, mmax=None, grid="legendre-gauss"):
        radius = 6.37122e6
        omega = 7.292e-5
        gravity = 9.80616
        havg = 10.0e3
        hamp = 120.0
        super().__init__(
            nlat, nlon, dt, lmax, mmax, grid, radius, omega, gravity, havg, hamp
        )

    def timestep(
        self, uspec: torch.Tensor, nsteps: int, burn_in_steps: int, save_steps: int
    ) -> torch.Tensor:
        """
        Integrate the solution using Adams-Bashforth / forward Euler for nsteps steps.
        """

        dudtspec = torch.zeros(
            3, 3, self.lmax, self.mmax, dtype=uspec.dtype, device=uspec.device
        )
        save_array = torch.zeros(
            (nsteps // save_steps) + 1,
            *uspec.shape,
            dtype=uspec.dtype,
            device=uspec.device,
        )

        # pointers to indicate the most current result
        inew = 0
        inow = 1
        iold = 2
        save_index = 0

        for iter in range(burn_in_steps + nsteps):
            # if iter == burn_in_steps:
            #     save_array[0] = uspec

            dudtspec[inew] = self.dudtspec(uspec)

            # update vort,div,phiv with third-order adams-bashforth.
            # forward euler, then 2nd-order adams-bashforth time steps to start.
            if iter == 0:
                dudtspec[inow] = dudtspec[inew]
                dudtspec[iold] = dudtspec[inew]
            elif iter == 1:
                dudtspec[iold] = dudtspec[inew]

            uspec = uspec + self.dt * (
                (23.0 / 12.0) * dudtspec[inew]
                - (16.0 / 12.0) * dudtspec[inow]
                + (5.0 / 12.0) * dudtspec[iold]
            )

            # implicit hyperdiffusion for vort and div.
            uspec[1:] = self.hyperdiff * uspec[1:]

            # cycle through the indices
            inew = (inew - 1) % 3
            inow = (inow - 1) % 3
            iold = (iold - 1) % 3

            # Save the solution at the specified time steps
            if (iter+1) >= burn_in_steps and (iter+1 - burn_in_steps) % save_steps == 0:
                save_array[save_index] = uspec
                save_index += 1
        return save_array


class SSWEGenerator:
    def __init__(
        self,
        nlat,
        nlon,
        pred_horizon,
        burn_in_steps=1,
        save_steps=1,
        dt_solver=150,
        device=torch.device("cpu"),
        normalize=True,
    ):
        self.dt = pred_horizon * 3600
        self.nsteps = self.dt // dt_solver
        self.burn_in_steps = burn_in_steps * 3600 // dt_solver
        self.save_steps = save_steps * 3600 // dt_solver
        self.nlat = nlat
        self.nlon = nlon
        self.device = device
        self.normalize = normalize

        lmax = ceil(nlat / 3)
        mmax = lmax
        self.solver = (
            SSWESolver(nlat, nlon, dt_solver, lmax=lmax, mmax=mmax, grid="equiangular")
            .to(device)
            .float()
        )

        if self.normalize:
            inp = self.get_ics()
            inp0 = self.solver.spec2grid(inp)
            self.inp_mean = torch.mean(inp0, dim=(-1, -2)).reshape(-1, 1, 1)
            self.inp_var = torch.var(inp0, dim=(-1, -2)).reshape(-1, 1, 1)

    def get_ics(self):
        inp = self.solver.random_initial_condition(mach=0.2)
        return inp

    def simulate(self):
        inp = self.get_ics()
        # solve pde for n steps to return the target and save hourly steps
        res = self.solver.timestep(
            inp, self.nsteps, self.burn_in_steps, self.save_steps
        )
        res = self.solver.spec2grid(res)
        if self.normalize:
            res = (res - self.inp_mean) / torch.sqrt(self.inp_var)
        return res


def simulate(config: dict, output_dir:str, var: str = "train"):
    """ Simulation method for generating SSWE data.

    Args:
        config (dict): The config file for the simulation.
        var (str, optional): Whether to generate train or test data. Defaults to "train".
    """
    pde_config = config["pde"]
    sim_config = config[var]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nlat = pde_config["nlat"]*int(sim_config["scaling_factor"])
    nlon = pde_config["nlon"]*int(sim_config["scaling_factor"])
    generator = SSWEGenerator(
        nlat=nlat,
        nlon=nlon,
        pred_horizon=sim_config["pred_horizon"],
        burn_in_steps=sim_config["burn_in_steps"],
        save_steps=sim_config["save_steps"],
        dt_solver=pde_config["dt_solver"],
        device=device,
    )

    # Get lat and lon
    Lons = generator.solver.lons.squeeze() - torch.pi
    Lats = generator.solver.lats.squeeze()
    Lons = Lons * 180 / np.pi
    Lats = Lats * 180 / np.pi

    # Check for and create data directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create nc file for forecast
    filename_out = os.path.join(output_dir, f"{var}.nc")
    # Check if file already exists
    if os.path.exists(filename_out):
        os.remove(filename_out)
    # Create file
    f = Dataset(filename_out, "a")

    # Create dimensions
    f.createDimension("samples", None)
    f.createDimension("t", sim_config["pred_horizon"] + 1)
    f.createDimension("c", 3)
    f.createDimension("lat", nlat)
    f.createDimension("lon", nlon)

    # Create variables
    lat_dim = f.createVariable("latitude", "f4", ("lat",))
    lon_dim = f.createVariable("longitude", "f4", ("lon",))
    t_dim = f.createVariable("time", "f4", ("t",))
    u = f.createVariable("u", "f4", ("samples", "t", "c", "lat", "lon"))

    # Fill coordinate variables
    lat_dim[:] = Lats.cpu().numpy()
    lon_dim[:] = Lons.cpu().numpy()
    t_dim[:] = np.arange(0, sim_config["pred_horizon"] + 1)

    for i in range(sim_config["num_samples"]):
        sample = generator.simulate()
        u[i] = sample.cpu().numpy()
    f.close()


def load_config(path: str) -> dict:
    """Load the configuration file for the simulation.

    Args:
        path (str, optional): Path to config file. Defaults to "data/generation/sswe_config.yaml".

    Returns:
        dict: Configuration dictionary.
    """
    with open(path, "r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


# Main method
if __name__ == "__main__":
    config_path = "data/generation/sswe_config.yaml"
    output_dir = "data/SSWE/processed"
    torch.manual_seed(0)
    
    config = load_config(path = config_path)
    # Generate training data
    simulate(config, output_dir, "train")
    # Generate test data
    simulate(config, output_dir, "test")
