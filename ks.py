import pde
import numpy as np



def get_sim(grid, n_steps, dim, t_range = 150, dt = 0.001):
    state = pde.ScalarField.random_uniform(grid, vmin = -1, vmax = 1)  # generate initial condition
    if dim == 1:
        eq = pde.PDE({"u": "-u * d_dx(u) / 2 - laplace(u + laplace(u))"})
    else:
        eq = pde.PDE({"u": "-gradient_squared(u) / 2 - laplace(u + laplace(u))"})  # define the pde

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