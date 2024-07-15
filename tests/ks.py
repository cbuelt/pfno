import pde
import numpy as np



def get_sim(grid, n_steps, dim, mesh = None, u = None, t_range = 150, dt = 0.001):
    if dim == 1:
        eq = pde.PDE({"u": "-u * d_dx(u) / 2 - laplace(u + laplace(u))"})
        state = pde.ScalarField.random_uniform(grid, vmin = -1, vmax = 1)  # generate initial condition
    else:
      #  X,Y = mesh        
      #  data = np.sin(np.pi/25*(X+Y)+u[0])+np.sin(np.pi/25*X+u[1])+np.sin(np.pi/25*Y+u[2])
      #  state = pde.ScalarField(grid, data = data)
        state = pde.ScalarField.random_normal(grid, mean = 0)
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