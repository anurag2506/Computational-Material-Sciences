
import numpy as np
import matplotlib.pyplot as plt

# Parameters
nx, ny = 100, 100 # grid size
dx = dy = 1.0 # space step
dt = 0.01 # time step
steps = 20000 # time steps

L = 1.0 # mobility
kappa = 1.0 # gradient energy coefficient

# Initialize order parameter (random noise around -1 to 1)
phi = np.random.rand(nx, ny) * 2 - 1

def laplacian(field):
    """Compute Laplacian with periodic BCs using finite difference"""
    return (
        np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
        np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) -
        4*field
    ) / dx**2

# Time integration
for n in range(steps+1):
    phi_old = phi.copy()

    # Allen-Cahn update (explicit Euler)
    phi += -L * dt * ( -kappa * laplacian(phi_old) + (phi_old**3 - phi_old) )

    # Plot snapshots
    if n in [0, 100, 500, 1000, 20000]:
        plt.imshow(phi, cmap="coolwarm", origin="lower")
        plt.colorbar(label="Order parameter φ")
        plt.title(f"Allen–Cahn evolution (step {n})")
        plt.savefig(f"Allen–Cahn evolution (step {n})")
        plt.show()