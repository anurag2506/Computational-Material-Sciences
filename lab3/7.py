import numpy as np
import matplotlib.pyplot as plt

# ==================================
# Parameters
# ==================================
nx, ny = 256, 256 # grid size (use powers of 2 for fast FFTs)
dx = dy = 1.0 # spatial step
M = 1.0 # mobility
kappa = 1.0 # gradient energy coefficient
dt = 0.5 # time step (stable for this semi-implicit scheme)
steps = 5000 # number of time steps
plot_every = 100 # plot interval

# Wavenumbers (for FFT Laplacians with periodic BCs)
kx = 2.0*np.pi*np.fft.fftfreq(nx, dx)[:, None] * np.ones((nx, ny))
ky = 2.0*np.pi*np.fft.fftfreq(ny, dy)[None, :] * np.ones((nx, ny))
k2 = kx**2 + ky**2
k4 = k2**2

# Avoid division by zero at k=0 in denominators by noting:
# For k=0, k2=0 so the nonlinear term is multiplied by k^2 -> 0, and the linear term has 1+dt*M*kappa*k^4 = 1.
denominator = 1.0 + dt * M * kappa * k4

# Initial condition: small noise around mean composition (spinodal)
# Choose c0 in (-1, 1). c0=0 is a classic symmetric spinodal case.
rng = np.random.default_rng(42)
c0 = 0.0
amplitude = 0.05
c = c0 + amplitude * rng.standard_normal((nx, ny))

# ==================================
# Functions
# ==================================
def free_energy_prime(c):
    """Derivative of double-well: f'(c) = c^3 - c"""
    return c**3 - c

def step_CH(c):
    """One semi-implicit timestep in Fourier space"""
    c_hat = np.fft.fft2(c)
    nonlinear_hat = np.fft.fft2(free_energy_prime(c))
    numerator = c_hat - dt * M * k2 * nonlinear_hat
    c_hat_next = numerator / denominator
    c_next = np.real(np.fft.ifft2(c_hat_next))
    return c_next

# ==================================
# Time integration
# ==================================
for n in range(steps + 1):
    if n % plot_every == 0:
        plt.figure()
        im = plt.imshow(c, origin='lower', cmap='coolwarm')
        plt.colorbar(im, label='composition c')
        plt.title(f'Cahn-Hilliard spinodal decomposition (step {n})')
        plt.savefig(f"Cahn Hilliard for step{n}")
        plt.tight_layout()
        plt.show()

        # Optional: monitor mass conservation (should stay ~c0)
        print(f"Step {n} | mean(c) = {c.mean():.6f}, var(c) = {c.var():.6f}")

    c = step_CH(c)