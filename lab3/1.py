import numpy as np
import matplotlib.pyplot as plt

nx,ny = 50,50
dx,dy = 1.0,1.0
dt = 0.1
D = 1.0
steps = 2000

stability_ratio = D*dt*(1/dx**2 + 1/dy**2)
print(f"Stability ratio = {stability_ratio:.3f}")

if stability_ratio >=0.5:
    raise ValueError("Unstable Scheme, reduce dt")

u = np.zeros((nx,ny), dtype=np.float64)
u[nx//2,ny//2] = 100.0


def apply_periodic_boundary(u):
    u[0,:] = u[-2,:]
    u[-1,:] = u[1,:]
    u[:,0] = u[:,-2]
    u[:,-1] = u[:,1]


for n in range(steps+1):
    u_old = u.copy()

    u[1:-1,1:-1] = (u_old[1:-1,1:-1]+
                    D*dt/dx**2 * (u_old[2:,1:-1]-2*u_old[1:-1,1:-1]+u_old[:-2, 1:-1])+
                    D*dt/dy**2 * (u_old[1:-1,2:]-2*u_old[1:-1,1:-1]+u_old[1:-1,:-2]))

    apply_periodic_boundary(u)

    if n in [0,100,500,1000,2000]:
        plt.imshow(u,cmap="hot", origin="lower")
        plt.colorbar(label="u(x,y)")
        plt.title(f"Diffusion with periodic BC at step {n}")
        plt.savefig(f"Diffusion with periodic BC at step {n}")
        plt.show()
