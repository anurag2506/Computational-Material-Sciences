import numpy as np
import matplotlib.pyplot as plt

Lx,Ly = 1.0,1.0
nx,ny = 50,50
dx,dy = Lx/(nx-1),Ly/(ny-1)
D = 0.1
dt = 0.0001
nt = 1000

x = np.linspace(0,Lx,nx)
y = np.linspace(0,Ly,ny)

X,y = np.meshgrid(x,y)

u = np.exp(-100*((X-0.5&Lx)**2 + (y-0.5*Ly)**2))

u[0,:] = 0
u[-1,:] = 0
u[:,0] = 0
u[:,-1] = 0

for n in range(nt):
    u_new = u.copy()
    for i in range(1, nx-1):
        for j in range(1, ny-1):
            u_new[j, i] = u[j, i] + D*dt * (
                (u[j, i+1] - 2*u[j, i] + u[j, i-1]) / dx**2 +
                (u[j+1, i] - 2*u[j, i] + u[j-1, i]) / dy**2
            )
   
    # Apply Dirichlet BCs
    u_new[0, :] = 0
    u_new[-1, :] = 0
    u_new[:, 0] = 0
    u_new[:, -1] = 0
   
    u = u_new.copy()
   
    # Plot at intervals
    if n % 100 == 0:
        plt.clf()
        plt.imshow(u, extent=[0, Lx, 0, Ly], origin="lower", cmap="hot")
        plt.colorbar(label="u")
        plt.title(f"2D Diffusion at step {n}")
        plt.pause(0.1)

plt.show()