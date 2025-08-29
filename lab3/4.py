import numpy as np
import matplotlib.pyplot as plt

L=1.0
nx=50
dx=L/(nx-1)
D=0.1
dt=0.0005
nt=500

x=np.linspace(0,L,nx)

u=np.exp(-100*(x-0.5*L)**2)

u[0]=0
u[-1]=0

solutions=[u.copy()]

for n in range(nt):
    u_new=u.copy()
    for i in range(1,nx-1):
        u_new[i]=u[i]+D*dt/dx**2 * (u[i+1]-2*u[i]+u[i-1])
    u_new[0]=0
    u_new[-1]=0
    
    u=u_new.copy()
    
    if n%50==0:
        solutions.append(u.copy())

for idx,sol in enumerate(solutions):
    plt.plot(x,sol,label=f"t={idx*50*dt:.3f}")
plt.xlabel("x")
plt.ylabel("u(x,t)")
plt.title("1D Diffusion Equation (Explicit FD)")
plt.savefig("Dirchlet Boundary condition")
plt.legend()
plt.show()