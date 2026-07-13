# First test for the POSETS code:
# This test is intended to solve check the working of the fluid module
# It corresponds to the shock tube problem presented in Diego's thesis
# As the code is two-dimensional the domain is two dimensional even if the problem is 1D
# For this reason only 1/2 cells on the second component is used.
# The code is intended to run from the terminal and save the outputs in a directory named 'test_results/test_1/'
# However it is possible to run it in an interactive environment in vs code as visuaslization using wsl is more limited.
# To do so one might need to change some path names in order to point to the correct directory but it should be trivial.

# One can experiment here with change in the size of the mesh, the polynomial order, timestep, etcetera.
# It can also be interesting to check the visualization tools in FEniCS, save the file to XDMF and open it in paraview for inspection.

#%%
#Import modules
import sys

POSETS_path = '/mnt/c/Users/manue/Desktop/POSETS/POSETS-master'  #Inlude here path to th e location of the POSETS liberaries
sys.path.append(POSETS_path)
save_path = 'test_results/test_1/'

import matplotlib.pyplot as plt #Plotting
import numpy as np  #Numpy for postprocessing
import dolfin as df #dolfin is the fenics environment for solving PDEs
import os 

import src #This contains all the code for the discretization, time evolution...
from src import Plasma

#%%
#This creates a very simple mesh with N triangles in the z direction and 2 in the x
N_z = 200
N_x =2
# As the code is two-dimensional the domain is two dimensional even if the problem is 1D
# For this reason only 1/2 cells on the second component is used.
mesh = df.RectangleMesh(df.Point(0,0), df.Point(1, 0.1), N_z, N_x, 'left/right') 

#Here we choose all the points on the boundary of out domain.
class All(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

# In this case the simulation could represent a standard gas not necesarily a plasma, in this case this is the polytropic index of such gas 
#Electron model 
ElectronModel = Plasma.ElectronModel(gamma=1.4)

# Magnetic field df.Fuctions 
cg_degree = 2 # Degree of the polynomial for the magnetic field
_mesh = df.refine(mesh) # Use a finer mesh as we will later refine the solution
V = df.FunctionSpace(_mesh, 'CG', cg_degree) # Function space for preliminary setup work (Bfield, etc)
#See that continuous Galerkin is used for the magnetic field. 

#This function appears because the code is intended for plasma sim. But for an unmagnetised gas this is 0. (df.Function() initializes as 0)
dHe_dpsi = df.Function(V)

#%%
#Here we call the physical model, this is the simplest one included. We initialize everything related to magnetic fields and such to 0. 
# Physical flux
PhysicalFlux = Plasma.PlasmaModel2D(ElectronModel=ElectronModel, Bz=df.Constant(0), Bx=df.Constant(0), dHe_dpsi=dHe_dpsi, electron_mag = False, ion_mag= False, collision_param = 0.0, val_check= True, positivity_flag=True, stabilization_params= (0, 0, 0), axisymmetry=False) 

# Boundary conditions 
bcs = {All() : Plasma.BC_Wall2D()} #We set everything to a wall (mandatory for upper and lower bounds).


# Other parameters of the simulation
NumericalFlux = src.NumericalFluxes.LocalLaxFriedrichs()
Propagator = src.Propagators.RK()
dg_degree=0 #Degree of the polynomial for the solution   
dt= 0.5/(2*dg_degree + 1) * mesh.hmin()
#%%
# Create simulation object
P = src.Simulation(mesh=mesh, 
                    PhysicalModel=PhysicalFlux,
                    NumericalFlux=NumericalFlux,  
                    Propagator=Propagator,
                    bcs=bcs, 
                    dg_degree=dg_degree,
                    dt=dt, refinement=0)

P.XDMFpath = save_path

P.Propagator.initialize_solver(P.du, P.a_form, P.b_form, P.u, P.dt)

P.u.assign(df.Expression(('x[0] < 0.5 ? 1 : 0.125',
                        '0.',
                        '0.',
                        '0.'),degree=4)) #Initial boundary conditions for the shock tube problem

print(P.u(0.5,0))

plt.colorbar(df.plot(P.u[0]))
plt.show(block=False)


a = np.max(df.assemble(P.b_form(P.u)).get_local())
b = np.min(df.assemble(P.b_form(P.u)).get_local())

print(a)
print(b)


#%%
#This a simple postprocessing function that we can pass toi the simulation environment
z = np.linspace(0, 1, 1000)

if not os.path.exists(save_path):
            os.makedirs(save_path)

def plot_sol(P):
    dens = np.array([P.u(i,0)[0] for i in z])
    jz = np.array([P.u(i,0)[1] for i in z])

    fig, ax = plt.subplots()

    ax.plot(z, dens, label = r'$n$')
    ax.plot(z, jz/dens, label = r'$u$')

    ax.set_xlim(0,1)
    ax.set_ylim(-0.05, 1.1)
    ax.set_xlabel('z')
    ax.set_ylabel('Values')
    ax.legend()
    fig.savefig(save_path + 'dens_vel.png', dpi = 100, bbox_inches = 'tight')

    plt.show()
    print(dens)
    return 

#%%
P.simulate([300, plot_sol])

print(P.u(0.5,0)[1])

src.save_xdmf_component(P, 0, 0.0)
src.save_xdmf_component(P, 1, 0.0)
src.save_xdmf_component(P, 2, 0.0)
src.save_xdmf_component(P, 3, 0.0)

# %%
