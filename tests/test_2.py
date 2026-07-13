# Second test for the POSETS code:
# This test is intended to solve check the working of the fluid module
# It corresponds to the Prandtl-Mayer expansion problem presented in Diego's thesis
# This problem is already fully 2D
# The code is intended to run from the terminal and save the outputs in a directory named 'test_results/test_2/'
# However it is possible to run it in an interactive environment in vs code as visuaslization using wsl is more limited.
# To do so one might need to change some path names in order to point to the correct directory but it should be trivial.

# This already teaches the main workflow for more difficult simulations: 
# Evolve in time and then solve the steady state when the state is close enough to convergence.
# One can experiment with how much time evolution does the simulation need to reach convergence. 
# Here one can check already refinement of the mesh with P.refine_mesh/ P.refine_p and converegence against the analytical mach number.
# One can also check the effect of the slope limiting/ stabilization mechanism.

#%%
#Import modules
import sys

POSETS_path = '/mnt/c/Users/manue/Desktop/POSETS/POSETS-master' #Inlude here path to the location of the POSETS liberaries
sys.path.append(POSETS_path)
save_path = 'test_results/test_2/'

import matplotlib.pyplot as plt #Plotting
import numpy as np  #Numpy for postprocessing
import dolfin as df #dolfin is the fenics environment for solving PDEs
import ufl_legacy as ufl
import os 

import src #This contains all the code for the discretization, time evolution...
import src.physics.Plasma as Plasma #This contains the physics of the plasma model

#%%
df.parameters["ghost_mode"] = "shared_facet"

#Load mesh
mesh = df.Mesh()
df.XDMFFile('/mnt/c/Users/manue/Desktop/POSETS/POSETS-master/tests/meshes/fan_mesh.xdmf').read(mesh)
mesh = df.refine(mesh)
#%%

#Set different boundary conditions for different parts of the boundary
class All(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Inlet(df.SubDomain):  
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[0], 0)

class Out(df.SubDomain):  
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[0], 25)

# Electron model 
ElectronModel = Plasma.ElectronModel(gamma=1.4)

M0 = 1.1 #Initial Mach number


#Set boundary conditions for the inlet
def n0(x,lib=np):
    return 1 #Constant density across all inlet
def dn_dx0(x,lib=np):
    return 0
def uz0(x,lib=np):    
    return M0*ElectronModel.c(n0(x, lib = lib)) #Constant mach Number across all inlet
def ux0(x,lib=np):
    return x*0 #Zero radial velocity



# Magnetic field df.Fuctions 
cg_degree = 2
_mesh = df.refine(mesh) # Use a finer mesh as we will later refine the solution
V = df.FunctionSpace(_mesh, 'CG', cg_degree) # Function space for preliminary setup work (Bfield, etc)

# Quite convenient: z,x as df.Function
z = df.interpolate(df.Expression('x[0]',degree=cg_degree), V)
x = df.interpolate(df.Expression('x[1]',degree=cg_degree), V)

dHe_dpsi = df.Function(V)

#This is ion out-of-plane velocity, makes no sense for unmagnetised gas but the code needs it  
def uy0(x,lib=np):
    return x*0

#%%
#####
# Physical flux
PhysicalFlux = Plasma.PlasmaModel2D(ElectronModel=ElectronModel, Bz=df.Constant(0), Bx=df.Constant(0), dHe_dpsi=dHe_dpsi, electron_mag = False, ion_mag= False, collision_param = 0.0, val_check= True, positivity_flag=True, stabilization_params= (0, 0, 0), axisymmetry=False) 

# Boundary conditions 
bcs = {All() : Plasma.BC_Wall2D(),
       Out() : Plasma.BC_SupersonicOutflow2D(ElectronModel),
       Inlet() : Plasma.BC_SupersonicInflow2D(n_outside=n0(x,lib=ufl), v_normal_outside=uz0(x,lib=ufl), v_parallel_outside=ux0(x,lib=ufl), vyi_outside=uy0(x,lib=ufl))}


# Other parameters of the simulation
NumericalFlux = src.NumericalFluxes.LocalLaxFriedrichs()
Propagator = src.Propagators.RK()
dg_degree=0
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
           
#Provide some non-zero density to start the simulation            
P.u.assign(df.Expression(('pow(10,-6)',
                        'pow(10,-6)',  
                        '0.',
                        '0.'), degree=4)) 

# Postproc function to pass to the simulate strategy

if not os.path.exists(save_path):
            os.makedirs(save_path)

def plot(P):

    fig = plt.figure()
    ax = fig.add_axes([0,0, 1.1, 0.9])
    ax.axes.set_aspect('equal')
    plt.colorbar(df.plot(ufl.ln(P.u[0])/ufl.ln(10)))
    ax.set_title(r'$log_{10}(n)$' + '\n t = ' + str(np.round(P.t, 5)))
    fig.savefig(save_path + 'expansion.png', dpi = 100, bbox_inches = 'tight')
    plt.show(block=False)

    fig1 = plt.figure()
    ax1 = fig1.add_axes([0,0, 1.1, 0.9])
    ax1.axes.set_aspect('equal')
    plt.colorbar(df.plot(ufl.ln(P.u[1])))
    fig1.savefig(save_path + 'expansion1.png', dpi = 100, bbox_inches = 'tight')
    plt.show()
    

#%%
# Analitical solution for mach number after turn
# One can compare the analytical solution against this TODO
from scipy.optimize import fsolve
from matplotlib import colors

gamma = ElectronModel.gamma

def nu(M):
    return np.sqrt((gamma+1)/(gamma-1))*np.arctan(np.sqrt((gamma-1)/(gamma+1)*(M**2-1))) - np.arctan(np.sqrt(M**2-1))

func = lambda M : nu(M) - nu(M0) - np.arctan(2/3)

M1 = fsolve(func, 2)[0]

#####
def M(P,z,x):
    q = P.u(z,x)
    return (q[1]**2 + q[2]**2)**0.5/(gamma*q[0]**(gamma+1))**0.5



#Run first simulation for p = 0
P.simulate([200, 'steady', plot])
src.save_xdmf_component(P, 0, 0.0)
src.save_xdmf_component(P, 1, 0.0)
src.save_xdmf_component(P, 2, 0.0)
src.save_xdmf_component(P, 3, 0.0)


# %%
