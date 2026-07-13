# Third test for the POSETS code:
# This test is intended to solve check the working of the fluid module along with the magnetisation one
# It corresponds to Planar Plasma Column presented in Diego's thesis
# The code is intended to run from the terminal and save the outputs in a directory named 'test_results/test_3/'
# However it is possible to run it in an interactive environment in vs code as visuaslization using wsl is more limited.
# To do so one might need to change some path names in order to point to the correct directory but it should be trivial.

# This shows how to define a magnetic field and the initial conditions for the thermalised
# potential and its derivative. 
# This also shows the convergence properties of the code, it is usefull to use the plot_det() function
# in order to visalize the ion streamlines converging to a straight column
# It is interesting to check here the time consumption to solve a given problem for the different
# orders and refinements and the number of degrees of freedom.

#%%
#Import modules
import sys

POSETS_path = '/mnt/c/Users/manue/Desktop/POSETS/POSETS-master'
sys.path.append(POSETS_path)
save_path = 'test_results/test_3/'

import matplotlib.pyplot as plt #Plotting
import numpy as np  #Numpy for postprocessing
import dolfin as df #dolfin is the fenics environment for solving PDEs
import ufl_legacy as ufl
import os 

import src #This contains all the code for the discretization, time evolution...
import src.physics.Plasma as Plasma #import src.np_postproc as post #Postprocessing libraries

df.parameters["ghost_mode"] = "shared_facet"


mesh = df.RectangleMesh(df.Point(0,0), df.Point(2, 1), 20, 10, 'left/right')

class All(df.SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

class Inlet(df.SubDomain):  
    def inside(self, x, on_boundary):
        return on_boundary and df.near(x[0], 0) and df.between(x[1], (0, 1)) 

class Wall(df.SubDomain):  
    def inside(self, x, on_boundary):
        return on_boundary and (df.near(x[1], 0)) 
    #We set a wall boundary condition only in the symmetry plane as no flow should cross it

class Interior(df.SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= 0.75 #This is a subdomain in the interior of the column to check convergence

interior = Interior()

# Initialize mesh function for interior domain
domains = df.MeshFunction("size_t", mesh, mesh.topology().dim())
domains.set_all(0)
interior.mark(domains, 1)

# Straight Magnetic Field
class BField2D():
    def psi(self, z=0., x=0., lib=np):
        psi = x+0.
        return psi

    def Bz(self, z=0., x=0., lib=np):
        Bz = z*0. + 1 # allocate and zero
        return Bz

    def Br(self, z=0., x=0., lib=np):
        Br = z*0.  # allocate and zero
        return Br
    
# This magnetic field is built specifically for this test but the 
# structure of the BField class inside src is basically the same as this one   

BField2D = BField2D()

#%%
# A sample inlet condition (callable with lib=np or lib=ufl)
np.ln = np.log # Alias, to avoid problems switching between ufl and numpy in calls

# Electron model 
gamma_e = 1.2
ElectronModel = Plasma.ElectronModel(gamma=gamma_e)

G = 3
R = 1
Mi0 = 1
eps = 0
#%%
def n0(x,lib=np):
    return (lib.exp(-G*lib.ln(10.)*(x/R)**2.) + eps)/(1+eps) #Gaussian inlet profile
def dn_dx0(x,lib=np):
    return -2*G*lib.ln(10)*(x/R**2)*lib.exp(-G*lib.ln(10)*(x/R)**2)/(1+eps) #Analytical derivative
def uz0(x,lib=np):    
    return Mi0*ElectronModel.c(n0(x, lib= lib)) #Sonic Inlet
def ux0(x,lib=np):
    return 0.*x
def uy0(x,lib=np):
    return 0.*x
def phi0(x,lib=np):
    return 0.*x
def dphi_dx0(x,lib=np):
    return 0.*x

# Interpolation libraries for He, dHe_dpsi
_x = np.linspace(0, 1, 5000) 
_n0 = n0(_x)
_dn_dx0 = dn_dx0(_x)
_phi0 = phi0(_x)
_dphi_dx0 = dphi_dx0(_x)
_psi0 = BField2D.psi(0,_x)
_Bz0 = BField2D.Bz(0,_x)
_He0 = ElectronModel.He(_n0, _phi0)
_dHe_dpsi0 = ElectronModel.dHe_dxi(_n0, _dn_dx0, _dphi_dx0) / (_Bz0)
ElectronModel.setup_interpolators(_psi0,_He0,_dHe_dpsi0)


#%%
# Magnetic field df.Fuctions 
cg_degree = 2
_mesh = df.refine(mesh) # Use a finer mesh as we will later refine the solution
V = df.FunctionSpace(_mesh, 'CG', cg_degree) # Function space for preliminary setup work (Bfield, etc)
V_cont = df.FunctionSpace(mesh, 'CG', 1)

# Quite convenient: z,x as df.Function
x = df.interpolate(df.Expression('x[1]',degree=cg_degree), V)

coords = V.tabulate_dof_coordinates()
psi = df.Function(V)
Bz = df.Function(V)
Br = df.Function(V)
psi.vector()[:] = BField2D.psi(coords[:,0], coords[:,1])
Bz.vector()[:] = BField2D.Bz(coords[:,0], coords[:,1])
Br.vector()[:] = BField2D.Br(coords[:,0], coords[:,1])

dHe_dpsi = df.Function(V)
dHe_dpsi.vector()[:] = ElectronModel.interpolate_dHe_dpsi(psi.vector())
He = df.Function(V)
He.vector()[:] = ElectronModel.interpolate_He(psi.vector())

#%%
#####
# Physical flux
PhysicalFlux = Plasma.PlasmaModel2D(ElectronModel=ElectronModel, Bz=Bz, Bx=Br, dHe_dpsi=dHe_dpsi, electron_mag = True, ion_mag= False, collision_param = 0.0, val_check= True, positivity_flag=True, stabilization_params= (0, 0, 0), axisymmetry=False) 

# Boundary conditions 
bcs = {All() : Plasma.BC_SupersonicOutflow2D(ElectronModel),
       Wall() : Plasma.BC_Wall2D(),
       Inlet() : Plasma.BC_SupersonicInflow2D(n_outside=n0(x,lib=ufl), v_normal_outside=uz0(x,lib=ufl), v_parallel_outside=ux0(x,lib=ufl), vyi_outside=uy0(x,lib=ufl))}


# Other parameters of the simulation
NumericalFlux = src.NumericalFluxes.LocalLaxFriedrichs()
Propagator = src.Propagators.RK()
dg_degree=0
dt=1/(2*dg_degree + 1)*mesh.hmin()/(Mi0+1)

#%%
# Create simulation object
P = src.Simulation(mesh = mesh, 
                      PhysicalModel=PhysicalFlux,
                      NumericalFlux=NumericalFlux,  
                      Propagator=Propagator,
                      bcs=bcs, 
                      dg_degree=dg_degree,
                      dt=dt, refinement=0)
P.XDMFpath = save_path

P.Propagator.initialize_solver(P.du, P.a_form, P.b_form, P.u, P.dt)

#%%
P.u.assign(df.Expression(('pow(10,-3)',
                        'pow(10,-3)',  
                        '0.',
                        '0.'), degree=4)) 



#%%
if not os.path.exists(save_path):
            os.makedirs(save_path)

#This function plots and save the log_10(n) and ion streamlines
def plot_det(P, save = True):
    x = np.linspace(0, 1, 50)
    #z = np.linspace(0, 2, 50)
    #X,Z = np.meshgrid(z,x)
    #postpro = post.postpro_object(x, z, P, Bz, Br, psi, He, dHe_dpsi, ElectronModel)

    #dens = postpro.dens_grid()
    #jx = postpro.j_x_grid()
    #jz = postpro.j_z_grid()
    #psi_grid = postpro.psi_grid()

    #fig = plt.figure()
    #ax= fig.add_axes([0, 0, 1.25, 0.75])
    #ax.axes.set_aspect('equal')

    #pcol = ax.pcolor(X,Z, np.log10(dens))
    #cbar = fig.colorbar(pcol, ax = ax)
    #plt.title(f't = {P.t}')

    #ax.streamplot(X,Z, jz, jx, color ='magenta')

    #ax.set_xlim(0,2)
    #ax.set_ylim(0,1)

    #if save:
        #plt.savefig(save_path + f'sol_p_{P.dg_degree}_ref_{P.refinement}.png', dpi = 100, bbox_inches= 'tight')
#%%

error_list_0 = [] #List to store the error of each simulation of order 0
n_dof_list_0 = []

def solve(P):
    P.simulate(['steady'])
    while not P.converged:
        P.simulate([200, 'steady'])
    plot_det(P)

dx = df.Measure('dx', domain=mesh, subdomain_data= domains)

#%%
solve(P)
print(f'dg degree = {P.dg_degree} with refinement {P.refinement} is solved.\n')
error_list_0.append(df.assemble(P.u[2]**2*dx(1)))
n_dof_list_0.append(len(P.u.vector()[:]))
#As the ion current should be in the z direction this measures the L^2 error of the solution
P.simulate(['refine'])
P.converged = False

#%%

solve(P)
print(f'dg degree = {P.dg_degree} with refinement {P.refinement} is solved.\n')
error_list_0.append(df.assemble(P.u[2]**2*dx(1)))
n_dof_list_0.append(len(P.u.vector()[:]))
P.simulate(['refine'])
P.converged = False

solve(P)
print(f'dg degree = {P.dg_degree} with refinement {P.refinement} is solved.\n')
error_list_0.append(df.assemble(P.u[2]**2*dx(1)))
n_dof_list_0.append(len(P.u.vector()[:]))
P.simulate(['refine'])
P.converged = False

solve(P)
print(f'dg degree = {P.dg_degree} with refinement {P.refinement} is solved.\n')
error_list_0.append(df.assemble(P.u[2]**2*dx(1)))
n_dof_list_0.append(len(P.u.vector()[:]))

#%%
error_list_1 = []
n_dof_list_1 = []
P1 = src.Simulation(mesh = mesh, 
                      PhysicalModel=PhysicalFlux,
                      NumericalFlux=NumericalFlux,  
                      Propagator=Propagator,
                      bcs=bcs, 
                      dg_degree=1,
                      dt=dt, refinement=0, u = P.u) 
# Here we use the solution to the previous problem as initial guess in order to save some time

P1.XDMFpath = save_path

solve(P1)
print(f'dg degree = {P1.dg_degree} with refinement {P1.refinement} is solved.\n')
error_list_1.append(df.assemble(P1.u[2]**2*dx(1)))
n_dof_list_1.append(len(P1.u.vector()[:]))
P1.simulate(['refine'])
P1.converged = False

solve(P1)
print(f'dg degree = {P1.dg_degree} with refinement {P1.refinement} is solved.\n')
error_list_1.append(df.assemble(P1.u[2]**2*dx(1)))
n_dof_list_1.append(len(P1.u.vector()[:]))
P1.simulate(['refine'])
P1.converged = False

solve(P1)
print(f'dg degree = {P1.dg_degree} with refinement {P1.refinement} is solved.\n')
error_list_1.append(df.assemble(P1.u[2]**2*dx(1)))
n_dof_list_1.append(len(P1.u.vector()[:]))
P1.simulate(['refine'])
P1.converged = False

solve(P1)
print(f'dg degree = {P1.dg_degree} with refinement {P1.refinement} is solved.\n')
error_list_1.append(df.assemble(P1.u[2]**2*dx(1)))
n_dof_list_1.append(len(P1.u.vector()[:]))

#%%
error_list_2 = []
n_dof_list_2 = []
P2 = src.Simulation(mesh = mesh, 
                      PhysicalModel=PhysicalFlux,
                      NumericalFlux=NumericalFlux,  
                      Propagator=Propagator,
                      bcs=bcs, 
                      dg_degree=2,
                      dt=dt/8, refinement=0, u = P.u)

P2.dt=0.5/(2*P2.dg_degree + 1)*P2.mesh.hmin()/(Mi0+1)

P2.XDMFpath = save_path

#Here we only do 3 refinement levels but the 
solve(P2)
print(f'dg degree = {P2.dg_degree} with refinement {P2.refinement} is solved.\n')
error_list_2.append(df.assemble(P2.u[2]**2*dx(1)))
n_dof_list_2.append(len(P2.u.vector()[:]))
P2.simulate(['refine'])
P2.converged = False

solve(P2)
print(f'dg degree = {P2.dg_degree} with refinement {P2.refinement} is solved.\n')
error_list_2.append(df.assemble(P2.u[2]**2*dx(1)))
n_dof_list_2.append(len(P2.u.vector()[:]))
P2.simulate(['refine'])
P2.converged = False

solve(P2)
print(f'dg degree = {P2.dg_degree} with refinement {P2.refinement} is solved.\n')
error_list_2.append(df.assemble(P2.u[2]**2*dx(1)))
n_dof_list_2.append(len(P2.u.vector()[:]))

#%%
#
h0 = mesh.hmin()

fig = plt.figure()
ax= fig.add_axes([0, 0, 1.25, 0.75])

ax.plot([h0,h0/2,h0/4,h0/8], error_list_0)
ax.scatter([h0,h0/2,h0/4,h0/8], error_list_0, marker = '+', label = r'$p = 0$')

ax.plot([h0,h0/2,h0/4,h0/8], error_list_1)
ax.scatter([h0,h0/2,h0/4,h0/8], error_list_1, marker = '+', label = r'$p = 1$')

ax.plot([h0,h0/2,h0/4], error_list_2)
ax.scatter([h0,h0/2,h0/4], error_list_2, marker = '+', label = r'$p = 2$')

ax.legend()
ax.set_xscale('log')
ax.set_yscale('log')

fig.savefig(save_path + f'convergence.png', dpi = 200, bbox_inches= 'tight')


