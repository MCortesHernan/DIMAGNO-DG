'''
Early postprocessing functions (for debugging only)
Real postprocessing SHALL be done in Paraview, after writing mesh, u to file. TODO
'''
import os
import matplotlib.pyplot as plt
import dolfin as df
from numpy import isinf
from numpy import where
import ufl_legacy as ufl
import numpy as np


def plot_2D(Simulation,i_component=None,tstr=None):
    '''
    Plot the 2D map of the solution component i
    '''  
    i_list = range(Simulation.PhysicalModel.u_dim()) if i_component is None else [i_component]
    tstr = Simulation.tstr() if tstr is None else tstr
    for i in i_list: 
        title = "u["+str(i)+"], t = "+tstr
        plt.figure()
        p = df.plot(Simulation.u[i], title=title)
        plt.colorbar(p)
        os.makedirs(Simulation.figpath+"u["+str(i)+"]", exist_ok=True)
        plt.savefig(Simulation.figpath+"u["+str(i)+"]/"+tstr+".png")
        plt.close()
    return None

def plot_ratio_2D(Simulation,i_num,i_den=0,tstr=None):
    '''
    Plot the 2D map of the solution component ratio i_num/i_den
    '''  
    tstr = Simulation.tstr() if tstr is None else tstr
    title = "u["+str(i_num)+"] over u["+str(i_den)+"], t = "+tstr
    plt.figure()
    p = df.plot(Simulation.u[i_num]/Simulation.u[i_den], title=title)
    plt.colorbar(p)
    os.makedirs(Simulation.figpath+"u["+str(i_num)+"] over u["+str(i_den)+"]", exist_ok=True)
    plt.savefig(Simulation.figpath+"u["+str(i_num)+"] over u["+str(i_den)+"]/"+tstr+".png")
    plt.close()
    return None

def save_xdmf(Simulation, t=None):
    '''
    Save whole solution vector to XDMF file.
    '''

    t = Simulation.t if t is None else t
    if isinf(t):
        t = 0
    t_r = round(100000*t)/100000
    os.makedirs(Simulation.XDMFpath+ "full_sol", exist_ok=True)
    file = df.XDMFFile(Simulation.XDMFpath + "full_sol/" + "u_t_"+str(t_r)+".xdmf")
    file.write_checkpoint(Simulation.u, "Solution Vector", time_step=t_r)
    
    return None

def save_df_function(path, f, name):
    '''
    Save a dolfin function to XDMF file.
    '''
    os.makedirs(path, exist_ok=True)
    file = df.XDMFFile(path + name + ".xdmf")
    file.write_checkpoint(f, name, time_step=0.0)
    
    return None

def load_df_function(V, name, path):
    '''
    Load a dolfin function from XDMF file.
    '''
    xdmf = df.XDMFFile(path + name + ".xdmf")
    f = df.Function(V)
    xdmf.read_checkpoint(f, name ,0) # The name "f_0" can be found in the XDMF file metadata
    return f

def save_xdmf_component(Simulation, i_component=None, t = None):
    '''
    Save a component of the solution vector to XDMF file
    '''

    i_list = range(Simulation.PhysicalModel.u_dim()) if i_component is None else [i_component]
    t = Simulation.t if t is None else t
    t_r = round(100000*t)/100000
    for i in i_list: 
        os.makedirs(Simulation.figpath+"u["+str(i)+"]", exist_ok=True)
        os.makedirs(Simulation.XDMFpath+ "u["+str(i)+']', exist_ok=True)
        file = df.XDMFFile(Simulation.XDMFpath + "u["+str(i)+"]/" + "u["+str(i)+"]_t_"+str(t_r)+".xdmf")
        file.write_checkpoint(Simulation.u.sub(i_component), "u["+str(i_component)+"]", time_step=t_r)

    return None

def load_xdmf_component(i_component=None, V = None, sol_name =None, file_path = None):
    # Read the mesh and solution from the XDMF file
    xdmf = df.XDMFFile(file_path)
    u=df.Function(V)
    xdmf.read_checkpoint(u, sol_name ,0) # The name "f_0" can be found in the XDMF file metadata
    return u

def save_dHe_dpsi(Simulation, dHe_dpsi):
    '''
    Save dHe_dpsi in an xdmf file.
    '''
    os.makedirs(Simulation.XDMFpath+ "dHe_dpsi", exist_ok=True)
    file = df.XDMFFile(Simulation.XDMFpath + "dHe_dpsi/" + "dHe_dpsi.xdmf")
    file.write_checkpoint(dHe_dpsi, "dHe_dpsi")
    return None


def mass_in_domain(Simulation, mesh):

    return df.assemble(Simulation.u[0]*df.dx(mesh))

def mass_flux(Simulation, mesh):

    '''Calculate the mass flux through an external boundary.'''

    j_vect = df.as_vector((Simulation.u[1],Simulation.u[2]))

    n = df.FacetNormal(mesh)

    return df.assemble(df.dot(j_vect,n)*df.ds)


def save_setup(Simulation, xc, R_L, R, G, u_z_0, gamma, setback_coils, path):

    Setup_file = open(path + 'setup.txt', 'a')
    Setup_file.write(' +++++++++++++++++++++++++++++ \n')
    Setup_file.write(' Setup file for simulation. \n')
    Setup_file.write(' +++++++++++++++++++++++++++++ \n')
    Setup_file.write(' Physical Parameters of the (gaussian) plasma distribution: \n')
    Setup_file.write(' Radius: R = ' + str(R) + '\n')
    Setup_file.write(' Centre: xc = ' + str(xc) + '\n')
    Setup_file.write(' Density at x = xc + R: G = 10e-' + str(G) + '\n')
    Setup_file.write(' Velocity: u_z = ' + str(u_z_0) + '\n')
    Setup_file.write(' +++++++++++++++++++++++++++++ \n')
    Setup_file.write(' Model parameters: \n')
    Setup_file.write(' gamma = ' + str(gamma) + '\n')
    Setup_file.write(' Ion magnetization:  '+ str(Simulation.PhysicalModel.ion_mag) + '\n')
    Setup_file.write(' +++++++++++++++++++++++++++++ \n')
    Setup_file.write(' Physical Parameters of the applied B Field: \n')
    Setup_file.write(' Radius of the coils: R_L = ' + str(R_L) + '\n')
    Setup_file.write(' Setback of coils: z_setb = ' + str(setback_coils) + '\n')
    Setup_file.write(' Current: I = ' + str(R) + '\n')
    Setup_file.write(' +++++++++++++++++++++++++++++ \n')
    Setup_file.write(' Parameters of the simulation: \n')
    Setup_file.write(' Time Stepping method: ' + str(Simulation.Propagator) + '\n')
    Setup_file.write(' Numerical Flux: ' + str(Simulation.NumericalFlux) + '\n')
    if Simulation.mesh.hmin() == Simulation.mesh.hmax():
        Setup_file.write(' Type of mesh: Regular \n')
    else:
        Setup_file.write(' Type of mesh: Iregular \n')

    Setup_file.write(' Number of cells : ' + str(Simulation.mesh.num_cells()) + '\n')
    Setup_file.write(' DG-Order : ' + str(Simulation.dg_degree) + '\n')
    Setup_file.close()

    return None

def j_e_source(mesh, jxi, Bz, Bx, R, jn, S, dg_degree = 0, j_e_perp = 0, nu_e = 0, axisymmetry = False, throat_ambipolarity = True):
    '''
    Calculate the electron in-plane velocity from the electron continuity equation
    with source terms.

    It uses the conservation law: B \cdot \grad(\phi) = S_i

    Here \phi = j_e/B which is conserved along magnetic streamlines.

    Inputs:

    P: Simulation object
    R: Radius of the inlet of plasma
    jz0: ion/electron current at the entrance should be of the type:
    n0(x, lib = ufl)*uz0(x, lib = ufl)
    if upstream_ambipolarity is imposed, if downstream write it as:
    jn_boundary(z,x, lib = ufl) where jn is the current normal to the boundary

    S: source term for the continuity equation
    axisymmetry: Boolean choose between axial or planar symmetry
    throat_ambipolarity: Boolean, True imposes ambipolarity in the throat, False imposes ambipolarity downstream.
    '''

    #Mesh
    mesh = (df.refine((mesh)))
    xx = df.SpatialCoordinate(mesh)

    coords = mesh.coordinates()
    min_z = np.min(coords[:,0])
    max_z = np.max(coords[:,0])
    min_x = np.min(coords[:,1])
    max_x = np.max(coords[:,1])
    
    #Vector Function space to solve the advection equation:

    V = df.FunctionSpace(mesh, 'DG', dg_degree + 1)
    V_vec = df.VectorFunctionSpace(mesh, 'CG', 2)

    #Reverse the direction of the magnetic field is ambipolarity is imposed downstream
    B = df.project(df.as_vector(((2*int(throat_ambipolarity) - 1)*Bz, (2*int(throat_ambipolarity) - 1)*Bx)), V_vec)
    B_perp = df.project(df.as_vector((-(2*int(throat_ambipolarity) - 1)*Bx, (2*int(throat_ambipolarity) - 1)*Bz)), V_vec)

    j_e_perp = df.project(j_e_perp, V)
    #Boundary conditions

    class Inlet(df.SubDomain):
        def inside(self, x, on_boundary):
            if throat_ambipolarity:
                return on_boundary and (df.near(x[0], 0) and df.between(x[1], (0,R))) 
            else:                
                return on_boundary and not (df.near(x[0], 0) and df.between(x[1], (0,R))) and not (df.near(x[1], 0) and not df.near(x[0], max_z))
   
   
    # Test function
    v = df.TestFunction(V)
    phi = df.TrialFunction(V) #Here phi is the variable to solve j_e_par/B

    # Mesh-related functions
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)

    #Measure for internal and external facets

    bound_mshfunc = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #Facets
    bound_mshfunc.set_all(0)

    INLET, OUTFLOW = 1, 2

    # Mark the whole exterior as outflow
    df.AutoSubDomain(lambda x, on_boundary: on_boundary).mark(bound_mshfunc, OUTFLOW)

    # Then overwrite the inflow markers
    inlets = Inlet()
    inlets.mark(bound_mshfunc, INLET)

    ds = df.Measure('ds', domain=mesh, subdomain_data=bound_mshfunc)

    #Physical Flux

    def F_u(phi):
        return phi*B

    #Upwinding
    Bn = (df.dot(B, n) + abs(df.dot(B, n)))/2.0

    #Magnetic field strength
    B_abs = df.project(df.sqrt(df.dot(B,B)), V)

    # Bilinear form
    a_int = df.dot(df.grad(v), - F_u(phi))*df.dx(mesh)

    a_vel = df.dot(df.jump(v), Bn('+')*phi('+') - Bn('-')*phi('-') )*df.dS(mesh)  + df.dot(v, Bn*phi)*ds(mesh)

    a = a_int + a_vel

    # Linear form (Source term)
    f = S - df.dot(B_perp, df.grad(j_e_perp/B_abs)) - (phi*B[1])/(xx[1]+1e-5)
    
    L = v*f*df.dx(mesh)
    

    # Set up boundary condition (apply strong BCs)
    class BC(df.UserExpression):
        def eval(self,value,x):
            
            if df.near(x[0], 0):
                j_e_par_bc = (-jn(x[0],x[1])*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(Bx(x[0],x[1])))/(-Bz(x[0],x[1])+1e-6)
                value[0] = -j_e_par_bc/B_abs(x[0],x[1])
            elif df.near(x[0], max_z):
                j_e_par_bc = (-jn(x[0],x[1])*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(-Bx(x[0],x[1])))/(Bz(x[0],x[1])+1e-6)
                value[0] = -j_e_par_bc/B_abs(x[0],x[1])
            elif df.near(x[1], max_x):
                j_e_par_bc = (-jn(x[0],x[1])*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(Bz(x[0],x[1])))/(Bx(x[0],x[1])+1e-6)
                value[0] = -j_e_par_bc/B_abs(x[0],x[1])
            
            # elif df.near(x[1], 0):
            #     j_e_par_bc = (-jn(x[0],x[1])*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(-Bz(x[0],x[1])))/(-Bx(x[0],x[1])+1e-6)
            #     value[0] = j_e_par_bc/B_abs(x[0],x[1])
            

        def value_shape(self):
            return ()
        
    g = BC()
    bc = df.DirichletBC(V, g, Inlet(), "geometric")

    #Total residual
    res = a - L

    #Separate lhs and rhs

    a = df.lhs(res)
    L = df.rhs(res)

    # Solution function
    phi_h = df.Function(V)

    # Assemble and apply boundary conditions
    A = df.assemble(a)
    b = df.assemble(L)

    bc.apply(A, b)

    # Solve system
    df.solve(A, phi_h.vector(), b)
    

    return B_abs*(phi_h)
    

def He_advection(He0, mesh, R_L, Bz, Bx, S):
    '''
    Calculates the new map of He solving an advection equation.

    It uses the conservation law: B \cdot \grad(\He) = S

    Inputs:

    He0: Initial bernoulli function along the throat (extended to the position of the coil)
    mesh: obvious
    R_L: Radius of the coil
    Bz, Bx: magnetic field components
    S: source term for the advection equation
        In the case of He advection with parallel collisions: 
          -B^2*\chi^-1 u_e_parallel
    '''

    #Vector Function space to solve the advection equation:

    V = df.FunctionSpace(mesh, 'DG', 1)
    V_vec = df.VectorFunctionSpace(mesh, 'CG', 2, 2)

    B = df.project(df.as_vector((Bz, Bx)), V_vec)

    #Boundary conditions

    class Inlet(df.SubDomain):
        def inside(self, x, on_boundary):
            return (df.near(x[0], 0) and df.between(x[1], (0,R_L))) and on_boundary
        
    # Test function

    v = df.TestFunction(V)
    phi = df.TrialFunction(V) #Here phi is the variable to solve j_e/B

    # Mesh-related functions
    n = df.FacetNormal(mesh)
    h = df.CellDiameter(mesh)

    #Measure for internal and external facets

    bound_mshfunc = df.MeshFunction("size_t", mesh, mesh.topology().dim() - 1) #Facets
    bound_mshfunc.set_all(0)

    INLET, OUTFLOW = 1, 2

    # Mark the whole exterior as outflow
    df.AutoSubDomain(lambda x, on_boundary: on_boundary).mark(bound_mshfunc, OUTFLOW)

    # Then overwrite the inflow markers
    inlets = Inlet()
    inlets.mark(bound_mshfunc, INLET)

    ds = df.Measure('ds', domain=mesh, subdomain_data=bound_mshfunc)

    #Physical Flux 
    def F_u(phi):
        return phi*B

    #Upwinding
    Bn = (df.dot(B, n) + abs(df.dot(B, n)))/2.0

    #Absolute value of B
    C = df.project(df.sqrt(df.dot(B,B)), V)

    # Bilinear form
    a_int = df.dot(df.grad(v), - F_u(phi))*df.dx(mesh)

    a_vel = df.dot(df.jump(v), Bn('+')*phi('+') - Bn('-')*phi('-') )*df.dS(mesh)  + df.dot(v, Bn*phi)*ds(mesh)

    a = a_int + a_vel

    # Linear form (Source term)
    L = v*S*df.dx(mesh)

    # Set up boundary condition (apply strong BCs)
    class BC(df.UserExpression):
        def eval(self, value, x):
            value[0] = He0(x[1])

        def value_shape(self):
            return ()
        
    g = BC()    
    bc = df.DirichletBC(V, g, Inlet(), "geometric")

    #Total residual
    res = a - L

    #Separate lhs and rhs

    a = df.lhs(res)
    L = df.rhs(res)

    # Solution function
    He = df.Function(V)

    # Assemble and apply boundary conditions
    A = df.assemble(a)
    b = df.assemble(L)

    bc.apply(A, b)

    # Solve system
    df.solve(A, He.vector(), b)
    He_limited_vec = where(He.vector()[:]>0, 0, He.vector()[:])
    He_limited_vec = where(He_limited_vec< 2*He0(R_L), He0(R_L), He_limited_vec)
    He.vector()[:] = He_limited_vec
    return He




