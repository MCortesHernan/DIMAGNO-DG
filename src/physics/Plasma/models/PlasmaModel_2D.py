
''' --- BASIC 2D MODEL WITH/OUT ENERGY CONSERVATION --- '''

'''

This module contains the definition of the plasma model with: 
- 2D 
- Energy conservation (optional) 

The state vector is: u = [n, gzi, gxi, gyi, (Ei)] where Ei is the ion mechanical energy (optional).
The source terms include ionization and charge exchange collisions.

The boundary conditions implemented are:
- Wall BCs: the normal velocity of both species is reversed and the parallel and out-of-plane velocities are unchanged.
- Supersonic inflow BCs: density and velocities for both species are prescribed.
- Supersonic outflow BCs: the normal velocity of both species is unchanged and the parallel and out-of-plane velocities are extrapolated from the interior. 

'''

# IMPORTS

import numpy as np
import ufl_legacy as ufl

import src

n_min = 1e-10 # minimum value of the density, when computing velocities

# Simple variable extraction functions (If axisymmetric x becomes r  and y becomes theta)
def n(u): return ufl.conditional(u[0] > 1e-20, u[0], 1e-20)
def gzi(u): return u[1]
def gxi(u): return u[2]
def gyi(u): return u[3]
def E(u): return u[4]
def vzi(u): return gzi(u)/(n(u) + n_min)
def vxi(u): return gxi(u)/(n(u) + n_min)
def vyi(u): return gyi(u)/(n(u) + n_min)
def v_i(u): return ufl.as_vector((vzi(u), vxi(u) , vyi(u)))
def v_squared(u): return ufl.inner(v_i(u), v_i(u))

#Spatial coordinates (only needed for axisymmetric form)
#x = ufl.SpatialCoordinate()

class PlasmaModel2D(src.PhysicalModels.PhysicalModel):
    '''
    Plasma physical fluxes.  
    The state vector is: u = [n, gzi, gxi, gyi] for non-energy conserving model
    The state vector is: u = [n, gzi, gxi, gyi, E] for energy conserving model
    ''' 
    def __init__(self, ElectronModel, IonModel, Bz, Bx, energy_eq = True, val_check= True, electron_mag = True, ion_mag = True, collision_param = 0.0,  positivity_flag = False, stabilization_params = (0,0), axisymmetry = False, mesh = None):

        self.ElectronModel = ElectronModel 
        self.IonModel = IonModel
        self.Bz = Bz
        self.Bx = Bx
        self.energy_eq = energy_eq
        self.val_check = val_check
        self.electron_mag = electron_mag
        self.ion_mag = ion_mag
        self.nu = collision_param
        self.stabilization_params = stabilization_params
        self.positivity_flag = positivity_flag
        self.axisymmetry = axisymmetry
        self.mesh = mesh
        self.x = ufl.SpatialCoordinate(self.ElectronModel.dHe_dpsi.function_space().mesh())

    def x_dim(self):
        return 2

    def u_dim(self):
        if self.energy_eq:
            return 5
        else:
            return 4
    

    def flux(self, u, sign = None): 

        if sign is None:
            He_sign = self.ElectronModel.He
        else:
            He_sign = self.ElectronModel.He(sign)

        pe = self.ElectronModel.pe(n(u))  # Electron pressure
        
        '''
        See that the values are not being directly modified on the function,
        but the system is indirectly forcing the simulation to be physically consistent.        
        '''

        if self.energy_eq:
            phi = self.ElectronModel.phi(n(u), He_sign)
            pi_unbounded = self.IonModel.pi(n(u), v_squared(u), E(u), phi)  # Ion pressure
            pi_min = 1e-10  # Replace with the desired threshold 
            #pi = ufl.conditional(ufl.ge(pi_unbounded, pi_min), pi_unbounded, self.IonModel.Ti0_val*n(u)**self.IonModel.gamma)  # Ensure pi >= pi_min
            pi = ufl.conditional(ufl.ge(pi_unbounded, pi_min), pi_unbounded, pi_min)  # Ensure pi >= pi_min

            return [[gzi(u),                    gxi(u)],
                    [gzi(u)*vzi(u) + pi + pe,   gxi(u)*vzi(u)],
                    [gzi(u)*vxi(u),             gxi(u)*vxi(u) + pi + pe],
                    [gzi(u)*vyi(u),             gxi(u)*vyi(u)],
                    [(E(u) + pi)*vzi(u),        (E(u) + pi)*vxi(u)]]
        else:
            return [[gzi(u),             gxi(u)],
                    [gzi(u)*vzi(u) + pe, gxi(u)*vzi(u)],
                    [gzi(u)*vxi(u),      gxi(u)*vxi(u) + pe],
                    [gzi(u)*vyi(u),      gxi(u)*vyi(u)]]

             
    def eigenvalues(self, u, normal, sign = None):   
        
        un = vzi(u)*normal[0] + vxi(u)*normal[1] # Inplane velocity component along normal
        
        if sign is None:
            He_sign = self.ElectronModel.He
        else:
            He_sign = self.ElectronModel.He(sign)

        if self.energy_eq:
            gamma_e = self.ElectronModel.gamma
            gamma_i = self.IonModel.gamma

            phi = self.ElectronModel.phi(n(u), He_sign)
            pi = self.IonModel.pi(n(u), v_squared(u), E(u), phi)  # Ion pressure
            pi_min = 1e-10  # Replace with the desired threshold
            #pi = ufl.conditional(ufl.ge(pi, pi_min), pi, self.IonModel.Ti0_val*n(u)**self.IonModel.gamma)  # Ensure pi >= pi_min
            pi = ufl.conditional(ufl.ge(pi, pi_min), pi, pi_min)  # Ensure pi >= pi_min
            Ti =  pi/(n(u)+n_min)# Ion temperature
            #Ti_min = 1e-12  # Replace with the desired threshold 
            #Ti = ufl.conditional(ufl.ge(Ti, Ti_min), Ti, self.IonModel.Ti0_val*n(u)**(self.IonModel.gamma-1))  # Ensure Ti >= Ti_min
            
            c = (gamma_i*Ti+gamma_e*self.ElectronModel.Te(n(u)))**0.5
        else:
            c = self.ElectronModel.c(n(u))

        return [un - c, un, un + c]
        

    def rhs(self, u):

        xx = self.x[1]  # Radial coordinate in axisymmetric case

        def vye(u):
            return  -(xx*int(self.axisymmetry) + (1- int(self.axisymmetry)))*self.ElectronModel.dHe_dpsi
        
        if self.energy_eq:
            
            phi = self.ElectronModel.phi(n(u), self.ElectronModel.He)
            pi = self.IonModel.pi(n(u), v_squared(u), E(u), phi) 
             
            return [self.nu*n(u) - int(self.axisymmetry)*n(u)*vxi(u)/xx,
                    -n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bx - int(self.axisymmetry)*n(u)*vxi(u)*vzi(u)/xx,
                    n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bz - int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/xx,
                    int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - int(self.axisymmetry)*2*n(u)*vxi(u)*vyi(u)/xx, 
                    - int(self.axisymmetry)*(E(u) + pi)*vxi(u)/xx]
        else:

            return [self.nu*n(u) -int(self.axisymmetry)*n(u)*vxi(u)/self.x[1],
                    -n(u)*(int(self.electron_mag)*self.ElectronModel.dHe_dpsi + int(self.ion_mag)*vyi(u))*self.Bx - int(self.axisymmetry)*n(u)*vxi(u)*vzi(u)/self.x[1],
                    n(u)*(int(self.electron_mag)*self.ElectronModel.dHe_dpsi + int(self.ion_mag)*vyi(u))*self.Bz - int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/self.x[1],
                    int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - int(self.axisymmetry)*2*n(u)*vxi(u)*vyi(u)/self.x[1]]


    def is_valid(self, u):
        if self.val_check:
            uflat = u.vector().get_local()
            # Check both 0::5 (density) 
            if self.energy_eq:
                return all(uflat[0::5] > 0)
            else:
                return all(uflat[0::4] > 0)
        else:
            return True  # Default case if validation is disabled


    def force_positivity(self, u):
        '''Return the vector of dof with a solution changing al negs to positive.'''
        
        if self.positivity_flag:
            q_vals = u.vector().get_local()
            if self.energy_eq:
                q_vals[0::5][q_vals[0::5] <= 1e-7] = 1e-7
            else:
                q_vals[0::4][q_vals[0::4] <= 1e-7] = 1e-7 # Density
            #q_vals[4::5][q_vals[4::5] <= 1e-7] = 1e-7 # Energy
            
            '''
            IN CASE FORCING POSITIVITY IN THE ENERGY IS NECESSARY - UNCOMMENT THE FOLLOWING BLOCK
            '''

            '''
            # Assuming `u` is defined in a certain FunctionSpace `V`
            W = df.VectorFunctionSpace(self.mesh, 'DG', 0, dim=self.u_dim()) 
            dof_coords = W.tabulate_dof_coordinates()
            dof_coords = dof_coords[0::5]
            He_values = []
            for i in range(len(dof_coords)):  # Iterate over DOF coordinates
                # Define or calculate the value you want to assign
                He_value = self.He(df.Point(dof_coords[i][0], dof_coords[i][1]))
                He_values.append(He_value)
            He_values = np.array(He_values)
            gamma_i = self.IonModel.gamma
            pi_vals = (gamma_i-1)*(q_vals[4::5] -  0.5*q_vals[0::5]*((q_vals[1::5]/(q_vals[0::5]))**2 + (q_vals[2::5]/(q_vals[0::5] ))**2 + (q_vals[3::5]/(q_vals[0::5]))**2) - q_vals[0::5]*self.ElectronModel.phi(q_vals[0::5], He_values))
            pi_min = 1e-6  # Replace with the desired threshold
            pi_vals[pi_vals <= pi_min] = pi_min
            q_vals[4::5] = (1/(gamma_i-1))*pi_vals +  0.5*q_vals[0::5]*((q_vals[1::5]/(q_vals[0::5]))**2 + (q_vals[2::5]/(q_vals[0::5]))**2 + (q_vals[3::5]/(q_vals[0::5]))**2) + q_vals[0::5]*self.ElectronModel.phi(q_vals[0::5], He_values)
            '''

            return q_vals

        else:
            return u.vector().get_local()


    def get_negatives(self, u):

        dens = u.sub(0, deepcopy=True)
        dens_vec = u.sub(0, deepcopy=True).vector()[:]
        dens_vec[dens_vec < 0] = 10

        dens.vector()[:] = dens_vec

        return dens

  
class BCPlasma2D(src.PhysicalModels.BC):
    '''
    Abstract class for plasma boundary conditions.
    '''    
    def v_normal(self,um,normal):
        '''
        Outward normal velocity.
        ''' 
        return vzi(um)*normal[0] + vxi(um)*normal[1]
    def v_parallel(self,um,normal):
        '''
        Parallel velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        ''' 
        return vzi(um)*normal[1] - vxi(um)*normal[0]  
 

class BC_Wall2D(BCPlasma2D):
    '''
    Wall boundary conditions. Also good for symmetry planes.  
    ''' 
    def __init__(self, energy_eq = False): 
        self.energy_eq = energy_eq

    def u_outside(self, um, normal):   
        v_normal  = - self.v_normal(um,normal)
        v_parallel = self.v_parallel(um,normal)

        # u_outside is the same as um except that u_normal has been reversed

        if self.energy_eq:
            return [n(um), 
                    n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                    n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                    n(um)*vyi(um),
                    E(um)] 
        else:
            return [n(um), 
                    n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                    n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                    n(um)*vyi(um)] 


class BC_SupersonicInflow2D(BCPlasma2D):
    '''
    Supersonic inflow boundary conditions in which we prescribe density and velocities. 
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., E_outside = 1., energy_eq = False):  
        
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.E_outside = E_outside #Ion mechanical energy outside
        self.energy_eq = energy_eq

    def u_outside(self, um, normal):   

        if self.energy_eq:
            return [self.n_outside, 
                    self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                    self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                    self.n_outside*self.vyi_outside,
                    self.E_outside] 
        else:
            return [self.n_outside, 
                    self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                    self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                    self.n_outside*self.vyi_outside] 

    
class BC_SupersonicOutflow2D(BCPlasma2D):
    '''
    Supersonic outflow boundary conditions.  
    The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.
    ''' 
    from src.utils.MathAux import max
    
    def __init__(self, ElectronModel, IonModel, energy_eq = False):
        self.energy_eq = energy_eq 
        self.ElectronModel = ElectronModel
        self.IonModel = IonModel

    def u_outside(self, um, normal):  
        if self.energy_eq:

            #gamma_e = self.ElectronModel.gamma
            #gamma_i = self.IonModel.gamma

            #phi = self.ElectronModel.phi(n(um), He_sign)
            #pi = self.IonModel.pi(n(um), v_squared(um), E(um), phi)  # Ion pressure
            #pi_min = 1e-10  # Replace with the desired threshold
            #pi = ufl.conditional(ufl.ge(pi, pi_min), pi, self.IonModel.Ti0_val*n(um)**self.IonModel.gamma)  # Ensure pi >= pi_min
            #pi = ufl.conditional(ufl.ge(pi, pi_min), pi, pi_min)  # Ensure pi >= pi_min
            #Ti =  pi/(n(um)+n_min)# Ion temperature

            #c = (gamma_i*Ti+gamma_e*self.ElectronModel.Te(n(um)))**0.5  # Sound speed
            
            c = self.ElectronModel.c(n(um))  # Sound speed

            v_normal = BC_SupersonicOutflow2D.max(self.v_normal(um, normal), c)  # Outward normal velocity (should be at least c) 
            #v_normal = self.v_normal(um, normal)
            
            v_parallel = self.v_parallel(um, normal)  # Parallel velocity

            return [n(um), 
                    n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                    n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                    n(um)*vyi(um),
                    E(um)] 
        
        else:

            c = self.ElectronModel.c(n(um))  # Sound speed
            v_normal = BC_SupersonicOutflow2D.max(self.v_normal(um, normal), c)  # Outward normal velocity (should be at least c) 
            v_parallel = self.v_parallel(um, normal)  # Parallel velocity

            return [n(um), 
                    n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                    n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                    n(um)*vyi(um)] 

