
''' --- 2D MODEL WITH NEUTRALS AND COLLISIONS --- '''

'''

This module contains the definition of the plasma model with: 
- 2D 
- Neutrals
- Collisions 

The state vector is: u = [n, gzi, gxi, gyi, nn, gzn, gxn]
The source terms include ionization and charge exchange collisions.

The boundary conditions implemented are:
- Wall BCs: the normal velocity of both species is reversed and the parallel and out-of-plane velocities are unchanged.
- Supersonic inflow BCs: density and velocities for both species are prescribed.
- Subsonic inflow BCs: density and velocities for both species and pressure (Energy) for neutrals are prescribed as ions are subsonic.
- Supersonic outflow BCs: the normal velocity of both species is unchanged and the parallel and out-of-plane velocities are extrapolated from the interior. 

'''

import numpy as np
import ufl_legacy as ufl

import src

n_min = 1e-10 # minimum value of the density, when computing velocities

# Simple variable extraction functions (If axisymmetric x becomes r  and y becomes theta)
#Ion variables:
def n(u): return ufl.conditional(u[0] > 1e-10, u[0], 1e-10)
def gzi(u): return u[1]
def gxi(u): return u[2]
def gyi(u): return u[3]
def vzi(u): return gzi(u)/(n(u) + 0*n_min)
def vxi(u): return gxi(u)/(n(u) + 0*n_min)
def vyi(u): return gyi(u)/(n(u) + 0*n_min)
#Neutrals Variables:
def nn(u): return ufl.conditional(u[4] > 1e-10, u[4], 1e-10)
def gzn(u): return u[5]
def gxn(u): return u[6]
def En(u): return u[7]
def vzn(u): return gzn(u)/(nn(u) + 0*n_min)
def vxn(u): return gxn(u)/(nn(u) + 0*n_min)


class PlasmaModel2D_neutrals(src.PhysicalModels.PhysicalModel):
    '''
    Plasma physical fluxes.  
    The state vector is: u = [n, gzi, gxi, gyi]
    '''
    def x_dim(self):
        return 2

    def u_dim(self):
        return 7
 
    def __init__(self, ElectronModel, NeutralModel, Bz, Bx, He, dHe_dpsi, CollisionModel, energy_eq=False, val_check= True, electron_mag = True, ion_mag = True, positivity_flag = False, stabilization_params = (0,0), axisymmetry = False):

        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel
        self.Bz = Bz
        self.Bx = Bx
        self.He = He
        self.dHe_dpsi = dHe_dpsi
        self.CollisionModel = CollisionModel
        self.energy_eq = energy_eq
        self.val_check = val_check
        self.electron_mag = electron_mag
        self.ion_mag = ion_mag
        self.stabilization_params = stabilization_params #Must be a tuple, first value is C_eps, the second is beta        self.positivity_flag = positivity_flag
        self.axisymmetry = axisymmetry
        self.x = ufl.SpatialCoordinate(dHe_dpsi.function_space().mesh())
     
    def flux(self, u, He_sign = None): 
        def nn(u):  return ufl.conditional(u[4] > self.NeutralModel.back_dens, u[4], self.NeutralModel.back_dens)
        pe = self.ElectronModel.pe(n(u))  # Electron pressure
        pn = self.NeutralModel.pn(nn(u)) # Ion pressure 

        return [[gzi(u),             gxi(u)],
                [gzi(u)*vzi(u) + pe, gxi(u)*vzi(u)],
                [gzi(u)*vxi(u),      gxi(u)*vxi(u) + pe],
                [gzi(u)*vyi(u),      gxi(u)*vyi(u)], 
                [gzn(u),            gxn(u)],
                [gzn(u)*vzn(u) + pn,    gxn(u)*vzn(u)],
                [gzn(u)*vxn(u),    gxn(u)*vxn(u) + pn]]
             
    def eigenvalues(self, u, normal, He_sign):   
        un = vzi(u)*normal[0] + vxi(u)*normal[1] # Inplane velocity component along normal
        unn = vzn(u)*normal[0] + vxn(u)*normal[1] # Inplane neutral velocity component along normal
        c = self.ElectronModel.c(n(u)) # Sound speed for electrons
        cn = self.NeutralModel.c(nn(u)) # Sound speed for neutrals
        return [un - c, un, un + c, unn - cn, unn, unn + cn]
        
    def rhs(self, u):
        #self.collision_param allows to eliminate collisions or switch them on.
        
        nu_i = self.CollisionModel.Ionization(nn(u), self.ElectronModel.Te(n(u)))
        vi = ufl.as_vector((vzi(u), vxi(u)))
        vn = ufl.as_vector((vzn(u), vxn(u)))
        nu_cex = self.CollisionModel.CEX(nn(u), vi, vn, self.ElectronModel.Te(n(u)))
        S_i = nu_i *n(u)
        S_cex = nu_cex * n(u)
        return [S_i - int(self.axisymmetry)*n(u)*vxi(u)/(self.x[1]),
               -n(u)*(int(self.electron_mag)*self.dHe_dpsi + int(self.ion_mag)*vyi(u))*self.Bx + S_i*vzn(u) + S_cex*(vzn(u) -  vzi(u)) - int(self.axisymmetry)*n(u)*vxi(u)*vzi(u)/(self.x[1]),
               n(u)*(int(self.electron_mag)*self.dHe_dpsi + int(self.ion_mag)*vyi(u))*self.Bz + S_i*vxn(u) + S_cex*(vxn(u) -  vxi(u)) - int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/(self.x[1]),
               int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - int(self.axisymmetry)*2*n(u)*vxi(u)*vyi(u)/(self.x[1]),
               - S_i - int(self.axisymmetry)*nn(u)*vxn(u)/(self.x[1]),
               - S_i*vzn(u) + S_cex*(vzi(u) -  vzn(u)) - int(self.axisymmetry)*nn(u)*vxn(u)*vzn(u)/(self.x[1]),
               - S_i*vxn(u) + S_cex*(vxi(u) -  vxn(u)) - int(self.axisymmetry)*nn(u)*(vxn(u)**2)/(self.x[1])]

    def neutral_background(self, u):
        background_dens = self.NeutralModel.back_dens

        q_vals = u.vector().get_local()
        q_vals[4::7][q_vals[4::7] <= background_dens] = background_dens

        return q_vals

    def is_valid(self, u):
        if self.val_check:
            uflat = u.vector().get_local() # Ensure that u is a 1D np.array
            return all(uflat[0::7] > 0) # Density must be positive
        else:
            return True

    def force_positivity(self, u):
        '''Return the vector of dof with a solution changing all negs to positive.'''
        
        q_vals = u.vector().get_local()
        q_vals[0::8][q_vals[0::8] <= 1e-8] = 1e-8

        return q_vals
    
    def get_negatives(self, u):

        dens = u.sub(0, deepcopy=True)
        dens_vec = u.sub(0, deepcopy=True).vector()[:]
        dens_vec[dens_vec < 0] = 10

        dens.vector()[:] = dens_vec

        return dens

  
class BCPlasma2D_neutrals(src.PhysicalModels.BC):
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
    
    def vn_normal(self,um,normal):
        '''
        Outward neutral normal velocity.
        ''' 
        return vzn(um)*normal[0] + vxn(um)*normal[1]
    
    def vn_parallel(self,um,normal):
        '''
        Parallel normal velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        ''' 
        return vzn(um)*normal[1] - vxn(um)*normal[0]  
 

class BC_Wall2D_neutrals(BCPlasma2D_neutrals):
    '''
    Wall boundary conditions. Also good for symmetry planes.  
    ''' 
    def u_outside(self, um, normal):   
        v_normal  = - self.v_normal(um,normal)
        v_parallel = self.v_parallel(um,normal)
        vn_normal  = - self.vn_normal(um,normal)
        vn_parallel = self.vn_parallel(um,normal)



        # u_outside is the same as um except that u_normal has been reversed
        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                n(um)*vyi(um), 
                nn(um),
                nn(um)*(vn_normal*normal[0] + vn_parallel*normal[1]), 
                nn(um)*(vn_normal*normal[1] - vn_parallel*normal[0])] 


class BC_SupersonicInflow2D_neutrals(BCPlasma2D_neutrals):
    '''
    Supersonic inflow boundary conditions in which we prescribe density and velocities. 
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., nn_outside = 1., vn_normal_outside = 1.1, vn_parallel_outside = 0.):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.nn_outside = nn_outside
        self.vn_normal_outside = -vn_normal_outside
        self.vn_parallel_outside = vn_parallel_outside

    def u_outside(self, um, normal):   
        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                self.n_outside*self.vyi_outside,
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel_outside*normal[1]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] - self.vn_parallel_outside*normal[0])] 

    
class BC_SupersonicOutflow2D_neutrals(BCPlasma2D_neutrals):
    '''
    Supersonic outflow boundary conditions.  
    The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.
    ''' 
    from src.utils.MathAux import max
    
    def __init__(self, ElectronModel, NeutralModel): 
        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel

    def u_outside(self, um, normal):  
        c = self.ElectronModel.c(n(um))  # Sound speed
        cn = self.NeutralModel.c(nn(um))# Sound speed for neutrals
        v_normal = BC_SupersonicOutflow2D_neutrals.max(self.v_normal(um, normal), c)# Outward normal velocity (should be at least c) 
        v_parallel = self.v_parallel(um, normal)  # Parallel velocity
        vn_normal = BC_SupersonicOutflow2D_neutrals.max(self.vn_normal(um, normal), cn)# Outward normal velocity (should be at least cn) 
        vn_parallel = self.vn_parallel(um, normal)  # Parallel velocity

        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                n(um)*vyi(um),
                nn(um), 
                nn(um)*(vn_normal*normal[0] + vn_parallel*normal[1]), 
                nn(um)*(vn_normal*normal[1] - vn_parallel*normal[0])] 
