
''' --- 2D MODEL WITH NEUTRALS, COLLISIONS AND TOTAL ENERGY CONSERVATION --- '''

'''

This module contains the definition of the plasma model with: 
- 2D 
- Neutrals
- Collisions 
- Total energy conservation

The state vector is: u = [n, gzi, gxi, gyi, Ei, nn, gzn, gxn, En]
The source terms include ionization and charge exchange collisions, and the energy equation includes the corresponding energy losses.

The boundary conditions implemented are:
- Wall BCs: the normal velocity of both species is reversed and the parallel and out-of-plane velocities are unchanged.
- Supersonic inflow BCs: density and velocities for both species are prescribed.
- Subsonic inflow BCs: density and velocities for both species and pressure (Energy) for neutrals are prescribed as ions are subsonic.
- Supersonic outflow BCs: the normal velocity of both species is unchanged and the parallel and out-of-plane velocities are extrapolated from the interior. 

'''

# --- IMPORTS --- #

import src

import numpy as np
import ufl_legacy as ufl
from dolfin import project
import dolfin as df
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp 
from scipy.interpolate import griddata
import gc

from scipy.integrate import simpson

from scipy.interpolate import interp1d

n_min = 1e-10 # minimum value of the density, when computing velocities

# --- MODEL DEFINITION AND VARIABLES --- #

# Simple variable extraction functions (If axisymmetric x becomes r  and y becomes theta)

#Ion variables:
def n(u): return ufl.conditional(u[0] > 1e-12, u[0], 1e-12)
def gzi(u): return u[1]
def gxi(u): return u[2]
def gyi(u): return u[3]
def Ei(u): return u[4]
def vzi(u): return gzi(u)/(n(u) + 0.*n_min)
def vxi(u): return gxi(u)/(n(u) + 0.*n_min)
def vyi(u): return gyi(u)/(n(u) + 0.*n_min)
def vi(u): return ufl.as_vector((vzi(u), vxi(u)))
def vi2(u): return ufl.inner(vi(u), vi(u))

#Neutrals Variables:
def nn(u): return ufl.conditional(u[5] > 1e-12, u[5], 1e-12)
def gzn(u): return u[6]
def gxn(u): return u[7]
def En(u): return u[8]
def vzn(u): return gzn(u)/(nn(u) + 0.*n_min)
def vxn(u): return gxn(u)/(nn(u) + 0.*n_min)
def vn(u): return ufl.as_vector((vzn(u), vxn(u)))
def vn2(u): return ufl.inner(vn(u), vn(u))

# Note: the energy equation is written in terms of the total energy density Ei and En, which include both thermal and kinetic energy. The pressure can be obtained as pi = (gamma-1)*(Ei - 0.5*n*vi2) for ions and pn = (gamma-1)*(En - 0.5*nn*vn2) for neutrals.

class PlasmaModel2D_neutrals_E(src.PhysicalModels.PhysicalModel):
    
    '''
    Plasma physical fluxes.  
    The sTate vector is: u = [n, gzi, gxi, gyi, Ei, nn, gzn, gxn, En]
    '''

    def x_dim(self):
        return 2

    def u_dim(self):
        return 9

    def __init__(self, ElectronModel = None, IonModel = None, NeutralModel = None,
                 CollisionModel = None, ElectronSolver = None, 
                 Bz = None, Bx = None, 
                 V_DG = None, V_CG = None, mesh = None,
                 val_check= True, electron_mag = True, ion_mag = True, positivity_flag = False, stabilization_params = (0,0,0), axisymmetry = False):

        self.ElectronModel = ElectronModel
        self.ElectronSolver = ElectronSolver
        self.IonModel = IonModel
        self.NeutralModel = NeutralModel
        self.Bz = Bz
        self.Bx = Bx
        self.CollisionModel = CollisionModel
        self.V_DG = V_DG
        self.V_CG = V_CG
        self.val_check = val_check
        self.electron_mag = electron_mag
        self.ion_mag = ion_mag
        self.stabilization_params = stabilization_params #Must be a tuple, first value is C_eps, the second is beta        
        self.positivity_flag = positivity_flag
        self.axisymmetry = axisymmetry
        self.mesh = mesh
        self.x = ufl.SpatialCoordinate(mesh)
     
    def flux(self, u, sign = None): 
        
        if sign is None:
            He_sign = self.ElectronModel.He
            A_sign = self.ElectronModel.A_Te
        else:
            He_sign = self.ElectronModel.He(sign)
            A_sign = self.ElectronModel.A_Te(sign)

        #A_sign = self.ElectronModel.A_Te_cont
            
        phi = self.ElectronModel.phi(n(u), He_sign, A_sign)
        
        # Compute pressures:
        
        pi = self.IonModel.pi(n(u), vi2(u), Ei(u), phi)
        pi = ufl.conditional(pi > 0, pi, 1e-12)
        pn = self.NeutralModel.pn(nn(u), vn2(u), En(u))  # Neutral pressure
        pn = ufl.conditional(pn > 0, pn, 1e-12)
        pe = self.ElectronModel.pe(n(u), A_sign)  # Electron pressure

        return [[gzi(u),                       gxi(u)],
                [gzi(u)*vzi(u) + pe + pi,      gxi(u)*vzi(u)],
                [gzi(u)*vxi(u),                gxi(u)*vxi(u) + pe + pi],
                [gzi(u)*vyi(u),                gxi(u)*vyi(u)],
                [(Ei(u) + pi)*vzi(u),          (Ei(u) + pi)*vxi(u)],
                #Neutrals flux:
                [gzn(u),                  gxn(u)],
                [gzn(u)*vzn(u) + pn,      gxn(u)*vzn(u)],
                [gzn(u)*vxn(u),           gxn(u)*vxn(u) + pn],
                [(En(u) + pn)*vzn(u),         (En(u) + pn)*vxn(u)]]

    def eigenvalues(self, u, normal, sign = None):   
        un = vzi(u)*normal[0] + vxi(u)*normal[1] # Inplane velocity component along normal
        unn = vzn(u)*normal[0] + vxn(u)*normal[1] # Inplane neutral velocity component along normal

        if sign is None:
            He_sign = self.ElectronModel.He
            A_sign = self.ElectronModel.A_Te
        else:
            He_sign = self.ElectronModel.He(sign)
            A_sign = self.ElectronModel.A_Te(sign)

        #A_sign = self.ElectronModel.A_Te_cont

        phi = self.ElectronModel.phi(n(u), He_sign, A_sign)
        pi = self.IonModel.pi(n(u), vi2(u), Ei(u), phi) #Ion pressure
        pi = ufl.conditional(pi > 0, pi, 1e-12)

        Ti =  self.IonModel.Ti(n(u), vi2(u), Ei(u), phi) #Ion temperature
        Ti = ufl.conditional(Ti > 0, Ti, 1e-12)

        Tn =  self.NeutralModel.Tn(nn(u), vn2(u), En(u)) #Neutral temperature
        Tn = ufl.conditional(Tn > 0, Tn, 1e-12)
        
        # Altough generally negligible, Ti is included, as it will increase under collisional effects
        
        c = ( A_sign*self.ElectronModel.gamma*n(u)**(self.ElectronModel.gamma-1))**0.5
        #c = self.ElectronModel.c(n(u), A_sign) #self.IonModel.c(n(u), (vzi(u)**2 + vxi(u)**2 + vyi(u)**2)**0.5 , Ei(u), phi) # Sound speed for ions
        cn = self.NeutralModel.c(Tn) # Sound speed for neutrals
        #cn = (self.NeutralModel.gamma*nn(u)**(self.NeutralModel.gamma-1))**0.5
    
        return [un - c, un, un + c, unn - cn, unn, unn + cn]
        
    def rhs(self, u):
        #self.collision_param allows to eliminate collisions or switch them on.
        v_i = ufl.as_vector((vzi(u), vxi(u)))
        v_n = ufl.as_vector((vzn(u), vxn(u)))
        
        phi = self.ElectronModel.phi(n(u), self.ElectronModel.He, self.ElectronModel.A_Te)

        Ti =  self.IonModel.Ti(n(u), vi2(u), Ei(u), phi) #Ion temperature
        Ti = ufl.conditional(Ti > 0, Ti, 1e-12)
        Tn =  self.NeutralModel.Tn(nn(u), vn2(u), En(u)) #Neutral temperature
        Tn = ufl.conditional(Tn > 0, Tn, 1e-12)

        pi = self.IonModel.pi(n(u), vi2(u), Ei(u), phi)
        pn = self.NeutralModel.pn(nn(u), vn2(u), En(u))

        nu_i = self.CollisionModel.Ionization(nn(u), self.ElectronModel.Te(n(u), self.ElectronModel.A_Te))
        nu_cex = self.CollisionModel.CEX(nn(u), v_i, v_n, self.ElectronModel.Te(n(u), self.ElectronModel.A_Te))
        S_i = nu_i * n(u)
        S_cex = nu_cex * n(u)

        Ei_no_phi = Ei(u) - n(u)*phi

        Q_i = S_i * En(u)/n(u) + S_cex * (En(u)/nn(u) - Ei_no_phi/n(u)) #Energy loss for ions due to ionization and charge exchange

        xx = self.x[1]

        def dHe_dpsi(u):
            return (xx*int(self.axisymmetry) + (1- int(self.axisymmetry)))*abs(self.ElectronModel.dHe_dpsi)

        def dA_dpsi(u):
            return (xx*int(self.axisymmetry) + (1- int(self.axisymmetry)))*abs(self.ElectronModel.dA_Te_dpsi)   

        def vye(u):
            uye_df = - dHe_dpsi(u) + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n(u)**(self.ElectronModel.gamma - 1) * dA_dpsi(u) #Electron diamagnetic velocity (assuming polytropic electrons)
            uye_df = ufl.conditional(uye_df > 1e-6, uye_df, 1e-6)
            return uye_df

        '''return [# Ions RHS: 
                S_i ,
               -n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bx + S_i*vzn(u) + S_cex*(vzn(u) -  vzi(u)),
               n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bz + S_i*vxn(u) + S_cex*(vxn(u) -  vxi(u))  - n(u)*vyi(u)**2/xx,
               int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) + n(u)*vxi(u)*vyi(u)/xx,
                Q_i + phi*S_i,
               
               # Neutrals RHS:
               - S_i ,
               - S_i*vzn(u) + S_cex*(vzi(u) -  vzn(u)),
               - S_i*vxn(u) + S_cex*(vxi(u) -  vxn(u)),
               - Q_i]'''

        return [# Ions RHS: 
                S_i - int(self.axisymmetry)*gxi(u)/xx,
               -n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bx + S_i*vzn(u) + S_cex*(vzn(u) -  vzi(u)) - int(self.axisymmetry)*vxi(u)*gzi(u)/xx,
               n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bz + S_i*vxn(u) + S_cex*(vxn(u) -  vxi(u)) - int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/xx,
               int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - int(self.axisymmetry)*2*gxi(u)*vyi(u)/xx,
                Q_i + phi*S_i - int(self.axisymmetry)*(Ei(u) + pi)*vxi(u)/xx,
               
               # Neutrals RHS:
               - S_i - int(self.axisymmetry)*gxn(u)/xx,
               - S_i*vzn(u) + S_cex*(vzi(u) -  vzn(u)) - int(self.axisymmetry)*vxn(u)*gzn(u)/xx,
               - S_i*vxn(u) + S_cex*(vxi(u) -  vxn(u)) - int(self.axisymmetry)*nn(u)*(vxn(u)**2)/xx,
                - Q_i - int(self.axisymmetry)*(En(u) + pn)*vxn(u)/xx ]
        
    def neutral_background(self, u):
        background_dens = self.NeutralModel.back_dens

        q_vals = u.vector().get_local()
        q_vals[5::9][q_vals[5::9] <= background_dens] = background_dens

        return q_vals

    def is_valid(self, u):
        if self.val_check:
            uflat = u.vector().get_local() # Ensure that u is a 1D np.array
            return all(uflat[0::9] > 0) # Density must be positive
        else:
            return True

    def force_positivity(self, u):
        '''Return the vector of dof with a solution changing all negs to positive.'''
        
        if self.positivity_flag:
            q_vals = u.vector().get_local()
            q_vals[0::9][q_vals[0::9] <= 1e-12] = 1e-12
            return q_vals
        else:
            return u.vector().get_local()

    def get_negatives(self, u):

        dens = u.sub(0, deepcopy=True)
        dens_vec = u.sub(0, deepcopy=True).vector()[:]
        dens_vec[dens_vec < 0] = 10

        dens.vector()[:] = dens_vec

        return dens
    
class BCPlasma2D_neutrals_E(src.PhysicalModels.BC):
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
        Parallel neutral velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        ''' 
        return vzn(um)*normal[1] - vxn(um)*normal[0]  
 

class BC_Wall2D_neutrals_E(BCPlasma2D_neutrals_E):
    '''
    Wall boundary conditions. Also good for symmetry planes.  
    ''' 

    def __init__(self, strong_imposition = False):  
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)
    
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
                Ei(um),
                nn(um),
                nn(um)*(vn_normal*normal[0] + vn_parallel*normal[1]), 
                nn(um)*(vn_normal*normal[1] - vn_parallel*normal[0]),
                En(um)]


class BC_SupersonicInflow2D_neutrals_E(BCPlasma2D_neutrals_E):
    '''
    Supersonic inflow boundary conditions in which we prescribe density and velocities. 
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., Ei_outside = 0., nn_outside = 1., vn_normal_outside = 1.1, vn_parallel_outside = 0., En_outside = 0, strong_imposition = False):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.Ei_outside = Ei_outside #Total Ion Energy outside
        self.nn_outside = nn_outside
        self.vn_normal_outside = -vn_normal_outside
        self.vn_parallel_outside = vn_parallel_outside
        self.En_outside = En_outside #Total Ion Energy outside
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def Fn_outside(self, um, normal):
        
        return 

    def u_outside(self, um, normal):

        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                self.n_outside*self.vyi_outside,
                self.Ei_outside,
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel_outside*normal[1]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] - self.vn_parallel_outside*normal[0]),
                self.En_outside] 

class BC_SubsonicInflow2D_neutrals_E(BCPlasma2D_neutrals_E):
    '''
    Subsonic inflow boundary conditions in which we prescribe density and velocities for both species and pressure (Energy) for neutrals as ions are subsonic.
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., nn_outside = 1., vn_normal_outside = 1.1, vn_parallel_outside = 0., En_outside = 0, strong_imposition = False):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.nn_outside = nn_outside
        self.vn_normal_outside = -vn_normal_outside
        self.vn_parallel_outside = vn_parallel_outside
        self.En_outside = En_outside #Total Neutral Energy outside
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)
    def Fn_outside(self, um, normal):
        
        return 

    def u_outside(self, um, normal):

        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                self.n_outside*self.vyi_outside,
                Ei(um),
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel_outside*normal[1]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] - self.vn_parallel_outside*normal[0]),
                self.En_outside] 

    
class BC_SupersonicOutflow2D_neutrals_E(BCPlasma2D_neutrals_E):
    '''
    Supersonic outflow boundary conditions.  
    The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.
    ''' 
    from src.utils.MathAux import max
    
    def __init__(self, ElectronModel, IonModel, NeutralModel, strong_imposition = False): 
        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel
        self.IonModel = IonModel
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def u_outside(self, um, normal):  

        Tn =  self.NeutralModel.Tn(nn(um), vn2(um), En(um)) #Neutral temperature
        #Tn = 0.0030169999999999997*nn(um)**(gamma_n - 1) # Neutral temperature from polytropic relation
        #Ti = self.IonModel.Ti(n(um), vi2(um), Ei(um), self.ElectronModel.phi(n(um), self.ElectronModel.He, self.ElectronModel.A_Te)) #Ion temperature
        phi = self.ElectronModel.phi(n(um), self.ElectronModel.He, self.ElectronModel.A_Te)
        Ti =  self.IonModel.Ti(n(um), vi2(um), Ei(um), phi) #Ion temperature
        Ti = ufl.conditional(Ti > 0, Ti, 1e-12)

        #c = self.ElectronModel.c(n(um), self.ElectronModel.A_Te) #self.IonModel.c(n(um), (vzi(um)**2 + vxi(um)**2 + vyi(um)**2)**0.5 , Ei(um), phi) # Sound speed for ions
        c = (self.IonModel.gamma*Ti + self.ElectronModel.A_Te*self.ElectronModel.gamma*n(um)**(self.ElectronModel.gamma-1))**0.5
        cn = self.NeutralModel.c(Tn) # Sound speed for neutrals

        v_normal = BC_SupersonicOutflow2D_neutrals_E.max(self.v_normal(um, normal), c)  # Outward normal velocity (should be at least c)
        #v_normal = self.v_normal(um, normal)  # Normal velocity
        v_parallel = self.v_parallel(um, normal)  # Parallel velocity
        vn_normal = BC_SupersonicOutflow2D_neutrals_E.max(self.vn_normal(um, normal), cn)  # Outward normal velocity (should be at least cn) 
        #vn_normal = self.vn_normal(um, normal)  # Normal velocity
        vn_parallel = self.vn_parallel(um, normal)  # Parallel velocity

        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel*normal[1]), 
                n(um)*(v_normal*normal[1] - v_parallel*normal[0]),
                n(um)*vyi(um),
                Ei(um),
                nn(um), 
                nn(um)*(vn_normal*normal[0] + vn_parallel*normal[1]), 
                nn(um)*(vn_normal*normal[1] - vn_parallel*normal[0]),
                En(um)]

