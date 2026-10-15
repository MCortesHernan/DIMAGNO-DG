
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

        error_axis_1 = 1.1
        error_axis_2 = 1.1

        return [# Ions RHS: 
                S_i - error_axis_1*int(self.axisymmetry)*gxi(u)/xx,
               -n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bx + S_i*vzn(u) + S_cex*(vzn(u) -  vzi(u)) - error_axis_2*int(self.axisymmetry)*vxi(u)*gzi(u)/xx,
               n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bz + S_i*vxn(u) + S_cex*(vxn(u) -  vxi(u)) - error_axis_2*int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/xx,
               int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - error_axis_2*int(self.axisymmetry)*2*gxi(u)*vyi(u)/xx,
                Q_i + phi*S_i - error_axis_1*int(self.axisymmetry)*(Ei(u) + pi)*vxi(u)/xx,
               
               # Neutrals RHS:
               - S_i - error_axis_1*int(self.axisymmetry)*gxn(u)/xx,
               - S_i*vzn(u) + S_cex*(vzi(u) -  vzn(u)) - error_axis_2*int(self.axisymmetry)*vxn(u)*gzn(u)/xx,
               - S_i*vxn(u) + S_cex*(vxi(u) -  vxn(u)) - error_axis_2*int(self.axisymmetry)*nn(u)*(vxn(u)**2)/xx,
                - Q_i - error_axis_1*int(self.axisymmetry)*(En(u) + pn)*vxn(u)/xx ]
        
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



class ElectronSolver:
    def __init__(self, z_grid, x_grid, z_len, x_len, V_CG, V_DG, BField2D, ElectronModel, CollisionModel,
                 throat_lim=[0, 1], N_l=200, N_psi=500, je_condition='throat_ambipolarity', phi_w = 0.0):
        self.z_grid = z_grid
        self.x_grid = x_grid
        self.z_len = z_len
        self.x_len = x_len
        self.V_CG = V_CG
        self.V_DG = V_DG
        self.BField2D = BField2D
        self.throat_lim = throat_lim
        self.N_l = N_l
        self.N_psi = N_psi
        self.ElectronModel = ElectronModel
        self.gamma_e = float(self.ElectronModel.gamma)
        self.CollisionModel = CollisionModel
        self.je_condition = je_condition
        self.phi_w = phi_w

        # Cache to store streamlines and avoid recomputation
        self._streamline_cache = {}
    
    def A_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):

        pts = streamline[:, :2]
        # Numerical samples along the streamline (arrays)
        n_s   = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s  = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        jpare_s = np.array([self.ElectronModel.jpare_cont(df.Point(z, x)) for z, x in pts])
        dHe_dpsi_s = np.array([self.ElectronModel.dHe_dpsi_cont(df.Point(z, x)) for z, x in pts])
        dA_Te_dpsi_s = np.array([self.ElectronModel.dA_Te_dpsi_cont(df.Point(z, x)) for z, x in pts])
        ue_par_s = - (jpare_s) / (n_s + 1e-5)
        x_s = pts[:, 1]
        ue_y_s = -x_s*dHe_dpsi_s + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n_s**(self.ElectronModel.gamma - 1) * x_s* dA_Te_dpsi_s #Electron diamagnetic velocity (assuming polytropic electrons)
        ue_s = (ue_par_s**2 + ue_y_s**2)**0.5

        E_i_ = 12.1 / 10.0
        gamma = self.gamma_e

        valid = (
            np.isfinite(n_s) &
            np.isfinite(nn_s) &
            np.isfinite(ue_s) &
            np.isfinite(ue_par_s)
        )

        n_s = n_s[valid]
        nn_s = nn_s[valid]
        ue_s = ue_s[valid]
        s = s[valid]
        ue_par_s = ue_par_s[valid]


        # Interpolants (numeric)
        n_interp = interp1d(s, n_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_interp = interp1d(s, ue_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        nn_interp = interp1d(s, nn_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_par_interp = interp1d(s, ue_par_s, kind='linear', fill_value="extrapolate", assume_sorted=True)

        def rhs(ds, A):
            # enforce scalar numeric A
            A = float(A)
            A = max(A, 1e-12)

            # safe interpolated values
            n_i = float(n_interp(ds))
            ue_i = float(ue_interp(ds))
            ue_par_i = float(ue_par_interp(ds))
            nn_i = float(nn_interp(ds))
            # clamp / protect interpolated values
            n_i = max(n_i, 1e-12)
            ue_par_i = ue_par_i

            Te = self.ElectronModel.Te(n_i, A)
            Te = max(Te, 1e-3)  # Avoid unphysically low temperatures that can cause issues in collision rates
            Te = min(Te, 1)  # Avoid unphysically high temperatures that can cause overflow in exp

            # safe inelastic energy term (cap exponent)
            exp_arg = (2.0 * E_i_) / (3.0 * Te + 1e-12)
            exp_arg = np.clip(exp_arg, -50, 50)
            E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

            ioniz = float(self.CollisionModel.Ionization(nn_i, Te, lib=np))
            nu_en = float(self.CollisionModel.E_N(nn_i, Te, self.CollisionModel.me, np))
            nu_ei = float(self.CollisionModel.E_I(nn_i, Te, self.CollisionModel.me, np))
            nu_exc = nu_ei*float(self.CollisionModel.Excitation(nn_i, Te, np)*(self.CollisionModel.Te0/self.CollisionModel.E_ion))
            nu_e = (nu_ei + nu_en + nu_exc)

            #rhs_val = - ioniz/(ue_i + 1e-12) * A * (gamma + (gamma-1) * E_inel/(Te + 1e-12) )
            rhs_val_1 = - ioniz*(gamma/(gamma-1)*Te + E_inel)
            rhs_val_2 = nu_e * self.CollisionModel.me * ue_i**2
            rhs_val = (gamma-1)/(ue_par_i * Te + 1e-12) * (rhs_val_1 + rhs_val_2)
            #rhs_val = max(rhs_val, -1e6)  # Avoid excessively large negative values that can cause solver issues
            #rhs_val = min(rhs_val, 1e6)   # Avoid excessively large positive values that can cause solver issues

            # Avoid NaN/infs
            if not np.isfinite(rhs_val):
                # clamp to a safe large value (or zero)
                ioniz = float(self.CollisionModel.Ionization(1e-5, Te, lib=np))
                rhs_val = - ioniz/(ue_i + 1e-12) * A * (gamma + (gamma-1) * E_inel/(Te + 1e-12) )

            return float(rhs_val)

        # initial Te0: use analytic or numeric safe scalar
        A0 = 1

        # Use a stiff solver if collisions make it stiff
        try:
            sol = solve_ivp(rhs, (s[0], s[-1]), [A0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        except Exception:
            # fallback to RK45 if BDF raises problems
            sol = solve_ivp(rhs, (s[0], s[-1]), [A0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)

        A_out = np.maximum(sol.y[0], 1e-12)
        A_out = np.minimum(A_out, 3)  # Avoid unphysically high A that can cause overflow in exp


        if A_out.shape[0] != s.shape[0]:
            A_out = np.interp(s, sol.t, A_out, left=A_out[0], right=A_out[-1])

        if not sol.success:
            print(f"Integration failed or produced NaN at streamline. Filling with A0.")
            A_out = np.full_like(s, np.nan)

        # cleanup
        del n_s, ue_par_s, nn_s
        del n_interp, ue_interp, nn_interp

        gc.collect()

        return s, A_out
    
    def He_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):
        
        pts = streamline[:, :2]
        n_s = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        A_Te_s = np.array([self.ElectronModel.A_Te_cont(df.Point(z, x)) for z, x in pts])
        dA_Te_dpsi_s = np.array([self.ElectronModel.dA_Te_dpsi_cont(df.Point(z, x)) for z, x in pts])
        dHe_dpsi_s = np.array([self.ElectronModel.dHe_dpsi_cont(df.Point(z, x)) for z, x in pts])
        x_s = pts[:, 1]
        ue_y_s = -x_s*dHe_dpsi_s + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n_s**(self.ElectronModel.gamma - 1) * x_s* dA_Te_dpsi_s #Electron diamagnetic velocity (assuming polytropic electrons)
        jpare_s = np.array([self.ElectronModel.jpare_cont(df.Point(z, x)) for z, x in pts])
        ue_par_s = - (jpare_s) / (n_s + 1e-5)
        n_s = np.maximum(n_s, 1e-6)

        E_i_ = 12.1 / 10.0        

        valid = (
        np.isfinite(n_s) &
        np.isfinite(nn_s) &
        np.isfinite(ue_y_s) &
        np.isfinite(A_Te_s) &
        np.isfinite(ue_par_s)
        )

        n_s = n_s[valid]
        nn_s = nn_s[valid]
        ue_y_s = ue_y_s[valid]
        s = s[valid]
        A_Te_s = A_Te_s[valid]
        ue_par_s = ue_par_s[valid]

        n_interp = interp1d(s, n_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        uye_interp = interp1d(s, ue_y_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        nn_interp = interp1d(s, nn_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        A_interp = interp1d(s, A_Te_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_par_interp = interp1d(s, ue_par_s, kind='linear', fill_value="extrapolate", assume_sorted=True)

        He0 = 6 #from value of function to avoid evaluating and introducing fenics errors

        def rhs(ds, He):

            # safe interpolated values
            n_i = float(n_interp(ds))
            uye_i = float(uye_interp(ds))
            nn_i = float(nn_interp(ds))
            A_i = float(A_interp(ds))
            ue_par_i = float(ue_par_interp(ds))

            Te = self.ElectronModel.Te(n_i, A_i)
            Te = max(Te, 1e-3)  # Avoid unphysically low temperatures that can cause issues in collision rates
            Te = min(Te, 1)  # Avoid unphysically high temperatures that can cause overflow in exp
            gamma = self.gamma_e

            nu_i = float(self.CollisionModel.Ionization(nn_i, Te, lib=np))
            nu_en = float(self.CollisionModel.E_N(nn_i, Te, self.CollisionModel.me, lib=np))
            nu_ei = float(self.CollisionModel.E_I( n_i, Te, self.CollisionModel.me, lib=np))
            nu_exc = nu_i*float(self.CollisionModel.Excitation(n_i, Te, lib=np))*(self.CollisionModel.Te0/self.CollisionModel.E_ion)

            nu_e = (nu_ei + nu_en + nu_i + nu_exc)
            
            exp_arg = (2.0 * E_i_) / (3.0 * Te + 1e-12)
            # hard numerical cap
            exp_arg = np.clip(exp_arg, -50, 50)
            E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

            rhs_val_1 = - nu_i * (gamma/(gamma - 1)*Te + E_inel)
            rhs_val_2 = - nu_e * self.CollisionModel.me * uye_i**2
        
            rhs_val = 1/(ue_par_i + 1e-12) * (rhs_val_1 + rhs_val_2)

            if not np.isfinite(rhs_val):
                # clamp to a safe large value (or zero)
                rhs_val = 1e-8

            rhs_val = np.real(rhs_val)
            
            return float(rhs_val)
            
        try:
            sol = solve_ivp(rhs, (s[0], s[-1]), [He0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        except Exception:
            # fallback to RK45 if BDF raises problems
            sol = solve_ivp(rhs, (s[0], s[-1]), [He0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)

        He_out = np.maximum(sol.y[0], 1e-12)


        if He_out.shape[0] != s.shape[0]:
            He_out = np.interp(s, sol.t, He_out, left=He_out[0], right=He_out[-1])

        if not sol.success:
            print(f"Integration failed or produced NaN at streamline. Filling with He0.")
            He_out = np.full_like(s, np.nan)
        
        del n_s, A_Te_s, ue_par_s
        del n_interp, nn_interp, A_interp, uye_interp, ue_par_interp
        gc.collect()

        return s, He_out
    
    def jn_sheath_model(self, z, x, n_df, phi_w):
        Te = self.ElectronModel.Te(n_df(df.Point(z,x)), self.ElectronModel.A_Te(df.Point(z,x)))
        c_se = (8*Te/(np.pi*self.CollisionModel.me))**0.5
        phi = self.ElectronModel.phi(n_df(df.Point(z,x)), self.ElectronModel.He(df.Point(z,x)), self.ElectronModel.A_Te(df.Point(z,x)))
        n_w = n_df(df.Point(z,x))*np.exp((phi_w - phi)/Te)
        jne = n_w*c_se/(4)

        return jne
    
    def col_i_current(self, z, x, jzi_df, jxi_df):
        jzi = jzi_df
        jxi = jxi_df
        downst_array = np.array([jzi(z[-1], x_i) for x_i in x])
        lat_array = np.array([jxi(z_i, x[-1]) for z_i in z])
        downst_bound = 2*np.pi*simpson(x*downst_array, x)
        lat_bound = 2*np.pi*simpson(x[-1]*lat_array, z)

        collected_ion_curr = downst_bound + lat_bound

        return collected_ion_curr

    def col_e_current(self, z, x, n_df, phi_w):
        jn_d = np.array([self.jn_sheath_model(z[-1], x_i, n_df, phi_w) for x_i in x])
        downst_bound = 2*np.pi*simpson(x*jn_d, x)

        jn_l =  np.array([self.jn_sheath_model(z_i, x[-1], n_df, phi_w) for z_i in z])
        lat_bound = 2*np.pi*simpson(x[-1]*jn_l, z)

        collected_e_curr = downst_bound + lat_bound

        return - collected_e_curr
    
    def zero_finder(self, phi_w_0, phi_w_1, n_df, jzi_df, jxi_df, z_s, x_s):
        '''This finds the wall potential that returns 0 collected current.'''
        eps = 1e-6
        collected_curr_0 = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_0)
        collected_curr_1 = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_1)
        if collected_curr_0 * collected_curr_1 > 0:
            print('Initial potentials must be chosen so that a zero exists in the middle. \n')
            print(f'Collected curents for the chosen initial potentials are: {collected_curr_0} and {collected_curr_1}.\n')
            return None
        else:
            iter = 0
            while True:
                phi_w_aux = 0.5*(phi_w_1 + phi_w_0)
                collected_curr_aux = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_aux)
                print(f'Potential at iteration {iter}: {phi_w_aux}')
                print(f'Collected current: {collected_curr_aux}\n')
                
                if abs(collected_curr_aux ) < eps:
                    break

                if collected_curr_aux * collected_curr_0 > 0:
                    phi_w_0 = phi_w_aux
                else:
                    phi_w_1= phi_w_aux   
                iter += 1
                
            return phi_w_aux
    
    def j_par_e_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                                n_df, nn_df, jzi_df, jxi_df, axis_df):
        pts = streamline[:, :2]
        n_s = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        jzi_s = np.array([jzi_df(df.Point(z, x)) for z, x in pts])
        jxi_s = np.array([jxi_df(df.Point(z, x)) for z, x in pts])
        A_Te_s = np.array([self.ElectronModel.A_Te_cont(df.Point(z, x)) for z, x in pts])
        Te_s = self.ElectronModel.Te(n_s, A_Te_s)
        Si_s = self.CollisionModel.Ionization(nn_s, Te_s, lib=np) * n_s
        axis_s = np.array([axis_df(df.Point(z, x)) for z, x in pts])

        bz_s = np.array([self.BField2D.Bz(z, x) for z, x in pts])
        bx_s = np.array([self.BField2D.Br(z, x) for z, x in pts])
        Bnorm = np.sqrt(bz_s**2 + bx_s**2)
        Bnorm[Bnorm == 0.0] = 1e-16

        bz_hat = bz_s / Bnorm
        bx_hat = bx_s / Bnorm
        j_par_i = jzi_s * bz_hat + jxi_s * bx_hat

        valid = (
        np.isfinite(Si_s) &
        np.isfinite(j_par_i) &
        np.isfinite(Bnorm)
        )

        Si_s = Si_s[valid]
        j_par_i = j_par_i[valid]
        Bnorm = Bnorm[valid]
        s = s[valid]

        Si_interp = interp1d(s, Si_s, kind='linear', fill_value="extrapolate")
        Bnorm_interp = interp1d(s, Bnorm, kind='linear', fill_value="extrapolate")
        axis_interp = interp1d(s, axis_s, kind='linear', fill_value="extrapolate")

        def rhs(ds, Y):
            val = 1/(Bnorm_interp(ds) + 1e-8) * (Si_interp(ds) + axis_interp(ds)) 
            val = float(val)
            return val

        if self.je_condition == 'throat_ambipolarity':
            Y0 = - j_par_i[0]/(Bnorm[0]+1e-8)          
            try:
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        
            j_par_e_out = sol.y[0] * Bnorm

            if j_par_e_out.shape[0] != s.shape[0]:
                j_par_e_out = np.interp(s, sol.t, j_par_e_out, left=j_par_e_out[0], right=j_par_e_out[-1])

            if not sol.success or np.any(~np.isfinite(j_par_e_out)):
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                j_par_e_out = np.full_like(s, np.nan)

        elif self.je_condition == 'wall_ambipolarity':
            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len
            
            if abs(pts[-1,0]) < eps:
                jn = jzi_s[-1] 
                j_e_par_bc = - jn / bz_hat[-1]
            elif abs(diff_x) < eps: 
                jn = jxi_s[-1]
                j_e_par_bc = - jn / bx_hat[-1]
            elif abs(diff_z) < eps:
                jn = jzi_s[-1]
                j_e_par_bc = - jn / bz_hat[-1]
            else:
                j_e_par_bc = - j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose ambipolarity at WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            # integrate from wall → throat
            sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev,
                            method='RK45', rtol=1e-8, atol=1e-8)

            # reverse solution back to original order
            Y_s = sol.y[0][::-1]
            j_par_e_out = Y_s * Bnorm
        
        elif self.je_condition == 'metal_wall':

            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len

            if abs(pts[-1,0]) < eps:
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = - jn / bz_hat[-1]
            elif abs(diff_x) < eps: 
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = - jn / bx_hat[-1]
            elif abs(diff_z) < eps:
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = - jn / bz_hat[-1]
            else:
                j_e_par_bc = j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose ambipolarity at WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            # integrate from wall → throat
            sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev,
                            method='RK45', rtol=1e-8, atol=1e-8)

            # reverse solution back to original order
            Y_s = sol.y[0][::-1]
            j_par_e_out = Y_s * Bnorm
            
        else:
            raise ValueError(f"Unknown je_condition: {self.je_condition}")

        del n_s, nn_s, jzi_s, jxi_s, A_Te_s, Te_s, Si_s
        del bz_s, bx_s, Bnorm, bz_hat, bx_hat, j_par_i
        del Si_interp, Bnorm_interp
        gc.collect()

        return s, j_par_e_out

    def G_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                                n_df, nn_df, jzi_df, jxi_df):
        pts = streamline[:, :2]
        n_s = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        jzi_s = np.array([jzi_df(df.Point(z, x)) for z, x in pts])
        jxi_s = np.array([jxi_df(df.Point(z, x)) for z, x in pts])
        A_Te_s = np.array([self.ElectronModel.A_Te_cont(df.Point(z, x)) for z, x in pts])
        Te_s = self.ElectronModel.Te(n_s, A_Te_s)
        Si_s = self.CollisionModel.Ionization(nn_s, Te_s, lib=np) * n_s

        bz_s = np.array([self.BField2D.Bz(z, x) for z, x in pts])
        bx_s = np.array([self.BField2D.Br(z, x) for z, x in pts])
        Bnorm = np.sqrt(bz_s**2 + bx_s**2)
        Bnorm[Bnorm == 0.0] = 1e-12

        bz_hat = bz_s / Bnorm
        bx_hat = bx_s / Bnorm
        j_par_i = jzi_s * bz_hat + jxi_s * bx_hat

        valid = (
        np.isfinite(Si_s) &
        np.isfinite(j_par_i) &
        np.isfinite(Bnorm)
        )

        Si_s = Si_s[valid]
        j_par_i = j_par_i[valid]
        Bnorm = Bnorm[valid]
        s = s[valid]

        Si_interp = interp1d(s, Si_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        Bnorm_interp = interp1d(s, Bnorm, kind='linear', fill_value="extrapolate", assume_sorted=True)

        def rhs(ds, Y):
            val = 1/(Bnorm_interp(ds) + 1e-8) * (Si_interp(ds)) 
            val = float(val)
            return val

        if self.je_condition == 'throat_ambipolarity':
            Y0 = j_par_i[0]/(Bnorm[0]+1e-8)          
            try:
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        
            G_out = sol.y[0] 

            if G_out.shape[0] != s.shape[0]:
                G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

            if not sol.success:
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                G_out = np.full_like(s, np.nan)
        
        elif self.je_condition == 'metal_wall':

            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len

            if abs(pts[-1,0]) < eps:
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = jn / bz_hat[-1]
            elif abs(diff_x) < eps: 
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = jn / bx_hat[-1]
            elif abs(diff_z) < eps:
                jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                j_e_par_bc = jn / bz_hat[-1]
            else:
                j_e_par_bc = j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose ambipolarity at WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            # integrate from wall → throat

            try:
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='BDF',
                            rtol=1e-6, atol=1e-8)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='RK45',
                            rtol=1e-6, atol=1e-8)
                
            # reverse solution back to original order
            G_out = sol.y[0][::-1]
            if G_out.shape[0] != s.shape[0]:
                G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

            if not sol.success:
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                G_out = np.full_like(s, np.nan)
        
        elif self.je_condition == 'wall_ambipolarity':
            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len
            
            if abs(pts[-1,0]) < eps:
                jn = jzi_s[-1] 
                j_e_par_bc = jn / bz_hat[-1]
            elif abs(diff_x) < eps: 
                jn = jxi_s[-1]
                j_e_par_bc = jn / bx_hat[-1]
            elif abs(diff_z) < eps:
                jn = jzi_s[-1]
                j_e_par_bc = jn / bz_hat[-1]
            else:
                j_e_par_bc = j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose ambipolarity at WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            # integrate from wall → throat

            try:
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='BDF',
                            rtol=1e-6, atol=1e-8)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='RK45',
                            rtol=1e-6, atol=1e-8)
                
            # reverse solution back to original order
            G_out = sol.y[0][::-1]
            if G_out.shape[0] != s.shape[0]:
                G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

            if not sol.success:
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                G_out = np.full_like(s, np.nan)
             
        return s, G_out
    
    # ---------------------- Cached streamline solver ----------------------
    def solve_grid_streamline_it(self, eq: str = 'Continuity', V = None, n_df = 1, nn_df = 1, jzi_df = 1, jxi_df = 1, axis_df = 1):
        
        # Cached streamlines
        if not self._streamline_cache:
            throat_1_array = np.linspace(self.throat_lim[0],self.throat_lim[1], int(2*self.N_psi))
            psi_1_array = np.array([self.BField2D.psi(0, x) for x in throat_1_array[:]])
            psi_vals = psi_1_array

            S_Tracing = src.BFields.Streamline_Tracing(self.BField2D, self.z_grid, self.x_grid, self.N_l)
            for psi_val in psi_vals:
                streamline, s = S_Tracing.streamline_mapping(psi_val)
                if streamline is not None:
                    self._streamline_cache[psi_val] = (streamline, s)

            for psi_val, (streamline, s) in self._streamline_cache.items():
                plt.plot(streamline[:,0], streamline[:,1], 'k-', linewidth=0.5)

            del S_Tracing
            gc.collect()

        all_points, all_values = [], []
        it=1
        for psi_val, (streamline, s) in self._streamline_cache.items():
            try:
                if eq == 'Continuity':
                    _, f_out = self.j_par_e_streamline_f_ivp(streamline, s, n_df, nn_df, jzi_df, jxi_df, axis_df)
                elif eq == 'G':
                    _, f_out = self.G_streamline_f_ivp(streamline, s, n_df, nn_df, jzi_df, jxi_df)
                elif eq == 'A':
                    _, f_out = self.A_streamline_f_ivp(streamline, s, n_df, nn_df)
                elif eq == 'He':
                    _, f_out = self.He_streamline_f_ivp(streamline, s, n_df, nn_df)
                else:
                    raise ValueError(f"Unknown equation type: {eq}")
                all_points.append(streamline[:, :2])
                all_values.append(f_out)
                print(f"Streamline {it}: ψ={psi_val} done.\n")
                it+=1
            except Exception as e:
                print(f"Streamline {it}: ψ={psi_val} failed: {e}")
                it+=1
                continue

        if not all_points:
            coords = V.tabulate_dof_coordinates().reshape((-1, 2))
            return np.zeros(coords.shape[0])

        all_points = np.vstack(all_points)
        all_values = np.concatenate(all_values)
        coords = V.tabulate_dof_coordinates().reshape((-1, 2))
        '''
        function_grid = griddata(
            points=all_points,
            values=all_values,
            xi=coords,
            method='linear',
            fill_value=np.nan
        )

        # --- Step 2: fill NaNs using nearest-neighbor fallback ---
        nan_mask = np.isnan(function_grid)
        if np.any(nan_mask):
            nearest_fill = griddata(
                points=all_points,
                values=all_values,
                xi=coords[nan_mask],
                method='nearest'
            )
            function_grid[nan_mask] = nearest_fill

        # --- Step 3: final cleanup ---
        function_grid = np.nan_to_num(function_grid)'''

        valid_mask = ~np.isnan(all_values)

        clean_points = all_points[valid_mask]
        clean_values = all_values[valid_mask]

        function_grid = griddata(
            clean_points,
            clean_values,
            coords,
            method='linear'
        )

        nan_mask = np.isnan(function_grid)
        function_grid[nan_mask] = griddata(
            clean_points,
            clean_values,
            coords[nan_mask],
            method='nearest'
        )

        del all_points, all_values
        gc.collect()

        return function_grid

    # ---------------------- Update wrapper ----------------------
    def update_jpare(self, V, n_df, nn_df, jzi_df, jxi_df, axis_df):
        jpare_grid = self.solve_grid_streamline_it(eq='Continuity', V=V, n_df = n_df, nn_df = nn_df, jzi_df = jzi_df, jxi_df = jxi_df, axis_df = axis_df)
        #if some value lower than 1e-12 appears, set it to 1e-12 to avoid NaNs
        jpare_grid[jpare_grid > -1e-12] = -1e-12
        jpare_df_new = df.Function(V)
        jpare_df_new.vector()[:] = jpare_grid
        if self.je_condition == 'throat_ambipolarity':
            jpare_df_new = ufl.conditional(abs(jpare_df_new) < 1e-5, (jpare_df_new/abs(jpare_df_new))*1e-5, jpare_df_new)
            jpare_df_new_cont = df.project(jpare_df_new, self.V_CG)
        elif self.je_condition == 'wall_ambipolarity':
            #jpare_df_new = ufl.conditional(abs(jpare_df_new) < 1e-10, 1e-10, jpare_df_new)
            jpare_df_new_smooth_cont , jpare_df_new_smooth_DG = self.smooth_Helmholtz(jpare_df_new, alpha_val=0.01)   
        elif self.je_condition == 'metal_wall':
            #jpare_df_new = ufl.conditional(abs(jpare_df_new) < 1e-5, (jpare_df_new/abs(jpare_df_new))*1e-5, jpare_df_new)
            jpare_df_new = df.project(jpare_df_new, self.V_DG)
            jpare_df_new_cont = df.project(jpare_df_new, self.V_CG)     
        else:
            raise ValueError(f"Unknown je_condition: {self.je_condition}")
        
        self.ElectronModel.jpare_cont.assign(jpare_df_new_cont)
        self.ElectronModel.jpare.assign(jpare_df_new)
        del jpare_grid
        del jpare_df_new
        del jpare_df_new_cont
        gc.collect()
    
    def compute_G(self, V, n_df, nn_df, jzi_df, jxi_df):
        G_grid = self.solve_grid_streamline_it(eq='G', V=V, n_df = n_df, nn_df = nn_df, jzi_df = jzi_df, jxi_df = jxi_df)
        G_df_new = df.Function(V)
        G_df_new.vector()[:] = G_grid
        if self.je_condition == 'metal_wall':
            G_df_new = ufl.conditional(abs(G_df_new) < 1e-5, 1e-5, G_df_new)
        G_df_new_smooth_cont, G_df_new_smooth = self.smooth_Helmholtz(G_df_new, alpha_val=0.01)
        self.ElectronModel.G_e.assign(G_df_new_smooth)
        self.ElectronModel.G_e_cont.assign(G_df_new_smooth_cont)

        del G_grid
        del G_df_new, G_df_new_smooth_cont, G_df_new_smooth
        gc.collect()
    
    def compute_G_nocoll(self, V, n_df, nn_df, jzi_df, jxi_df):
        G_grid = self.solve_grid_streamline_it(eq='G', V=V, n_df = n_df, nn_df = nn_df, jzi_df = jzi_df, jxi_df = jxi_df)
        G_df_new = df.Function(V)
        G_df_new.vector()[:] = G_grid
        G_df_new_smooth_cont, G_df_new_smooth = self.smooth_Helmholtz(G_df_new, alpha_val=0.01)
        self.ElectronModel.G_e_nocoll.assign(G_df_new_smooth)

        del G_grid
        del G_df_new, G_df_new_smooth_cont, G_df_new_smooth
        gc.collect()

    def update_He(self, V, n_df, nn_df):
        He_grid = self.solve_grid_streamline_it(eq='He', V=V, n_df = n_df, nn_df = nn_df)
        He_df_new = df.Function(V)
        He_df_new.vector()[:] = He_grid
        He_df_new_smooth_cont_1, He_df_new_smooth_1 = self.smooth_Helmholtz(He_df_new, alpha_val=0.01)
        self.ElectronModel.He_cont.assign(He_df_new_smooth_cont_1)
        self.ElectronModel.He.assign(He_df_new_smooth_1)

        diff_He_grid = He_grid - self.ElectronModel.He_nocoll.vector()[:]
        diff_He_df = df.Function(V)
        diff_He_df.vector()[:] = diff_He_grid
        diff_He_df_smooth_cont, diff_He_df_smooth = self.smooth_Helmholtz(diff_He_df, alpha_val=0.2)
        self.diff_He_df_cont = diff_He_df_smooth_cont

        del He_grid
        del diff_He_grid
        del He_df_new, He_df_new_smooth_cont_1, He_df_new_smooth_1
        del diff_He_df, diff_He_df_smooth_cont, diff_He_df_smooth
        gc.collect()

    def update_A(self, V, n_df, nn_df):
        A_grid = self.solve_grid_streamline_it(eq='A', V=V, n_df = n_df, nn_df = nn_df)
        A_df_new = df.Function(V)
        A_df_new.vector()[:] = A_grid * self.ElectronModel.A_Te_nocoll.vector()[:]
        A_df_new_smooth_cont, A_df_new_smooth = self.smooth_Helmholtz(A_df_new, alpha_val=0.01)
        self.ElectronModel.A_Te.assign(A_df_new_smooth)
        self.ElectronModel.A_Te_cont.assign(A_df_new_smooth_cont)

        diff_A_grid = A_grid * self.ElectronModel.A_Te_nocoll.vector()[:] - self.ElectronModel.A_Te_nocoll.vector()[:]
        diff_A_df = df.Function(V)
        diff_A_df.vector()[:] = diff_A_grid
        diff_A_df_smooth_cont, diff_A_df_smooth = self.smooth_Helmholtz(diff_A_df, alpha_val=0.2)
        self.diff_A_df_cont = diff_A_df_smooth_cont

        del A_grid
        del A_df_new, A_df_new_smooth_cont, A_df_new_smooth
        del diff_A_grid, diff_A_df, diff_A_df_smooth_cont, diff_A_df_smooth
        gc.collect()

    def update_dHe_dpsi(self, Bz, Br, xx, V, n_df):
        ddiff_He_dpsi = df.dot(df.grad(self.diff_He_df_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_He_dpsi, alpha_val=0.01)

        dHe_dpsi_new_grid = abs(ddiff_He_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dHe_dpsi_nocoll.vector()[:]
        self.ElectronModel.dHe_dpsi.vector()[:] = dHe_dpsi_new_grid
        

        '''dHe_dpsi_new = df.dot(df.grad(df.project(self.ElectronModel.He, V)), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(dHe_dpsi_new, alpha_val=0.01)

        self.ElectronModel.dHe_dpsi.vector()[:] = ddiff_He_dpsi_df_smooth.vector()[:]'''

        del  ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth
        gc.collect()

    def update_dA_dpsi(self, Bz, Br, xx, V, n_df):
        ddiff_A_dpsi = df.dot(df.grad(self.diff_A_df_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_A_dpsi, alpha_val=0.01)

        dA_Te_dpsi_new_grid = abs(ddiff_A_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dA_Te_dpsi_nocoll.vector()[:]
        self.ElectronModel.dA_Te_dpsi.vector()[:] = dA_Te_dpsi_new_grid
        

        '''dA_dpsi_new = df.dot(df.grad(df.project(self.ElectronModel.A_Te, V)), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(dA_dpsi_new, alpha_val=0.01)

        self.ElectronModel.dA_Te_dpsi.vector()[:] = ddiff_A_dpsi_df_smooth.vector()[:]'''

        del  ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth
        gc.collect()

    def update_dHe_dpsi_extra_smooth(self, Bz, Br, xx, V, n_df):
        diff_He_df_cont_smooth_cont, diff_He_df_cont_smooth = self.smooth_Helmholtz(self.diff_He_df_cont, alpha_val=0.1)
        ddiff_He_dpsi = df.dot(df.grad(diff_He_df_cont_smooth_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_He_dpsi, alpha_val=0.1)

        dHe_dpsi_new_grid = abs(ddiff_He_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dHe_dpsi_nocoll.vector()[:]
        self.ElectronModel.dHe_dpsi_smooth.vector()[:] = dHe_dpsi_new_grid
        

        '''dHe_dpsi_new = df.dot(df.grad(df.project(self.ElectronModel.He, V)), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(dHe_dpsi_new, alpha_val=0.01)

        self.ElectronModel.dHe_dpsi.vector()[:] = ddiff_He_dpsi_df_smooth.vector()[:]'''

        del  ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth
        gc.collect()

    def update_dA_dpsi_extra_smooth(self, Bz, Br, xx, V, n_df):
        diff_A_df_cont_smooth_cont, diff_A_df_cont_smooth = self.smooth_Helmholtz(self.diff_A_df_cont, alpha_val=0.1)
        ddiff_A_dpsi = df.dot(df.grad(diff_A_df_cont_smooth_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_A_dpsi, alpha_val=0.1)

        dA_Te_dpsi_new_grid = abs(ddiff_A_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dA_Te_dpsi_nocoll.vector()[:]
        self.ElectronModel.dA_Te_dpsi_smooth.vector()[:] = dA_Te_dpsi_new_grid
        

        '''dA_dpsi_new = df.dot(df.grad(df.project(self.ElectronModel.A_Te, V)), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(dA_dpsi_new, alpha_val=0.01)

        self.ElectronModel.dA_Te_dpsi.vector()[:] = ddiff_A_dpsi_df_smooth.vector()[:]'''

        del  ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth
        gc.collect()


    def smooth_Helmholtz(self, f, alpha_val = 0.1):
        '''
        Smooth the function f using a Helmholtz filter.
        ''' 
        u = df.TrialFunction(self.V_CG)
        v = df.TestFunction(self.V_CG)
        alpha = df.Constant(alpha_val)   # controls smoothness
        a = (u*v + alpha**2*df.dot(df.grad(u), df.grad(v)))*df.dx
        L = f*v*df.dx
        f_smooth = df.Function(self.V_CG)
        df.solve(a == L, f_smooth)

        del u, v, alpha, a, L
        gc.collect()
        print("Smoothing applied to function.")
        f_smooth_DG = df.project(f_smooth, self.V_DG)
        print("Projection to DG done.")

        return  f_smooth, f_smooth_DG



class ElectronSolver_Fenics:
    def __init__(self, mesh, Bz, Bx, R, ji = [0,1], S = 0, phi_w = 0, dg_degree = 0, j_e_perp = 0, nu_e = 0, axisymmetry = False, throat_ambipolarity = True):

        self.mesh = mesh
        self.Bz = Bz
        self.Bx = Bx
        self.R = R
        self.jzi = ji[0]
        self.jxi = ji[1]
        self.S = S
        self.phi_w = phi_w
        self.dg_degree = dg_degree
        self.j_e_perp = j_e_perp
        self.nu_e = nu_e
        self.axisymmetry = axisymmetry
        self.throat_ambipolarity = throat_ambipolarity

    def j_e_source(self):
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
        phi_w: can be set to anything as long as no metallic boundaries are set, in such case a good estimation is needed.
        '''

        #Mesh
        mesh = (df.refine((self.mesh)))
        xx = df.SpatialCoordinate(mesh)

        coords = mesh.coordinates()
        min_z = np.min(coords[:,0])
        max_z = np.max(coords[:,0])
        min_x = np.min(coords[:,1])
        max_x = np.max(coords[:,1])
        
        #Vector Function space to solve the advection equation:

        V = df.FunctionSpace(mesh, 'DG', self.dg_degree + 1)
        V_vec = df.VectorFunctionSpace(mesh, 'CG', 2)

        #Reverse the direction of the magnetic field is ambipolarity is imposed downstream
        B = df.project(df.as_vector(((2*int(self.throat_ambipolarity) - 1)*self.Bz, (2*int(self.throat_ambipolarity) - 1)*self.Bx)), V_vec)
        B_perp = df.project(df.as_vector((-(2*int(self.throat_ambipolarity) - 1)*self.Bx, (2*int(self.throat_ambipolarity) - 1)*self.Bz)), V_vec)

        j_e_perp = df.project(self.j_e_perp, V)
        jzi = df.project(self.jzi, V)
        jxi = df.project(self.jxi, V)
        #Boundary conditions

        throat_ambipolarity = self.throat_ambipolarity
        R = self.R
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
        f = self.S - df.dot(B_perp, df.grad(j_e_perp/B_abs)) - (phi*B[1])/(xx[1]+1e-5)
        
        L = v*f*df.dx(mesh)
        
        Bx = self.Bx
        Bz = self.Bz
        phi_w = self.phi_w

        def jn(z, x, phi_w): #This specifies the current normal to the boundary where ambipolarity is imposed
            '''Here phi_w is a dummy argument its only included for compatibility with the sheath module'''
            eps = 1e-3
            if z > max_z-eps:
                return jzi(z,x)
            elif x> max_x-eps:
                return jxi(z,x)
            elif z < eps:
                return jzi(z,x)
            elif x < eps:
                return -jxi(z,x)
            else:
                return 0

        # Set up boundary condition (apply strong BCs)
        class BC(df.UserExpression):
            def eval(self,value,x):
                
                if df.near(x[0], 0):
                    j_e_par_bc = (-jn(x[0],x[1], phi_w)*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(Bx(x[0],x[1])))/(-Bz(x[0],x[1])+1e-6)
                    value[0] = -j_e_par_bc/B_abs(x[0],x[1])
                elif df.near(x[0], max_z):
                    j_e_par_bc = (-jn(x[0],x[1], phi_w)*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(-Bx(x[0],x[1])))/(Bz(x[0],x[1])+1e-6)
                    value[0] = -j_e_par_bc/B_abs(x[0],x[1])
                elif df.near(x[1], max_x):
                    j_e_par_bc = (-jn(x[0],x[1], phi_w)*B_abs(x[0], x[1]) - j_e_perp(x[0], x[1])*(Bz(x[0],x[1])))/(Bx(x[0],x[1])+1e-6)
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

    
