
''' --- 2D MODEL WITH NEUTRALS, COLLISIONS AND INTERNAL ENERGY CONSERVATION --- '''

'''

This module contains the definition of the plasma model with: 
- 2D 
- Neutrals
- Collisions 
- Internal energy conservation

The state vector is: u = [n, gzi, gxi, gyi, ei, nn, gzn, gxn, en]
The source terms include ionization and charge exchange collisions, and the energy equation includes the corresponding energy losses.

The boundary conditions implemented are:
- Wall BCs: the normal velocity of both species is reversed and the parallel and out-of-plane velocities are unchanged.
- Supersonic inflow BCs: density and velocities for both species are prescribed.
- Subsonic inflow BCs: density and velocities for both species and pressure (Energy) for neutrals are prescribed as ions are subsonic.
- Supersonic outflow BCs: the normal velocity of both species is unchanged and the parallel and out-of-plane velocities are extrapolated from the interior. 

'''


import numpy as np
import ufl_legacy as ufl
from dolfin import project
import dolfin as df

import src

n_min = 1e-10 # minimum value of the density, when computing velocities

# Simple variable extraction functions (If axisymmetric x becomes r  and y becomes theta)
#Ion variables:
def n(u): return ufl.conditional(u[0] > 1e-12, u[0], 1e-12)
def gzi(u): return u[1]
def gxi(u): return u[2]
def gyi(u): return u[3]
def pi(u): return ufl.conditional(2/3*u[4] > 1e-12, 2/3*u[4], 1e-12)
def vzi(u): return gzi(u)/(n(u) + 0*n_min)
def vxi(u): return gxi(u)/(n(u) + 0*n_min)
def vyi(u): return gyi(u)/(n(u) + 0*n_min)
#Neutrals Variables:
def nn(u): return ufl.conditional(u[5] > 1e-12, u[5], 1e-12)
def gzn(u): return u[6]
def gxn(u): return u[7]
def pn(u): return ufl.conditional(2/3*u[8] > 1e-12, 2/3*u[8], 1e-12)
def vzn(u): return gzn(u)/(nn(u) + 0*n_min)
def vxn(u): return gxn(u)/(nn(u) + 0*n_min)

#Spatial coordinates (only needed for axisymmetric form)
#x = ufl.SpatialCoordinate()


class PlasmaModel2D_neutrals_energy(src.PhysicalModels.PhysicalModel):
    '''
    Plasma physical fluxes.  
    The state vector is: u = [n, gzi, gxi, gyi, 3/2*pi, nn, gzn, gxn, 3/2*pn]
    '''
    def x_dim(self):
        return 2

    def u_dim(self):
        return 9

    def __init__(self, ElectronModel, IonModel, NeutralModel, Bz, Bx, He, dHe_dpsi, CollisionModel, V_cont, val_check= True, electron_mag = True, ion_mag = True, positivity_flag = False, stabilization_params = (0,0,0), axisymmetry = False):

        self.ElectronModel = ElectronModel
        self.IonModel = IonModel
        self.NeutralModel = NeutralModel
        self.Bz = Bz
        self.Bx = Bx
        self.He = He
        self.dHe_dpsi = dHe_dpsi
        self.CollisionModel = CollisionModel
        self.V_cont = V_cont
        self.val_check = val_check
        self.electron_mag = electron_mag
        self.ion_mag = ion_mag
        self.stabilization_params = stabilization_params #Must be a tuple, first value is C_eps, the second is beta        
        self.positivity_flag = positivity_flag
        self.axisymmetry = axisymmetry
        self.x = ufl.SpatialCoordinate(dHe_dpsi.function_space().mesh())
     
    def flux(self, u, sign = None): 
        pe = self.ElectronModel.pe(n(u))  # Electron pressure
        #pi_unbounded = pi(u)
        #pi_min = 1e-10
        #pi_ = ufl.conditional(ufl.ge(pi_unbounded, pi_min), pi_unbounded, pi_min)  # Ensure pi >= pi_min
        #pn_unbounded = pn(u)
        #pn_min = 1e-10
        #pn_ = ufl.conditional(ufl.ge(pn_unbounded, pn_min), pn_unbounded, pn_min)  # Ensure pn >= pn_min
        
        # pn = self.NeutralModel.pn(n = nn(u), u = (vzn(u)**2 + vxn(u)**2)**0.5 , E = En(u))# Neutral pressure 
        # pi_ = self.IonModel.pi(n(u), (vzi(u)**2 + vxi(u)**2 + vyi(u)**2)**0.5 , Ei(u))
        # pi = ufl.conditional(pi_ > 0, pi_, 0)


        return [[gzi(u),                       gxi(u)],
                [gzi(u)*vzi(u) + pe + pi(u),      gxi(u)*vzi(u)],
                [gzi(u)*vxi(u),                gxi(u)*vxi(u) + pe + pi(u)],
                [gzi(u)*vyi(u),                gxi(u)*vyi(u)],
                [(5/2*pi(u))*vzi(u),              (5/2*pi(u))*vxi(u)],
                #Neutrals flux:
                [gzn(u),                  gxn(u)],
                [gzn(u)*vzn(u) + pn(u),      gxn(u)*vzn(u)],
                [gzn(u)*vxn(u),           gxn(u)*vxn(u) + pn(u)],
                [(5/2*pn(u))*vzn(u),         (5/2*pn(u))*vxn(u)]]
             
    def eigenvalues(self, u, normal, sign):   
        un = vzi(u)*normal[0] + vxi(u)*normal[1] # Inplane velocity component along normal
        unn = vzn(u)*normal[0] + vxn(u)*normal[1] # Inplane neutral velocity component along normal
        c = self.ElectronModel.c(n(u)) # Sound speed for electrons
        #Here we tentatively use the sound speed of electrons as cse >> csi
        #If at some point the complete diagonalization is performed we should substitute it
        #pn_unbounded = pn(u)
        #pn_min = 1e-10
        #pn_ = ufl.conditional(ufl.ge(pn_unbounded, pn_min), pn_unbounded, pn_min)  # Ensure pi >= pi_min
        cn = (pn(u)/(nn(u) + n_min))**0.5 #self.NeutralModel.c(nn(u), (vzn(u)**2 + vxn(u)**2)**0.5 , En(u)) # Sound speed for neutrals 
        
        #cn = (self.NeutralModel.gamma*nn(u)**(self.NeutralModel.gamma-1))**0.5
        '''IMPORTANT: perform complete diagonalization to determine the correct eigenvalues - ions temperature is considered practically negligible'''
    
        return [un - c, un, un + c, unn - cn, unn, unn + cn]
        
    def rhs(self, u):
        #self.collision_param allows to eliminate collisions or switch them on.
        v_i = ufl.as_vector((vzi(u), vxi(u)))
        v_n = ufl.as_vector((vzn(u), vxn(u)))
        
        #Ti_unbounded = pi(u)/(n(u) + n_min)
        #Ti_min = 1e-10
        #Tn_unbounded = pn(u)/(nn(u) + n_min)
        #Tn_min = 1e-10
        #Tn = ufl.conditional(ufl.ge(Tn_unbounded, Tn_min), Tn_unbounded, Tn_min)  # Ensure pn >= pn_min
        #Ti = ufl.conditional(ufl.ge(Ti_unbounded, Ti_min), Ti_unbounded, Ti_min)  # Ensure pi >= pi_min

        Ti = pi(u)/(n(u) + n_min)
        Tn = pn(u)/(nn(u) + n_min)

        nu_i = self.CollisionModel.Ionization(nn(u), self.ElectronModel.Te(n(u)))
        nu_cex = self.CollisionModel.CEX(nn(u), v_i, v_n, self.ElectronModel.Te(n(u)))
        S_i = nu_i * n(u)
        S_cex = nu_cex * n(u)
        #def Ei(u): return 3/2*pi(u) + 0.5*n(u)*(vzi(u)**2 + vxi(u)**2)
        #def En(u): return 3/2*pn(u) + 0.5*nn(u)*(vzn(u)**2 + vxn(u)**2)

        pn_cont = project(pn(u), self.V_cont)
        pi_cont = project(pi(u), self.V_cont)

        xx = self.x[1]

        def vye(u):
            return  -(xx*int(self.axisymmetry) + (1- int(self.axisymmetry)))*self.dHe_dpsi
        
        #ufl.conditional(self.x[1]>1e-2, self.x[1], 1e-2)

        return [S_i - int(self.axisymmetry)*n(u)*vxi(u)/xx,
               -n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bx + S_i*vzn(u) + S_cex*(vzn(u) -  vzi(u)) - int(self.axisymmetry)*n(u)*vxi(u)*vzi(u)/xx,
               n(u)*(-int(self.electron_mag)*vye(u) + int(self.ion_mag)*vyi(u))*self.Bz + S_i*vxn(u) + S_cex*(vxn(u) -  vxi(u)) - int(self.axisymmetry)*n(u)*(vxi(u)**2 - vyi(u)**2)/xx,
               int(self.ion_mag)*n(u)*(self.Bx*vzi(u) - self.Bz*vxi(u)) - int(self.axisymmetry)*2*n(u)*vxi(u)*vyi(u)/xx,
               S_i*(3/2*Tn + 0.5*(v_i - v_n)**2) + S_cex*(3/2*(Tn - Ti) + 0.5*(v_i - v_n)**2) + ufl.dot(v_i, ufl.grad(pi_cont)) - int(self.axisymmetry)*5/2*pi(u)*vxi(u)/xx,
               #Neutrals rhs:
               - S_i - int(self.axisymmetry)*nn(u)*vxn(u)/xx,
               - S_i*vzn(u) + S_cex*(vzi(u) -  vzn(u)) - int(self.axisymmetry)*nn(u)*vxn(u)*vzn(u)/xx,
               - S_i*vxn(u) + S_cex*(vxi(u) -  vxn(u)) - int(self.axisymmetry)*nn(u)*(vxn(u)**2)/xx,
               - S_i*3/2*Tn + S_cex*(3/2*(Ti - Tn) + 0.5*(v_i - v_n)**2) + ufl.dot(v_n, ufl.grad(pn_cont)) - int(self.axisymmetry)*5/2*pn(u)*vxn(u)/xx]

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
            q_vals[0::9][q_vals[0::9] <= 1e-8] = 1e-8
            #q_vals[0::9][q_vals[0::9] >= 1] = 1
            #q_vals[5::9][q_vals[5::9] <= 1e-7] = 1e-7

            return q_vals

        else:
            return u.vector().get_local()


    def get_negatives(self, u):

        dens = u.sub(0, deepcopy=True)
        dens_vec = u.sub(0, deepcopy=True).vector()[:]
        dens_vec[dens_vec < 0] = 10

        dens.vector()[:] = dens_vec

        return dens

  
class BCPlasma2D_neutrals_energy(src.PhysicalModels.BC):
    '''
    Abstract class for plasma boundary conditions.
    '''    
    def g_normal(self,um,normal):
        '''
        Outward normal velocity.
        ''' 
        return gzi(um)*normal[0] + gxi(um)*normal[1]

    def g_parallel(self,um,normal):
        '''
        Parallel velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        ''' 
        return gzi(um)*normal[1] - gxi(um)*normal[0]  
    
    def gn_normal(self,um,normal):
        '''
        Outward neutral normal velocity.
        ''' 
        return gzn(um)*normal[0] + gxn(um)*normal[1]
    
    def gn_parallel(self,um,normal):
        '''
        Parallel neutral velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        ''' 
        return gzn(um)*normal[1] - gxn(um)*normal[0]  


class BC_Wall2D_neutrals_energy(BCPlasma2D_neutrals_energy):
    '''
    Wall boundary conditions. Also good for symmetry planes.  
    ''' 
    def u_outside(self, um, normal):   
        g_normal  = - self.g_normal(um,normal)
        g_parallel = self.g_parallel(um,normal)
        gn_normal  = - self.gn_normal(um,normal)
        gn_parallel = self.gn_parallel(um,normal)



        # u_outside is the same as um except that u_normal has been reversed
        return [n(um), 
                (g_normal*normal[0] + g_parallel*normal[1]), 
                (g_normal*normal[1] - g_parallel*normal[0]),
                gyi(um), 
                3./2.*pi(um),
                nn(um),
                (gn_normal*normal[0] + gn_parallel*normal[1]), 
                (gn_normal*normal[1] - gn_parallel*normal[0]),
                3./2.*pn(um)] 


class BC_SupersonicInflow2D_neutrals_energy(BCPlasma2D_neutrals_energy):
    '''
    Supersonic inflow boundary conditions in which we prescribe density and velocities. 
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., pi_outside = 0., nn_outside = 1., vn_normal_outside = 1.1, vn_parallel_outside = 0., pn_outside = 0):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.pi_outside = pi_outside #Total Ion Energy outside
        self.nn_outside = nn_outside
        self.vn_normal_outside = -vn_normal_outside
        self.vn_parallel_outside = vn_parallel_outside
        self.pn_outside = pn_outside #Total Ion Energy outside

    
    def u_outside(self, um, normal):
        
        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                self.n_outside*self.vyi_outside,
                3./2.*self.pi_outside,
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel_outside*normal[1]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] - self.vn_parallel_outside*normal[0]),
                3./2.*self.pn_outside] 

class BC_SubsonicInflow2D_neutrals_energy(BCPlasma2D_neutrals_energy):
    '''
    Subsonic inflow boundary conditions in which we prescribe density and velocities for both species and pressure (Energy) for neutrals as ions are subsonic.
    '''
    def __init__(self, n_outside=1., v_normal_outside=1.1, v_parallel_outside=0., vyi_outside=0., nn_outside = 1., vn_normal_outside = 1.1, vn_parallel_outside = 0., pn_outside = 0):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel_outside = v_parallel_outside  # Parallel velocity outside. 
        self.vyi_outside = vyi_outside  # Out-of-plane velocity outside
        self.nn_outside = nn_outside
        self.vn_normal_outside = -vn_normal_outside
        self.vn_parallel_outside = vn_parallel_outside
        self.pn_outside = pn_outside #Total Neutral Energy outside

    def Fn_outside(self, um, normal):
        
        return 

    def u_outside(self, um, normal):
        
        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel_outside*normal[1]), 
                self.n_outside*(self.v_normal_outside*normal[1] - self.v_parallel_outside*normal[0]),
                self.n_outside*self.vyi_outside,
                3./2.*pi(um),
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel_outside*normal[1]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] - self.vn_parallel_outside*normal[0]),
                3./2.*self.pn_outside] 


class BC_SupersonicOutflow2D_neutrals_energy(BCPlasma2D_neutrals_energy):
    '''
    Supersonic outflow boundary conditions.  
    The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.
    ''' 
    from src.utils.MathAux import max
    
    def __init__(self, ElectronModel, NeutralModel): 
        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel

    def u_outside(self, um, normal):  
        c = self.ElectronModel.c(n(um))  #Electron Sound speed
        cn = (pn(um)/nn(um) + n_min)**0.5 #self.NeutralModel.c(nn(um), (vzn(um)**2 + vxn(um)**2)**0.5 , En(um)) # Sound speed for neutrals
        g_normal = BC_SupersonicOutflow2D_neutrals_energy.max(self.g_normal(um, normal), n(um)*c)  # Outward normal velocity (should be at least c) 
        g_parallel = self.g_parallel(um, normal)  # Parallel velocity
        gn_normal = BC_SupersonicOutflow2D_neutrals_energy.max(self.gn_normal(um, normal), nn(um)*cn)  # Outward normal velocity (should be at least cn) 
        gn_parallel = self.gn_parallel(um, normal)  # Parallel velocity

        return [n(um), 
                (g_normal*normal[0] + g_parallel*normal[1]), 
                (g_normal*normal[1] - g_parallel*normal[0]),
                gyi(um),
                3./2.*pi(um),
                nn(um), 
                (gn_normal*normal[0] + gn_parallel*normal[1]), 
                (gn_normal*normal[1] - gn_parallel*normal[0]),
                3./2.*pn(um)]
