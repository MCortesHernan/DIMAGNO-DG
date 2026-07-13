
''' --- 3D COMPLETE MODEL --- '''

'''

This module contains the definition of the plasma model with: 
- 3D 
- Neutrals
- Collisions
- Total energy conservation

The state vector is: u = [n, gzi, gxi, gyi, Ei, nn, gzn, gxn, gyn, En]

The boundary conditions implemented are:
- Wall BCs: the normal velocity of both species is reversed and the parallel and out-of-plane velocities are unchanged.
- Supersonic inflow BCs: density and velocities for both species are prescribed.
- Supersonic outflow BCs: the normal velocity of both species is unchanged and the parallel and out-of-plane velocities are extrapolated from the interior. 
'''

# --- IMPORTS --- #

from sympy import gamma

import src

import numpy as np
import ufl_legacy as ufl
import dolfin as df

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
def vi(u): return ufl.as_vector((vzi(u), vxi(u), vyi(u)))
def vi2(u): return ufl.inner(vi(u), vi(u))

#Neutrals Variables:
def nn(u): return ufl.conditional(u[5] > 1e-12, u[5], 1e-12)
def gzn(u): return u[6]
def gxn(u): return u[7]
def gyn(u): return u[8]
def En(u): return u[9]
def vzn(u): return gzn(u)/(nn(u) + 0.*n_min)
def vxn(u): return gxn(u)/(nn(u) + 0.*n_min)
def vyn(u): return gyn(u)/(nn(u) + 0.*n_min)
def vn(u): return ufl.as_vector((vzn(u), vxn(u), vyn(u)))
def vn2(u): return ufl.inner(vn(u), vn(u))


class PlasmaModel3D(src.PhysicalModels.PhysicalModel):
    '''
    Plasma physical fluxes.  
    The state vector is: u = [n, gzi, gxi, gyi, Ei, nn, gzn, gxn, En]
    '''
    def x_dim(self):
        return 3

    def u_dim(self):
        return 10
 
    def __init__(self, ElectronModel = None, IonModel = None, NeutralModel = None,
                 CollisionModel = None, ElectronSolver = None, 
                 Bz = None, Bx = None, By = None, 
                 V_DG = None, V_CG = None, 
                 val_check= True, electron_mag = True, ion_mag = True, positivity_flag = False, stabilization_params = (0,0,0), axisymmetry = False,
                 mesh = None):

        self.ElectronModel = ElectronModel
        self.ElectronSolver = ElectronSolver
        self.IonModel = IonModel
        self.NeutralModel = NeutralModel
        self.Bz = Bz
        self.Bx = Bx
        self.By = By
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
        self.x = ufl.SpatialCoordinate(self.mesh)
     
    def flux(self, u, sign = None): 
        
        pe = self.ElectronModel.pe(n(u))  # Electron pressure

        phi = self.ElectronModel.phi(n(u), self.ElectronModel.He, self.ElectronModel.A_Te)

        pi = self.IonModel.pi(n(u), vi2(u), Ei(u), phi)
        pi = ufl.conditional(pi > 0, pi, 1e-12) # Avoid negative ion pressure values when computing the fluxes.

        pn = self.NeutralModel.pn(nn(u), vn2(u), En(u))
        pn = ufl.conditional(pn > 0, pn, 1e-12) # Avoid negative neutral pressure values when computing the fluxes.

        return [[gzi(u),                    gxi(u),                     gyi(u)],
                [gzi(u)*vzi(u) + pe + pi,   gxi(u)*vzi(u),              gyi(u)*vzi(u)],
                [gzi(u)*vxi(u),             gxi(u)*vxi(u) + pe + pi,    gyi(u)*vxi(u)],
                [gzi(u)*vyi(u),             gxi(u)*vyi(u),              gyi(u)*vyi(u) + pe + pi],
                [(Ei(u) + pi)*vzi(u),       (Ei(u) + pi)*vxi(u),        (Ei(u) + pi)*vyi(u)],
                [gzn(u),                    gxn(u),                     gyn(u)],
                [gzn(u)*vzn(u) + pn,        gxn(u)*vzn(u),              gyn(u)*vzn(u)],
                [gzn(u)*vxn(u),             gxn(u)*vxn(u) + pn,        gyn(u)*vxn(u)],
                [gzn(u)*vyn(u),             gxn(u)*vyn(u),              gyn(u)*vyn(u) + pn],
                [(En(u) + pn)*vzn(u),       (En(u) + pn)*vxn(u),        (En(u) + pn)*vyn(u)]]
    
             
    def eigenvalues(self, u, normal, sign = None):   
        
        un = vzi(u)*normal[0] + vxi(u)*normal[1] + vyi(u)*normal[2]  # Inplane velocity component along normal
        unn = vzn(u)*normal[0] + vxn(u)*normal[1] + vyn(u)*normal[2]  # Inplane velocity component along normal for neutrals
        
        phi = self.ElectronModel.phi(n(u), self.ElectronModel.He, self.ElectronModel.A_Te)
        pi = self.IonModel.pi(n(u), vi2(u), Ei(u), phi) #Ion pressure
        pi = ufl.conditional(pi > 0, pi, 1e-12)

        Ti =  self.IonModel.Ti(n(u), vi2(u), Ei(u), phi) #Ion temperature
        Ti = ufl.conditional(Ti > 0, Ti, 1e-12)

        Tn =  self.NeutralModel.Tn(nn(u), vn2(u), En(u)) #Neutral temperature
        Tn = ufl.conditional(Tn > 0, Tn, 1e-12)
        
        # Altough generally negligible, Ti is included, as it will increase under collisional effects
        
        c = (self.ElectronModel.A_Te*self.ElectronModel.gamma*n(u)**(self.ElectronModel.gamma-1))**0.5
        #c = self.ElectronModel.c(n(u), A_sign) #self.IonModel.c(n(u), (vzi(u)**2 + vxi(u)**2 + vyi(u)**2)**0.5 , Ei(u), phi) # Sound speed for ions
        cn = self.NeutralModel.c(Tn) # Sound speed for neutrals
        #cn = (self.NeutralModel.gamma*nn(u)**(self.NeutralModel.gamma-1))**0.5
    
        return [un - c, un, un + c, unn - cn, unn, unn + cn]
        
    def rhs(self, u):

        #self.collision_param allows to eliminate collisions or switch them on.
        v_i = ufl.as_vector((vzi(u), vxi(u), vyi(u)))
        v_n = ufl.as_vector((vzn(u), vxn(u), vyn(u)))
        
        phi = self.ElectronModel.phi(n(u), self.ElectronModel.He, self.ElectronModel.A_Te)

        nu_i = self.CollisionModel.Ionization(nn(u), self.ElectronModel.Te(n(u), self.ElectronModel.A_Te))
        nu_cex = self.CollisionModel.CEX(nn(u), v_i, v_n, self.ElectronModel.Te(n(u), self.ElectronModel.A_Te))
        S_i = nu_i * n(u)
        S_cex = nu_cex * n(u)

        Ei_no_phi = Ei(u) - n(u)*phi

        Q_i = S_i * En(u)/n(u) + S_cex * (En(u)/nn(u) - Ei_no_phi/n(u)) #Energy loss for ions due to ionization and charge exchange

        B = ufl.as_vector((self.Bz, self.Bx, self.By))
        B_norm = ufl.sqrt(ufl.dot(B, B))
        b = B / (B_norm + 1e-10) # Unit vector in the direction of the magnetic field, with a small term added to avoid division by zero when B is zero.

        def grad_perp(f):
            grad_f = ufl.grad(f)
            return grad_f - b * ufl.dot(b, grad_f)

        gradHe_perp = grad_perp(self.ElectronModel.He)
        gradA_perp  = grad_perp(self.ElectronModel.A_Te)

        electron_mag_force = (
            n(u) * gradHe_perp
            - 1/(self.ElectronModel.gamma - 1) * n(u)**self.ElectronModel.gamma * gradA_perp
            )
        
        ion_mag_force = n(u) * ufl.cross(vi(u), B)

        return [S_i,
                electron_mag_force[0] + ion_mag_force[0] + S_i*vzn(u) + S_cex * (vzn(u) - vzi(u)),
                electron_mag_force[1] + ion_mag_force[1] + S_i*vxn(u) + S_cex * (vxn(u) - vxi(u)),
                electron_mag_force[2] + ion_mag_force[2] + S_i*vyn(u) + S_cex * (vyn(u) - vyi(u)),
                Q_i,
                -S_i,
                S_i*vzi(u) + S_cex * (vzi(u) - vzn(u)),
                S_i*vxi(u) + S_cex * (vxi(u) - vxn(u)),
                S_i*vyi(u) + S_cex * (vyi(u) - vyn(u)),
                -Q_i
                ]

    def is_valid(self, u):
        if self.val_check:
            uflat = u.vector().get_local()
            # Check both 0::5 (density) 
            return all(uflat[0::self.u_dim()] > 0)
        else:
            return True

    def force_positivity(self, u):
        '''Return the vector of dof with a solution changing al negs to positive.'''
        
        if self.positivity_flag:
            q_vals = u.vector().get_local()
            q_vals[0::self.u_dim()][q_vals[0::self.u_dim()] <= 1e-6] = 1e-6

            return q_vals

        else:
            return u.vector().get_local()


    def get_negatives(self, u):

        dens = u.sub(0, deepcopy=True)
        dens_vec = u.sub(0, deepcopy=True).vector()[:]
        dens_vec[dens_vec < 0] = 10

        dens.vector()[:] = dens_vec

        return dens

  
class BCPlasma3D(src.PhysicalModels.BC):
    '''
    Abstract class for plasma boundary conditions.
    '''    
    def v_normal(self,um,normal):
        '''
        Outward normal velocity.
        ''' 
        return vzi(um)*normal[0] + vxi(um)*normal[1] + vyi(um)*normal[2]

    def vn_normal(self,um,normal):
        '''
        Outward normal velocity for neutrals.
        ''' 
        return vzn(um)*normal[0] + vxn(um)*normal[1] + vyn(um)*normal[2]
    
    

    def parallel_vectors(self, n):
        '''
        Given a normal vector n, compute two orthogonal vectors u1 and u2
        that lie in the plane orthogonal to n.
        '''
        # Ensure the normal vector is a unit vector
        norm_n = df.sqrt(df.dot(n, n))
        n = n / norm_n
    
        # Compute u1 as an arbitrary vector orthogonal to n
        u1 = ufl.as_vector([n[1] - n[2], -n[0], n[0]])
        
        # Normalize u1
        norm_u1 = df.sqrt(df.dot(u1, u1))
        u1 = u1 / norm_u1
    
        # Compute u2 as the cross product of n and u1
        u2 = df.cross(n, u1)
    
        # Normalize u2 (optional, since cross product of unit vectors is already unit)
        norm_u2 = df.sqrt(df.dot(u2, u2))
        u2 = u2 / norm_u2
        
        p1 = u1
        p2 = u2

        return p1, p2

    def v_parallel(self,um,normal):
        '''
        Parallel velocity.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        '''
        p1, p2 = self.parallel_vectors(normal)

        v_parallel1 = vzi(um)*p1[0] + vxi(um)*p1[1] + vyi(um)*p1[2]
        v_parallel2 = vzi(um)*p2[0] + vxi(um)*p2[1] + vyi(um)*p2[2]

        return v_parallel1, v_parallel2
        #return vzi(um)*normal[1] - vxi(um)*normal[0]

    def vn_parallel(self,um,normal):
        '''
        Parallel velocity for neutrals.
        The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 
        '''
        p1, p2 = self.parallel_vectors(normal)

        vn_parallel1 = vzn(um)*p1[0] + vxn(um)*p1[1] + vyn(um)*p1[2]
        vn_parallel2 = vzn(um)*p2[0] + vxn(um)*p2[1] + vyn(um)*p2[2]

        return vn_parallel1, vn_parallel2  
 

class BC_Wall3D(BCPlasma3D):
    '''
    Wall boundary conditions. Also good for symmetry planes.  
    ''' 
    def __init__(self, strong_imposition = False):  
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)
    
    def u_outside(self, um, normal):   
        v_normal  = - self.v_normal(um,normal)
        v_parallel1, v_parallel2 = self.v_parallel(um,normal)
        p1, p2 = self.parallel_vectors(normal)
        vn_normal  = - self.vn_normal(um,normal)
        vn_parallel1, vn_parallel2 = self.vn_parallel(um,normal)

        # u_outside is the same as um except that u_normal has been reversed
        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel1*p1[0] + v_parallel2*p2[0]), 
                n(um)*(v_normal*normal[1] + v_parallel1*p1[1] + v_parallel2*p2[1]),
                n(um)*(v_normal*normal[2] + v_parallel1*p1[2] + v_parallel2*p2[2]),
                Ei(um),
                nn(um),
                nn(um)*(vn_normal*normal[0] + vn_parallel1*p1[0] + vn_parallel2*p2[0]), 
                nn(um)*(vn_normal*normal[1] + vn_parallel1*p1[1] + vn_parallel2*p2[1]),
                nn(um)*(vn_normal*normal[2] + vn_parallel1*p1[2] + vn_parallel2*p2[2]),
                En(um)]

class BC_SupersonicInflow3D(BCPlasma3D):
    '''
    Supersonic inflow boundary conditions in which we prescribe density and velocities. 
    '''
    def __init__(self, n_outside=1., 
                 v_normal_outside=1.1, v_parallel1_outside=0., v_parallel2_outside=0.,
                 Ei_outside=0.,
                 nn_outside=1.,
                 vn_normal_outside=1.1, vn_parallel1_outside=0., vn_parallel2_outside=0.,
                 En_outside=0.,
                 strong_imposition = False
                 ):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel1_outside = v_parallel1_outside  # Parallel velocity outside. 
        self.v_parallel2_outside = v_parallel2_outside  # Out-of-plane velocity outside
        self.Ei_outside = Ei_outside  # Ion energy outside
        self.nn_outside = nn_outside  # Neutral density outside
        self.vn_normal_outside = -vn_normal_outside  # Outward-normal velocity for neutrals outside (hence the minus sign)
        self.vn_parallel1_outside = vn_parallel1_outside  # Parallel velocity for neutrals outside
        self.vn_parallel2_outside = vn_parallel2_outside  # Out-of-plane velocity for neutrals outside
        self.En_outside = En_outside  # Neutral energy outside
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def u_outside(self, um, normal):
        
        p1, p2 = self.parallel_vectors(normal)

        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel1_outside*p1[0] + self.v_parallel2_outside*p2[0]), 
                self.n_outside*(self.v_normal_outside*normal[1] + self.v_parallel1_outside*p1[1] + self.v_parallel2_outside*p2[1]),
                self.n_outside*(self.v_normal_outside*normal[2] + self.v_parallel1_outside*p1[2] + self.v_parallel2_outside*p2[2]),
                self.Ei_outside,
                self.nn_outside,
                self.nn_outside*(self.vn_normal_outside*normal[0] + self.vn_parallel1_outside*p1[0] + self.vn_parallel2_outside*p2[0]), 
                self.nn_outside*(self.vn_normal_outside*normal[1] + self.vn_parallel1_outside*p1[1] + self.vn_parallel2_outside*p2[1]),
                self.nn_outside*(self.vn_normal_outside*normal[2] + self.vn_parallel1_outside*p1[2] + self.vn_parallel2_outside*p2[2]),
                self.En_outside]

    
class BC_SupersonicOutflow3D(BCPlasma3D):
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

        c = (self.ElectronModel.A_Te*self.ElectronModel.gamma*n(um)**(self.ElectronModel.gamma-1))**0.5
        cn = self.NeutralModel.c(Tn) # Sound speed for neutrals

        v_normal = BC_SupersonicOutflow3D.max(self.v_normal(um, normal), c)  # Outward normal velocity (should be at least c)
        #v_normal = self.v_normal(um, normal)  # Normal velocity
        v_parallel1, v_parallel2 = self.v_parallel(um, normal)  # Parallel velocity
        vn_normal = BC_SupersonicOutflow3D.max(self.vn_normal(um, normal), cn)  # Outward normal velocity (should be at least cn) 
        #vn_normal = self.vn_normal(um, normal)  # Normal velocity
        vn_parallel1, vn_parallel2 = self.vn_parallel(um, normal)  # Parallel velocity
        p1, p2 = self.parallel_vectors(normal)

        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel1*p1[0] + v_parallel2*p2[0]), 
                n(um)*(v_normal*normal[1] + v_parallel1*p1[1] + v_parallel2*p2[1]),
                n(um)*(v_normal*normal[2] + v_parallel1*p1[2] + v_parallel2*p2[2]),
                Ei(um),
                nn(um),
                nn(um)*(vn_normal*normal[0] + vn_parallel1*p1[0] + vn_parallel2*p2[0]), 
                nn(um)*(vn_normal*normal[1] + vn_parallel1*p1[1] + vn_parallel2*p2[1]),
                nn(um)*(vn_normal*normal[2] + vn_parallel1*p1[2] + vn_parallel2*p2[2]),
                En(um)] 
