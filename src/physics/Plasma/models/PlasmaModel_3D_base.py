
''' --- BASIC 3D MODEL --- '''

'''

This module contains the definition of the plasma model with: 
- 3D 

The state vector is: u = [n, gzi, gxi, gyi]

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

n_min = 1e-6 # minimum value of the density, when computing velocities

# --- MODEL DEFINITION AND VARIABLES --- #

# Simple variable extraction functions (If axisymmetric x becomes r  and y becomes theta)
#Ion variables:
def n(u): return ufl.conditional(u[0] > 1e-6, u[0], 1e-6)
def gzi(u): return u[1]
def gxi(u): return u[2]
def gyi(u): return u[3]
def vzi(u): return gzi(u)/(n(u) + n_min)
def vxi(u): return gxi(u)/(n(u) + n_min)
def vyi(u): return gyi(u)/(n(u) + n_min)


class PlasmaModel3D_base(src.PhysicalModels.PhysicalModel):
    '''
    Plasma physical fluxes.  
    The state vector is: u = [n, gzi, gxi, gyi, Ei, nn, gzn, gxn, En]
    '''
    def x_dim(self):
        return 3

    def u_dim(self):
        return 4
 
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

        return [[gzi(u),                    gxi(u),                     gyi(u)],
                [gzi(u)*vzi(u) + pe ,   gxi(u)*vzi(u),              gyi(u)*vzi(u)],
                [gzi(u)*vxi(u),             gxi(u)*vxi(u) + pe ,    gyi(u)*vxi(u)],
                [gzi(u)*vyi(u),             gxi(u)*vyi(u),              gyi(u)*vyi(u) + pe ]]
    
             
    def eigenvalues(self, u, normal, sign = None):   
        
        un = vzi(u)*normal[0] + vxi(u)*normal[1] + vyi(u)*normal[2]  # Inplane velocity component along normal
        
        # Altough generally negligible, Ti is included, as it will increase under collisional effects
        
        c = (self.ElectronModel.A_Te*self.ElectronModel.gamma*n(u)**(self.ElectronModel.gamma-1))**0.5
        #c = self.ElectronModel.c(n(u), A_sign) #self.IonModel.c(n(u), (vzi(u)**2 + vxi(u)**2 + vyi(u)**2)**0.5 , Ei(u), phi) # Sound speed for ions
        #cn = (self.NeutralModel.gamma*nn(u)**(self.NeutralModel.gamma-1))**0.5
    
        return [un - c, un, un + c]
        
    def rhs(self, u):

        #self.collision_param allows to eliminate collisions or switch them on.
        v_i = ufl.as_vector((vzi(u), vxi(u), vyi(u)))
        
        B = ufl.as_vector((self.Bz, self.Bx, self.By))
        B_norm = ufl.sqrt(ufl.dot(B, B))
        b = B / (B_norm + 1e-10) # Unit vector in the direction of the magnetic field, with a small term added to avoid division by zero when B is zero.

        def grad_perp(f):
            grad_f = ufl.grad(f)
            return grad_f - b * ufl.dot(b, grad_f)

        gradHe_perp = grad_perp(self.ElectronModel.He)
        gradA_perp  = grad_perp(self.ElectronModel.A_Te)

        electron_mag_force = (
            
             - 1/(self.ElectronModel.gamma - 1) * n(u)**self.ElectronModel.gamma * gradA_perp # + n(u) * gradHe_perp
            )
        
        ion_mag_force = n(u) * ufl.cross(v_i, B)

        return [0,
                electron_mag_force[0] + ion_mag_force[0],
                electron_mag_force[1] + ion_mag_force[1],
                electron_mag_force[2] + ion_mag_force[2]
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
    


class BCPlasma3D_base(src.PhysicalModels.BC):

    #Abstract class for plasma boundary conditions.
     
    def v_normal(self,um,normal):
      
        #Outward normal velocity.
        
        return vzi(um)*normal[0] + vxi(um)*normal[1] + vyi(um)*normal[2]

    

    def parallel_vectors(self, n):
     
        #Given a normal vector n, compute two orthogonal vectors u1 and u2
        #that lie in the plane orthogonal to n.
       
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

        #Parallel velocity.
        #The basis {1p,1n,1y} is right-handed and 1n is always the outward normal. 

        p1, p2 = self.parallel_vectors(normal)

        v_parallel1 = vzi(um)*p1[0] + vxi(um)*p1[1] + vyi(um)*p1[2]
        v_parallel2 = vzi(um)*p2[0] + vxi(um)*p2[1] + vyi(um)*p2[2]

        return v_parallel1, v_parallel2
        #return vzi(um)*normal[1] - vxi(um)*normal[0]
 

class BC_Wall3D_base(BCPlasma3D_base):

    #Wall boundary conditions. Also good for symmetry planes.  

    def __init__(self, strong_imposition = False):  
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)
    
    def u_outside(self, um, normal):   
        v_normal  = - self.v_normal(um,normal)
        v_parallel1, v_parallel2 = self.v_parallel(um,normal)
        p1, p2 = self.parallel_vectors(normal)

        # u_outside is the same as um except that u_normal has been reversed
        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel1*p1[0] + v_parallel2*p2[0]), 
                n(um)*(v_normal*normal[1] + v_parallel1*p1[1] + v_parallel2*p2[1]),
                n(um)*(v_normal*normal[2] + v_parallel1*p1[2] + v_parallel2*p2[2])]

class BC_SupersonicInflow3D_base(BCPlasma3D_base):

    #Supersonic inflow boundary conditions in which we prescribe density and velocities. 

    def __init__(self, n_outside=1., 
                 v_normal_outside=1.1, v_parallel1_outside=0., v_parallel2_outside=0.,
                 strong_imposition = False
                 ):  
        self.n_outside = n_outside  # Density outside 
        self.v_normal_outside = -v_normal_outside  # Outward-normal velocity outside (hence the minus sign)
        self.v_parallel1_outside = v_parallel1_outside  # Parallel velocity outside. 
        self.v_parallel2_outside = v_parallel2_outside  # Out-of-plane velocity outside
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def u_outside(self, um, normal):
        
        p1, p2 = self.parallel_vectors(normal)

        return [self.n_outside, 
                self.n_outside*(self.v_normal_outside*normal[0] + self.v_parallel1_outside*p1[0] + self.v_parallel2_outside*p2[0]), 
                self.n_outside*(self.v_normal_outside*normal[1] + self.v_parallel1_outside*p1[1] + self.v_parallel2_outside*p2[1]),
                self.n_outside*(self.v_normal_outside*normal[2] + self.v_parallel1_outside*p1[2] + self.v_parallel2_outside*p2[2])]

    
class BC_SupersonicOutflow3D_base(BCPlasma3D_base):

    #Supersonic outflow boundary conditions.  
    #The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.

    from src.utils.MathAux import max
    
    def __init__(self, ElectronModel, IonModel, NeutralModel, strong_imposition = False): 
        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel
        self.IonModel = IonModel
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def u_outside(self, um, normal):  
        

        c = (self.ElectronModel.A_Te*self.ElectronModel.gamma*n(um)**(self.ElectronModel.gamma-1))**0.5

        #check 
        #v_normal = BC_SupersonicOutflow3D_base.max(self.v_normal(um, normal), c)  # Outward normal velocity (should be at least c)
        v_normal = self.v_normal(um, normal)  # Normal velocity
        v_parallel1, v_parallel2 = self.v_parallel(um, normal)  # Parallel velocity
        p1, p2 = self.parallel_vectors(normal)

        return [n(um), 
                n(um)*(v_normal*normal[0] + v_parallel1*p1[0] + v_parallel2*p2[0]), 
                n(um)*(v_normal*normal[1] + v_parallel1*p1[1] + v_parallel2*p2[1]),
                n(um)*(v_normal*normal[2] + v_parallel1*p1[2] + v_parallel2*p2[2])] 




# ---------------------------------------------------------------------------------


'''class BCPlasma3D_base(PhysicalModels.BC):

    #Abstract class for plasma boundary conditions.
  
    def velocity(self, um):

        return ufl.as_vector((vzi(um), vxi(um), vyi(um)))
    
    def v_normal(self,um,normal):

        #Outward normal velocity.

        return vzi(um)*normal[0] + vxi(um)*normal[1] + vyi(um)*normal[2]
    
    def v_normal_vector(self, um, normal):

        #Outward normal velocity vector.
        
        v_normal = self.v_normal(um, normal)
        return v_normal * normal
    
    def v_tangential_vector(self, um, normal):
        
        #Tangential velocity vector.
        
        v = self.velocity(um)
        v_normal_vec = self.v_normal_vector(um, normal)
        return v - v_normal_vec
    
    def _cartesian_tangential_vectors(self, n):
        n = n / ufl.sqrt(ufl.dot(n, n))

        # pick a safe reference axis
        a = ufl.as_vector([1.0, 0.0, 0.0])

        # if almost parallel, switch axis
        a = ufl.conditional(
            ufl.sqrt(ufl.dot(n, a)**2) > 0.9,
            ufl.as_vector([0.0, 1.0, 0.0]),
            a
        )

        p1 = ufl.cross(n, a)
        p1 = p1 / ufl.sqrt(ufl.dot(p1, p1))

        p2 = ufl.cross(n, p1)

        return p1, p2
 
class BC_Wall3D_base(BCPlasma3D_base):
    
    #Wall boundary conditions. Also good for symmetry planes.  
    
    def __init__(self, strong_imposition = False):  
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)
    
    def u_outside(self, um, normal):   
        v_normal_vec = self.v_normal_vector(um, normal)
        v = self.velocity(um)
        v_out = v - 2*v_normal_vec # Reverse the normal component of the velocity while keeping the tangential component unchanged

        # u_outside is the same as um except that u_normal has been reversed
        return [n(um), 
                n(um)*(v_out[0]), 
                n(um)*(v_out[1]), 
                n(um)*(v_out[2])]

class BC_SupersonicInflow3D_base(BCPlasma3D_base):

    def __init__(self,
                 n_outside=1.0,
                 v_normal_outside=1.1,
                 v_parallel1_outside=0.0,
                 v_parallel2_outside=0.0,
                 strong_imposition=False):

        self.n_outside = n_outside
        self.v_normal_outside = -v_normal_outside  # inflow sign convention
        self.v_parallel1_outside = v_parallel1_outside
        self.v_parallel2_outside = v_parallel2_outside
        self.strong_imposition = strong_imposition

    def u_outside(self, um, normal):

        p1, p2 = self._cartesian_tangential_vectors(normal)

        v_out = (
            self.v_normal_outside * normal +
            self.v_parallel1_outside * p1 +
            self.v_parallel2_outside * p2
        )

        return [self.n_outside,
                self.n_outside * v_out[0],
                self.n_outside * v_out[1],
                self.n_outside * v_out[2]]

    
class BC_SupersonicOutflow3D_base(BCPlasma3D_base):
    
    #Supersonic outflow boundary conditions.  
    #The basis {1p,1n,1y} is right-handed and 1n is always the outward normal.
     
    from src.MathAux import max
    
    def __init__(self, ElectronModel, IonModel, NeutralModel, strong_imposition = False): 
        self.ElectronModel = ElectronModel
        self.NeutralModel = NeutralModel
        self.IonModel = IonModel
        self.strong_imposition = strong_imposition # Whether to strongly impose the BC (True) or weakly impose it (False)

    def u_outside(self, um, normal):  
        
        v_out = self.velocity(um)

        #make normal component supersonic if it is not already
        c = (self.ElectronModel.A_Te*self.ElectronModel.gamma*n(um)**(self.ElectronModel.gamma-1))**0.5
        v_normal = self.v_normal(um, normal)
        v_normal = BC_SupersonicOutflow3D_base.max(v_normal, c)  # Outward normal velocity (should be at least c)

        v_out = (v_out - self.v_normal_vector(um, normal)) + v_normal * normal # Adjust the normal component of the velocity to ensure supersonic outflow

        return [n(um), 
                n(um)*v_out[0],
                n(um)*v_out[1],
                n(um)*v_out[2]] 
'''
