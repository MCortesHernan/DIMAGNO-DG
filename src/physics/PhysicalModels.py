'''
Physical fluxes in the bulk and at boundary conditions. Abstract classes
'''

class PhysicalModel:  
    '''
    Abstract class for the physical fluxes
    '''

    def x_dim(self):
        '''
        Dimension of geometric space.
        '''
        raise NotImplementedError("Abstract class")

    def u_dim(self):
        '''
        Dimension of state vector.
        '''
        raise NotImplementedError("Abstract class")

    def flux(self, u):
        '''
        Computes the flux function as a list of lists.
        First index is the component, second index is the coordinate.
        '''
        raise NotImplementedError("Abstract class")

    def flux_n(self, u, normal, sign = None):
        '''
        Computes the flux function multiplied by a unit normal vector as list. 
        '''
        f = self.flux(u, sign)

        return [sum([fij*nj for (fij, nj) in zip(fi, normal)]) for fi in f]

    def eigenvalues(self, u, normal): 
        '''
        Computes the eigenvalues of the Jacobian of the flux function in the n direction as a list.
        '''
        raise NotImplementedError("Abstract class")

    def rhs(self, u):
        '''
        Computes the right hand side of the PDE.

        Parameters
        ---------- 
        u : list. State vector.
        '''
        raise NotImplementedError("Abstract class")

    def is_valid(self, u):
        '''
        Checks if the state vector is valid.

        Parameters
        ----------
        u : State vector. If state vector at many points are given, this should be a list of state vectors: u = [[u1], [u2], ..., [un]].
        '''
        raise NotImplementedError("Abstract class")
 
 
class BC:
    '''
    Abstract class for the boundary conditions
    ''' 
    def F_boundary(self, um, normal):
        pass
        
    def u_outside(self, um, normal):
        '''
        Computes the state vector u outside of the domain.
        
        Parameters:
        -----------
        um: state vector inside
        normal: outward normal vector

        Returns:
        --------
        u_outside: state vector outside
        '''
        raise NotImplementedError("Abstract class")

class BC_Neumann(BC):
    '''
    Neumann boundary conditions.
    ''' 
    def u_outside(self, um, normal): 
        return um


class BC_Dirichlet(BC):
    '''
    Dirichlet boundary conditions.
    ''' 
    def u_outside(self, um, normal): 
        return [0.]*len(um)
        
