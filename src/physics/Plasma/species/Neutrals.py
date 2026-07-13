'''
Dimensionless Neutral model
'''

class NeutralModel:
    '''
    Dimensionless neutral model class, grouping all thermodynamic functions.
    '''
    def __init__(self, r_n=1, r_T = 1, gamma=5./3., back_dens = 0, e_internal = True, is_polytropic = False):
        '''
        -is_polytropic = Boolean, if True Tn = Tn0 *nn^gamma/nn0^gamma, if False use energy equation
        If not polytropic care to give all required magnitudes for the calulation of the temperature
        
        -gamma = Polytropic index
        
        -r_n = ratio of neutral to plasma densities in the entrance n_n0/n_i0
        
        -r_T = ratio of neutral to plasma temperatures in the entrance T_n0/T_e0
        -back_dens = background density of neutrals in units of ion density at the nozzle throat
        '''
        self.is_polytropic = is_polytropic
        self.gamma = gamma
        self.r_n = r_n
        self.r_T = r_T
        self.back_dens = back_dens
        self.e_internal = e_internal

    def Tn(self, n=1., u_squared = 0., E = 0.):
        
        #Computes the electron temperature
        
        if self.is_polytropic:
            return self.Tn_poly(n)
        else:
            return self.pn(n, u_squared, E)/(n + 1e-10) #Avoid division by zero

    def pn(self, n=1., u_squared = 0., E = 0.):
        '''
        Computes the electron pressure
        '''
        if self.e_internal:
            return 2./3.*E
        else:
            return (self.gamma-1)*(E - 0.5*n*u_squared)
        
    def Tn_poly(self, n=1.):
        '''
        Computes the neutral temperature assuming polytropic profile 
        '''
        return self.r_T*self.r_n**(self.gamma-1)*n**(self.gamma-1)
    
    def pn_poly(self, n=1.):
        '''
        Computes the neutral pressure assuming polytropic profile 
        '''
        return self.Tn_poly(n)*n
    
    def c(self, Tn=1):
        '''
        Computes the sound speed
        '''
        return (self.gamma*Tn)**0.5
