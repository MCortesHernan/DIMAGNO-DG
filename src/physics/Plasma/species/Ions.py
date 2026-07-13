'''
Dimensionless Ion model
'''

class IonModel:
    '''
    Dimensionless neutral model class, grouping all thermodynamic functions.
    '''
    def __init__(self, gamma = 5./3., Ti0_val = 1., e_internal = True, ElectronModel = None):
        self.gamma = gamma
        self.Ti0_val = Ti0_val
        self.e_internal = e_internal
        self.ElectronModel = ElectronModel

        #Introducing ElectronModel because c for ions system will depend on Te aswell

    def pi(self, n=1., u_squared = 1., E = 1., phi = 1.):
        '''
        Computes the ion temperature
        '''
        if self.e_internal:
            return 2./3.*E
        else:
            return (self.gamma-1)*(E - 0.5*n*u_squared - n*phi)

    def Ti(self, n = 1., u_squared = 1., E = 1., phi = 1.):
        '''
        Computes the ion pressure
        '''
        return self.pi(n, u_squared, E, phi)/(n + 1e-12) #Avoid division by zero
    
    def c(self, n = 1., u_squared = 1., E = 1., phi = 1.):
        '''
        Computes the sound speed
        '''
        gamma_e = self.ElectronModel.gamma
        #return (self.gamma*self.Ti(n, u_squared, E, phi) + gamma_e*n**(gamma_e-1))**0.5
        return (gamma_e*self.ElectronModel.Te(n))**0.5
    
    def Ti_poly(self, n = 1.):
        '''
        Computes the ion temperature assuming polytropic profile 
        '''
        return self.Ti0_val*n**(self.gamma-1)
    
    def pi_poly(self, n = 1.):
        '''
        Computes the ion pressure assuming polytropic profile 
        '''
        return self.Ti_poly(n)*n
    

