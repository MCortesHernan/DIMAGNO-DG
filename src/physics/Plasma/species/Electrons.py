'''
Dimensionless electron model
'''
import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson
import dolfin as df

class ElectronModel:
    '''
    Dimensionless electron model class, grouping all thermodynamic functions.
    '''
    def __init__(self, ne0 = 1, Te0 = 1, gamma=5./3., 
                 He = 1, He_cont = 1, He_nocoll = 1, 
                 dHe_dpsi = 1, dHe_dpsi_cont = 1, dHe_dpsi_nocoll = 1, dHe_dpsi_smooth = 1,
                 jpare = 0, jpare_cont = 1, G_e = 1, G_e_cont = 1, G_e_nocoll = 1,
                 A_Te = 1, A_Te_cont = 1, A_Te_nocoll = 1, 
                 dA_Te_dpsi = 1, dA_Te_dpsi_cont = 1, dA_Te_dpsi_nocoll = 1, dA_Te_dpsi_smooth = 1,
                 add_A = False):

        '''
        Initialize electron model parameters.

        Most inputs are precomputed dimensionless profiles
        or interpolation-ready arrays/functions.
        '''

        # Reference normalization quantities
        self.ne0 = ne0   # Reference electron density
        self.Te0 = Te0   # Reference electron temperature
        self.gamma = gamma  # Adiabatic index

        # Energy-related fields
        self.He = He
        self.He_cont = He_cont
        self.He_nocoll = He_nocoll

        self.dHe_dpsi = dHe_dpsi
        self.dHe_dpsi_cont = dHe_dpsi_cont
        self.dHe_dpsi_nocoll = dHe_dpsi_nocoll
        self.dHe_dpsi_smooth = dHe_dpsi_smooth

        # Parallel electron current
        self.jpare = jpare
        self.jpare_cont = jpare_cont

        # Electron energy flux terms
        self.G_e = G_e
        self.G_e_cont = G_e_cont
        self.G_e_nocoll = G_e_nocoll

        # Electron temperature functions
        self.A_Te = A_Te
        self.A_Te_cont = A_Te_cont
        self.A_Te_nocoll = A_Te_nocoll

        self.dA_Te_dpsi = dA_Te_dpsi
        self.dA_Te_dpsi_cont = dA_Te_dpsi_cont
        self.dA_Te_dpsi_nocoll = dA_Te_dpsi_nocoll
        self.dA_Te_dpsi_smooth = dA_Te_dpsi_smooth

        # Model configuration flags
        self.add_A = add_A

    def Te(self, n=1, A=1):
        '''
        Computes the electron temperature
        '''
        if self.gamma == 1.:
            return n*0. + 1. # Multiplication is to keep array shape
        else:
            return A*self.Te0*(n/self.ne0)**(self.gamma-1)
            
    def Te_poly(self, n=1):
        '''
        Computes the electron temperature assuming a polytropic model
        '''
        if self.gamma == 1.:
            return n*0. + 1. # Multiplication is to keep array shape
        else:
            return self.Te0*(n/self.ne0)**(self.gamma-1)

    def c(self, n=1., A=1.):
        '''
        Sound speed
        '''
        return (self.gamma*self.Te(n,A))**0.5
    
    def pe(self, n=1., A=1.):
        '''
        Computes the electron pressure
        '''
        return n*self.Te(n, A)

    def he(self, n=1., A=1.): 
        '''
        Computes the electron barotropy function
        ''' 
        if self.add_A:
            return self.gamma/(self.gamma-1) * self.Te(n, A)
        else:
            return self.gamma/(self.gamma-1) * self.Te0*(A*(n/self.ne0)**(self.gamma-1) - 1)
    
    def phi(self, n=1., He=0., A=1.):
        '''
        Computes phi given n and He
        '''
        return self.he(n, A) - He
    
    def He_init(self, n=1., phi=0., A=1.):
        '''
        Computes the electron He function
        '''
        return self.he(n, A) - phi
    
    def dhe_dn(self, n=1., A=1.): 
        '''
        Computes the derivative of the electron barotropy function with respect to n
        ''' 
        if self.gamma == 1.:    
            return 1./n 
        else:
            return self.gamma * self.Te0 * A * n**(self.gamma-2)/self.ne0**(self.gamma-1)
    
    def dHe_dxi(self, n=1., dn_dxi=0., dphi_dxi=0., A=1.):
        '''
        Computes the derivative of the electron He function in the generic direction xi
        ''' 
        return self.dhe_dn(n, A) * dn_dxi - dphi_dxi

    def setup_interpolators(self, _psi, _He, _dHe_dpsi):
        '''
        Stores vectors _psi, _He, and _dHe_dpsi and creates interpolation methods.
        '''
        # Create internal interpolation libraries and store inputs
        self._psi = _psi 
        self._He = _He
        self._dHe_dpsi = _dHe_dpsi

        # Create interpolator functions and set them as callable methods of the instance
        self.interpolate_He = interp1d(self._psi, self._He, fill_value=(self._He[0], self._He[-1]), bounds_error=False) 
        self.interpolate_dHe_dpsi = interp1d(self._psi, self._dHe_dpsi, fill_value=0., bounds_error=False)
    
    def electron_flux(self, _x, _n, _uz0, x_sep):
        '''
        Calculate the needed electron current for zero net current.
        '''

        j_i_inlet = simpson(_n*_uz0, _x)
        
        #Find closest point to the location of the separatrix

        idx = (np.abs(_x - x_sep)).argmin()

        j_i_sep = simpson( _n[idx:]*_uz0[idx:], _x[idx:])

        heaviside = np.ones_like(_x)
        heaviside[:idx] = 0.

        return np.array(-heaviside * (j_i_inlet/j_i_sep) * _n*_uz0)


    def electron_current_interp_set(self, _psi, _B, _j_e):
        '''
        Stores vectors _psi, _B, _n and _u_e_par and creates interpolation methods.
        '''
        self._psi = _psi
        self._B = _B
        self._j_e = _j_e
        self._j_e_par_B = self._j_e / self._B

        self.interpolate_j_e_par_B = interp1d(self._psi, self._j_e_par_B, fill_value=0., bounds_error=False)
    

