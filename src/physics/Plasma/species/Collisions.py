
'''
Collisions Module
'''

#from math import exp, log10
import ufl_legacy as ufl
import numpy as np

class CollisionModels:
    
    '''
    Class containing models for different types of collisions.
    
    All methods return dimensionless collision frequencies
    or related normalized quantities.
    '''

    def __init__(self, sigma_0_i = 5e-20 , sigma_0_cex = 81e-20,
                  sigma_0_en = 27e-20, n0 = 7e18, R0 = 18e-3, Te0 = 10, E_ion = 12.1, A = 131, B0 = 0.0875):
        
        '''
        Init function. 
        Default values for n0 and c0 have been taken from the low beta case in Ahed10f.
        '''

        self.sigma_0_i = sigma_0_i #Cross section for ionization default is values on enrique's paper
        self.sigma_0_cex = sigma_0_cex #Cross section for ionization default is values on enrique's paper
        self.sigma_0_en = sigma_0_en
        self.n0 = n0 #Initial plasma density in m^-3 for normalization
        self.R0 = R0 #Nozzle radius in m for normalization, default value is Radius of Marco's ECR.
        self.Te0 = Te0 #Electron temperature in electronvolts
        self.E_ion = E_ion
        self.alpha = self.n0 * self.R0**3
        self.A = A #Atomic number of the gas used, 131 corresponds to xenon
        self.me =  1/(1836*self.A) #Electon mass in units of the ion mass
        self.c0 = (1.6e-19 * self.Te0/(self.A * 1.67e-27))**0.5 #Speed of sound in ms^-1
        self.B0 = B0

        #Non dimensionalisation with n0 and R0 
        #leads to discrepancies alpha corrects them.

    def Ionization(self, nn, Te, lib = ufl):
        
        '''
        Returns the dimensionless ionization frequency.
        '''

        vth = 3e8*(self.Te0*Te/0.51e6)**0.5
        vth_ = vth/self.c0
        sigma_0_ = self.sigma_0_i/self.R0**2
        
        E_ion_ = self.E_ion/self.Te0

        sigma = sigma_0_*(1 + Te*E_ion_/(Te + E_ion_)**2)* lib.exp(-E_ion_/Te)

        nu_i = self.alpha * nn*vth_*sigma

        return nu_i
        

    def CEX(self, nn, vi, vn, Te, lib = ufl):
        
        '''
        Returns the dimensionless CEX frequency.
        '''

        np.ln = np.log
        vth = 3e8*(self.Te0*Te/0.51e6)**0.5
        vth_ = vth/self.c0
        sigma_0_ = self.sigma_0_cex/self.R0**2
        
        vel = 1000 #ms^-1 This factor appears in the model for the cex cross section
        vel_ = vel/self.c0
        if lib == ufl:
            c_in = lib.conditional(lib.sqrt(lib.dot(vi-vn, vi-vn)) > 1e-6, lib.sqrt(lib.dot(vi-vn, vi-vn)), 1e-6)
        else:
            c_in = lib.sqrt((vi[0]-vn[0])**2 + (vi[1]-vn[1])**2)

        sigma = sigma_0_*(1- 0.2*(lib.ln(c_in/vel_)/lib.ln(10)))**2

        nu_cex = self.alpha* nn *c_in * sigma

        return nu_cex
    
    def E_N(self, nn, Te, me, lib = ufl):
        
        '''
        Electron-Neutral drag collision frequency
        '''

        sigma = self.sigma_0_en/self.R0**2
        ce = lib.sqrt(Te/me)

        return self.alpha * nn * ce * sigma
    
    def E_I(self, ne, Te, me, lib = ufl):
        
        '''
        Electron-Ion drag collision frequency
        '''

        np.ln = np.log
        ln_Delta = 9 + 0.5*lib.ln((1e18/(self.n0*ne))*(self.Te0*Te/1))
        Rei = (1e-12*self.n0*self.R0/self.c0)*2.9*(1/(self.Te0*Te))**1.5*ln_Delta
        ni = ne

        return ni*Rei
    
    def Excitation(self, ne, Te, lib = ufl):
        
        '''
        Returns the effective ionization energy, this is, the ionization energy corrected to
        account for inelastic excitation collisions.
        '''
        
        E_ion_ = self.E_ion/self.Te0
        E_inel = (1 + 0.25*lib.exp(2*E_ion_/(3*Te)))*E_ion_

        return E_inel 
