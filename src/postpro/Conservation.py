'''
Postprocessing to determine conservation of magnitudes assesed by simulation
Mass, momentum and total energy assesed 
See that conservation is evaluated from numpy arrays - this could result in a source of error

To be considered an alternative conservation evaluation directly on fenics functions
'''
import numpy as np
import src
from scipy.integrate import simpson
import os
import matplotlib.pyplot as plt
import dolfin as df
from numpy import isinf
from numpy import where
import ufl_legacy as ufl

class Conservation:
    '''
    Class to evaluate conservation of mass, mmentum and energy '
    '''

    def __init__(self, z, x, dens_grid, jx_grid, jz_grid, pi_grid, force_grid, E_grid, gamma, axisymmetric = False):
        '''
        Constructor
        '''
        self.z = z
        self.x = x
        self.dens_grid = dens_grid
        self.jx_grid = jx_grid
        self.jz_grid = jz_grid
        self.pi_grid = pi_grid
        self.force_grid = force_grid
        self.E_grid = E_grid
        self.gamma = gamma
        self.axisymmetric = axisymmetric

    def momentum_conservation_surface(self):
        '''
        Evaluate mass conservation
        '''

        '''S1 is the integral for the surface in the 
        x= 0 plane, this is, the symmetry plane.'''

        int_S1 = np.zeros_like(self.z)

        for i in range(1,len(self.z)):
            if self.axisymmetric:
                #This evaluates to zero if x_min = 0
                int_S1[i] = 2*np.pi*simpson(-self.x[0]*self.jz_grid[0, 0:i]*self.jx_grid[0, 0:i]/self.dens_grid[0, 0:i], self.z[0:i])
            else:
                int_S1[i] = simpson(-self.jz_grid[0, 0:i]*self.jx_grid[0, 0:i]/self.dens_grid[0, 0:i], self.z[0:i])

        '''S2 is the integral for the surface in a certain z
        the last value is the momentum leaving through 
        the z = z_max surface.'''

        int_S2 = np.zeros_like(self.z)

        for i in range(1,len(self.z)):
            if self.axisymmetric:
                int_S2[i] = 2*np.pi*simpson(self.x*(self.jz_grid[:, i]**2/self.dens_grid[:, i] + self.dens_grid[:, i]**self.gamma + self.pi_grid[:, i]), self.x)
            else:
                int_S2[i] = simpson(self.jz_grid[:, i]**2/self.dens_grid[:, i] + self.dens_grid[:, i]**self.gamma + self.pi_grid[:, i], self.x)
        
        '''S3 is the integral for the surface in the 
        x = x_max plane, this is the external wall paralel
        to the symmetry plane.'''

        int_S3 = np.zeros_like(self.z)

        for i in range(1,len(self.z)):

            if self.axisymmetric:
                int_S3[i] = 2*np.pi*simpson(self.x[-1]*self.jz_grid[-1, 0:i]*self.jx_grid[-1, 0:i]/self.dens_grid[-1, 0:i], self.z[0:i])
            else:
                int_S3[i] = simpson(self.jz_grid[-1, 0:i]*self.jx_grid[-1, 0:i]/self.dens_grid[-1, 0:i], self.z[0:i])

        '''S4 is the integral for the surface at z = 0
        this is the inlet surface.'''

        if self.axisymmetric:
            int_S4 = 2*np.pi*simpson(self.x*(-self.jz_grid[:, 0]**2/self.dens_grid[:, 0] - self.dens_grid[:, 0]**self.gamma-self.pi_grid[:, 0]), self.x)*np.ones_like(self.z)
        else:
            int_S4 = simpson(-self.jz_grid[:, 0]**2/self.dens_grid[:, 0] - self.dens_grid[:, 0]**self.gamma - self.pi_grid[:, 0], self.x)*np.ones_like(self.z)
        int_S4[0] = 0

        return int_S1, int_S2, int_S3, int_S4

    def momentum_conservation_volume(self):
        
        field = self.force_grid
        
        '''
        Calculate de volume integral of a 2D scalar map.
        return is:
            - int_0^rmax (f(z,r)) dr
            - int_0^z dz' int_0^rmax (f(z',r)) dr
        '''

        integral_z = np.zeros(len(self.z))

        for i in range(len(self.z)):
            if self.axisymmetric:
                integral_z[i] = 2*np.pi* simpson( self.x* field[:,i], self.x)
            else:
                integral_z[i] = simpson(field[:,i], self.x)

        integral = np.zeros(len(self.z))

        for i in range(len(self.z)):
            integral[i] = simpson(integral_z[:i+1], self.z[:i+1])

        return integral, integral_z
    
    def mass_conservation_surface(self):

        '''S1 is the integral for the surface in the 
        x= 0 plane, this is, the symmetry plane.'''

        int_S1 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):
            if self.axisymmetric:
                int_S1[i] = 2*np.pi*simpson(-self.x[0]*self.jz_grid[0, 0:i], self.z[0:i])
            else:
                int_S1[i] = simpson(-self.jx_grid[0, 0:i], self.z[0:i])

        '''S2 is the integral for the surface in a certain z
        the last value is the mass leaving through 
        the z = z_max surface.'''

        int_S2 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):
            if self.axisymmetric:
                int_S2[i] = 2*np.pi*simpson(self.x*self.jz_grid[:, i], self.x)
            else:
                int_S2[i] = simpson(self.jz_grid[:, i], self.x)

        '''S3 is the integral for the surface in the 
        x = x_max plane, this is the external wall parallel
        to the symmetry plane.'''

        int_S3 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):

            if self.axisymmetric:
                int_S3[i] = 2*np.pi*simpson(self.x[-1]*self.jx_grid[-1, 0:i], self.z[0:i])
            else:
                int_S3[i] = simpson(self.jx_grid[-1, 0:i], self.z[0:i])

        '''S4 is the integral for the surface at z = 0
        this is the inlet surface.'''

        if self.axisymmetric:
            int_S4 = 2*np.pi*simpson(self.x*(-self.jz_grid[:, 0]), self.x)*np.ones_like(self.z)
        else:
            int_S4 = simpson(-self.jz_grid[:, 0], self.x)*np.ones_like(self.z)
        int_S4[0] = 0

        return int_S1, int_S2, int_S3, int_S4
    
    def energy_conservation_surface(self):
        '''
        Evaluate energy conservation
        '''

        '''S1 is the integral for the surface in the 
        x= 0 plane, this is, the symmetry plane.'''

        int_S1 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):
            if self.axisymmetric:
                int_S1[i] = 2*np.pi*simpson(-self.x[0]*self.jz_grid[0, 0:i]*self.pi_grid[0, 0:i], self.z[0:i])
            else:
                int_S1[i] = simpson(-(self.E_grid[0, 0:i] + self.pi_grid[0, 0:i])*(self.jx_grid[0, 0:i]/self.dens_grid[0, 0:i]), self.z[0:i])

        '''S2 is the integral for the surface in a certain z
        the last value is the energy leaving through 
        the z = z_max surface.'''

        int_S2 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):
            if self.axisymmetric:
                int_S2[i] = 2*np.pi*simpson(self.x*self.jz_grid[:, i]*self.pi_grid[:, i], self.x)
            else:
                int_S2[i] = simpson((self.E_grid[:, i] + self.pi_grid[:, i])*(self.jz_grid[:, i]/self.dens_grid[:, i]), self.x)

        '''S3 is the integral for the surface in the 
        x = x_max plane, this is the external wall parallel
        to the symmetry plane.'''

        int_S3 = np.zeros_like(self.z)

        for i in range(1, len(self.z)):

            if self.axisymmetric:
                int_S3[i] = 2*np.pi*simpson(self.x[-1]*self.jx_grid[-1, 0:i]*self.pi_grid[-1, 0:i], self.z[0:i])
            else:
                int_S3[i] = simpson((self.E_grid[-1, 0:i] + self.pi_grid[-1, 0:i])*(self.jx_grid[-1, 0:i]/self.dens_grid[-1, 0:i]), self.z[0:i])

        '''S4 is the integral for the surface at z = 0
        this is the inlet surface.'''

        if self.axisymmetric:
            int_S4 = 2*np.pi*simpson(self.x*(-self.jz_grid[:, 0])*self.pi_grid[:, 0], self.x)*np.ones_like(self.z)
        else:
            int_S4 = simpson(-(self.E_grid[:, 0] + self.pi_grid[:, 0])*(self.jz_grid[:, 0]/self.dens_grid[:, 0]), self.x)*np.ones_like(self.z)
        int_S4[0] = 0

        return int_S1, int_S2, int_S3, int_S4




                