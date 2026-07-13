'''
Numerical fluxes
'''
 
class NumericalFlux: 
    def flux_n(self, u_p, u_m, normal_p, PhysicalModel): 
        '''
        Computes the numerical flux 
        
        Parameters:
        -----------
        u_p, u_m: state vectors on plus and minus sides
        normal_p: Outward normal vector from plus side
        PhysicalFlux: Object with flux and eigenvalues methods 

        Returns:
        --------
        F_num: numerical flux as a list of same size as u
        '''
        raise NotImplementedError("Abstract class")


class LocalLaxFriedrichs(NumericalFlux):
    '''
    Local Lax-Friedrichs numerical flux
    '''
    from src.utils.MathAux import maxabs

    def flux_n(self, u_p, u_m, normal_p = None, PhysicalModel = None, bc = False): 

        if bc:
            F_p = PhysicalModel.flux_n(u_p, normal_p, sign = '+')
            F_m = PhysicalModel.flux_n(u_m, normal_p, sign = None)

            eigenvalues_p = PhysicalModel.eigenvalues(u_p, normal_p, sign = '+')
            eigenvalues_m = PhysicalModel.eigenvalues(u_m, normal_p, sign = None)
            alpha = LocalLaxFriedrichs.maxabs(*eigenvalues_p,*eigenvalues_m)
        
            return  [0.5*(F_p[i] + F_m[i] + alpha*(u_p[i] - u_m[i])) for i in range(len(u_p))]
            #return [0.5 * (F_p[i] + F_m[i]) for i in range(len(u_p))]

        else:
            F_p = PhysicalModel.flux_n(u_p, normal_p, sign = '+')
            F_m = PhysicalModel.flux_n(u_m, normal_p, sign = '-')
        
            eigenvalues_p = PhysicalModel.eigenvalues(u_p, normal_p, sign = '+')
            eigenvalues_m = PhysicalModel.eigenvalues(u_m, normal_p, sign = '-')
            alpha = LocalLaxFriedrichs.maxabs(*eigenvalues_p,*eigenvalues_m)
            
            return  [0.5*(F_p[i] + F_m[i] + alpha*(u_p[i] - u_m[i])) for i in range(len(u_p))]
            #return  [0.5*(F_p[i] + F_m[i] + 0.5*alpha*(u_p[i] - u_m[i])) for i in range(len(u_p))]


class GlobalLaxFriedrichs(NumericalFlux):
    '''
    Global Lax-Friedrichs numerical flux
    '''  
    def __init__(self, alpha):
        self.alpha = alpha  # Global Lax-Friedrichs viscosity

    def flux_n(self, u_p, u_m, normal_p, PhysicalModel): 
        F_p = PhysicalModel.flux_n(u_p, normal_p)
        F_m = PhysicalModel.flux_n(u_m, normal_p)
     
        return  [0.5*(F_p[i] + F_m[i] + self.alpha*(u_p[i] - u_m[i])) for i in range(len(u_p))]

class HLLE(NumericalFlux):
    '''
    HLLE as in Nate Sime's Paper.
    '''
    #TODO complete, not working at the moment.
    from src.utils.MathAux import maxabs, minabs, max, min
    from math import fabs

    tol = 1e-14
    
    def flux_n(self, u_p, u_m, He_p = None, He_m = None, normal_p = None, PhysicalModel = None):


        F_p = PhysicalModel.flux_n(u_p, normal_p, He_p)
        F_m = PhysicalModel.flux_n(u_m, normal_p, He_m)

        eigens_p = PhysicalModel.eigenvalues(u_p, normal_p, He_p)
        eigens_m = PhysicalModel.eigenvalues(u_m, normal_p, He_m)

        lam_p = HLLE.max(*eigens_p, 0)
        lam_m = HLLE.min(*eigens_m, 0)

        safe = 1/((lam_p-lam_m)+1e-10)

        return [safe*(lam_p*F_m[i] - lam_m*F_p[i]) + safe*(lam_p*lam_m)*(u_p[i]-u_m[i]) for i in range(len(u_p))]
    