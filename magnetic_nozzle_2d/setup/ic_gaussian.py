
"""
ic.py — Initial conditions for the magnetic nozzle case
=========================================================
All functions accept (x, lib=np) so they work both with
numpy arrays (setup) and ufl expressions (boundary conditions).
"""
 
import numpy as np
 
 
def build_ic_functions(cfg, pp, ElectronModel, IonModel, NeutralModel):
    """
    Returns a dict of callables, each with signature f(x, lib=np).
    pp   : physical params dict from build_physical_params()
    n0_  : normalized ion density at throat
    nn0_ : normalized neutral density at throat
    Mi0  : initial ion Mach number at throat
    """
    r_T_i     = pp["r_T_i"]
    r_T_n     = pp["r_T_n"]
    G       = cfg["initial_conditions"]["G"]
    eps     = cfg["initial_conditions"]["eps"]
    R       = cfg["mesh"]["R"]
    phi_max = cfg["initial_conditions"].get("phi_max", 0.0)
 
    G = cfg["initial_conditions"]["G"] #Density falls to 10^-G at x = R
    eps = cfg["initial_conditions"]["eps"] #To avoid zero densities

    n0_ = pp['n0_']
    nn0_ = pp['nn0_']
    Mi0 = pp['Mi0']

    def n0(x,lib=np):
        return (n0_* lib.exp(-G*lib.ln(10.)*(x/R)**2.) + eps)/(1 + eps)
    def dn_dx0(x,lib=np):
        return -n0_*2*G*lib.ln(10)*(x/R**2)*lib.exp(-G*lib.ln(10)*(x/R)**2)/(1+eps)
    def A0(x,lib=np):
        return 1/((n0(x, lib=lib)/ElectronModel.ne0)**(ElectronModel.gamma-1))
    def dA_dx0(x,lib=np):
        return -(ElectronModel.gamma-1)*n0(x,lib=lib)**(-ElectronModel.gamma)*dn_dx0(x,lib=lib)
    def Te0_func(x,lib=np):
        return ElectronModel.Te0*A0(x,lib=lib)*(n0(x, lib=lib)/ElectronModel.ne0)**(ElectronModel.gamma-1)
    def He0(x,lib=np):
        return ElectronModel.gamma/(ElectronModel.gamma-1) * Te0_func(x,lib=lib) - phi0(x,lib=lib)
    def dHe_dx0(x,lib=np):
        return ElectronModel.gamma/(ElectronModel.gamma-1) * ElectronModel.Te0/ElectronModel.ne0**(ElectronModel.gamma-1) * (n0(x, lib=lib)**(ElectronModel.gamma-1)*dA_dx0(x,lib=lib) + A0(x,lib=lib)*(ElectronModel.gamma-1)*n0(x, lib=lib)**(ElectronModel.gamma-2)*dn_dx0(x,lib=lib)) - dphi_dx0(x,lib=lib)
    def uzi0(x,lib=np):    
        return Mi0*((ElectronModel.c(n0(x, lib= lib), A0(x,lib=lib)))**2 + IonModel.gamma*r_T_i)**0.5
    def pi0(x, lib= np):
        return (n0(x, lib = lib)*r_T_i)/(1)
    def nn0(x,lib=np):
        return nn0_*lib.exp(-G*lib.ln(10.)*(x/R)**2. + eps)/(1+eps)
    def uzn0(x,lib=np):    
        return (5/3*r_T_n)**0.5 + 0.*x
    def pn0(x, lib= np):
        return (nn0(x, lib = lib)*r_T_n)
    def ux0(x,lib=np):
        return x*0
    def uy0(x,lib=np):
        return 0.*x
    def phi0(x,lib=np):
        return phi_max + 0.*x
    def dphi_dx0(x,lib=np):
        return phi_max * 10/lib.sqrt(lib.pi) * lib.exp(-(10*(x-R))**2) + 0.*x

    def Ei0(x,lib=np):
        return (0.5*n0(x, lib=lib)*uzi0(x, lib=lib)**2 + pi0(x, lib=lib)/(IonModel.gamma - 1.) + n0(x, lib=lib)*phi0(x, lib=lib))
    def En0(x,lib=np):
        return (0.5*nn0(x, lib=lib)*uzn0(x, lib=lib)**2 + pn0(x, lib=lib)/(NeutralModel.gamma - 1.))

 
    return dict(
        n0=n0, nn0=nn0,
        uzi0=uzi0, uzn0=uzn0,
        ux0=ux0, uy0=uy0,
        Ei0=Ei0, En0=En0,
        phi0=phi0, A0=A0,
        He0=He0, dHe_dx0=dHe_dx0,
        dn_dx0=dn_dx0, dA_dx0=dA_dx0, dphi_dx0=dphi_dx0,
    )
 
