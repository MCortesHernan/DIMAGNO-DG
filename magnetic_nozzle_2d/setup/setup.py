
"""
setup.py — Mesh, models, function spaces, and field interpolation
==================================================================
Everything that needs to run once before the time loop:
  - mesh + subdomains
  - magnetic field
  - species models + collision model
  - FEniCS function spaces
  - interpolation of He, A_Te, dHe_dpsi, dA_dpsi onto DG/CG spaces
  - PhysicalFlux, boundary conditions
  - Simulation object
"""
 
import numpy as np
import dolfin as df
import ufl_legacy as ufl
 
import src
from src import Plasma

def build_parameters(cfg):
    phys   = cfg["physical"]
    e      = 1.6e-19
    k_B    = 1.38e-23
    K_to_eV = 8.62e-5
 
    R0     = phys["R0"]
    Te0    = phys["Te0"]
    A      = phys["A"]
    mi_dim = A * 1.67e-27
    c0     = (e * Te0 / mi_dim) ** 0.5
    m_dot_i = phys["m_dot_i_ng"] * 1e-9       # ng/s → kg/s
 
    T_n_eV  = cfg["neutrals"]["T_n_eV"]
    r_T_n   = T_n_eV / Te0
    T_n_K   = T_n_eV / K_to_eV

    T_i_eV  = cfg["ions"]["T_i_eV"]
    r_T_i   = T_i_eV / Te0
    T_i_K   = T_i_eV / K_to_eV  

    # ratio_dens and uz_throat are normalization constants specific
    # to this case — derived from a converged reference run.
    Mi0 = cfg["initial_conditions"]["Mi0"]
    ratio_dens = 0.14547458113783976
    gamma_e = cfg["electrons"]["gamma"]
    gamma_i = cfg["ions"]["gamma"]
    uz_throat  = Mi0*np.sqrt(gamma_e + gamma_i*r_T_i) * c0
    n0_m3 = m_dot_i / (np.pi * R0**2 * mi_dim * uz_throat * c0 * ratio_dens)
 
    back_pres = cfg["initial_conditions"].get("background_pressure_pa", 0.0)
    back_dens = (back_pres / (k_B * T_n_K)) / n0_m3 if back_pres > 0 else 0.0
 
    # Magnetic field. In normalized units the ion magnetic force is n u_i x B with coefficient 1,
    # so the reference field is the one with ion Hall parameter Omega_i R0 / c0 = 1.
    B0      = phys["B0"]                          # physical field at the throat centre [G]
    B_ref   = mi_dim * c0 / (e * R0) * 1e4        # [G]
    B0_norm = B0 / B_ref                          # normalized |B| at the throat = ion Hall parameter
    if "B_norm" in phys and abs(phys["B_norm"] / B_ref - 1) > 1e-2:
        print(f"WARNING: config B_norm = {phys['B_norm']:.0f} G is ignored; B_ref = {B_ref:.0f} G is computed from A, Te0 and R0")

    n0_ = 1.0
    gamma_n = cfg["neutrals"]["gamma"]
    csn0 = (gamma_n*r_T_n)**0.5
    eta_u = cfg["initial_conditions"]["eta_u"]
    m_dot_n = ((1-eta_u)/eta_u)*m_dot_i
    nn0_ = (m_dot_n/(np.pi*R0**2*mi_dim*csn0*c0*ratio_dens))/n0_m3 #Neutral density at the throat from mass flow conservation

    print(f"Mi0      = {Mi0:.3f}")
    print(f"c0       = {c0:.1f} m/s")
    print(f"n0_m3    = {n0_m3:.3e} m⁻³")
    print(f"m_dot_i  = {m_dot_i:.3e} kg/s")
    print(f"B0       = {B0:.1f} G,  B_ref = {B_ref:.1f} G,  B0_norm = {B0_norm:.4f}")
    print(f"r_T_n    = {r_T_n:.4f}")
    print(f"r_T_i    = {r_T_i:.4f}")
    print(f'back_pres = {back_pres:.3e} Pa')
    print(f"back_dens = {back_dens:.3e} (normalized)")
    print(f"n0_      = {n0_:.3e} (normalized)")
    print(f"nn0_     = {nn0_:.3e} (normalized)")

    return dict(
        e=e, k_B=k_B, K_to_eV=K_to_eV,
        R0=R0, Te0=Te0, A=A, mi_dim=mi_dim, c0=c0,
        m_dot_i=m_dot_i, n0_m3=n0_m3, ratio_dens=ratio_dens,
        r_T_n=r_T_n, T_n_K=T_n_K,
        r_T_i=r_T_i, T_i_K=T_i_K,
        back_pres=back_pres, back_dens=back_dens,
        B0=B0, B_ref=B_ref, B0_norm=B0_norm,
        n0_ = n0_, nn0_ = nn0_, Mi0=Mi0,
    )

def build_mesh(cfg):
    """Builds the FEniCS mesh and subdomain objects."""
    m    = cfg["mesh"]
    z_len = m["z_length"]
    x_len = m["x_length"]
    param = m["param"]
    R_L   = m["R_L"]
 
    mesh = df.RectangleMesh(
        df.Point(0, 0), df.Point(z_len, x_len),
        z_len * param, x_len * param,
        "left/right",
    )
    mesh = df.refine(mesh)
 
    class All(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
 
    class Inlet(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[0], 0) and df.between(x[1], (0, R_L))
 
    class Wall(df.SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and df.near(x[1], 0)
 
    return mesh, All(), Inlet(), Wall()
 
def build_magnetic_field(cfg, B0_norm):
    """Instantiates and rescales the magnetic field."""
    mf  = cfg["magnetic_field"]
    R_L = mf["R_L"]
    I   = mf["I"]
 
    if mf["type"] == "Loop":
        BField = Plasma.MagneticField_Loop(R_L=R_L, I=I)
    else:
        raise NotImplementedError(f"Field type not implemented: {mf['type']}")
 
    # |B| = B0_norm at the throat centre (z, r) = (0, 0) (B0_norm = B0/B_ref, see build_parameters)
    BField.rescale(0.0, 0.0, B_target=B0_norm)
    return BField
 
def build_models(cfg, pp):
    """
    Creates ElectronModel, IonModel, NeutralModel, CollisionModels.
    """
    ic      = cfg["initial_conditions"]
    eta_u   = ic["eta_u"]
    Mi0     = ic["Mi0"]
    r_T_i   = pp["r_T_i"]
    r_T_n   = pp["r_T_n"]
    gamma_n = cfg["neutrals"]["gamma"]
 
    ElectronModel = Plasma.ElectronModel(
        ne0=pp['n0_'],
        gamma=cfg["electrons"]["gamma"],
        add_A=True,
    )
    IonModel = Plasma.IonModel(
        gamma=cfg["ions"]["gamma"],
        Ti0_val=r_T_i,
        e_internal=cfg["ions"]["e_internal"],
        ElectronModel=ElectronModel,
    )
    NeutralModel = Plasma.NeutralModel(
        gamma=cfg["neutrals"]["gamma"],
        r_n=pp['nn0_'] / pp['n0_'],
        r_T=r_T_n,
        back_dens=pp["back_dens"],
        e_internal=cfg["neutrals"]["e_internal"],
        is_polytropic=cfg["neutrals"]["is_polytropic"],
    )
    coll = cfg["collisions"]
    phys = cfg["physical"]
    CollisionModel = Plasma.CollisionModels(
        sigma_0_i=coll["sigma_0_i"],
        sigma_0_cex=coll["sigma_0_cex"],
        sigma_0_en=coll["sigma_0_en"],
        n0=pp["n0_m3"],
        R0=pp["R0"],
        Te0=pp["Te0"],
        E_ion=phys["E_ion"],
        A=phys["A"],
        B0=pp["B0"],
    )
 
    return ElectronModel, IonModel, NeutralModel, CollisionModel
 

def build_function_spaces(cfg, mesh):
    dg_degree = cfg["numerics"]["dg_degree"]
    cg_degree = cfg["numerics"]["cg_degree"]
    V_DG = df.FunctionSpace(mesh, "DG", dg_degree)
    V_CG = df.FunctionSpace(mesh, "CG", cg_degree)
    return V_DG, V_CG
 
 
def interpolate_fields_on_mesh(cfg, BField, ElectronModel, ic_funcs, V_DG, V_CG):
    """
    Interpolates psi, Bz, Br onto the mesh and sets up He, A_Te,
    dHe_dpsi, dA_Te_dpsi in both DG and CG spaces.
    Attaches all functions as attributes of ElectronModel.
    """
    R_L = cfg["mesh"]["R_L"]
 
    # ── B field on DG ──────────────────────────────────────────
    coords_dg = V_DG.tabulate_dof_coordinates()
    psi = df.Function(V_DG);  psi.vector()[:]  = BField.psi(coords_dg[:, 0], coords_dg[:, 1])
    Bz  = df.Function(V_DG);  Bz.vector()[:]   = BField.Bz(coords_dg[:, 0], coords_dg[:, 1])
    Br  = df.Function(V_DG);  Br.vector()[:]   = BField.Br(coords_dg[:, 0], coords_dg[:, 1])
 
    # ── Reference arrays along the axis (x=0) ─────────────────
    _x         = np.linspace(0, R_L, 20000)
    _psi0      = BField.psi(0, _x)
    _Bz0       = BField.Bz(0, _x)
    _He0       = ic_funcs["He0"](_x)
    _dHe_dx0   = ic_funcs["dHe_dx0"](_x)
    _A0        = ic_funcs["A0"](_x)
    _dA_dx0    = ic_funcs["dA_dx0"](_x)
    _dHe_dpsi0 = _dHe_dx0 / (_x * _Bz0 + 1e-12)
    _dA_dpsi0  = _dA_dx0  / (_x * _Bz0 + 1e-12)
 
    # ── DG functions ───────────────────────────────────────────
    He            = df.Function(V_DG)
    dHe_dpsi      = df.Function(V_DG)
    A_Te          = df.Function(V_DG)
    dA_Te_dpsi    = df.Function(V_DG)
    G_e           = df.Function(V_DG)
 
    ElectronModel.setup_interpolators(_psi0, _He0, _dHe_dpsi0)
    dHe_dpsi.vector()[:] = ElectronModel.interpolate_dHe_dpsi(psi.vector())
    # Fix NaNs on the symmetry axis
    failed = np.where(np.isnan(dHe_dpsi.vector()[:]))
    dHe_dpsi.vector()[failed[0]] = _dHe_dpsi0[1]
    He.vector()[:] = ElectronModel.interpolate_He(psi.vector())
 
    ElectronModel.setup_interpolators(_psi0, _A0, _dA_dpsi0)
    A_Te.vector()[:]       = ElectronModel.interpolate_He(psi.vector())
    dA_Te_dpsi.vector()[:] = ElectronModel.interpolate_dHe_dpsi(psi.vector())
 
    # No-collision copies (initial state)
    He_nocoll         = df.Function(V_DG);  He_nocoll.vector()[:]         = He.vector()[:]
    dHe_dpsi_nocoll   = df.Function(V_DG);  dHe_dpsi_nocoll.vector()[:]   = dHe_dpsi.vector()[:]
    A_Te_nocoll       = df.Function(V_DG);  A_Te_nocoll.vector()[:]       = A_Te.vector()[:]
    dA_Te_dpsi_nocoll = df.Function(V_DG);  dA_Te_dpsi_nocoll.vector()[:] = dA_Te_dpsi.vector()[:]
    G_e_nocoll        = df.Function(V_DG)
 
    # ── CG functions ───────────────────────────────────────────
    coords_cg  = V_CG.tabulate_dof_coordinates()
    psi_cont   = df.Function(V_CG)
    psi_cont.vector()[:] = BField.psi(coords_cg[:, 0], coords_cg[:, 1])
 
    He_cont         = df.Function(V_CG)
    dHe_dpsi_cont   = df.Function(V_CG)
    A_Te_cont       = df.Function(V_CG)
    dA_Te_dpsi_cont = df.Function(V_CG)
    G_e_cont        = df.Function(V_CG)
 
    ElectronModel.setup_interpolators(_psi0, _He0, _dHe_dpsi0)
    He_cont.vector()[:]       = ElectronModel.interpolate_He(psi_cont.vector())
    dHe_dpsi_cont.vector()[:] = ElectronModel.interpolate_dHe_dpsi(psi_cont.vector())
 
    ElectronModel.setup_interpolators(_psi0, _A0, _dA_dpsi0)
    A_Te_cont.vector()[:]      = ElectronModel.interpolate_He(psi_cont.vector())
    dA_Te_dpsi_cont.vector()[:] = ElectronModel.interpolate_dHe_dpsi(psi_cont.vector())
 
    jpare      = df.Function(V_DG)
    jpare_cont = df.Function(V_CG)
 
    # ── Attach everything to ElectronModel ────────────────────
    ElectronModel.He            = He
    ElectronModel.dHe_dpsi      = dHe_dpsi
    ElectronModel.A_Te          = A_Te
    ElectronModel.dA_Te_dpsi    = dA_Te_dpsi
    ElectronModel.G_e           = G_e
    ElectronModel.He_nocoll     = He_nocoll
    ElectronModel.dHe_dpsi_nocoll = dHe_dpsi_nocoll
    ElectronModel.A_Te_nocoll   = A_Te_nocoll
    ElectronModel.dA_Te_dpsi_nocoll = dA_Te_dpsi_nocoll
    ElectronModel.G_e_nocoll    = G_e_nocoll    
    ElectronModel.He_cont       = He_cont
    ElectronModel.dHe_dpsi_cont = dHe_dpsi_cont
    ElectronModel.A_Te_cont     = A_Te_cont
    ElectronModel.dA_Te_dpsi_cont = dA_Te_dpsi_cont
    ElectronModel.G_e_cont      = G_e_cont
    ElectronModel.jpare         = jpare
    ElectronModel.jpare_cont    = jpare_cont
 
    return Bz, Br, psi
 
 
def build_simulation(cfg, mesh, All, Inlet, Wall,
                     BField, ElectronModel, IonModel, NeutralModel,
                     CollisionModel, ic_funcs,
                     V_DG, V_CG, Bz, Br):
    """
    Assembles PhysicalFlux, BCs and the Simulation object.
    Returns (P).
    """
    m    = cfg["mesh"]
    num  = cfg["numerics"]
    esol = cfg["electron_solver"]
    dg_degree = num["dg_degree"]
 
    z_array = np.linspace(0, m["z_length"], m["z_length"] * 50)
    x_array = np.linspace(0, m["x_length"], m["x_length"] * 50)
    z_grid, x_grid = np.meshgrid(z_array, x_array)
 
    # ── Physical flux ─────────────────────────────────────────
    if cfg['initial_conditions']['eta_u'] > 0:
        PhysicalFlux = Plasma.PlasmaModel2D_neutrals_E(
            ElectronModel=ElectronModel,
            ElectronSolver=None,
            IonModel=IonModel,
            NeutralModel=NeutralModel,
            Bz=Bz, Bx=Br,
            CollisionModel=CollisionModel,
            V_DG=V_DG, V_CG=V_CG, mesh=mesh,
            electron_mag=True, ion_mag=True,
            stabilization_params=tuple(num["stabilization"]),
            axisymmetry=True,
            positivity_flag=True,
        )
    else:
        PhysicalFlux = Plasma.PlasmaModel2D_neutrals_E_back_p(
            ElectronModel=ElectronModel,
            ElectronSolver=None,
            IonModel=IonModel,
            NeutralModel=NeutralModel,
            Bz=Bz, Bx=Br,
            CollisionModel=CollisionModel,
            V_DG=V_DG, V_CG=V_CG, mesh=mesh,
            electron_mag=True, ion_mag=True,
            stabilization_params=tuple(num["stabilization"]),
            axisymmetry=True,
            positivity_flag=False,
        )
     
    # ── Boundary conditions ───────────────────────────────────
    x_sym = df.SpatialCoordinate(mesh)[1]
    if cfg['initial_conditions']['eta_u'] > 0:
        bcs = {
            All: Plasma.BC_SupersonicOutflow2D_neutrals_E(
                ElectronModel, IonModel, NeutralModel, strong_imposition=False),
            Wall: Plasma.BC_Wall2D_neutrals_E(strong_imposition=False),
            Inlet: Plasma.BC_SupersonicInflow2D_neutrals_E(
                n_outside           = ic_funcs["n0"](x_sym,   lib=ufl),
                v_normal_outside    = ic_funcs["uzi0"](x_sym,  lib=ufl),
                v_parallel_outside  = ic_funcs["ux0"](x_sym,   lib=ufl),
                vyi_outside         = ic_funcs["uy0"](x_sym,   lib=ufl),
                Ei_outside          = ic_funcs["Ei0"](x_sym,   lib=ufl),
                nn_outside          = ic_funcs["nn0"](x_sym,   lib=ufl),
                vn_normal_outside   = ic_funcs["uzn0"](x_sym,  lib=ufl),
                vn_parallel_outside = ic_funcs["ux0"](x_sym,   lib=ufl),
                En_outside          = ic_funcs["En0"](x_sym,   lib=ufl),
                strong_imposition=False,
            ),
        }
    else:
        bcs = {
            All: Plasma.BC_SupersonicOutflow2D_neutrals_E_back_p(
                ElectronModel, IonModel, NeutralModel, strong_imposition=False),
            Wall: Plasma.BC_Wall2D_neutrals_E_back_p(strong_imposition=False),
            Inlet: Plasma.BC_SupersonicInflow2D_neutrals_E_back_p(
                n_outside           = ic_funcs["n0"](x_sym,   lib=ufl),
                v_normal_outside    = ic_funcs["uzi0"](x_sym,  lib=ufl),
                v_parallel_outside  = ic_funcs["ux0"](x_sym,   lib=ufl),
                vyi_outside         = ic_funcs["uy0"](x_sym,   lib=ufl),
                Ei_outside          = ic_funcs["Ei0"](x_sym,   lib=ufl),
                strong_imposition=False,
            ),
        }
 
    # ── Simulation object ─────────────────────────────────────
    dt = 1.0 / (2 * dg_degree + 1) * mesh.hmin() / (cfg['initial_conditions']['Mi0'] + 1)
    P = src.Simulation(
        mesh=mesh,
        PhysicalModel=PhysicalFlux,
        NumericalFlux=src.NumericalFluxes.LocalLaxFriedrichs(),
        Propagator=src.Propagators.RK(),
        bcs=bcs,
        dg_degree=dg_degree,
        dt=dt,
        refinement=num["refinements"],
    )
 
    return P

    