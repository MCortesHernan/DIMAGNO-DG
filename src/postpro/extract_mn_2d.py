"""
postpro/extract.py — Field extraction to numpy grids
=====================================================
Loads a converged solution and electron parameters, samples all
fields onto a regular numpy grid, and saves them as .npy files.

This is the slow step — run once after convergence.
Everything downstream (metrics, plots) works from the saved grids.

Called from run.py via --extract, or directly:
    from postpro.extract import run_extraction
    run_extraction(P, cfg, pp, V_DG, V_CG, Bz, Br, psi,
                   ElectronModel, IonModel, NeutralModel, CollisionModel,
                   je_condition, it_number)
"""

import os
import numpy as np
import dolfin as df
import ufl_legacy as ufl

import src

def make_grid(cfg):
    """Returns z_grid, x_grid for the mesh dimensions in cfg."""
    m = cfg["mesh"]
    z_array = np.linspace(0, m["z_length"], m["z_length"] * 50)
    x_array = np.linspace(0, m["x_length"], m["x_length"] * 50)
    return np.meshgrid(z_array, x_array)


def sample_grid(Z, X, f, name="data"):
    """Evaluates a dolfin Function f on every point of the (Z, X) grid."""
    grid = np.zeros_like(Z)
    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            grid[i, j] = f(df.Point(Z[i, j], X[i, j]))
    print(f"  Sampled: {name}")
    return grid

def load_electron_params(save_path, V_DG, ElectronModel):
    """
    Loads He, A_Te, G_e and their no-collision counterparts
    from the updated_df_param/ folder and attaches them
    to ElectronModel.
    """
    load_path = save_path + "updated_df_param/"
    print(f"Loading electron params from: {load_path}")

    G_e            = src.save_load.load_df_function(V_DG, "G_e_df",              load_path)
    G_e_nocoll     = src.save_load.load_df_function(V_DG, "G_e_nocoll_df",       load_path)
    He             = src.save_load.load_df_function(V_DG, "He_df",               load_path)
    He_nocoll      = src.save_load.load_df_function(V_DG, "He_nocoll_df",        load_path)
    dHe_dpsi       = src.save_load.load_df_function(V_DG, "dHe_dpsi_df",         load_path)
    dHe_dpsi_nocoll= src.save_load.load_df_function(V_DG, "dHe_dpsi_nocoll_df",  load_path)
    A_Te           = src.save_load.load_df_function(V_DG, "A_Te_df",             load_path)
    A_Te_nocoll    = src.save_load.load_df_function(V_DG, "A_Te_nocoll_df",      load_path)
    dA_Te_dpsi     = src.save_load.load_df_function(V_DG, "dA_Te_dpsi_df",       load_path)
    dA_Te_dpsi_nocoll = src.save_load.load_df_function(V_DG, "dA_Te_dpsi_nocoll_df", load_path)

    for attr, val in [
        ("G_e", G_e), ("G_e_nocoll", G_e_nocoll),
        ("He", He), ("He_nocoll", He_nocoll),
        ("dHe_dpsi", dHe_dpsi), ("dHe_dpsi_nocoll", dHe_dpsi_nocoll),
        ("A_Te", A_Te), ("A_Te_nocoll", A_Te_nocoll),
        ("dA_Te_dpsi", dA_Te_dpsi), ("dA_Te_dpsi_nocoll", dA_Te_dpsi_nocoll),
    ]:
        setattr(ElectronModel, attr, val)

    return G_e, G_e_nocoll, He, He_nocoll, dHe_dpsi, dHe_dpsi_nocoll, \
           A_Te, A_Te_nocoll, dA_Te_dpsi, dA_Te_dpsi_nocoll

def get_neutral_fields(P, cfg, NeutralModel):
    """
    Returns (nn_df, jzn_df, jxn_df, En_df) accounting for tests
    where neutrals are background-only (back_pres > 0, no neutral flow).
    This replaces the hardcoded test_num > 5 logic.
    """
    back_pres = cfg["initial_conditions"].get("background_pressure_pa", 0.0)
    back_dens = NeutralModel.back_dens

    if back_pres > 0:
        # Background pressure test: neutrals are a uniform background,
        # no flowing neutral component
        nn_df  = df.Constant(back_dens)
        jzn_df = df.Constant(1e-12)
        jxn_df = df.Constant(1e-12)
        Tn_const = df.Constant(NeutralModel.r_T)
        En_df = df.Constant(NeutralModel.back_dens*Tn_const / (NeutralModel.gamma - 1))
        print("  Background-pressure test: using uniform neutral background")
    else:
        # Standard test: neutrals are part of the solution
        nn_df  = P.u.sub(5)
        jzn_df = P.u.sub(6)
        jxn_df = P.u.sub(7)
        En_df  = P.u.sub(8)

    return nn_df, jzn_df, jxn_df, En_df

def compute_derived_grids(grids, ElectronModel, IonModel, NeutralModel, CollisionModel):
    """
    Computes all derived quantities from the raw sampled grids.
    Returns a dict of name → numpy array.
    """
    g = grids   # shorthand

    # Velocities
    uzi  = g["jzi"] / (g["n"]  + 1e-8)
    uxi  = g["jxi"] / (g["n"]  + 1e-8)
    uyi  = g["jyi"] / (g["n"]  + 1e-8)
    uzn  = g["jzn"] / (g["nn"] + 1e-8)
    uxn  = g["jxn"] / (g["nn"] + 1e-8)

    ui_plane = np.sqrt(uzi**2 + uxi**2)
    ui       = np.sqrt(uzi**2 + uxi**2 + uyi**2)
    un       = np.sqrt(uzn**2 + uxn**2)

    ji_plane = np.sqrt(g["jzi"]**2 + g["jxi"]**2)
    jn_plane = np.sqrt(g["jzn"]**2 + g["jxn"]**2)

    # Pressures and temperatures
    phi  = ElectronModel.phi(g["n"], g["He"], g["A_Te"])
    Te   = ElectronModel.Te(g["n"], g["A_Te"])
    pi   = (g["Ei"] - 0.5 * g["n"] * ui**2 - g["n"] * phi) * (IonModel.gamma - 1.0)
    pn   = (g["En"] - 0.5 * g["nn"] * un**2) * (NeutralModel.gamma - 1.0)
    Ti   = pi   / (g["n"]  + 1e-8)
    Tn   = pn   / (g["nn"] + 1e-8)

    # Speed of sound and Mach
    c    = ElectronModel.c(g["n"], g["A_Te"])
    cn   = (NeutralModel.gamma * Tn) ** 0.5
    Mi   = ui / (c  + 1e-8)
    Mn   = un / (cn + 1e-8)

    # Magnetic
    B    = np.sqrt(g["Bz"]**2 + g["Br"]**2)

    # Electron potential energy
    Phi_big = phi + ElectronModel.gamma / (ElectronModel.gamma - 1.0) \
              * (-g["A_Te"] * (g["n"] / ElectronModel.ne0) ** (ElectronModel.gamma - 1.0))

    # Collisions
    nu_i   = CollisionModel.Ionization(g["nn"], Te, lib=np)
    nu_cex = CollisionModel.CEX(g["nn"], [uzi, uxi], [uzn, uxn], Te, lib=np)
    nu_en  = CollisionModel.E_N(g["nn"], Te, CollisionModel.me, np)
    nu_ei  = CollisionModel.E_I(g["n"],  Te, CollisionModel.me, np)
    nu_exc = nu_i * CollisionModel.Excitation(g["n"], Te, np) \
             * (CollisionModel.Te0 / CollisionModel.E_ion)
    nu_e   = nu_ei + nu_en + nu_i + nu_exc
    Si     = nu_i   * g["n"]
    S_cex  = nu_cex * g["n"]

    # Hall parameters
    Hall_e  = B / (CollisionModel.me * nu_e  + 1e-12)
    Hall_en = B / (CollisionModel.me * nu_e  + 1e-12)
    Hall_ei = B / (CollisionModel.me * nu_i  + 1e-12)

    # Electron azimuthal and cross-field currents
    x_grid = grids["x_grid"]
    uye    = (-x_grid * g["dHe_dpsi"]
              + 1.0 / (ElectronModel.gamma - 1.0)
              * (ElectronModel.Te0 / ElectronModel.ne0 ** (ElectronModel.gamma - 1.0))
              * g["n"] ** (ElectronModel.gamma - 1.0)
              * x_grid * g["dA_Te_dpsi"])

    jye_nocoll = -uye * g["n"]
    jye_coll   = -jye_nocoll / (Hall_e ** 2)
    jye        = jye_nocoll + jye_coll
    jperpe     = jye_nocoll / (Hall_e)

    jze = g["jpare"] * g["Bz"] / (B + 1e-12) + jperpe * g["Br"] / (B + 1e-12)
    jxe = g["jpare"] * g["Br"] / (B + 1e-12) - jperpe * g["Bz"] / (B + 1e-12)

    je_plane = np.sqrt(g["jpare"]**2 + jperpe**2)
    je       = np.sqrt(g["jpare"]**2 + jperpe**2 + jye**2)
    jt_plane = np.sqrt((jze + g["jzi"])**2 + (jxe + g["jxi"])**2)

    return dict(
        uzi=uzi, uxi=uxi, uyi=uyi, uzn=uzn, uxn=uxn,
        ui_plane=ui_plane, ui=ui, un=un,
        ji_plane=ji_plane, jn_plane=jn_plane,
        phi=phi, Te=Te, pi=pi, pn=pn, Ti=Ti, Tn=Tn,
        c=c, cn=cn, Mi=Mi, Mn=Mn, B=B, Phi_big=Phi_big,
        nu_i=nu_i, nu_cex=nu_cex, nu_en=nu_en, nu_ei=nu_ei,
        nu_exc=nu_exc, nu_e=nu_e, Si=Si, S_cex=S_cex,
        Hall_e=Hall_e, Hall_en=Hall_en, Hall_ei=Hall_ei,
        uye=uye, jye_nocoll=jye_nocoll, jye_coll=jye_coll, jye=jye,
        jperpe=jperpe, jze=jze, jxe=jxe,
        je_plane=je_plane, je=je, jt_plane=jt_plane,
    )


# ─────────────────────────────────────────────────────────────
# Main extraction function
# ─────────────────────────────────────────────────────────────

def run_extraction(P, cfg, pp, V_DG, V_CG, Bz, Br, psi,
                   ElectronModel, IonModel, NeutralModel, CollisionModel,
                   je_condition):
    """
    Full extraction pipeline:
      1. Build save/load paths
      2. Load electron params from disk
      3. Get neutral fields (handles background-pressure tests)
      4. Save all df functions
      5. Sample all fields onto numpy grids
      6. Compute derived quantities
      7. Save all grids as .npy
    """
    m    = cfg["mesh"]
    dg   = cfg["numerics"]["dg_degree"]
    base = cfg["output"]["base_dir"]
    test = cfg["test_name"]

    save_path      = f"{base}/Test_{test}_{m['z_length']}x{m['x_length']}_p_{dg}/{je_condition}/"
    save_grid_path = save_path + "grid_data/"
    save_df_path   = save_path + "df_data/"

    os.makedirs(save_grid_path, exist_ok=True)
    os.makedirs(save_df_path,   exist_ok=True)
    print(f"Extraction paths:\n  grids → {save_grid_path}\n  df    → {save_df_path}")

    # ── 1. Load electron params ───────────────────────────────
    (G_e, G_e_nocoll, He, He_nocoll,
     dHe_dpsi, dHe_dpsi_nocoll,
     A_Te, A_Te_nocoll,
     dA_Te_dpsi, dA_Te_dpsi_nocoll) = load_electron_params(save_path, V_DG, ElectronModel)

    # Recompute jpare with actual Bz, Br
    jpare = ElectronModel.jpare
    jpare.assign(df.project(-G_e * (Bz**2 + Br**2)**0.5, V_DG))

    # ── 2. Solution fields ────────────────────────────────────
    n_df   = P.u.sub(0)
    jzi_df = P.u.sub(1)
    jxi_df = P.u.sub(2)
    jyi_df = P.u.sub(3)
    Ei_df  = P.u.sub(4)

    nn_df, jzn_df, jxn_df, En_df = get_neutral_fields(P, cfg, NeutralModel)

    # ── 3. Save df functions ──────────────────────────────────
    print("Saving df functions...")
    fields_df = [
        (n_df,               "n_df"),
        (jzi_df,             "jzi_df"),
        (jxi_df,             "jxi_df"),
        (jyi_df,             "jyi_df"),
        (Ei_df,              "Ei_df"),
        (nn_df,              "nn_df"),
        (jzn_df,             "jzn_df"),
        (jxn_df,             "jxn_df"),
        (En_df,              "En_df"),
        (G_e,                "G_e_df"),
        (G_e_nocoll,         "G_e_nocoll_df"),
        (He,                 "He_df"),
        (He_nocoll,          "He_nocoll_df"),
        (dHe_dpsi,           "dHe_dpsi_df"),
        (dHe_dpsi_nocoll,    "dHe_dpsi_nocoll_df"),
        (A_Te,               "A_Te_df"),
        (A_Te_nocoll,        "A_Te_nocoll_df"),
        (dA_Te_dpsi,         "dA_Te_dpsi_df"),
        (dA_Te_dpsi_nocoll,  "dA_Te_dpsi_nocoll_df"),
        (jpare,              "jpare_df"),
        (Bz,                 "Bz_df"),
        (Br,                 "Br_df"),
        (psi,                "psi_df"),
    ]
    for field, name in fields_df:
        src.save_load.save_df_function(save_df_path, field, name)

    # ── 4. Build sampling grid ────────────────────────────────
    z_grid, x_grid = make_grid(cfg)
    np.save(save_grid_path + "z_grid.npy", z_grid)
    np.save(save_grid_path + "x_grid.npy", x_grid)

    # ── 5. Sample raw fields ──────────────────────────────────
    print("Sampling raw fields onto grid...")
    raw = dict(
        n          = sample_grid(z_grid, x_grid, n_df,              "n"),
        jzi        = sample_grid(z_grid, x_grid, jzi_df,            "jzi"),
        jxi        = sample_grid(z_grid, x_grid, jxi_df,            "jxi"),
        jyi        = sample_grid(z_grid, x_grid, jyi_df,            "jyi"),
        Ei         = sample_grid(z_grid, x_grid, Ei_df,             "Ei"),
        nn         = sample_grid(z_grid, x_grid, nn_df,             "nn"),
        jzn        = sample_grid(z_grid, x_grid, jzn_df,            "jzn"),
        jxn        = sample_grid(z_grid, x_grid, jxn_df,            "jxn"),
        En         = sample_grid(z_grid, x_grid, En_df,             "En"),
        G_e        = sample_grid(z_grid, x_grid, G_e,               "G_e"),
        G_e_nocoll = sample_grid(z_grid, x_grid, G_e_nocoll,        "G_e_nocoll"),
        He         = sample_grid(z_grid, x_grid, He,                "He"),
        He_nocoll  = sample_grid(z_grid, x_grid, He_nocoll,         "He_nocoll"),
        dHe_dpsi   = sample_grid(z_grid, x_grid, dHe_dpsi,          "dHe_dpsi"),
        dHe_dpsi_nocoll = sample_grid(z_grid, x_grid, dHe_dpsi_nocoll, "dHe_dpsi_nocoll"),
        A_Te       = sample_grid(z_grid, x_grid, A_Te,              "A_Te"),
        A_Te_nocoll= sample_grid(z_grid, x_grid, A_Te_nocoll,       "A_Te_nocoll"),
        dA_Te_dpsi = sample_grid(z_grid, x_grid, dA_Te_dpsi,        "dA_Te_dpsi"),
        dA_Te_dpsi_nocoll = sample_grid(z_grid, x_grid, dA_Te_dpsi_nocoll, "dA_Te_dpsi_nocoll"),
        jpare      = sample_grid(z_grid, x_grid, jpare,             "jpare"),
        Bz         = sample_grid(z_grid, x_grid, Bz,                "Bz"),
        Br         = sample_grid(z_grid, x_grid, Br,                "Br"),
        psi        = sample_grid(z_grid, x_grid, psi,               "psi"),
        x_grid     = x_grid,
    )

    # ── 6. Compute derived quantities ─────────────────────────
    print("Computing derived quantities...")
    derived = compute_derived_grids(raw, ElectronModel, IonModel, NeutralModel, CollisionModel)

    # ── 7. Save all grids ─────────────────────────────────────
    print("Saving grids...")
    all_grids = {**raw, **derived}
    all_grids.pop("x_grid")   # already saved separately

    for name, grid in all_grids.items():
        np.save(save_grid_path + name + ".npy", grid)

    print(f"Extraction complete. {len(all_grids)} grids saved to {save_grid_path}")
    return save_grid_path