"""
run.py — Magnetic nozzle solver
================================
Usage:
    python run.py --test U3
    python run.py --test U3 --load 0.0
    python run.py --test U3 --load 0.0 --electron-update
    python run.py --list-tests
"""
 
import sys
import os
import copy
import argparse
import logging
 
import numpy as np
import yaml
import dolfin as df
import ufl_legacy as ufl
 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)
 
import src
import src.physics.Plasma as Plasma
 
from setup.setup import (
    build_parameters,
    build_mesh,
    build_magnetic_field,
    build_models,
    build_function_spaces,
    interpolate_fields_on_mesh,
    build_simulation,
)
from setup.ic_gaussian import build_ic_functions
 
df.parameters["ghost_mode"] = "shared_facet"
np.ln = np.log
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

def output_path(cfg, suffix=""):
    m    = cfg["mesh"]
    dg   = cfg["numerics"]["dg_degree"]
    base = cfg["output"]["base_dir"]
    test = cfg["test_name"]
    path = f"{base}/Test_{test}_{m['z_length']}x{m['x_length']}_p_{dg}/"
    return path + suffix + "/" if suffix else path

def parse_args():
    p = argparse.ArgumentParser(description="Magnetic nozzle solver")
    
    # ── Test selection ────────────────────────────────────────
    p.add_argument("--test",            default="U3",   help="Test name (e.g. U3, B1)")
    p.add_argument("--list-tests",      action="store_true")

    # ── Checkpoint ───────────────────────────────────────────
    p.add_argument("--load",            default=None,   metavar="T",    help="Load checkpoint at t=T")
    p.add_argument("--no-solve",        action="store_true",            help="Load checkpoint but do not run simulation")

    # ── Convergence ───────────────────────────────────────────
    p.add_argument("--tol",             default=None,   type=float,     help="Override plasma_tol from config")
    p.add_argument("--steps",           default=None,   type=int,       help="Override steps_per_check from config")
    p.add_argument("--max-iter",        default=None,   type=int,       help="Max convergence iterations (safety cap)")

    # ── Numerics ──────────────────────────────────────────────
    p.add_argument("--dg-degree",       default=None,   type=int,       help="Override DG polynomial degree")
    p.add_argument("--mesh-param",      default=None,   type=int,       help="Override mesh resolution parameter")

    # ── Postprocessing ────────────────────────────────────────
    p.add_argument("--extract",         action="store_true",         help="Sample all fields to numpy grids")
    p.add_argument("--metrics",         action="store_true",         help="Compute forces, powers, efficiencies")

    return p.parse_args()

def apply_overrides(args, cfg):
    """CLI args take priority over config file values."""
    if args.tol:
        cfg["convergence"]["plasma_tol"] = args.tol
    if args.steps:
        cfg["convergence"]["steps_per_check"] = args.steps
    if args.max_iter is not None:                          
        cfg["convergence"]["max_iter"] = args.max_iter
    if args.dg_degree is not None:
        cfg["numerics"]["dg_degree"] = args.dg_degree
    if args.mesh_param is not None:
        cfg["mesh"]["param"] = args.mesh_param

    return cfg
 
def main():
    args = parse_args()
 
    if args.list_tests:
        src.config_utils.list_tests()
        return
 
    # ── 1. Config ─────────────────────────────────────────────
    cfg = src.config_utils.load_config(args.test)
    cfg = apply_overrides(args, cfg)
    log.info("Test: %s", cfg["test_name"])
 
    # ── 2. Physical params ────────────────────────────────────
    pp = build_parameters(cfg)
 
    # ── 3. Mesh + magnetic field ──────────────────────────────
    mesh, All, Inlet, Wall = build_mesh(cfg)
    BField = build_magnetic_field(cfg, pp["B0_norm"])
 
    # ── 4. Species models ─────────────────────────────────────
    ElectronModel, IonModel, NeutralModel, CollisionModel = build_models(cfg, pp)
 
    # ── 5. Initial conditions ─────────────────────────────────
    ic_funcs = build_ic_functions(cfg, pp, ElectronModel, IonModel, NeutralModel)
 
    # ── 6. Function spaces + field interpolation ──────────────
    V_DG, V_CG = build_function_spaces(cfg, mesh)
    Bz, Br, psi = interpolate_fields_on_mesh(cfg, BField, ElectronModel, ic_funcs, V_DG, V_CG)
 
    # ── 7. Simulation object ──────────────────────────────────
    P = build_simulation(
        cfg, mesh, All, Inlet, Wall,
        BField, ElectronModel, IonModel, NeutralModel,
        CollisionModel, ic_funcs, V_DG, V_CG, Bz, Br,
    )
    P.XDMFpath = output_path(cfg)
    log.info("Output path: %s", P.XDMFpath)
 
    # ── 8. Load checkpoint or set initial state ───────────────
    if args.load is not None:
        path = output_path(cfg) + f"full_sol/u_t_{args.load}.xdmf"
        log.info("Loading checkpoint t=%s", args.load)
        P.read_u_from_file(path)
        P.t = float(args.load)
    else:
       if cfg['initial_conditions']['eta_u'] > 0:
            P.u.assign(df.Expression(
                ("pow(10,-8)", "0.", "0.", "0.",
                 "1.5*pow(10,-8)*r_T_i - pow(10,-8)*0.",
                 "back_dens + pow(10,-10)", "0.", "0.",
                 "1.5*pow(10,-8)*r_T_n"),
                back_dens=pp["back_dens"], r_T_i=pp["r_T_i"], r_T_n=pp["r_T_n"], degree=4,
            ))
       else:
            P.u.assign(df.Expression(
                ('pow(10,-8)',
                '0.*pow(10,-8)',  #1.2*exp(-G*std::log(10.)*(x[1]/R)*(x[1]/R))
                '0.',
                '0.',
                '1.5*pow(10,-8)*r_T*0. + 0.5*pow(10,-8)*0.*pow(10,-8)*0.*pow(10,-8) - pow(10,-8)*0.'),
                back_dens= pp["back_dens"], r_T_i=pp["r_T_i"], degree=4,
            ))

    P.Propagator.initialize_solver(P.du, P.a_form, P.b_form, P.u, P.dt)
 
    # ── 9. Solve (optional) ──────────────────────────────────────────────
    if args.no_solve:
        if args.load is None:
            log.error("--no-solve requires --load to have a starting state")
            return
        log.info("Skipping solver (--no-solve)")
    else:
        P.run_to_convergence(cfg, log)
        P.t = 0
        if not args.no_save:
            src.save_xdmf(P)

    # ── 11. Extract grids (optional) ─────────────────────────
    if args.extract:
        from src.postpro.extract_mn_2d import run_extraction
        run_extraction(P, cfg, pp, V_DG, V_CG, Bz, Br, psi,
                       ElectronModel, IonModel, NeutralModel, CollisionModel,
                       cfg["electron_solver"]["je_condition"])

    '''# ── 12. Metrics (optional) ────────────────────────────────
    if args.metrics:
        from postpro.metrics import run_metrics
        run_metrics(P, cfg, pp, V_DG, V_CG, Bz, Br,
                    ElectronModel, IonModel, NeutralModel, CollisionModel,
                    ic_funcs, cfg["electron_solver"]["je_condition"], args.it_number)
    '''
if __name__ == "__main__":
    main()
