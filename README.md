# DIMAGNO-DG

**Authors:** Manuel Cortés Hernán (manuelco@pa.uc3m.es), Mario Merino (mario.merino@uc3m.es), Diego Garcia-Lahuerta (dieggarc@ing.uc3m.es) — Universidad Carlos III de Madrid.

**DIMAGNO-DG** is a Discontinuous Galerkin (DG) multi-fluid PDE solver built on [FEniCS](https://fenicsproject.org/) (legacy / `dolfin` + `ufl_legacy`) aimed at the simulation of plasma flows in the context of space plasma propulsion. 
The model assumes well-magnetized electrons and any magnetization degree for ions, and it is thus appropriate to study the plasma flow in magnetic nozzles and similar systems. It integrates systems of conservation laws (mass, momentum, energy) for an arbitrary number of fluid species toward their **steady state**, using a Runge–Kutta time-marching scheme as a pseudo-time relaxation. 

The framework is quite **physics-agnostic** by design: the numerical core (DG assembly, numerical fluxes, time propagators) operates on any model that implements a small abstract interface (`PhysicalModel`, `BC`). The repository currently ships one fully worked instance of this interface: a **2D axisymmetric magnetic-nozzle plasma expansion** solver, coupling ion and neutral conservation equations with a magnetized, collisional electron closure.

---

## Table of contents

- [1. Features](#1-features)
- [2. Repository structure](#2-repository-structure)
- [3. Installation](#3-installation)
- [4. Quick start](#4-quick-start)
- [5. Configuration system](#5-configuration-system)
- [6. The magnetic nozzle physics model](#6-the-magnetic-nozzle-physics-model)
- [7. Numerical method](#7-numerical-method)
- [8. Extending DIMAGNO-DG to a new fluid / problem](#8-extending-dimagno-dg-to-a-new-fluid--problem)
- [9. Output and postprocessing](#9-output-and-postprocessing)
- [10. Known issues / TODO](#10-known-issues--todo)

---

## 1. Features

- **Generic DG conservation-law framework**: arbitrary-order Discontinuous Galerkin in space (`dg_degree` configurable), with weak forms (mass, volume flux, surface flux, source term) written once in `WeakForms.py` and reused by any physical model.
- **Time integration**: Strong-Stability-Preserving RK3 (`Propagators.RK`), with an optional row-sum mass-lumping mode for a fully explicit, matrix-free update; Forward Euler and Backward Euler (implicit, Newton-based) propagators are also available.
- **Numerical fluxes**: Local Lax–Friedrichs (default, robust), Global Lax–Friedrichs, and a partially implemented HLLE flux.
- **Reference implementation — magnetic nozzle plasma expansion**:
  - Two-fluid model: ions + neutrals, each with full mass/momentum/energy conservation (`PlasmaModel2D_neutrals_E`, state vector of 9 components).
  - Collisional source terms: electron-impact ionization and charge-exchange (CEX), with optional background neutral pressure.
  - Analytic magnetic fields (current loop / infinite wire arrays) computed via elliptic integrals, plus a generic streamline tracer.
  - Wall, supersonic/subsonic inflow, and supersonic outflow boundary conditions, strongly or weakly imposed.
- **h/p adaptivity hooks**: mesh refinement (`refine_mesh`) and polynomial order increase (`refine_p`) as steps in a simulation "strategy" list.
- **Artificial stabilization**: optional Hartmann–Houston-type diffusive term scaled by the local flux divergence, for shock-capturing/robustness in coarse DG.
- **Convergence-to-steady-state driver**: time-marches in batches and monitors `‖du/dt‖` against a tolerance (`run_to_convergence`), with periodic XDMF checkpointing.
- **Checkpoint/restart**: full solution state saved to XDMF and reloadable via `--load <t>`.
- **Mesh pipeline**: parametric rectangular meshes built directly in `dolfin`, or arbitrary polygonal domains meshed with `gmsh` and converted to XDMF via `meshio`.
- **YAML-based hierarchical configuration**: global `constants.yaml` + per-test override files, deep-merged at load time, so each physical test case only declares what differs from the defaults.

## 2. Repository structure

DIMAGNO-DG is organized as a **generic core** (`src/`) plus one **case folder per physical problem** — e.g. `magnetic_nozzle_2d/` for the reference plasma-expansion case. A case folder is self-contained: it holds the `run.py` entry point, the `setup/` wiring, the `config/` YAML files, and (once executed) the output/results tree for that case.

```
DIMAGNO-DG/
├── src/                                 # generic DG core + reference physics package (case-agnostic)
│   ├── __init__.py                      # public-API facade (see below)
│   ├── numerics/
│   │   ├── Simulations.py               # class Simulation: assembly, time stepping, convergence loop
│   │   ├── Propagators.py               # ForwardEuler, BackwardEuler, RK (SSP-RK3)
│   │   ├── WeakForms.py                 # generic DG weak-form builders (a_form, flux_volume_form, ...)
│   │   └── NumericalFluxes.py           # LocalLaxFriedrichs, GlobalLaxFriedrichs, HLLE (WIP)
│   ├── physics/
│   │   ├── PhysicalModels.py            # abstract PhysicalModel / BC base classes
│   │   └── Plasma/                      # physics package for the magnetic-nozzle case (a "model package")
│   │       ├── __init__.py              # aggregates the species models below into `src.Plasma.*` (not seen — verify!)
│   │       ├── magn/
│   │       │   └── BFields.py           # BField2D/3D, MagneticField_Loop/Planar, Streamline_Tracing
│   │       ├── species/
│   │       │   ├── Ions.py              # IonModel — ion thermodynamic closure
│   │       │   ├── Neutrals.py          # NeutralModel — neutral thermodynamic closure
│   │       │   ├── Electrons.py         # ElectronModel — electron thermodynamic + quasineutral closure
│   │       │   └── Collisions.py        # CollisionModels — ionization, CEX, drag frequencies
│   │       └── models/
│   │           └── PlasmaModel_2D_n_E.py  # PlasmaModel2D_neutrals_E, plasma BCs
│   ├── postpro/
│   │   ├── extract.py                   # grid extraction for postprocessing (called via --extract)
│   │   └── Plotting/
│   │       └── plot_class.py            # plotting utilities (separate from the quick-look helpers in utils/save_load.py)
│   └── utils/
│       ├── save_load.py                 # XDMF I/O (incl. save_xdmf), quick-look plotting, mass/flux diagnostics
│       ├── mesh_creator.py              # gmsh-based polygonal mesh generation → XDMF
│       ├── mesh_reader.py               # .msh → .xdmf conversion (meshio)
│       ├── Elliptic_UFL.py              # UFL-compatible elliptic integrals K, E, Π (power series)
│       ├── config_utils.py              # load_config / list_tests (YAML loading + deep merge)
│       └── MathAux.py                   # smooth max/min/maxabs/minabs (branch-free)
│
└── magnetic_nozzle_2d/                  # case folder — reference plasma-expansion case (one of possibly several)
    ├── run.py                           # CLI entry point for this case (test selection, solve, restart, postpro)
    ├── setup/
    │   ├── setup.py                     # build_parameters, build_mesh, build_magnetic_field, build_models,
    │   │                                 # build_function_spaces, interpolate_fields_on_mesh, build_simulation
    │   └── ic_gaussian.py               # build_ic_functions — analytic Gaussian initial/boundary profiles
    ├── config/
    │   ├── constants.yaml               # global defaults for this case
    │   └── tests/
    │       └── test_<NAME>.yaml         # one file per test/run, overrides only what changes
    └── results/                         # created at run time: output_path(cfg) → results/Test_<name>_.../
```

**Convention**: each new physical *case* gets its own case folder (mirroring `magnetic_nozzle_2d/`, with its own `run.py`, `setup/`, and `config/`), and — if it needs new physics rather than a reparametrization — its own model package under `src/physics/` (mirroring `Plasma/`). The generic `src/numerics core is reused untouched across every case.

## 3. Installation

DIMAGNO-DG depends on **legacy FEniCS 2019** (`dolfin` + `ufl_legacy`), which must be installed through `conda-forge`. The tested platform is **WSL2 / Ubuntu** with Miniconda or Anaconda.

### Step 1 — FEniCS via conda

A ready-to-run install script is provided at the repo root:

```bash
bash Fenics_Installation.sh
```

This script: creates a conda environment named `fenics2019` with Python 3.10, installs `fenics=2019.1.0` from `conda-forge`, applies the required MPI workaround for WSL, and finishes with a Poisson-equation smoke test to confirm the installation is working. To run it manually instead, the equivalent steps are:

```bash
# Create and activate the environment
conda create -n fenics2019 -c conda-forge python=3.10 fenics=2019.1.0 -y
conda activate fenics2019

# WSL only: prevent an MPI shared-memory error
export OMPI_MCA_btl_vader_single_copy_mechanism=none
```

### Step 2 — Python dependencies

With the `fenics2019` environment active, install the remaining pure-Python packages:

```bash
pip install -r requirements.txt
```

`requirements.txt` contents:

```
ufl_legacy      # legacy UFL API shim required by DIMAGNO-DG source files
numpy
scipy
matplotlib
PyYAML
meshio
gmsh
```

### Step 3 — Verify

```bash
conda activate fenics2019
cd DIMAGNO-DG/magnetic_nozzle_2d
python run.py --list-tests
```

## 4. Quick start

```bash
# List available test cases (reads config/tests/test_*.yaml descriptions)
python run.py --list-tests

# Run test "U3" from scratch, to convergence
python run.py --test U3

# Resume test "U3" from a saved checkpoint at t = 0.0
python run.py --test U3 --load 0.0

# Override numerics/convergence from the CLI
python run.py --test U3 --tol 1e-6 --steps 1000 --dg-degree 1 --mesh-param 15

# Run and sample the converged fields onto numpy grids
python run.py --test U3 --extract
```

### CLI reference (`run.py`)

| Flag | Purpose |
|---|---|
| `--test NAME` | Selects `config/tests/test_NAME.yaml` (merged on top of `constants.yaml`) |
| `--list-tests` | Prints all available tests with their `description` field |
| `--load T` | Loads the checkpoint at time `T` from `full_sol/u_t_T.xdmf` before (re)running |
| `--no-run` | Loads a checkpoint but skips the solve step (inspection/postpro only) |
| `--tol` | Overrides `convergence.plasma_tol` |
| `--steps` | Overrides `convergence.steps_per_check` |
| `--max-iter` | Safety cap on convergence-check iterations |
| `--dg-degree` | Overrides `numerics.dg_degree` |
| `--mesh-param` | Overrides `mesh.param` (element-size parameter) |
| `--extract` | Samples all fields to numpy grids via `postpro/extract_mn_2d.py` |
| `--metrics` | Computes forces/powers/efficiencies (currently disabled, see [§10](#10-known-issues--todo)) |

## 5. Configuration system

Configuration is split into two layers:

1. **`config/constants.yaml`** — defaults shared by (almost) every test of a given physical case: reference physical parameters, species properties, mesh sizing, magnetic field geometry, numerics, and convergence criteria.
2. **`config/tests/test_<NAME>.yaml`** — only the keys that differ for that particular run (e.g. utilization efficiency, Mach number, background pressure). Anything not redeclared is inherited from `constants.yaml`.

```yaml
# config/constants.yaml (excerpt)
physical:
  R0: 0.018          # Thruster radius [m]
  Te0: 10.0          # Electron temperature [eV]
  B0: 400.0          # Magnetic field [G]
  A: 131             # Atomic number (Xe)
  m_dot_i_ng: 100.0  # Ionic mass flow [ng/s]

mesh:
  z_length: 10       # Domain length in z [R0]
  x_length: 10       # Domain length in x [R0]
  param: 10           # Element-size parameter (smaller = finer)
  R: 1.0
  R_L: 2.0

numerics:
  dg_degree: 0
  cg_degree: 2
  stabilization: [0.2, 0.0, 0.0]   # (C_eps, beta, ext) — see §7

convergence:
  plasma_tol: 1.0e-7
  steps_per_check: 500
  max_iter: null
```

```yaml
# config/tests/test_U.yaml
test_name: U
initial_conditions:
  eta_u: 0.625                 # Ionization utilization n_i/(n_i+n_n)
  Mi0: 0.5                     # Ion Mach number at the throat
  background_pressure_pa: 0.0  # 0.0 disables the background-neutral-pressure variant
```

Key top-level config sections (see `constants.yaml` for the full set): `physical`, `collisions`, `electrons`, `ions`, `neutrals`, `mesh`, `magnetic_field`, `initial_conditions`, `numerics`, `convergence`, `output`.

`build_parameters(cfg)` (in `setup/setup.py`) turns these raw, dimensional config values into the dimensionless set actually used by the solver (reference density/velocity, normalized magnetic field, normalized background density, etc.), printing a normalization summary at start-up.

## 6. The magnetic nozzle physics model

The reference case solves the **2D axisymmetric expansion of a partially ionized plasma through a magnetic nozzle**, with the state vector

```
u = [n, gz_i, gx_i, gy_i, E_i,   nn, gz_n, gx_n, E_n]
     └──── ions (5) ────────┘    └── neutrals (4) ──┘
```

where `n`, `nn` are number densities, `g_*` are the in-plane/out-of-plane momentum densities, and `E_i`, `E_n` are total (kinetic + thermal) energy densities. All quantities are dimensionless, normalized by a reference density, the thruster radius `R0`, and a reference sound speed `c0`.

- **Ion and neutral fluid equations** (`PlasmaModel2D_neutrals_E.flux` / `.rhs`): standard Euler-type conservation laws, with the ion momentum equation forced by the magnetized `J×B` term (electron and ion diamagnetic/E×B drifts) and an electrostatic potential `φ` derived from the electron pressure balance.
- **Collisional coupling** (`CollisionModels`): dimensionless ionization and charge-exchange (CEX) frequencies, function of neutral density and electron temperature, feeding mass/momentum/energy source terms shared between the ion and neutral equations (`S_i`, `S_cex`, `Q_i`).
- **Electron closure** (`ElectronModel`): electrons are treated as a massless, magnetized, polytropic fluid in quasi-neutral equilibrium.
- **Magnetic field** (`BFields.py`): analytic field of a circular current loop (or planar wire array), computed via complete elliptic integrals (own UFL-compatible power-series implementation, `Elliptic_UFL.py`, plus SciPy for the NumPy path); `Streamline_Tracing` extracts and resamples field lines.
- **Boundary conditions** (`BCPlasma2D_neutrals_E` subclasses): wall/symmetry (normal velocity reflected), supersonic inflow (density + velocities prescribed for both species), subsonic inflow (energy also prescribed for neutrals, since ions remain subsonic), and supersonic outflow (velocity floored at the local sound speed, remaining fields extrapolated).

## 7. Numerical method

- **Spatial discretization**: Discontinuous Galerkin, vector-valued DG space of degree `dg_degree` with `u_dim()` components per cell. The weak form follows the standard DG conservation-law pattern:

  ```
  a(v, du) = b(v, u) · dt
  a(v, du) = ∫ du·v dx                                   (mass matrix)
  b(v, u)  = -∫ F(u):∇v dx + ∫ F̂_n(u⁺,u⁻)·[v] dS + ∫ rhs(u)·v dx + BC terms
  ```

  built generically in `WeakForms.py` and specialized per case only through `PhysicalModel.flux`, `.eigenvalues`, `.rhs`.
- **Numerical flux**: Local Lax–Friedrichs by default (`F̂ = ½(F⁺+F⁻) + ½α(u⁺-u⁻)`, with `α` the local max wave speed from `PhysicalModel.eigenvalues`); Global Lax–Friedrichs and an HLLE variant are also implemented (HLLE marked as not fully working).
- **Stabilization**: an optional Hartmann–Houston-style artificial diffusion term, `ε = C_eps·h^(2-β)·‖∇·F‖ + ext`, added as an extra DG flux-volume term when `stabilization_params[0]` is set in the config (`numerics.stabilization`).
- **Time integration**: third-order Strong-Stability-Preserving Runge–Kutta (`Propagators.RK`), used as a pseudo-time relaxation toward steady state. Each stage solves the same bilinear form `a(v,du)` (the DG mass matrix) against a different right-hand side; an optional row-sum **mass-lumping** mode bypasses the linear solves entirely (`du = b/M_lumped`) for a fully explicit, matrix-free update at the cost of first-order-accurate quadrature of the mass term.
- **Steady-state convergence**: `Simulation.run_to_convergence` advances the solution in batches of `convergence.steps_per_check` time steps, estimates `‖Δu‖ / (n·dt)` over a 20-step window (`time_convergence`), checkpoints to XDMF after every batch, and stops once this rate drops below `convergence.plasma_tol` (or `convergence.max_iter` batches are exhausted). A direct Newton steady-state solve (`Simulation.steady_state`) is also available as an alternative/complement to time marching.
- **Positivity / robustness**: density is floored after every step (`force_positivity`, gated by `positivity_flag`), state validity is asserted (`is_valid`, currently checks density positivity), and the neutral background density floor is reapplied when a `NeutralModel.back_dens` is configured.
- **h/p refinement**: `Simulation.refine_mesh()` (uniform `dolfin.refine`) and `Simulation.refine_p()` (DG degree + 1) can be inserted as steps in a `simulate(strategy=[...])` list, alongside integer step counts, `'steady'`, `'double'`/`'halve'` (time-step rescaling), and cleanup actions.

## 8. Extending DIMAGNO-DG to a new fluid / problem

Because the DG core only talks to the physics through `PhysicalModel`/`BC`, adding a new conservation-law system (a different gas, a different number of species, a non-plasma fluid problem, etc.) does **not** require touching `Simulations.py`, `WeakForms.py`, `Propagators.py`, or `NumericalFluxes.py`. The steps are:

1. **Subclass `PhysicalModel`** (`src/physics/PhysicalModels.py`) and implement:
   - `x_dim()` — spatial dimension (1, 2, or 3).
   - `u_dim()` — number of conserved components.
   - `flux(u)` — flux tensor `F_ij(u)` as a list of lists.
   - `eigenvalues(u, normal)` — wave speeds in the normal direction (used by the numerical flux for upwinding/dissipation).
   - `rhs(u)` — source terms (collisions, geometric/axisymmetric terms, body forces, ...).
   - `is_valid(u)` — physical-admissibility check (e.g. positivity).
   - Optionally `force_positivity`, `neutral_background`/equivalent floors, and `stabilization_params`.
2. **Subclass `BC`** for each boundary type your problem needs (wall, inflow, outflow, symmetry, ...), implementing `u_outside(um, normal)` — the exterior/ghost state used to build the numerical flux at that boundary (see `BC_Wall2D_neutrals_E`, `BC_SupersonicInflow2D_neutrals_E`, etc. for worked examples). Set `strong_imposition=True` on a BC instance to enforce it as a Dirichlet-type strong constraint instead of through the numerical flux.
3. **Write a `setup/<case>/setup.py`** mirroring the magnetic-nozzle one: `build_parameters` (dimensional → dimensionless config), `build_mesh`, any case-specific field builders (e.g. `build_magnetic_field`), `build_models` (your closure-law objects), `build_function_spaces`, and `build_simulation`, which finally instantiates `Simulation(mesh, PhysicalModel, NumericalFlux, Propagator, bcs, dg_degree, dt, ...)`.
4. **Write an initial-conditions module** (`ic_<case>.py`) exposing the analytic profiles needed both to initialize `u` and to evaluate Dirichlet/inflow boundary states.
5. **Add `config/constants.yaml`** for the new case and one `config/tests/test_<NAME>.yaml` per run/scenario.
6. Point `run.py` (or a copy of it) at the new `setup/` module.

## 9. Output and postprocessing

- **Checkpoints**: `src/utils/save_load.py::save_xdmf` (re-exported as `src.save_xdmf`) writes the full DG solution vector to `{output_path}/full_sol/u_t_<t>.xdmf` using `dolfin`'s checkpoint format; `Simulation.read_u_from_file` reloads it. `run.py --load T` restarts from such a checkpoint.
- **Quick-look diagnostics** (`save_load.py`): `plot_2D` / `plot_ratio_2D` (matplotlib field maps), `mass_in_domain`, `mass_flux` (boundary integrals), `save_xdmf_component` (per-component XDMF export). These are explicitly flagged in the source as debugging helpers — "real" postprocessing is intended for ParaView on the saved XDMF files.
- **Grid extraction** (`--extract`): samples all fields onto structured numpy grids for the converged (or loaded) state.
- **Metrics** (`--metrics`): forces/powers/efficiencies postprocessing exists in design but is currently commented out in `run.py` (see below).

## 10. Known issues / TODO

A few inconsistencies were spotted while documenting the code; flagging them here rather than silently working around them:

- **`--metrics`** postprocessing block is commented out in `run.py` (`postpro/metrics.py` referenced but not wired up).
- **`HLLE` numerical flux** is explicitly marked `# TODO complete, not working at the moment` in `NumericalFluxes.py`.

  
