
# --- IMPORTS --- #

import gc
import numpy as np
import dolfin as df
import ufl_legacy as ufl
from scipy.spatial import cKDTree


'''
ElectronSolver_3D

3D counterpart of ElectronSolver restricted to the parallel electron current (no A / He updates).

Electron continuity, with u_e = u_par_e b + u_e_perp and div(B) = 0:

    div(n u_par_e b) = B d/ds (n u_par_e / B)  ->  along each magnetic line, with G = n u_par_e / B:

        dG/ds = ( S - div(Gamma_perp) ) / B

    Gamma_perp = n u_e_perp = f_m x B / B^2
    f_m        = -n u_e x B = n grad_perp(He) - Te0/((gamma-1) ne0^(gamma-1)) n^gamma grad_perp(A)
                 (the electron magnetic force of PlasmaModel3D_base, plus the He term)
    S          = nu_ion n

In 2D (axisymmetric) div(Gamma_perp) = 0 and this reduces to ElectronSolver.G_streamline_f_ivp.

Same conventions as ElectronSolver:  jpare = -G |B|,  u_par_e = -jpare / n = G |B| / n.

Boundary conditions (je_condition). In a magnetic arch (Merino et al. 2023, PSST 32 065005) the lines that
leave the outlet are of two kinds: inner lines, that reach the symmetry plane and connect with the symmetric
thruster (they cannot carry electron current: u_par_e = 0 there by symmetry), and outer lines, that expand to
infinity (here: leave the domain through an outflow face).

    'ICA' (Inbound Current Ambipolarity, globally current-free emission, as POSETS MA_2D / Electrons.electron_flux):
          G = 0 at the throat on inner lines; on outer lines the electron flux through the throat is
              ica_profile = 'ion_flux': Gamma_e = (I_i / I_i,outer) n u_zi       (POSETS)
              ica_profile = 'uniform' : u_par_e = U, constant, with total Gamma_e = I_i   (paper, section 3)
          Inner / outer lines (separatrix):
              separatrix = 'traced': from the face through which every traced line leaves the domain
                                     (exact 3D separatrix: inner = lines reaching the symmetry plane)
              separatrix = '2D'    : 2D approximation, outer = throat points with x > x_sep, where x_sep is
                                     given or found as the zero of psi(0, x) of a planar field (Bfield2D_sep)
    'wall_ambipolarity': G imposed at the END of every line (integrated backwards), with the electron normal
          flux equal to the ion normal flux on the face the line reaches (as ElectronSolver 2D):
              G_end = (n u_i . n_f) / (B . n_f);  G_end = 0 on the symmetry plane;
              lines that do not reach a face: G_end = n u_par_i / B.
    'OFW' (Outbound Floating Wall): as 'wall_ambipolarity', with the sheath electron flux
              n exp((phi_w - phi)/Te) c_e/4 on the faces; phi_w makes the total collected current
              (ions - electrons through the outflow faces) zero (bisection, as ElectronSolver.zero_finder).

Numerics:
    - Lines traced once from the throat seeds (vectorized RK4 on the CG nodal B, as in MA_3D_base) and cached.
    - The RHS does not depend on G, so dG/ds is a quadrature (trapezoidal rule) along the cached lines.
    - Lines -> mesh with a KD-tree (as He and A in MA_3D_base), then Helmholtz smoothing (as ElectronSolver).
    - All derived fields live on CG1 (V1). Their nodes form a regular grid on a BoxMesh, so sampling
      along lines and on postprocessing planes is done with vectorized trilinear interpolation.
'''


class _RegularGridSampler:
    '''Trilinear sampler for nodal values of a CG function whose DOFs lie on a regular grid (BoxMesh).'''

    def __init__(self, coords):
        axes = [np.unique(np.round(coords[:, i], 8)) for i in range(3)]
        self.n = np.array([len(a) for a in axes])
        self.lo = np.array([a[0] for a in axes])
        self.hi = np.array([a[-1] for a in axes])
        self.h = (self.hi - self.lo) / (self.n - 1)
        idx = np.rint((coords - self.lo) / self.h).astype(np.intp)
        if np.prod(self.n) != len(coords) or not np.allclose(self.lo + idx * self.h, coords):
            raise ValueError('The DOFs do not form a regular grid (a BoxMesh with CG elements is required).')
        self.strides = np.array([self.n[1] * self.n[2], self.n[2], 1])
        self.flat_idx = idx @ self.strides

    def to_grid(self, values):
        '''Nodal values (Ndof,) or (Ndof, k) -> flattened grid array.'''
        values = np.asarray(values, dtype=float)
        g = np.zeros((int(np.prod(self.n)),) + values.shape[1:])
        g[self.flat_idx] = values
        return g

    def inside(self, pts):
        return np.all((pts >= self.lo) & (pts <= self.hi), axis=1)

    def __call__(self, grid_vals, pts):
        f = (pts - self.lo) / self.h
        i = np.clip(np.floor(f).astype(np.intp), 0, self.n - 2)
        t = np.clip(f - i, 0., 1.)
        base = i @ self.strides
        out = np.zeros((len(pts),) + grid_vals.shape[1:])
        shape_w = (-1,) + (1,) * (grid_vals.ndim - 1)
        for a in (0, 1):
            wa = t[:, 0] if a else 1 - t[:, 0]
            for b in (0, 1):
                wb = wa * (t[:, 1] if b else 1 - t[:, 1])
                for c in (0, 1):
                    w = wb * (t[:, 2] if c else 1 - t[:, 2])
                    out += w.reshape(shape_w) * grid_vals[base + a * self.strides[0] + b * self.strides[1] + c * self.strides[2]]
        return out


class ElectronSolver_3D:
    def __init__(self, mesh, V_CG, V_DG, BField3D, Bz, Bx, By, ElectronModel, CollisionModel,
                 seed_points, delta_s=0.01, max_steps=5000, stride=10, batch_size=8192,
                 alpha_smooth=None, n_min=1e-6, B_min=1e-3, je_condition='ICA',
                 separatrix='traced', x_sep=None, Bfield2D_sep=None, x_sep_range=None,
                 open_faces=('zmax', 'xmax', 'y0', 'ymax'), sym_face='x0', ica_profile='ion_flux',
                 phi_w_bracket=(-6.0, 0.0), n_face=60, bn_min=1e-2, perp_div=True, zero_inner=False,
                 topo_length=400.0, topo_ds=0.02):
        '''
        mesh, V_CG, V_DG : BoxMesh and its CG (any degree) and DG spaces
        BField3D         : analytic field (Plasma.MagneticField_Loop_3D), already rescaled
        Bz, Bx, By       : the same field as CG functions (used in the UFL forms and for tracing)
        seed_points      : (N, 3) throat points (z, x, y) where the lines start
        delta_s          : RK4 step (arc length, the tracer follows the unit vector b)
        stride           : store 1 of every `stride` steps along the lines
        alpha_smooth     : Helmholtz smoothing length (default: one CG1 cell)
        je_condition     : 'ICA' | 'wall_ambipolarity' | 'OFW'   (see the module docstring)
        separatrix       : 'topology' | 'traced' | '2D'   (ICA: which throat points are on outer, current-carrying lines)
                           'topology': exact. Each line is followed with the analytic field beyond the domain
                                       until it crosses the symmetry plane (inner: connects with the symmetric
                                       thruster) or returns to the thruster plane z = z_min around the coil (outer)
                           'traced'  : from the face through which the line leaves the domain (a line leaving
                                       downstream may still turn back to the symmetry plane outside the domain)
                           '2D'      : x > x_sep, with x_sep from a planar field
        topo_length      : maximum arc length followed for 'topology' (lines near the separatrix are very long)
        topo_ds          : base step for 'topology' (grows with the distance to the outlet)
        x_sep            : '2D' separatrix position on the throat (outer: x > x_sep). If None it is computed
                           from Bfield2D_sep.psi(0, x) inside x_sep_range (default: x range of the seeds)
        open_faces       : faces through which lines expand to infinity (outflow faces)
        sym_face         : face of the symmetry plane (inner lines end there)
        ica_profile      : 'ion_flux' (POSETS) | 'uniform' (constant u_par_e on the outer lines)
        phi_w_bracket    : initial bracket of the wall potential for OFW
        n_face           : points per direction to sample each outflow face (OFW current balance)
        bn_min           : floor of |b . n_f| when converting a normal flux into G at a face
        zero_inner       : ICA: if True, G = 0 along the whole of every line that starts on an inner throat point
                           (default False: G = 0 only at the throat; the perpendicular term may add current downstream)
        perp_div         : include -div(n u_e_perp) in the line equation (True, 3D). False drops it, so G is
                           constant along the lines (as in 2D, where div(n u_e_perp) = 0); useful for comparison
        Note: the throat sums assume equally spaced seeds (as the mgrid seeds of MA_3D_base).
        '''
        if je_condition not in ('ICA', 'wall_ambipolarity', 'OFW'):
            raise ValueError("je_condition must be 'ICA', 'wall_ambipolarity' or 'OFW'")
        if separatrix not in ('topology', 'traced', '2D'):
            raise ValueError("separatrix must be 'topology', 'traced' or '2D'")
        if ica_profile not in ('ion_flux', 'uniform'):
            raise ValueError("ica_profile must be 'ion_flux' or 'uniform'")

        self.mesh = mesh
        self.V_CG = V_CG
        self.V_DG = V_DG
        self.BField3D = BField3D
        self.ElectronModel = ElectronModel
        self.CollisionModel = CollisionModel
        self.gamma_e = float(ElectronModel.gamma)
        self.seed_points = np.asarray(seed_points, dtype=float)
        self.delta_s = delta_s
        self.max_steps = max_steps
        self.stride = stride
        self.batch_size = batch_size
        self.n_min = n_min
        self.B_min = B_min
        self.je_condition = je_condition
        self.separatrix = separatrix
        self.x_sep = x_sep
        self.Bfield2D_sep = Bfield2D_sep
        self.x_sep_range = x_sep_range
        self.open_faces = tuple(open_faces)
        self.sym_face = sym_face
        self.ica_profile = ica_profile
        self.phi_w_bracket = phi_w_bracket
        self.n_face = n_face
        self.bn_min = bn_min
        self.perp_div = perp_div
        self.zero_inner = zero_inner
        self.topo_length = topo_length
        self.topo_ds = topo_ds
        self.phi_w = None

        # Outward normals of the box faces, components (z, x, y)
        self._normals = {'z0': np.array([-1., 0., 0.]), 'zmax': np.array([1., 0., 0.]),
                         'x0': np.array([0., -1., 0.]), 'xmax': np.array([0., 1., 0.]),
                         'y0': np.array([0., 0., -1.]), 'ymax': np.array([0., 0., 1.])}

        # Work space (CG1) and its regular-grid sampler
        self.V1 = df.FunctionSpace(mesh, 'CG', 1)
        self.c1 = self.V1.tabulate_dof_coordinates().reshape((-1, 3))
        self.grid1 = _RegularGridSampler(self.c1)
        self.alpha_smooth = self.grid1.h.max() if alpha_smooth is None else alpha_smooth

        self.dx = df.dx(domain=mesh, metadata={'quadrature_degree': 4})
        self.ds = df.ds(domain=mesh, metadata={'quadrature_degree': 4})

        # B on the CG1 nodes (analytic, same field as Bz, Bx, By)
        self.B_1 = self._B_analytic(self.c1)
        self.Bn_1 = np.maximum(np.sqrt((self.B_1**2).sum(axis=1)), B_min)
        self.b_1 = self.B_1 / self.Bn_1[:, None]

        # B for the tracing: nodal values of Bz, Bx, By on V_CG (as B_grid in MA_3D_base)
        self.gridB = _RegularGridSampler(V_CG.tabulate_dof_coordinates().reshape((-1, 3)))
        self._B_grid = self.gridB.to_grid(np.column_stack(
            [Bz.vector().get_local(), Bx.vector().get_local(), By.vector().get_local()]))

        # Tracing direction: the lines must enter the domain (+z) from the throat
        self.q = 1 if self.BField3D.Bz(*self.seed_points.mean(axis=0)) > 0 else -1

        # UFL magnetic field
        self.B_ufl = ufl.as_vector((Bz, Bx, By))
        self.B2_ufl = ufl.dot(self.B_ufl, self.B_ufl)
        self.b_ufl = self.B_ufl / ufl.sqrt(self.B2_ufl + 1e-12)

        # Linear solvers (iterative: the 3D spaces are large)
        for pc in ('hypre_amg', 'amg', 'jacobi'):
            if df.has_krylov_solver_preconditioner(pc):
                self._pc = pc
                break
        u, v = df.TrialFunction(self.V1), df.TestFunction(self.V1)
        self._M = df.assemble(u * v * self.dx)
        self._H = {}

        # Cache to store streamlines and avoid recomputation
        self._streamline_cache = {}

        # Results
        self.fields_1 = {}
        self._fields_grid = None

    # ------------------------------------------------------------------ helpers

    def _B_analytic(self, pts):
        z, x, y = pts[:, 0], pts[:, 1], pts[:, 2]
        return np.column_stack([self.BField3D.Bz(z, x, y), self.BField3D.Bx(z, x, y), self.BField3D.By(z, x, y)])

    def _solve(self, A, L):
        f = df.Function(self.V1)
        solver = df.KrylovSolver('cg', self._pc)
        solver.parameters['relative_tolerance'] = 1e-10
        solver.solve(A, f.vector(), df.assemble(L))
        return f

    def _function(self, values):
        f = df.Function(self.V1)
        f.vector().set_local(np.asarray(values, dtype=float))
        f.vector().apply('insert')
        return f

    def project(self, expr):
        '''L2 projection onto V1.'''
        v = df.TestFunction(self.V1)
        return self._solve(self._M, expr * v * self.dx)

    def smooth_Helmholtz(self, f, alpha_val=None):
        '''
        Smooth f with a Helmholtz filter, (1 - alpha^2 Lap) f_s = f  (same filter as ElectronSolver).
        Also acts as L2 projection onto V1, so it accepts DG functions and UFL expressions.
        '''
        alpha_val = self.alpha_smooth if alpha_val is None else alpha_val
        if alpha_val not in self._H:
            u, v = df.TrialFunction(self.V1), df.TestFunction(self.V1)
            a = (u * v + df.Constant(alpha_val)**2 * df.dot(df.grad(u), df.grad(v))) * self.dx
            self._H[alpha_val] = df.assemble(a)
        v = df.TestFunction(self.V1)
        return self._solve(self._H[alpha_val], f * v * self.dx)

    def grad_perp(self, f):
        g = ufl.grad(f)
        return g - self.b_ufl * ufl.dot(self.b_ufl, g)

    def perp_flux(self, n_cg, He, A):
        '''Gamma_perp = n u_e_perp = f_m x B / B^2  (UFL).'''
        g = self.gamma_e
        kappa = float(self.ElectronModel.Te0) / ((g - 1) * float(self.ElectronModel.ne0)**(g - 1))
        n_pos = ufl.max_value(n_cg, self.n_min)
        f_m = n_pos * self.grad_perp(He) - kappa * n_pos**g * self.grad_perp(A)
        return ufl.cross(f_m, self.B_ufl) / (self.B2_ufl + 1e-10)

    def _assign(self, name, f_V1, V):
        '''Store a V1 function in ElectronModel.<name> (interpolated to V), as ElectronSolver does with .assign.'''
        g = df.interpolate(f_V1, V)
        current = getattr(self.ElectronModel, name, None)
        try:
            current.assign(g)
        except Exception:
            setattr(self.ElectronModel, name, g)

    # ------------------------------------------------------------------ streamlines

    def b_unit(self, pts):
        '''q * b at pts (zero outside the domain), from the CG nodal field.'''
        B = self.gridB(self._B_grid, pts)
        B[~self.gridB.inside(pts)] = 0.0
        B *= self.q
        return B / (np.sqrt((B**2).sum(axis=1)) + 1e-12)[:, None]

    def trace_streamlines(self):
        '''
        Vectorized RK4 tracing from the throat seeds (as streamline_trace_fast in MA_3D_base).
        Stores, in order along each line, records (line ids, points, arc length s): one every `stride`
        steps and the last point inside the domain. Also builds the KD-tree map lines -> V1 nodes.
        '''
        seeds = self.seed_points
        lo, hi = self.gridB.lo, self.gridB.hi
        ds = self.delta_s
        records = []
        exit_face = np.full(len(seeds), 'none', dtype=object)   # face through which each line leaves
        end_pts = seeds.copy()                                    # last point of each line inside the domain

        def face_of(p):
            '''Face crossed by points p that left the box (priority: symmetry plane first).'''
            f = np.full(len(p), 'none', dtype=object)
            for name, cond in (('ymax', p[:, 2] > hi[2]), ('y0', p[:, 2] < lo[2]), ('xmax', p[:, 1] > hi[1]),
                               ('zmax', p[:, 0] > hi[0]), ('z0', p[:, 0] < lo[0]), ('x0', p[:, 1] < lo[1])):
                f[cond] = name
            return f

        for start in range(0, len(seeds), self.batch_size):
            cur = seeds[start:start + self.batch_size].copy()
            ids = np.arange(start, start + len(cur))
            s_line = np.zeros(len(cur))
            active = np.arange(len(cur))

            for step in range(1, self.max_steps + 1):
                if active.size == 0:
                    break
                p = cur[active]
                k1 = self.b_unit(p)
                k2 = self.b_unit(p + 0.5 * ds * k1)
                k3 = self.b_unit(p + 0.5 * ds * k2)
                k4 = self.b_unit(p + ds * k3)
                new = p + (ds / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

                ok = np.all((new >= lo) & (new <= hi), axis=1)
                gone = active[~ok]
                if gone.size:  # last point inside the domain
                    records.append((ids[gone], cur[gone].copy(), s_line[gone].copy()))
                    exit_face[ids[gone]] = face_of(new[~ok])
                    end_pts[ids[gone]] = cur[gone]
                active = active[ok]
                s_line[active] += np.sqrt(((new[ok] - p[ok])**2).sum(axis=1))  # true step (shorter near the boundary)
                cur[active] = new[ok]

                if step % self.stride == 0 and active.size:
                    records.append((ids[active], cur[active].copy(), s_line[active].copy()))

            if active.size:  # lines that reached max_steps
                records.append((ids[active], cur[active].copy(), s_line[active].copy()))
                end_pts[ids[active]] = cur[active]

        all_pts = np.vstack([seeds] + [r[1] for r in records])
        line_ids = np.concatenate([np.arange(len(seeds))] + [r[0] for r in records])
        _, kd_idx = cKDTree(all_pts).query(self.c1, workers=-1)

        self._streamline_cache = {'records': records, 'kd_idx': kd_idx, 'n_points': len(all_pts),
                                  'line_ids': line_ids, 'exit_face': exit_face, 'end_pts': end_pts}
        faces, counts = np.unique(exit_face.astype(str), return_counts=True)
        print(f'ElectronSolver_3D: {len(seeds)} lines traced, {len(all_pts)} points stored.')
        print('  lines by exit face:', dict(zip(faces, counts.tolist())))
        gc.collect()

    def integrate_G(self, R_grid, G0):
        '''Trapezoidal integration of dG/ds = R along the cached lines. Returns G at all stored points.'''
        seeds = self.seed_points
        G = np.array(G0, dtype=float)
        R_prev = self.grid1(R_grid, seeds)
        s_prev = np.zeros(len(seeds))
        out = [G.copy()]
        for ids, pts, s in self._streamline_cache['records']:
            R = self.grid1(R_grid, pts)
            G[ids] += self.q * 0.5 * (R_prev[ids] + R) * (s - s_prev[ids])  # traced along q*b
            R_prev[ids] = R
            s_prev[ids] = s
            out.append(G[ids].copy())
        return np.concatenate(out), G

    # ------------------------------------------------------------------ boundary conditions

    def line_topology(self, pts, direction=1, batch_size=4096):
        '''
        Topology of the magnetic lines through pts (N, 3), followed along q*b with the analytic field and no
        domain limits: 'inner' if the line crosses the symmetry plane (x < x_min), 'outer' if it returns to the
        thruster plane (z < z_min) without crossing it, 'open' if neither happens within topo_length.
        direction = +1 follows q*b (downstream from the throat), -1 the opposite way.
        '''
        lo = self.gridB.lo
        centre = self.seed_points.mean(axis=0)
        res = np.full(len(pts), 'open', dtype=object)
        for start in range(0, len(pts), batch_size):
            cur = np.array(pts[start:start + batch_size], dtype=float)
            act = np.arange(len(cur)); s = 0.0
            def bq(p):
                B = self._B_analytic(p) * self.q * direction
                return B / (np.sqrt((B**2).sum(axis=1)) + 1e-300)[:, None]
            with np.errstate(all='ignore'):
                while act.size and s < self.topo_length:
                    p = cur[act]
                    h = self.topo_ds * np.maximum(1.0, 0.25 * np.linalg.norm(p - centre, axis=1))[:, None]
                    k1 = bq(p); k2 = bq(p + 0.5*h*k1); k3 = bq(p + 0.5*h*k2); k4 = bq(p + h*k3)
                    new = p + h/6.0 * (k1 + 2*k2 + 2*k3 + k4)
                    bad = ~np.isfinite(new).all(axis=1)              # coil singularity
                    inner = (new[:, 1] < lo[1]) & ~bad
                    outer = (new[:, 0] < lo[0]) & ~inner & ~bad
                    res[start + act[inner]] = 'inner'; res[start + act[outer]] = 'outer'
                    keep = ~(inner | outer | bad)
                    cur[act[keep]] = new[keep]; act = act[keep]; s += self.topo_ds
        return res

    def topology_map(self, pts):
        '''+1 where the line through each point is outer, -1 where it is inner (followed both ways).'''
        fwd = self.line_topology(pts, direction=+1)
        bwd = self.line_topology(pts, direction=-1)
        inner = (fwd == 'inner') | (bwd == 'inner')
        return np.where(inner, -1.0, 1.0)

    def separatrix_on_line(self, y, x_range, tol=1e-3):
        '''x where the throat line (z = z_min, y) changes from inner to outer lines (bisection), or None.'''
        z0 = self.gridB.lo[0]
        xs = np.linspace(x_range[0], x_range[1], 41)
        t = self.line_topology(np.column_stack([np.full_like(xs, z0), xs, np.full_like(xs, y)]))
        k = np.where((t[:-1] == 'inner') & (t[1:] == 'outer'))[0]
        if k.size == 0:
            return None
        a, b = xs[k[0]], xs[k[0] + 1]
        while b - a > tol:
            m = 0.5 * (a + b)
            if self.line_topology(np.array([[z0, m, y]]))[0] == 'outer': b = m
            else: a = m
        return 0.5 * (a + b)

    def field_line(self, p0, length=30.0, ds=0.02):
        '''Magnetic line through p0 (both directions) inside the domain, with the analytic field. Returns (N, 3).'''
        lo, hi = self.gridB.lo, self.gridB.hi
        out = []
        for sgn in (-1, 1):
            p = np.array(p0, dtype=float)[None]; seg = [p[0].copy()]
            for _ in range(int(length/ds)):
                def bq(x):
                    B = self._B_analytic(x) * sgn
                    return B / (np.sqrt((B**2).sum(axis=1)) + 1e-300)[:, None]
                with np.errstate(all='ignore'):
                    k1 = bq(p); k2 = bq(p+0.5*ds*k1); k3 = bq(p+0.5*ds*k2); k4 = bq(p+ds*k3)
                    p = p + ds/6.0*(k1+2*k2+2*k3+k4)
                if not np.isfinite(p).all() or np.any(p[0] < lo) or np.any(p[0] > hi):
                    break
                seg.append(p[0].copy())
            out.append(np.array(seg))
        return np.vstack([out[0][::-1], out[1][1:]])

    def outer_seeds(self):
        '''Throat points whose lines can carry electron current (outer lines, not connected to the symmetric thruster).'''
        if self.separatrix == 'topology':
            if 'topology' not in self._streamline_cache:
                print(f'  following {len(self.seed_points)} lines beyond the domain to find the 3D separatrix ...')
                topo = self.line_topology(self.seed_points)
                self._streamline_cache['topology'] = topo
                kinds, counts = np.unique(topo.astype(str), return_counts=True)
                print('  line topology at the throat:', dict(zip(kinds, counts.tolist())))
            return self._streamline_cache['topology'] == 'outer'
        if self.separatrix == 'traced':
            return np.isin(self._streamline_cache['exit_face'].astype(str), list(self.open_faces))
        if self.x_sep is None:
            if self.Bfield2D_sep is None:
                raise ValueError("separatrix='2D' needs x_sep or Bfield2D_sep")
            a, b = self.x_sep_range if self.x_sep_range is not None else \
                (self.seed_points[:, 1].min(), self.seed_points[:, 1].max())
            x = np.linspace(a, b, 5001)
            psi = self.Bfield2D_sep.psi(0 * x, x)
            j = np.where(psi[:-1] * psi[1:] < 0)[0]
            if j.size == 0:
                raise ValueError('No zero of psi(0, x) in x_sep_range: give x_sep explicitly.')
            self.x_sep = 0.5 * (x[j[0]] + x[j[0] + 1])
            print(f'  2D separatrix on the throat: x_sep = {self.x_sep:.4f}')
        return self.seed_points[:, 1] > self.x_sep

    def _throat_bc_ICA(self, u):
        '''G at the throat: zero on inner lines, current-free electron emission on outer lines.'''
        seeds = self.seed_points
        U = np.array([u(df.Point(*p)) for p in seeds])
        Bs = self._B_analytic(seeds)
        Bn = np.maximum(np.sqrt((Bs**2).sum(axis=1)), self.B_min)
        n = np.maximum(U[:, 0], self.n_min)
        Fi = U[:, 1]                                  # ion flux through the throat plane, n u_zi
        outer = self.outer_seeds()
        if not outer.any():
            raise ValueError('ICA: no throat point lies on an outer line (closed arch): no electron current can be emitted.')

        if self.ica_profile == 'ion_flux':
            c = Fi.sum() / Fi[outer].sum()
            Fe = np.where(outer, c * Fi, 0.0)         # electron flux through the throat
            Bz = Bs[:, 0]
            Bz = np.where(np.abs(Bz) < self.B_min, np.where(Bz < 0, -self.B_min, self.B_min), Bz)
            G0 = Fe / Bz                              # flux through z = 0 is G B_z
        else:
            bz = Bs[:, 0] / Bn
            U_par = Fi.sum() / (n * bz)[outer].sum()  # constant u_par_e on the outer lines
            G0 = np.where(outer, n * U_par / Bn, 0.0)

        print(f'  ICA: {outer.sum()} of {len(seeds)} throat points on outer lines '
              f'({100 * Fi[outer].sum() / Fi.sum():.1f}% of the ion flux), separatrix = {self.separatrix}')
        self.outer_mask = outer
        return G0

    def _sheath_flux(self, n, He, A, phi_w):
        '''Electron flux collected by a floating wall (as ElectronSolver.jn_sheath_model).'''
        EM = self.ElectronModel
        Te = EM.Te(n, A)
        c_se = (8 * Te / (np.pi * self.CollisionModel.me))**0.5
        phi = EM.phi(n, He, A)
        return n * np.exp((phi_w - phi) / Te) * c_se / 4

    def _face_samples(self, u):
        '''n, n u_i . n_f, He, A on a regular grid of every outflow face, with the integration weights.'''
        lo, hi = self.gridB.lo, self.gridB.hi
        axis = {'z': 0, 'x': 1, 'y': 2}
        data = []
        for f in self.open_faces:
            ax = axis[f[0]]
            val = lo[ax] if f.endswith('0') else hi[ax]
            o = [i for i in range(3) if i != ax]
            a = np.linspace(lo[o[0]], hi[o[0]], self.n_face)
            b = np.linspace(lo[o[1]], hi[o[1]], self.n_face)
            A_, B_ = np.meshgrid(a, b, indexing='ij')
            pts = np.zeros((A_.size, 3)); pts[:, ax] = val; pts[:, o[0]] = A_.ravel(); pts[:, o[1]] = B_.ravel()
            U = np.array([u(df.Point(*p)) for p in pts])
            He = np.array([self.ElectronModel.He(df.Point(*p)) for p in pts])
            A = np.array([self.ElectronModel.A_Te(df.Point(*p)) for p in pts])
            # trapezoidal weights of the face grid
            wa = np.full(len(a), a[1] - a[0]); wa[[0, -1]] *= 0.5
            wb = np.full(len(b), b[1] - b[0]); wb[[0, -1]] *= 0.5
            w = np.outer(wa, wb).ravel()
            Fi = U[:, 1:4] @ self._normals[f]
            ok = np.isfinite(U).all(axis=1) & np.isfinite(He) & np.isfinite(A)   # skip invalid samples
            data.append((np.maximum(U[ok, 0], self.n_min), Fi[ok], He[ok], A[ok], w[ok]))
        return data

    def zero_finder(self, u, phi_w_0=None, phi_w_1=None, max_iter=200, rtol=1e-6):
        '''Wall potential with zero total current collected through the outflow faces (bisection).'''
        phi_w_0, phi_w_1 = (self.phi_w_bracket if phi_w_0 is None else (phi_w_0, phi_w_1))
        data = self._face_samples(u)
        I_i = sum((w * Fi).sum() for n, Fi, He, A, w in data)
        if not np.isfinite(I_i) or I_i == 0:
            raise ValueError(f'OFW: collected ion current through the outflow faces is {I_i}.')

        def I_net(phi_w):
            return I_i - sum((w * self._sheath_flux(n, He, A, phi_w)).sum() for n, Fi, He, A, w in data)

        f0, f1 = I_net(phi_w_0), I_net(phi_w_1)
        if f0 * f1 > 0:
            raise ValueError(f'OFW: no zero of the collected current in [{phi_w_0}, {phi_w_1}] '
                             f'(currents {f0:.3e}, {f1:.3e}); change phi_w_bracket.')
        for it in range(max_iter):
            phi_m = 0.5 * (phi_w_0 + phi_w_1)
            fm = I_net(phi_m)
            if abs(fm) < rtol * abs(I_i):
                break
            if fm * f0 > 0:
                phi_w_0, f0 = phi_m, fm
            else:
                phi_w_1 = phi_m
        print(f'  OFW: phi_w = {phi_m:.6f}  (collected ion current {I_i:.4e}, residual {fm:.2e}, {it + 1} iterations)')
        return phi_m

    def _end_bc(self, u):
        '''G at the end of every line (wall_ambipolarity / OFW).'''
        cache = self._streamline_cache
        end_pts, faces = cache['end_pts'], cache['exit_face'].astype(str)
        U = np.array([u(df.Point(*p)) for p in end_pts])
        B = self._B_analytic(end_pts)
        Bn = np.maximum(np.sqrt((B**2).sum(axis=1)), self.B_min)
        n = np.maximum(U[:, 0], self.n_min)

        # Lines that do not reach a face: local parallel ion flux (as the 2D 'else' branch)
        G_end = (U[:, 1:4] * B).sum(axis=1) / Bn**2

        if self.je_condition == 'OFW':
            self.phi_w = self.zero_finder(u)
            He = np.array([self.ElectronModel.He(df.Point(*p)) for p in end_pts])
            A = np.array([self.ElectronModel.A_Te(df.Point(*p)) for p in end_pts])

        for f, nrm in self._normals.items():
            sel = faces == f
            if not sel.any():
                continue
            if f == self.sym_face:
                G_end[sel] = 0.0                      # u_par_e = 0 on the symmetry plane
                continue
            Bdn = B[sel] @ nrm
            floor = self.bn_min * Bn[sel]
            Bdn = np.where(np.abs(Bdn) < floor, np.where(Bdn < 0, -floor, floor), Bdn)
            if self.je_condition == 'wall_ambipolarity':
                Gam_n = U[sel, 1:4] @ nrm             # ion normal flux
            else:
                Gam_n = self._sheath_flux(n[sel], He[sel], A[sel], self.phi_w)
            G_end[sel] = Gam_n / Bdn                  # electron normal flux = G B . n_f
        return G_end

    # ------------------------------------------------------------------ G and electron velocity

    def compute_G(self, u, nn=0.0):
        '''
        u  : plasma solution P.u = (n, n u_zi, n u_xi, n u_yi)
        nn : neutral density for the ionization source (float or nodal array on V1). 0 -> S = 0
        Sets ElectronModel.G_e, G_e_cont, jpare, jpare_cont and the electron velocity fields.
        '''
        if not self._streamline_cache:
            self.trace_streamlines()

        EM = self.ElectronModel

        # 1. Continuous n, He, A (n is DG; He and A come from the KD-tree)
        n_cg = self.smooth_Helmholtz(u[0])
        He_s = self.smooth_Helmholtz(EM.He)
        A_s = self.smooth_Helmholtz(EM.A_Te)
        n_1 = np.maximum(n_cg.vector().get_local(), self.n_min)
        A_1 = A_s.vector().get_local()

        # 2. Perpendicular electron flux and its divergence (weak form, first derivatives only)
        Gam_perp = self.perp_flux(n_cg, He_s, A_s)
        v = df.TestFunction(self.V1)
        nrm = df.FacetNormal(self.mesh)
        L_div = -ufl.dot(Gam_perp, ufl.grad(v)) * self.dx + v * ufl.dot(Gam_perp, nrm) * self.ds
        div_Gam = self._solve(self._M, L_div)
        div_1 = div_Gam.vector().get_local()

        # 3. Ionization source and RHS of the line equation
        if np.any(np.asarray(nn) > 0):
            Te_1 = EM.Te(n_1, A_1)
            S_1 = self.CollisionModel.Ionization(nn, Te_1, lib=np) * n_1
        else:
            S_1 = np.zeros_like(n_1)
        R_1 = (S_1 - (div_1 if self.perp_div else 0.0)) / self.Bn_1

        # 4. Boundary condition at the throat (ICA) or at the end of the lines (wall_ambipolarity, OFW)
        G0 = None
        if self.je_condition == 'ICA':
            G0 = self._throat_bc_ICA(u)

        # 5. Integrate along the lines, map to the mesh, smooth
        R_grid = self.grid1.to_grid(R_1)
        if self.je_condition == 'ICA':
            G_pts, G_last = self.integrate_G(R_grid, G0)
            if self.zero_inner:
                # only lines starting on outer throat points carry electron current
                G_pts = np.where(self.outer_mask[self._streamline_cache['line_ids']], G_pts, 0.0)
                G_last = np.where(self.outer_mask, G_last, 0.0)
            inner = self._streamline_cache['exit_face'] == self.sym_face
            if inner.any():
                print(f'  ICA consistency: max |G| reaching the symmetry plane = {np.abs(G_last[inner]).max():.3e} '
                      f'(should be ~0; max |G0| = {np.abs(G0).max():.3e})')
        else:
            # Condition at the end of the lines: integrate backwards, G(s) = G_end - int_s^end R ds
            I_pts, I_last = self.integrate_G(R_grid, np.zeros(len(self.seed_points)))
            G_end = self._end_bc(u)
            G_pts = I_pts + (G_end - I_last)[self._streamline_cache['line_ids']]
        G_fn = self.smooth_Helmholtz(self._function(G_pts[self._streamline_cache['kd_idx']]))
        G_1 = G_fn.vector().get_local()

        jpare_1 = -G_1 * self.Bn_1
        jpare_fn = self._function(jpare_1)

        self._assign('G_e', G_fn, self.V_DG)
        self._assign('G_e_cont', G_fn, self.V_CG)
        self._assign('jpare', jpare_fn, self.V_DG)
        self._assign('jpare_cont', jpare_fn, self.V_CG)

        # 6. Electron velocity: u_e = u_par_e b + Gamma_perp / n
        Gam_1 = np.column_stack([self.project(Gam_perp[i]).vector().get_local() for i in range(3)])
        upare_1 = G_1 * self.Bn_1 / n_1
        ue_par_1 = upare_1[:, None] * self.b_1
        ue_perp_1 = Gam_1 / n_1[:, None]
        ue_1 = ue_par_1 + ue_perp_1

        self.G_fn, self.jpare_fn, self.div_Gam_fn = G_fn, jpare_fn, div_Gam
        self.upare_fn = self._function(upare_1)
        self.ue_fn = [self._function(ue_1[:, i]) for i in range(3)]

        self.fields_1 = {
            'n_e':       n_1,
            'G':         G_1,
            'jpare':     jpare_1,
            'div_Gperp': div_1,
            'upare':     upare_1,
            'uez_par':   ue_par_1[:, 0], 'uex_par':  ue_par_1[:, 1], 'uey_par':  ue_par_1[:, 2],
            'uez_perp':  ue_perp_1[:, 0], 'uex_perp': ue_perp_1[:, 1], 'uey_perp': ue_perp_1[:, 2],
            'uez':       ue_1[:, 0], 'uex': ue_1[:, 1], 'uey': ue_1[:, 2],
        }
        self.fields_1['ue_perp'] = np.sqrt(ue_perp_1[:, 0]**2 + ue_perp_1[:, 1]**2 + ue_perp_1[:, 2]**2)
        self.fields_1['ue'] = np.sqrt(ue_1[:, 0]**2 + ue_1[:, 1]**2 + ue_1[:, 2]**2)

        self._field_names = list(self.fields_1.keys())
        self._fields_grid = self.grid1.to_grid(np.column_stack([self.fields_1[k] for k in self._field_names]))

        del n_cg, He_s, A_s, Gam_1, ue_par_1, ue_perp_1, ue_1
        gc.collect()

    def sample(self, z, x, y):
        '''
        Electron fields at the points (z, x, y) (arrays of the same shape, or scalars), as `sample` in MA_3D_base.
        Keys: G, jpare, upare, div_Gperp, n_e, ue, ue_perp and the (z, x, y) components of
        ue_par (uez_par, ...), ue_perp (uez_perp, ...) and ue (uez, uex, uey).
        '''
        if self._fields_grid is None:
            raise RuntimeError('Run compute_G / run_electron_update first.')
        z, x, y = np.broadcast_arrays(z, x, y)
        pts = np.column_stack([z.ravel(), x.ravel(), y.ravel()]).astype(float)
        vals = self.grid1(self._fields_grid, pts)
        return {k: vals[:, i].reshape(z.shape) for i, k in enumerate(self._field_names)}

    # ------------------------------------------------------------------ driver and output

    def run_electron_update(self, P, nn=0.0, log=None):
        msg = f'Electron update (3D)  [je_condition = {self.je_condition}]'
        log.info(msg) if log is not None else print(msg)
        self.compute_G(P.u, nn)
        print('G_e, jpare, u_par_e and u_e updated.')

    def save_electron_params(self, path):
        import src
        funcs = {
            'G_e_df':          self.ElectronModel.G_e,
            'jpare_df':        self.ElectronModel.jpare,
            'G_e_cont_df':     self.G_fn,
            'upare_df':        self.upare_fn,
            'div_Gam_perp_df': self.div_Gam_fn,
            'uez_df':          self.ue_fn[0],
            'uex_df':          self.ue_fn[1],
            'uey_df':          self.ue_fn[2],
        }
        for name, func in funcs.items():
            src.save_load.save_df_function(path, func, name)
        print('Electron parameters saved in', path)
