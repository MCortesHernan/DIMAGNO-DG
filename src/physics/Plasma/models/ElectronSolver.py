
# --- IMPORTS --- #

import numpy as np
import ufl_legacy as ufl
from dolfin import project
import dolfin as df
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from scipy.integrate import simpson
from scipy.interpolate import griddata, LinearNDInterpolator
from scipy.interpolate import interp1d
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sp

import src
import gc

'''
ElectronSolver (optimized)

Same equations, boundary conditions, public methods and outputs as the original solver
(kept in ElectronSolver_old.py). What changes is how the work is done:

1. Field evaluation along the streamlines. The original evaluated every dolfin Function point by point
   (f(df.Point(z, x)), a cell search + basis evaluation per call, ~7 fields per point). Here, for every
   function space, a sparse matrix M (streamline points x DOFs) is built once with the same cell choice
   dolfin uses (first collision, closest cell when extrapolating). Afterwards, f on all the streamline
   points is a single product M @ f.vector(). Analytic B fields are evaluated on whole arrays.

2. G and He equations. Their right-hand sides do not depend on the unknown (dG/ds = S_i/B,
   dHe/ds = f(n, A, u_e, ...)), so they are integrals. With ode_mode='quadrature' (default) they are
   integrated with 3-point Gauss-Legendre per segment of the same piecewise-linear interpolants the
   original RHS used, instead of solve_ivp(BDF) per streamline (error well below the BDF rtol = 1e-6).
   ode_mode='ivp' reproduces the original solve_ivp calls.
   The A equation depends on A (through Te), so it is still integrated with solve_ivp (same settings),
   with np.interp on precomputed arrays instead of interp1d objects.

3. Streamline -> mesh. The Delaunay triangulation of the streamline points (the expensive part of
   griddata) and the KD-tree for the nearest fallback are computed once and reused (same result as griddata).
   If NaNs are present the original griddata path is used.

4. Helmholtz smoothing and DG projection: matrices assembled and LU-factorized once per alpha and reused.

5. zero_finder: the collected ion current does not depend on phi_w and the sheath quantities
   (n, A, He) on the collecting boundaries are evaluated once; each bisection step is then vectorized.

6. psi on the (z, x) grid is computed once for all the contours (it was recomputed for every streamline),
   and gc.collect() is called once per sweep instead of once per streamline.
'''

# Gauss-Legendre nodes / weights on [-1, 1]
_GL_X, _GL_W = np.polynomial.legendre.leggauss(3)


def _cumulative_integral(s, rhs_vec):
    '''
    Cumulative integral from s[0] of rhs(s), rhs evaluated (vectorized) on the piecewise-linear
    interpolants used by the original solve_ivp RHS. 3-point Gauss-Legendre on every segment.
    '''
    h = np.diff(s)
    mid = 0.5 * (s[1:] + s[:-1])
    pts = mid[:, None] + 0.5 * h[:, None] * _GL_X[None, :]
    f = np.asarray(rhs_vec(pts.ravel()), dtype=float).reshape(pts.shape)
    seg = 0.5 * h * (f @ _GL_W)
    return np.concatenate([[0.0], np.cumsum(seg)])


class ElectronSolver:
    def __init__(self, z_grid, x_grid, z_len, x_len, V_CG, V_DG, BField2D, ElectronModel, CollisionModel,
                 throat_lim=[0, 1], N_l=200, N_psi=500, je_condition='ICA', phi_w = 0.0, ode_mode='quadrature'):

        '''
        je_condition:
            - 'ICA' (Inbound Current Ambipolarity)
            - 'OFW' (Outbound Floating Wall)
            - 'wall_ambipolarity' (Wall Ambipolarity) - TODO: physical problems as upare extremely low and not consistent in periphery
        ode_mode:
            - 'quadrature' (default): G and He integrated as integrals (fast)
            - 'ivp': G and He integrated with solve_ivp exactly as in ElectronSolver_old
        '''
        self.z_grid = z_grid
        self.x_grid = x_grid
        self.z_len = z_len
        self.x_len = x_len
        self.V_CG = V_CG
        self.V_DG = V_DG
        self.BField2D = BField2D
        self.throat_lim = throat_lim
        self.N_l = N_l
        self.N_psi = N_psi
        self.ElectronModel = ElectronModel
        self.gamma_e = float(self.ElectronModel.gamma)
        self.CollisionModel = CollisionModel
        self.je_condition = je_condition
        self.phi_w = phi_w
        if ode_mode not in ('quadrature', 'ivp'):
            raise ValueError("ode_mode must be 'quadrature' or 'ivp'")
        self.ode_mode = ode_mode

        # Cache to store streamlines and avoid recomputation
        self._streamline_cache = {}

        # Caches of the optimized implementation
        self._all_line_pts = None      # all streamline points (z, x), in cache order
        self._line_slices = {}         # id(streamline) -> slice in _all_line_pts
        self._outside = None           # points outside the mesh (need extrapolation)
        self._eval_matrices = {}       # (mesh id, element) -> sparse evaluation matrix (or None)
        self._samples = None           # per-sweep cache: id(f) -> values on all streamline points
        self._interp_cache = {}        # included lines -> (Delaunay, cKDTree)
        self._helmholtz_solvers = {}   # alpha -> (LUSolver, test function)
        self._dg_solver = None         # (LUSolver, test function) for the DG projection

    # ------------------------------------------------------------------ optimized evaluation helpers

    def _build_line_index(self):
        '''Stack all cached streamline points and remember the slice of every streamline.'''
        pts, start = [], 0
        self._line_slices = {}
        for psi_val, (streamline, s) in self._streamline_cache.items():
            n = streamline.shape[0]
            self._line_slices[id(streamline)] = slice(start, start + n)
            pts.append(streamline[:, :2])
            start += n
        self._all_line_pts = np.vstack(pts) if pts else np.zeros((0, 2))
        self._outside = None
        self._eval_matrices = {}

    def _eval_matrix(self, V):
        '''
        Sparse matrix M such that M @ f.vector() = f at all the streamline points, for f in V.
        Same cell choice as dolfin's Function.eval (first collision, closest cell if outside).
        Returns None if it cannot be built (then the pointwise evaluation is used).
        '''
        mesh = V.mesh()
        key = (mesh.id(), str(V.ufl_element()))
        if key in self._eval_matrices:
            return self._eval_matrices[key]

        M = None
        try:
            if V.num_sub_spaces() != 0:
                raise ValueError('vector space')
            pts = self._all_line_pts
            tree = mesh.bounding_box_tree()
            element = V.element()
            dofmap = V.dofmap()
            ndofs = element.space_dimension()
            ncells = mesh.num_cells()
            rows = np.repeat(np.arange(len(pts)), ndofs)
            cols = np.empty(len(pts) * ndofs, dtype=np.int64)
            vals = np.empty(len(pts) * ndofs)
            outside = np.zeros(len(pts), dtype=bool)
            for i, (z, x) in enumerate(pts):
                p = df.Point(z, x)
                c = tree.compute_first_entity_collision(p)
                if c >= ncells:
                    # As dolfin's Function.eval: the closest cell is accepted without extrapolation
                    # if it is within DOLFIN_EPS (round-off on the boundary); otherwise extrapolation is needed
                    c, dist = tree.compute_closest_entity(p)
                    outside[i] = dist >= df.DOLFIN_EPS
                cell = df.Cell(mesh, c)
                try:
                    orientation = cell.orientation()
                except Exception:
                    orientation = 0
                phi = element.evaluate_basis_all(np.array([z, x]), cell.get_vertex_coordinates(), orientation)
                cols[i * ndofs:(i + 1) * ndofs] = dofmap.cell_dofs(c)
                vals[i * ndofs:(i + 1) * ndofs] = phi
            M = sp.csr_matrix((vals, (rows, cols)), shape=(len(pts), V.dim()))
            if self._outside is None:
                self._outside = outside
            print(f"Evaluation matrix built for {key[1]} ({len(pts)} streamline points).")
        except Exception as e:
            print(f"Evaluation matrix not available for {key[1]} ({e}); using pointwise evaluation.")
            M = None

        self._eval_matrices[key] = M
        return M

    def _sample_all(self, f):
        '''Values of f on all the streamline points (cached during a sweep).'''
        if self._samples is not None and id(f) in self._samples:
            return self._samples[id(f)]

        pts = self._all_line_pts
        vals = None
        if isinstance(f, df.Function):
            M = self._eval_matrix(f.function_space())
            if M is not None:
                vals = M @ f.vector().get_local()
        elif isinstance(f, df.Constant):
            try:
                vals = np.full(len(pts), float(f))
            except Exception:
                vals = None
        if vals is None:
            vals = np.array([f(df.Point(z, x)) for z, x in pts])

        if self._samples is not None:
            self._samples[id(f)] = vals
        return vals

    def _line_values(self, f, streamline):
        '''f on the points of a streamline (same values as [f(df.Point(z, x)) for z, x in pts]).'''
        sl = self._line_slices.get(id(streamline))
        if sl is None or self._samples is None:
            return np.array([f(df.Point(z, x)) for z, x in streamline[:, :2]])
        vals = self._sample_all(f)
        # Same behaviour as dolfin: points outside the mesh need allow_extrapolation
        if isinstance(f, df.Function) and self._outside is not None and self._outside[sl].any():
            if not f.get_allow_extrapolation():
                raise RuntimeError('Unable to evaluate function at point (outside the mesh, extrapolation not allowed).')
        return vals[sl].copy()

    # ------------------------------------------------------------------ streamline equations

    def A_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):

        pts = streamline[:, :2]
        # Numerical samples along the streamline (arrays)
        n_s   = self._line_values(n_df, streamline)
        nn_s  = self._line_values(nn_df, streamline)
        jpare_s = self._line_values(self.ElectronModel.jpare_cont, streamline)
        dHe_dpsi_s = self._line_values(self.ElectronModel.dHe_dpsi_cont, streamline)
        dA_Te_dpsi_s = self._line_values(self.ElectronModel.dA_Te_dpsi_cont, streamline)
        ue_par_s = - (jpare_s) / (n_s + 1e-5)
        x_s = pts[:, 0]
        ue_y_s = -x_s*dHe_dpsi_s + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n_s**(self.ElectronModel.gamma - 1) * x_s* dA_Te_dpsi_s #Electron diamagnetic velocity (assuming polytropic electrons)
        ue_s = (ue_par_s**2 + ue_y_s**2)**0.5

        E_i_ = 12.1 / 10.0
        gamma = self.gamma_e

        valid = (
            np.isfinite(n_s) &
            np.isfinite(nn_s) &
            np.isfinite(ue_s) &
            np.isfinite(ue_par_s)
        )

        n_s = n_s[valid]
        nn_s = nn_s[valid]
        ue_s = ue_s[valid]
        s = s[valid]
        ue_par_s = ue_par_s[valid]

        CM = self.CollisionModel
        EM = self.ElectronModel

        def rhs(ds, A):
            # enforce scalar numeric A
            A = float(np.asarray(A).ravel()[0])   # same as float(A) for a length-1 array (works on any NumPy version)
            A = max(A, 1e-12)

            # safe interpolated values
            n_i = float(np.interp(ds, s, n_s))
            ue_i = float(np.interp(ds, s, ue_s))
            ue_par_i = float(np.interp(ds, s, ue_par_s))
            nn_i = float(np.interp(ds, s, nn_s))
            # clamp / protect interpolated values
            n_i = max(n_i, 1e-12)

            Te = EM.Te(n_i, A)
            Te = max(Te, 1e-3)  # Avoid unphysically low temperatures that can cause issues in collision rates
            Te = min(Te, 1)  # Avoid unphysically high temperatures that can cause overflow in exp

            # safe inelastic energy term (cap exponent)
            exp_arg = (2.0 * E_i_) / (3.0 * Te + 1e-12)
            exp_arg = np.clip(exp_arg, -50, 50)
            E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

            ioniz = float(CM.Ionization(nn_i, Te, lib=np))
            nu_en = float(CM.E_N(nn_i, Te, CM.me, np))
            nu_ei = float(CM.E_I(nn_i, Te, CM.me, np))
            nu_exc = nu_ei*float(CM.Excitation(nn_i, Te, np)*(CM.Te0/CM.E_ion))
            nu_e = (nu_ei + nu_en + nu_exc)

            rhs_val_1 = - ioniz*(gamma/(gamma-1)*Te + E_inel)
            rhs_val_2 = nu_e * CM.me * ue_i**2
            rhs_val = (gamma-1)/(ue_par_i * Te + 1e-12) * (rhs_val_1 + rhs_val_2)

            # Avoid NaN/infs
            if not np.isfinite(rhs_val):
                # clamp to a safe large value (or zero)
                ioniz = float(CM.Ionization(1e-5, Te, lib=np))
                rhs_val = - ioniz/(ue_i + 1e-12) * A * (gamma + (gamma-1) * E_inel/(Te + 1e-12) )

            return float(rhs_val)

        # initial A: use analytic or numeric safe scalar
        A0 = 1

        # A appears in its own RHS (through Te): integrated with solve_ivp as in the original
        try:
            sol = solve_ivp(rhs, (s[0], s[-1]), [A0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        except Exception:
            # fallback to RK45 if BDF raises problems
            sol = solve_ivp(rhs, (s[0], s[-1]), [A0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)

        A_out = np.maximum(sol.y[0], 1e-12)

        if A_out.shape[0] != s.shape[0]:
            A_out = np.interp(s, sol.t, A_out, left=A_out[0], right=A_out[-1])

        if not sol.success:
            print(f"Integration failed or produced NaN at streamline. Filling with A0.")
            A_out = np.full_like(s, np.nan)

        return s, A_out

    def He_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):

        pts = streamline[:, :2]
        n_s = self._line_values(n_df, streamline)
        nn_s = self._line_values(nn_df, streamline)
        A_Te_s = self._line_values(self.ElectronModel.A_Te_cont, streamline)
        dA_Te_dpsi_s = self._line_values(self.ElectronModel.dA_Te_dpsi_cont, streamline)
        dHe_dpsi_s = self._line_values(self.ElectronModel.dHe_dpsi_cont, streamline)
        x_s = pts[:, 0]
        ue_y_s = -x_s*dHe_dpsi_s + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n_s**(self.ElectronModel.gamma - 1) * x_s* dA_Te_dpsi_s #Electron diamagnetic velocity (assuming polytropic electrons)
        jpare_s = self._line_values(self.ElectronModel.jpare_cont, streamline)
        ue_par_s = - (jpare_s) / (n_s + 1e-5)
        n_s = np.maximum(n_s, 1e-6)

        E_i_ = 12.1 / 10.0

        valid = (
        np.isfinite(n_s) &
        np.isfinite(nn_s) &
        np.isfinite(ue_y_s) &
        np.isfinite(A_Te_s) &
        np.isfinite(ue_par_s)
        )

        n_s = n_s[valid]
        nn_s = nn_s[valid]
        ue_y_s = ue_y_s[valid]
        s = s[valid]
        A_Te_s = A_Te_s[valid]
        ue_par_s = ue_par_s[valid]

        CM = self.CollisionModel
        EM = self.ElectronModel
        gamma = self.gamma_e

        def rhs_vec(ds):
            '''RHS of dHe/ds on an array of positions (same expressions as the original scalar RHS).'''
            n_i = np.interp(ds, s, n_s)
            uye_i = np.interp(ds, s, ue_y_s)
            nn_i = np.interp(ds, s, nn_s)
            A_i = np.interp(ds, s, A_Te_s)
            ue_par_i = np.interp(ds, s, ue_par_s)

            Te = np.clip(EM.Te(n_i, A_i), 1e-3, 1)

            with np.errstate(all='ignore'):
                nu_i = CM.Ionization(nn_i, Te, lib=np)
                nu_en = CM.E_N(nn_i, Te, CM.me, lib=np)
                nu_ei = CM.E_I(n_i, Te, CM.me, lib=np)
                nu_exc = nu_i*CM.Excitation(n_i, Te, lib=np)*(CM.Te0/CM.E_ion)

                nu_e = (nu_ei + nu_en + nu_i + nu_exc)

                exp_arg = np.clip((2.0 * E_i_) / (3.0 * Te + 1e-12), -50, 50)
                E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

                rhs_val_1 = - nu_i * (gamma/(gamma - 1)*Te + E_inel)
                rhs_val_2 = - nu_e * CM.me * uye_i**2
                rhs_val = 1/(ue_par_i + 1e-12) * (rhs_val_1 + rhs_val_2)

            rhs_val = np.where(np.isfinite(rhs_val), rhs_val, 1e-12)
            return np.real(rhs_val)

        He0 = 6 #from value of function to avoid evaluating and introducing fenics errors
        # TODO this must be generalized

        if self.ode_mode == 'quadrature':
            He_out = He0 + _cumulative_integral(s, rhs_vec)
            if not np.all(np.isfinite(He_out)):
                print(f"Integration failed or produced NaN at streamline. Filling with He0.")
                return s, np.full_like(s, np.nan)
            He_out = np.maximum(He_out, 1e-12)
            return s, He_out

        # ode_mode == 'ivp': original solve_ivp integration
        def rhs(ds, He):
            return float(rhs_vec(np.array([ds]))[0])

        try:
            sol = solve_ivp(rhs, (s[0], s[-1]), [He0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
        except Exception:
            # fallback to RK45 if BDF raises problems
            sol = solve_ivp(rhs, (s[0], s[-1]), [He0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)

        He_out = np.maximum(sol.y[0], 1e-12)

        if He_out.shape[0] != s.shape[0]:
            He_out = np.interp(s, sol.t, He_out, left=He_out[0], right=He_out[-1])

        if not sol.success:
            print(f"Integration failed or produced NaN at streamline. Filling with He0.")
            He_out = np.full_like(s, np.nan)

        return s, He_out

    # ------------------------------------------------------------------ sheath / collected currents

    def jn_sheath_model(self, z, x, n_df, phi_w):
        Te = self.ElectronModel.Te(n_df(df.Point(z,x)), self.ElectronModel.A_Te(df.Point(z,x)))
        c_se = (8*Te/(np.pi*self.CollisionModel.me))**0.5
        phi = self.ElectronModel.phi(n_df(df.Point(z,x)), self.ElectronModel.He(df.Point(z,x)), self.ElectronModel.A_Te(df.Point(z,x)))
        n_w = n_df(df.Point(z,x))*np.exp((phi_w - phi)/Te)
        jne = n_w*c_se/(4)

        return jne

    def col_i_current(self, z, x, jzi_df, jxi_df):
        jzi = jzi_df
        jxi = jxi_df
        downst_array = np.array([jzi(z[-1], x_i) for x_i in x])
        lat_array = np.array([jxi(z_i, x[-1]) for z_i in z])
        downst_bound = 2*np.pi*simpson(x*downst_array, x)
        lat_bound = 2*np.pi*simpson(x[-1]*lat_array, z)

        collected_ion_curr = downst_bound + lat_bound

        return collected_ion_curr

    def col_e_current(self, z, x, n_df, phi_w):
        jn_d = np.array([self.jn_sheath_model(z[-1], x_i, n_df, phi_w) for x_i in x])
        downst_bound = 2*np.pi*simpson(x*jn_d, x)

        jn_l =  np.array([self.jn_sheath_model(z_i, x[-1], n_df, phi_w) for z_i in z])
        lat_bound = 2*np.pi*simpson(x[-1]*jn_l, z)

        collected_e_curr = downst_bound + lat_bound

        return - collected_e_curr

    def _sheath_data(self, pts, n_df):
        '''n, Te and phi at boundary points (independent of phi_w), as in jn_sheath_model.'''
        EM = self.ElectronModel
        n = np.array([n_df(df.Point(z, x)) for z, x in pts])
        A = np.array([EM.A_Te(df.Point(z, x)) for z, x in pts])
        He = np.array([EM.He(df.Point(z, x)) for z, x in pts])
        Te = EM.Te(n, A)
        phi = EM.phi(n, He, A)
        return n, Te, phi

    def _jn_sheath_vec(self, data, phi_w):
        n, Te, phi = data
        c_se = (8*Te/(np.pi*self.CollisionModel.me))**0.5
        n_w = n*np.exp((phi_w - phi)/Te)
        return n_w*c_se/(4)

    def zero_finder(self, phi_w_0, phi_w_1, n_df, jzi_df, jxi_df, z_s, x_s, max_iter=200):
        '''This finds the wall potential that returns 0 collected current.'''
        eps = 1e-6

        # The collected ion current does not depend on phi_w: computed once
        col_i = self.col_i_current(z_s, x_s, jzi_df, jxi_df)
        # Sheath quantities on the downstream (z = z[-1]) and lateral (x = x[-1]) boundaries: evaluated once
        data_d = self._sheath_data([(z_s[-1], x_i) for x_i in x_s], n_df)
        data_l = self._sheath_data([(z_i, x_s[-1]) for z_i in z_s], n_df)

        def col_e(phi_w):
            downst_bound = 2*np.pi*simpson(x_s*self._jn_sheath_vec(data_d, phi_w), x_s)
            lat_bound = 2*np.pi*simpson(x_s[-1]*self._jn_sheath_vec(data_l, phi_w), z_s)
            return - (downst_bound + lat_bound)

        collected_curr_0 = col_i + col_e(phi_w_0)
        collected_curr_1 = col_i + col_e(phi_w_1)
        if collected_curr_0 * collected_curr_1 > 0:
            print('Initial potentials must be chosen so that a zero exists in the middle. \n')
            print(f'Collected curents for the chosen initial potentials are: {collected_curr_0} and {collected_curr_1}.\n')
            return None
        else:
            iter = 0
            while True:
                phi_w_aux = 0.5*(phi_w_1 + phi_w_0)
                collected_curr_aux = col_i + col_e(phi_w_aux)
                print(f'Potential at iteration {iter}: {phi_w_aux}')
                print(f'Collected current: {collected_curr_aux}\n')

                if abs(collected_curr_aux ) < eps or iter >= max_iter:
                    break

                if collected_curr_aux * collected_curr_0 > 0:
                    phi_w_0 = phi_w_aux
                else:
                    phi_w_1= phi_w_aux
                iter += 1

            return phi_w_aux

    # ------------------------------------------------------------------ G (electron continuity)

    def G_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                                n_df, nn_df, jzi_df, jxi_df):
        pts = streamline[:, :2]
        n_s = self._line_values(n_df, streamline)
        nn_s = self._line_values(nn_df, streamline)
        jzi_s = self._line_values(jzi_df, streamline)
        jxi_s = self._line_values(jxi_df, streamline)
        A_Te_s = self._line_values(self.ElectronModel.A_Te_cont, streamline)
        Te_s = self.ElectronModel.Te(n_s, A_Te_s)
        Si_s = self.CollisionModel.Ionization(nn_s, Te_s, lib=np) * n_s

        bz_s = np.asarray(self.BField2D.Bz(pts[:, 0], pts[:, 1]), dtype=float)
        bx_s = np.asarray(self.BField2D.Br(pts[:, 0], pts[:, 1]), dtype=float)
        Bnorm = np.sqrt(bz_s**2 + bx_s**2)
        Bnorm[Bnorm == 0.0] = 1e-12

        bz_hat = bz_s / Bnorm
        bx_hat = bx_s / Bnorm
        j_par_i = jzi_s * bz_hat + jxi_s * bx_hat

        valid = (
        np.isfinite(Si_s) &
        np.isfinite(j_par_i) &
        np.isfinite(Bnorm)
        )

        Si_s = Si_s[valid]
        j_par_i = j_par_i[valid]
        Bnorm = Bnorm[valid]
        s = s[valid]

        def rhs_vec(ds):
            return 1/(np.interp(ds, s, Bnorm) + 1e-8) * np.interp(ds, s, Si_s)

        def rhs(ds, Y):
            return float(rhs_vec(np.array([ds]))[0])

        quad = self.ode_mode == 'quadrature'

        if self.je_condition == 'ICA':
            Y0 = j_par_i[0]/(Bnorm[0]+1e-8)
            if quad:
                G_out = Y0 + _cumulative_integral(s, rhs_vec)
                if not np.all(np.isfinite(G_out)):
                    print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                    G_out = np.full_like(s, np.nan)
                return s, G_out

            try:
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='BDF',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s[0], s[-1]), [Y0], t_eval=s, method='RK45',
                            rtol=1e-6, atol=1e-8, max_step=(s[-1]-s[0])/50.0)

            G_out = sol.y[0]

            if G_out.shape[0] != s.shape[0]:
                G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

            if not sol.success:
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                G_out = np.full_like(s, np.nan)

        elif self.je_condition in ('OFW', 'wall_ambipolarity'):

            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len

            if self.je_condition == 'OFW':
                # Sheath electron flux at the wall
                if abs(pts[-1,0]) < eps:
                    jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                    j_e_par_bc = jn / bz_hat[-1]
                elif abs(diff_x) < eps:
                    jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                    j_e_par_bc = jn / bx_hat[-1]
                elif abs(diff_z) < eps:
                    jn = self.jn_sheath_model(pts[-1,0], pts[-1,1], n_df, self.phi_w)
                    j_e_par_bc = jn / bz_hat[-1]
                else:
                    j_e_par_bc = j_par_i[-1]
            else:
                # Ambipolarity at the wall
                if abs(pts[-1,0]) < eps:
                    j_e_par_bc = jzi_s[-1] / bz_hat[-1]
                elif abs(diff_x) < eps:
                    j_e_par_bc = jxi_s[-1] / bx_hat[-1]
                elif abs(diff_z) < eps:
                    j_e_par_bc = jzi_s[-1] / bz_hat[-1]
                else:
                    j_e_par_bc = j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose the condition at the WALL and integrate backwards
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            if quad:
                I = _cumulative_integral(s, rhs_vec)
                G_out = Y0 - (I[-1] - I)
                if self.je_condition == 'OFW' and not np.all(np.isfinite(G_out)):
                    print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                    G_out = np.full_like(s, np.nan)
                return s, G_out

            if self.je_condition == 'OFW':
                # The original called solve_ivp(ufl.rhs, ...) with BDF, which always raised and fell back to RK45
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='RK45',
                                rtol=1e-6, atol=1e-8)
                # reverse solution back to original order
                G_out = sol.y[0][::-1]
                if G_out.shape[0] != s.shape[0]:
                    G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

                if not sol.success:
                    print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                    G_out = np.full_like(s, np.nan)
            else:
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev,
                                method='RK45', rtol=1e-8, atol=1e-8)
                # reverse solution back to original order
                G_out = sol.y[0][::-1]

        return s, G_out

    # ------------------------------------------------------------------ sweep over all streamlines

    def _streamline_mapping(self, S_Tracing, psi_grid, psi_level):
        '''Same as Streamline_Tracing.streamline_mapping, with psi on the grid computed once.'''
        cs = plt.contour(self.z_grid, self.x_grid, psi_grid, levels=[psi_level])
        paths = cs.collections[0].get_paths()
        streamline_path = paths[0].vertices  # take first path
        plt.close()

        return S_Tracing.resample_streamline(streamline_path)

    def _to_dofs(self, line_keys, all_points, all_values, coords):
        '''Same result as griddata(linear) + griddata(nearest) on the NaN, with the triangulation cached.'''
        valid_mask = ~np.isnan(all_values)

        if not valid_mask.all():
            # Point set changes: original path
            clean_points = all_points[valid_mask]
            clean_values = all_values[valid_mask]
            function_grid = griddata(clean_points, clean_values, coords, method='linear')
            nan_mask = np.isnan(function_grid)
            function_grid[nan_mask] = griddata(clean_points, clean_values, coords[nan_mask], method='nearest')
            return function_grid

        key = (tuple(line_keys), all_points.shape[0])
        if key not in self._interp_cache:
            self._interp_cache = {key: (Delaunay(all_points), cKDTree(all_points))}
        tri, tree = self._interp_cache[key]

        function_grid = LinearNDInterpolator(tri, all_values)(coords)
        function_grid = np.asarray(function_grid, dtype=float).reshape(-1)
        nan_mask = np.isnan(function_grid)
        if nan_mask.any():
            _, idx = tree.query(coords[nan_mask])
            function_grid[nan_mask] = all_values[idx]
        return function_grid

    def solve_grid_streamline_it(self, eq: str = 'Continuity', V = None, n_df = 1, nn_df = 1, jzi_df = 1, jxi_df = 1, axis_df = 1):

        # Cached streamlines
        if not self._streamline_cache:
            throat_1_array = np.linspace(self.throat_lim[0],self.throat_lim[1], int(2*self.N_psi))
            psi_1_array = np.array([self.BField2D.psi(0, x) for x in throat_1_array[:]])
            psi_vals = psi_1_array

            S_Tracing = src.BFields.Streamline_Tracing(self.BField2D, self.z_grid, self.x_grid, self.N_l)
            psi_grid = self.BField2D.psi(self.z_grid, self.x_grid)  # computed once for all the contours
            for psi_val in psi_vals:
                streamline, s = self._streamline_mapping(S_Tracing, psi_grid, psi_val)
                if streamline is not None:
                    self._streamline_cache[psi_val] = (streamline, s)

            for psi_val, (streamline, s) in self._streamline_cache.items():
                plt.plot(streamline[:,0], streamline[:,1], 'k-', linewidth=0.5)

            del S_Tracing, psi_grid
            gc.collect()

        if self._all_line_pts is None:
            self._build_line_index()

        all_points, all_values, line_keys = [], [], []
        self._samples = {}   # field values on all streamline points, computed once per field in this sweep
        it=1
        try:
            for psi_val, (streamline, s) in self._streamline_cache.items():
                try:
                    if eq == 'Continuity':
                        _, f_out = self.j_par_e_streamline_f_ivp(streamline, s, n_df, nn_df, jzi_df, jxi_df, axis_df)
                    elif eq == 'G':
                        _, f_out = self.G_streamline_f_ivp(streamline, s, n_df, nn_df, jzi_df, jxi_df)
                    elif eq == 'A':
                        _, f_out = self.A_streamline_f_ivp(streamline, s, n_df, nn_df)
                    elif eq == 'He':
                        _, f_out = self.He_streamline_f_ivp(streamline, s, n_df, nn_df)
                    else:
                        raise ValueError(f"Unknown equation type: {eq}")
                    all_points.append(streamline[:, :2])
                    all_values.append(f_out)
                    line_keys.append(psi_val)
                    it+=1
                except Exception as e:
                    print(f"Streamline {it}: ψ={psi_val} failed: {e}")
                    it+=1
                    continue
        finally:
            self._samples = None

        print(f"{eq}: {len(line_keys)} of {len(self._streamline_cache)} streamlines done.")

        if not all_points:
            coords = V.tabulate_dof_coordinates().reshape((-1, 2))
            return np.zeros(coords.shape[0])

        all_points = np.vstack(all_points)
        all_values = np.concatenate(all_values)
        coords = V.tabulate_dof_coordinates().reshape((-1, 2))

        function_grid = self._to_dofs(line_keys, all_points, all_values, coords)

        del all_points, all_values
        gc.collect()

        return function_grid

    # ------------------------------------------------------------------ updates (unchanged interface)

    def compute_G(self, V, n_df, nn_df, jzi_df, jxi_df):
        G_grid = self.solve_grid_streamline_it(eq='G', V=V, n_df = n_df, nn_df = nn_df, jzi_df = jzi_df, jxi_df = jxi_df)
        G_df_new = df.Function(V)
        G_df_new.vector()[:] = G_grid
        if self.je_condition == 'OFW':
            G_df_new = ufl.conditional(abs(G_df_new) < 1e-5, (G_df_new/abs(G_df_new))*1e-5, G_df_new)
        G_df_new_smooth_cont, G_df_new_smooth = self.smooth_Helmholtz(G_df_new, alpha_val=0.01)
        self.ElectronModel.G_e.assign(G_df_new_smooth)
        self.ElectronModel.G_e_cont.assign(G_df_new_smooth_cont)

        del G_grid
        del G_df_new, G_df_new_smooth_cont, G_df_new_smooth
        gc.collect()

    def compute_G_nocoll(self, V, n_df, nn_df, jzi_df, jxi_df):
        G_grid = self.solve_grid_streamline_it(eq='G', V=V, n_df = n_df, nn_df = nn_df, jzi_df = jzi_df, jxi_df = jxi_df)
        G_df_new = df.Function(V)
        G_df_new.vector()[:] = G_grid
        G_df_new_smooth_cont, G_df_new_smooth = self.smooth_Helmholtz(G_df_new, alpha_val=0.01)
        self.ElectronModel.G_e_nocoll.assign(G_df_new_smooth)

        del G_grid
        del G_df_new, G_df_new_smooth_cont, G_df_new_smooth
        gc.collect()

    def update_He(self, V, n_df, nn_df):
        He_grid = self.solve_grid_streamline_it(eq='He', V=V, n_df = n_df, nn_df = nn_df)
        He_df_new = df.Function(V)
        He_df_new.vector()[:] = He_grid
        He_df_new_smooth_cont_1, He_df_new_smooth_1 = self.smooth_Helmholtz(He_df_new, alpha_val=0.01)
        self.ElectronModel.He_cont.assign(He_df_new_smooth_cont_1)
        self.ElectronModel.He.assign(He_df_new_smooth_1)

        diff_He_grid = He_grid - self.ElectronModel.He_nocoll.vector()[:]
        diff_He_df = df.Function(V)
        diff_He_df.vector()[:] = diff_He_grid
        diff_He_df_smooth_cont, diff_He_df_smooth = self.smooth_Helmholtz(diff_He_df, alpha_val=0.2)
        self.diff_He_df_cont = diff_He_df_smooth_cont

        del He_grid
        del diff_He_grid
        del He_df_new, He_df_new_smooth_cont_1, He_df_new_smooth_1
        del diff_He_df, diff_He_df_smooth_cont, diff_He_df_smooth
        gc.collect()

    def update_A(self, V, n_df, nn_df):
        A_grid = self.solve_grid_streamline_it(eq='A', V=V, n_df = n_df, nn_df = nn_df)
        A_df_new = df.Function(V)
        A_df_new.vector()[:] = A_grid * self.ElectronModel.A_Te_nocoll.vector()[:]
        A_df_new_smooth_cont, A_df_new_smooth = self.smooth_Helmholtz(A_df_new, alpha_val=0.01)
        self.ElectronModel.A_Te.assign(A_df_new_smooth)
        self.ElectronModel.A_Te_cont.assign(A_df_new_smooth_cont)

        diff_A_grid = A_grid * self.ElectronModel.A_Te_nocoll.vector()[:] - self.ElectronModel.A_Te_nocoll.vector()[:]
        diff_A_df = df.Function(V)
        diff_A_df.vector()[:] = diff_A_grid
        diff_A_df_smooth_cont, diff_A_df_smooth = self.smooth_Helmholtz(diff_A_df, alpha_val=0.2)
        self.diff_A_df_cont = diff_A_df_smooth_cont

        del A_grid
        del A_df_new, A_df_new_smooth_cont, A_df_new_smooth
        del diff_A_grid, diff_A_df, diff_A_df_smooth_cont, diff_A_df_smooth
        gc.collect()

    def update_dHe_dpsi(self, Bz, Br, xx, V, n_df):
        ddiff_He_dpsi = df.dot(df.grad(self.diff_He_df_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_He_dpsi, alpha_val=0.01)

        dHe_dpsi_new_grid = abs(ddiff_He_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dHe_dpsi_nocoll.vector()[:]
        self.ElectronModel.dHe_dpsi.vector()[:] = dHe_dpsi_new_grid

        del  ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth
        gc.collect()

    def update_dA_dpsi(self, Bz, Br, xx, V, n_df):
        ddiff_A_dpsi = df.dot(df.grad(self.diff_A_df_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_A_dpsi, alpha_val=0.01)

        dA_Te_dpsi_new_grid = abs(ddiff_A_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dA_Te_dpsi_nocoll.vector()[:]
        self.ElectronModel.dA_Te_dpsi.vector()[:] = dA_Te_dpsi_new_grid

        del  ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth
        gc.collect()

    def update_dHe_dpsi_extra_smooth(self, Bz, Br, xx, V, n_df):
        diff_He_df_cont_smooth_cont, diff_He_df_cont_smooth = self.smooth_Helmholtz(self.diff_He_df_cont, alpha_val=0.1)
        ddiff_He_dpsi = df.dot(df.grad(diff_He_df_cont_smooth_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_He_dpsi, alpha_val=0.1)

        dHe_dpsi_new_grid = abs(ddiff_He_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dHe_dpsi_nocoll.vector()[:]
        self.ElectronModel.dHe_dpsi_smooth.vector()[:] = dHe_dpsi_new_grid

        del  ddiff_He_dpsi_df_cont_smooth, ddiff_He_dpsi_df_smooth
        gc.collect()

    def update_dA_dpsi_extra_smooth(self, Bz, Br, xx, V, n_df):
        diff_A_df_cont_smooth_cont, diff_A_df_cont_smooth = self.smooth_Helmholtz(self.diff_A_df_cont, alpha_val=0.1)
        ddiff_A_dpsi = df.dot(df.grad(diff_A_df_cont_smooth_cont), ufl.as_vector([-Br, Bz])) / (xx*(Bz**2 + Br**2) + 1e-5)
        ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth = self.smooth_Helmholtz(ddiff_A_dpsi, alpha_val=0.1)

        dA_Te_dpsi_new_grid = abs(ddiff_A_dpsi_df_smooth.vector()[:]) + self.ElectronModel.dA_Te_dpsi_nocoll.vector()[:]
        self.ElectronModel.dA_Te_dpsi_smooth.vector()[:] = dA_Te_dpsi_new_grid

        del  ddiff_A_dpsi_df_cont_smooth, ddiff_A_dpsi_df_smooth
        gc.collect()

    def _project_DG(self, f):
        '''Same as df.project(f, V_DG), with the mass matrix assembled and factorized once.'''
        if self._dg_solver is None:
            u = df.TrialFunction(self.V_DG)
            w = df.TestFunction(self.V_DG)
            self._dg_solver = (df.LUSolver(df.assemble(u*w*df.dx)), w)
        solver, w = self._dg_solver
        f_DG = df.Function(self.V_DG)
        solver.solve(f_DG.vector(), df.assemble(f*w*df.dx))
        return f_DG

    def smooth_Helmholtz(self, f, alpha_val = 0.1):
        '''
        Smooth the function f using a Helmholtz filter.
        The matrix of every alpha is assembled and LU-factorized once (the factorization is reused).
        '''
        if alpha_val not in self._helmholtz_solvers:
            u = df.TrialFunction(self.V_CG)
            v = df.TestFunction(self.V_CG)
            alpha = df.Constant(alpha_val)   # controls smoothness
            a = (u*v + alpha**2*df.dot(df.grad(u), df.grad(v)))*df.dx
            self._helmholtz_solvers[alpha_val] = (df.LUSolver(df.assemble(a)), v)

        solver, v = self._helmholtz_solvers[alpha_val]
        f_smooth = df.Function(self.V_CG)
        solver.solve(f_smooth.vector(), df.assemble(f*v*df.dx))
        f_smooth_DG = self._project_DG(f_smooth)

        return  f_smooth, f_smooth_DG

    def run_electron_update(self, P, cfg, pp, Bz, Br, NeutralModel, log):
        je_cond = cfg["electron_solver"]["je_condition"]
        log.info("Electron update  [je_condition = %s]", je_cond)

        n_df   = df.project(P.u[0]);  n_df.set_allow_extrapolation(True)
        jzi_df = df.project(P.u[1]);  jzi_df.set_allow_extrapolation(True)
        jxi_df = df.project(P.u[2]);  jxi_df.set_allow_extrapolation(True)

        if cfg['initial_conditions']['background_pressure_pa'] > 0:
            nn_df  = df.project(P.u[5]);  nn_df.set_allow_extrapolation(True)
        else:
            nn_df  = df.Constant(NeutralModel.back_dens)

        self.je_condition = je_cond

        if je_cond == "OFW":
            m = cfg["mesh"]
            z_array = np.linspace(0, m["z_length"], m["z_length"] * 50)
            x_array = np.linspace(0, m["x_length"], m["x_length"] * 50)
            self.phi_w = self.zero_finder(-6, 0, n_df, jzi_df, jxi_df, z_array, x_array)

        self.compute_G(self.V_DG, n_df, nn_df, jzi_df, jxi_df)
        self.compute_G_nocoll(self.V_DG, n_df, df.Constant(1e-10), jzi_df, jxi_df)
        self.ElectronModel.jpare.assign(df.project(-self.ElectronModel.G_e * (Bz**2 + Br**2)**0.5, self.V_DG))
        self.ElectronModel.jpare_cont.assign(df.project(-self.ElectronModel.G_e * (Bz**2 + Br**2)**0.5, self.V_CG))

        self.update_A(self.V_DG, n_df, nn_df)
        self.update_He(self.V_DG, n_df, nn_df)

        xx = df.SpatialCoordinate(self.V_DG.mesh())[1]
        self.update_dHe_dpsi(Bz, Br, xx, self.V_CG, df.project(P.u[0], self.V_DG))
        self.update_dA_dpsi(Bz, Br, xx, self.V_CG, df.project(P.u[0], self.V_DG))

    def save_electron_params(self, cfg, output_path, log, subfolder="updated_df_param"):
        path = output_path(cfg, suffix=cfg["electron_solver"]["je_condition"]) + subfolder + "/"
        log.info("Saving electron params → %s", path)
        funcs = {
            "He_df":               self.ElectronModel.He,
            "dHe_dpsi_df":         self.ElectronModel.dHe_dpsi,
            "He_nocoll_df":        self.ElectronModel.He_nocoll,
            "dHe_dpsi_nocoll_df":  self.ElectronModel.dHe_dpsi_nocoll,
            "A_Te_df":             self.ElectronModel.A_Te,
            "dA_Te_dpsi_df":       self.ElectronModel.dA_Te_dpsi,
            "A_Te_nocoll_df":      self.ElectronModel.A_Te_nocoll,
            "dA_Te_dpsi_nocoll_df":self.ElectronModel.dA_Te_dpsi_nocoll,
            "G_e_df":              self.ElectronModel.G_e,
            "G_e_nocoll_df":       self.ElectronModel.G_e_nocoll,
        }
        for name, func in funcs.items():
            src.save_load.save_df_function(path, func, name)
