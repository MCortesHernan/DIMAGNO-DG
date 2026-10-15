
# --- IMPORTS --- #

import numpy as np
import ufl_legacy as ufl
from dolfin import project
import dolfin as df
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp 
from scipy.integrate import simpson
from scipy.interpolate import griddata
from scipy.interpolate import interp1d

import src
import gc

class ElectronSolver:
    def __init__(self, z_grid, x_grid, z_len, x_len, V_CG, V_DG, BField2D, ElectronModel, CollisionModel,
                 throat_lim=[0, 1], N_l=200, N_psi=500, je_condition='ICA', phi_w = 0.0):
        
        '''
        je_condition: 
            - 'ICA' (Inbound Current Ambipolarity) 
            - 'OFW' (Outbound Floating Wall)
            - 'wall_ambipolarity' (Wall Ambipolarity) - TODO: physical problems as upare extremely low and not consistent in periphery
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

        # Cache to store streamlines and avoid recomputation
        self._streamline_cache = {}

    
    def A_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):

        pts = streamline[:, :2]
        # Numerical samples along the streamline (arrays)
        n_s   = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s  = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        jpare_s = np.array([self.ElectronModel.jpare_cont(df.Point(z, x)) for z, x in pts])
        dHe_dpsi_s = np.array([self.ElectronModel.dHe_dpsi_cont(df.Point(z, x)) for z, x in pts])
        dA_Te_dpsi_s = np.array([self.ElectronModel.dA_Te_dpsi_cont(df.Point(z, x)) for z, x in pts])
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

        # Interpolants (numeric)
        n_interp = interp1d(s, n_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_interp = interp1d(s, ue_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        nn_interp = interp1d(s, nn_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_par_interp = interp1d(s, ue_par_s, kind='linear', fill_value="extrapolate", assume_sorted=True)

        def rhs(ds, A):
            # enforce scalar numeric A
            A = float(A)
            A = max(A, 1e-12)

            # safe interpolated values
            n_i = float(n_interp(ds))
            ue_i = float(ue_interp(ds))
            ue_par_i = float(ue_par_interp(ds))
            nn_i = float(nn_interp(ds))
            # clamp / protect interpolated values
            n_i = max(n_i, 1e-12)
            ue_par_i = ue_par_i

            Te = self.ElectronModel.Te(n_i, A)
            Te = max(Te, 1e-3)  # Avoid unphysically low temperatures that can cause issues in collision rates
            Te = min(Te, 1)  # Avoid unphysically high temperatures that can cause overflow in exp

            # safe inelastic energy term (cap exponent)
            exp_arg = (2.0 * E_i_) / (3.0 * Te + 1e-12)
            exp_arg = np.clip(exp_arg, -50, 50)
            E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

            ioniz = float(self.CollisionModel.Ionization(nn_i, Te, lib=np))
            nu_en = float(self.CollisionModel.E_N(nn_i, Te, self.CollisionModel.me, np))
            nu_ei = float(self.CollisionModel.E_I(nn_i, Te, self.CollisionModel.me, np))
            nu_exc = nu_ei*float(self.CollisionModel.Excitation(nn_i, Te, np)*(self.CollisionModel.Te0/self.CollisionModel.E_ion))
            nu_e = (nu_ei + nu_en + nu_exc)

            #rhs_val = - ioniz/(ue_i + 1e-12) * A * (gamma + (gamma-1) * E_inel/(Te + 1e-12) )
            rhs_val_1 = - ioniz*(gamma/(gamma-1)*Te + E_inel)
            rhs_val_2 = nu_e * self.CollisionModel.me * ue_i**2
            rhs_val = (gamma-1)/(ue_par_i * Te + 1e-12) * (rhs_val_1 + rhs_val_2)

            # Avoid NaN/infs
            if not np.isfinite(rhs_val):
                # clamp to a safe large value (or zero)
                ioniz = float(self.CollisionModel.Ionization(1e-5, Te, lib=np))
                rhs_val = - ioniz/(ue_i + 1e-12) * A * (gamma + (gamma-1) * E_inel/(Te + 1e-12) )

            return float(rhs_val)

        # initial A: use analytic or numeric safe scalar
        A0 = 1

        # Use a stiff solver if collisions make it stiff
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

        # cleanup
        del n_s, ue_par_s, nn_s
        del n_interp, ue_interp, nn_interp

        gc.collect()

        return s, A_out
    
    def He_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                        n_df, nn_df):
        
        pts = streamline[:, :2]
        n_s = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        A_Te_s = np.array([self.ElectronModel.A_Te_cont(df.Point(z, x)) for z, x in pts])
        dA_Te_dpsi_s = np.array([self.ElectronModel.dA_Te_dpsi_cont(df.Point(z, x)) for z, x in pts])
        dHe_dpsi_s = np.array([self.ElectronModel.dHe_dpsi_cont(df.Point(z, x)) for z, x in pts])
        x_s = pts[:, 0]
        ue_y_s = -x_s*dHe_dpsi_s + 1/(self.ElectronModel.gamma - 1) * (self.ElectronModel.Te0 / self.ElectronModel.ne0**(self.ElectronModel.gamma - 1)) * n_s**(self.ElectronModel.gamma - 1) * x_s* dA_Te_dpsi_s #Electron diamagnetic velocity (assuming polytropic electrons)
        jpare_s = np.array([self.ElectronModel.jpare_cont(df.Point(z, x)) for z, x in pts])
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

        n_interp = interp1d(s, n_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        uye_interp = interp1d(s, ue_y_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        nn_interp = interp1d(s, nn_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        A_interp = interp1d(s, A_Te_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        ue_par_interp = interp1d(s, ue_par_s, kind='linear', fill_value="extrapolate", assume_sorted=True)

        def rhs(ds, He):

            # safe interpolated values
            n_i = float(n_interp(ds))
            uye_i = float(uye_interp(ds))
            nn_i = float(nn_interp(ds))
            A_i = float(A_interp(ds))
            ue_par_i = float(ue_par_interp(ds))

            Te = self.ElectronModel.Te(n_i, A_i)
            Te = max(Te, 1e-3)  # Avoid unphysically low temperatures that can cause issues in collision rates
            Te = min(Te, 1)  # Avoid unphysically high temperatures that can cause overflow in exp
            gamma = self.gamma_e

            nu_i = float(self.CollisionModel.Ionization(nn_i, Te, lib=np))
            nu_en = float(self.CollisionModel.E_N(nn_i, Te, self.CollisionModel.me, lib=np))
            nu_ei = float(self.CollisionModel.E_I( n_i, Te, self.CollisionModel.me, lib=np))
            nu_exc = nu_i*float(self.CollisionModel.Excitation(n_i, Te, lib=np))*(self.CollisionModel.Te0/self.CollisionModel.E_ion)

            nu_e = (nu_ei + nu_en + nu_i + nu_exc)
            
            exp_arg = (2.0 * E_i_) / (3.0 * Te + 1e-12)
            # hard numerical cap
            exp_arg = np.clip(exp_arg, -50, 50)
            E_inel = E_i_ * (2.0 + 0.25 * np.exp(exp_arg))

            rhs_val_1 = - nu_i * (gamma/(gamma - 1)*Te + E_inel)
            rhs_val_2 = - nu_e * self.CollisionModel.me * uye_i**2
        
            
            rhs_val = 1/(ue_par_i + 1e-12) * (rhs_val_1 + rhs_val_2)

            if not np.isfinite(rhs_val):
                # clamp to a safe large value (or zero)
                rhs_val = 1e-12

            rhs_val = np.real(rhs_val)
            
            return float(rhs_val)
            
        He0 = 6 #from value of function to avoid evaluating and introducing fenics errors
        # TODO this must be generalized 

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
        
        del n_s, A_Te_s, ue_par_s
        del n_interp, nn_interp, A_interp, uye_interp, ue_par_interp
        gc.collect()

        return s, He_out
    
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
    
    def zero_finder(self, phi_w_0, phi_w_1, n_df, jzi_df, jxi_df, z_s, x_s):
        '''This finds the wall potential that returns 0 collected current.'''
        eps = 1e-6
        collected_curr_0 = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_0)
        collected_curr_1 = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_1)
        if collected_curr_0 * collected_curr_1 > 0:
            print('Initial potentials must be chosen so that a zero exists in the middle. \n')
            print(f'Collected curents for the chosen initial potentials are: {collected_curr_0} and {collected_curr_1}.\n')
            return None
        else:
            iter = 0
            while True:
                phi_w_aux = 0.5*(phi_w_1 + phi_w_0)
                collected_curr_aux = self.col_i_current(z_s, x_s, jzi_df, jxi_df) + self.col_e_current(z_s,x_s, n_df, phi_w_aux)
                print(f'Potential at iteration {iter}: {phi_w_aux}')
                print(f'Collected current: {collected_curr_aux}\n')
                
                if abs(collected_curr_aux ) < eps:
                    break

                if collected_curr_aux * collected_curr_0 > 0:
                    phi_w_0 = phi_w_aux
                else:
                    phi_w_1= phi_w_aux   
                iter += 1
                
            return phi_w_aux

    def G_streamline_f_ivp(self, streamline: np.ndarray, s: np.ndarray,
                                n_df, nn_df, jzi_df, jxi_df):
        pts = streamline[:, :2]
        n_s = np.array([n_df(df.Point(z, x)) for z, x in pts])
        nn_s = np.array([nn_df(df.Point(z, x)) for z, x in pts])
        jzi_s = np.array([jzi_df(df.Point(z, x)) for z, x in pts])
        jxi_s = np.array([jxi_df(df.Point(z, x)) for z, x in pts])
        A_Te_s = np.array([self.ElectronModel.A_Te_cont(df.Point(z, x)) for z, x in pts])
        Te_s = self.ElectronModel.Te(n_s, A_Te_s)
        Si_s = self.CollisionModel.Ionization(nn_s, Te_s, lib=np) * n_s

        bz_s = np.array([self.BField2D.Bz(z, x) for z, x in pts])
        bx_s = np.array([self.BField2D.Br(z, x) for z, x in pts])
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

        Si_interp = interp1d(s, Si_s, kind='linear', fill_value="extrapolate", assume_sorted=True)
        Bnorm_interp = interp1d(s, Bnorm, kind='linear', fill_value="extrapolate", assume_sorted=True)

        def rhs(ds, Y):
            val = 1/(Bnorm_interp(ds) + 1e-8) * (Si_interp(ds)) 
            val = float(val)
            return val

        if self.je_condition == 'ICA':
            Y0 = j_par_i[0]/(Bnorm[0]+1e-8)          
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
        
        elif self.je_condition == 'OFW':

            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len

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

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose metal WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            try:
                sol = solve_ivp(ufl.rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='BDF',
                            rtol=1e-6, atol=1e-8)
            except Exception:
                # fallback to RK45 if BDF raises problems
                sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev, method='RK45',
                            rtol=1e-6, atol=1e-8)
                
            # reverse solution back to original order
            G_out = sol.y[0][::-1]
            if G_out.shape[0] != s.shape[0]:
                G_out = np.interp(s, sol.t, G_out, left=G_out[0], right=G_out[-1])

            if not sol.success:
                print(f"Integration failed or produced NaN at streamline. Filling with zero.")
                G_out = np.full_like(s, np.nan)
        
        elif self.je_condition == 'wall_ambipolarity':
            eps = 1e-3
            diff_z = pts[-1,0] - self.z_len
            diff_x = pts[-1,1] - self.x_len
            
            if abs(pts[-1,0]) < eps:
                jn = jzi_s[-1] 
                j_e_par_bc = jn / bz_hat[-1]
            elif abs(diff_x) < eps: 
                jn = jxi_s[-1]
                j_e_par_bc = jn / bx_hat[-1]
            elif abs(diff_z) < eps:
                jn = jzi_s[-1]
                j_e_par_bc = jn / bz_hat[-1]
            else:
                j_e_par_bc = j_par_i[-1]

            s_rev     = s[::-1]
            B_rev     = Bnorm[::-1]

            # impose ambipolarity at WALL:
            Y0 = j_e_par_bc / (B_rev[0] + 1e-12)

            sol = solve_ivp(rhs, (s_rev[0], s_rev[-1]), [Y0], t_eval=s_rev,
                            method='RK45', rtol=1e-8, atol=1e-8)

            # reverse solution back to original order
            G_out = sol.y[0][::-1]        

        return s, G_out
    
    def solve_grid_streamline_it(self, eq: str = 'Continuity', V = None, n_df = 1, nn_df = 1, jzi_df = 1, jxi_df = 1, axis_df = 1):
        
        # Cached streamlines
        if not self._streamline_cache:
            throat_1_array = np.linspace(self.throat_lim[0],self.throat_lim[1], int(2*self.N_psi))
            psi_1_array = np.array([self.BField2D.psi(0, x) for x in throat_1_array[:]])
            psi_vals = psi_1_array

            S_Tracing = src.BFields.Streamline_Tracing(self.BField2D, self.z_grid, self.x_grid, self.N_l)
            for psi_val in psi_vals:
                streamline, s = S_Tracing.streamline_mapping(psi_val)
                if streamline is not None:
                    self._streamline_cache[psi_val] = (streamline, s)

            for psi_val, (streamline, s) in self._streamline_cache.items():
                plt.plot(streamline[:,0], streamline[:,1], 'k-', linewidth=0.5)

            del S_Tracing
            gc.collect()

        all_points, all_values = [], []
        it=1
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
                print(f"Streamline {it}: ψ={psi_val} done.\n")
                it+=1
            except Exception as e:
                print(f"Streamline {it}: ψ={psi_val} failed: {e}")
                it+=1
                continue

        if not all_points:
            coords = V.tabulate_dof_coordinates().reshape((-1, 2))
            return np.zeros(coords.shape[0])

        all_points = np.vstack(all_points)
        all_values = np.concatenate(all_values)
        coords = V.tabulate_dof_coordinates().reshape((-1, 2))

        valid_mask = ~np.isnan(all_values)

        clean_points = all_points[valid_mask]
        clean_values = all_values[valid_mask]

        function_grid = griddata(
            clean_points,
            clean_values,
            coords,
            method='linear'
        )

        nan_mask = np.isnan(function_grid)
        function_grid[nan_mask] = griddata(
            clean_points,
            clean_values,
            coords[nan_mask],
            method='nearest'
        )

        del all_points, all_values
        gc.collect()

        return function_grid
    
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

    def smooth_Helmholtz(self, f, alpha_val = 0.1):
        '''
        Smooth the function f using a Helmholtz filter.
        ''' 
        u = df.TrialFunction(self.V_CG)
        v = df.TestFunction(self.V_CG)
        alpha = df.Constant(alpha_val)   # controls smoothness
        a = (u*v + alpha**2*df.dot(df.grad(u), df.grad(v)))*df.dx
        L = f*v*df.dx
        f_smooth = df.Function(self.V_CG)
        df.solve(a == L, f_smooth)

        del u, v, alpha, a, L
        gc.collect()
        print("Smoothing applied to function.")
        f_smooth_DG = df.project(f_smooth, self.V_DG)
        print("Projection to DG done.")

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
    
