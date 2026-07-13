import os
import shutil
import numpy as np
import dolfin as df

import src
import matplotlib.pyplot as plt

class Simulation: 
    '''
    Class that performs the general assembly and update of the solution Function u
    '''
    def __init__(self, mesh, PhysicalModel, NumericalFlux, Propagator, bcs={}, 
                dg_degree=0, t=0., u=None, dt=0.01, savepath="saves/",
                figpath="figs/", XDMFpath="xdmf/", converged = False, refinement = None, test_num = 0):
        self.mesh = mesh
        self.PhysicalModel = PhysicalModel 
        self.NumericalFlux = NumericalFlux 
        self.Propagator = Propagator 
        self.bcs = bcs 
        self.dg_degree = dg_degree 
        self.t = t
        self.u = u
        self.dt = dt
        self.savepath = savepath
        self.figpath = figpath
        self.XDMFpath = XDMFpath
        self.converged = converged
        self.refinement = refinement
        self.test_num = test_num

        # Calculate mesh-derived quantities
        self._on_mesh_update()

    def _on_mesh_update(self):
        
        '''
        Internal method.
        Calculates mesh-derived quantities, finite element spaces, and applies BCs.
        Should be run initially and whenever the mesh is changed.
        This is handled automatically by the class.
        '''

        # Mesh-derived quantities
        self.normal = df.FacetNormal(self.mesh) 
        V = df.VectorFunctionSpace(self.mesh, 'CG', 1, dim=self.PhysicalModel.x_dim()) # Auxiliary space for coords
        self.coords = df.interpolate(df.Expression(tuple('x['+str(i)+']' for i in range(self.PhysicalModel.x_dim())),degree=1), V) 

        # Function space and function
        self.W = df.VectorFunctionSpace(self.mesh, 'DG', self.dg_degree, dim=self.PhysicalModel.u_dim()) 
        self.v = df.TestFunction(self.W)   
        self.du = df.Function(self.W)   
        self.u = df.Function(self.W) if self.u is None else df.interpolate(self.u,self.W) # An existing df.Function will be interpolated
        
        # Mark boundaries and measures
        boundary_markers = df.MeshFunction('size_t', self.mesh, dim=self.PhysicalModel.x_dim()-1) # A mesh function over facets (x_dim-1)
        i = 0
        for sd in self.bcs.keys(): # Apply SubDomain markers to boundary_markers, used to define ds, and give each piece an integer id
            sd.mark(boundary_markers, i)
            i += 1

        self.dx = df.Measure('dx', domain=self.mesh)
        self.dS = df.Measure('dS', domain=self.mesh)
        self.ds = df.Measure('ds', domain=self.mesh, subdomain_data=boundary_markers)

        # Clear memory of propagator
        self.Propagator.reset_memory()
 
    def a_form(self,du):
        '''
        form a in the time-dependent PDE weak form a(v,du) == b(v,f(t,u)) * dt
        '''
        return src.WeakForms.a_form(self.v,du,self.dx,self.u,self.PhysicalModel)
    
    def stabilize(self, b_form):
        
        '''Add a diffusive term in the DG discretization of the bilinear form as given by Hartmann and Houston.'''

        #Get diffusion coefficient:

        F = df.as_tensor(self.PhysicalModel.flux(self.u))
        h = df.CellDiameter(self.mesh)
        (C_eps, beta, ext) = self.PhysicalModel.stabilization_params
        div_F = df.sqrt(df.dot(df.div(F), df.div(F))) #Norm of the divergence of the flux

        eps = df.as_tensor((C_eps*h**(2-beta)*div_F + ext)*df.Identity(self.u.ufl_shape[0])) 

        b_form += src.WeakForms.flux_volume_form(self.v, eps*df.grad(self.u), self.dx, self.u, self.PhysicalModel)

        return b_form

    def b_form(self,u):
        '''
        form b in the time-dependent PDE weak form a(v,du) == b(v,f(t,u)) * dt
        Parameters:
        -----------
        t: Time in the right hand side
        u: State vector in the right hand side 
        '''
        # Volume flux integral
        F = df.as_tensor(self.PhysicalModel.flux(u))

        b  = -src.WeakForms.flux_volume_form(self.v,F,self.dx,self.u,self.PhysicalModel)
        b += src.WeakForms.flux_volume_ax_form(self.v, self.dx, self.u, self.PhysicalModel.ElectronModel, self.PhysicalModel)

        # Surface flux integral
        Fn = df.as_vector(self.NumericalFlux.flux_n(u('+'), u('-'), normal_p = self.normal('+'), PhysicalModel = self.PhysicalModel))
        b += -src.WeakForms.flux_surface_form(self.v('+')-self.v('-'),Fn,self.dS,df.Constant((1,0)),self.PhysicalModel)

        # RHS source term integral
        rhs = df.as_vector(self.PhysicalModel.rhs(u))
        b += src.WeakForms.rhs_volume_form(self.v,rhs,self.dx,self.u,self.PhysicalModel)
        
        # Weakly imposed boundary condition integrals
        i = 0
        for bc in self.bcs.values():
            u_outside = df.as_vector(bc.u_outside(u, self.normal))

            # Strongly imposed BCs: directly enforce u = u_outside on ds(i)
            if hasattr(bc, "strong_imposition") and bc.strong_imposition:
                print(f"Applying strong BC on ds({i})")
                Fn = df.as_vector(self.NumericalFlux.flux_n(u_outside, u_outside, -self.normal,
                                                            self.PhysicalModel, bc=True))
                b += -src.WeakForms.flux_surface_form(-self.v, Fn, self.ds(i), self.u, self.PhysicalModel)

            # Weakly imposed BCs: use numerical flux formulation
            else:
                Fn = df.as_vector(self.NumericalFlux.flux_n(u_outside, u, -self.normal,
                                                            self.PhysicalModel, bc=True))
                b += -src.WeakForms.flux_surface_form(-self.v, Fn, self.ds(i), self.u, self.PhysicalModel)
            i += 1

        if self.PhysicalModel.stabilization_params[0] is not None:
            b = self.stabilize(b)

        return b

    def step(self):
        '''
        Performs a single time step with chosen propagator
        ''' 
        # du = df.Function(self.W)
        # du.assign(self.du)
        
        # if isinstance(self.Propagator, Propagators.RK):
        #     self.Propagator.initialize_solver(du, self.a_form, self.b_form, self.u, self.dt)

        self.Propagator.update_du(self.du, self.a_form, self.b_form, self.u, self.dt) 
        #self.du.assign(du)
        
        self.t = self.t + self.dt

        if isinstance(self.Propagator, src.Propagators.RK):
            self.u.assign((1./3.)*self.u + (2./3.)*(self.Propagator.u_2 + self.du))
        else:
            self.u.assign(self.u + self.du)

        # Enforce positibity if such option is selected
        #self.u.vector()[:] = self.PhysicalModel.force_positivity(self.u)
        q_vals = self.PhysicalModel.force_positivity(self.u)

        self.u.vector().set_local(q_vals)
        self.u.vector().apply("insert")

        # Validity check before updating solution
        assert self.PhysicalModel.is_valid(self.u), "Invalid state reached!"

        #Force background_pressure if exists
        #if hasattr(self.PhysicalModel, 'NeutralModel'):
        try:
            self.u.vector()[:] = self.PhysicalModel.neutral_background(self.u)
        except AttributeError:
            self.u.vector()[:] = self.u.vector()[:]

    def steady_state(self):
        '''
        Solve for steady state directly. It does not assign solution to u in case it fails
        ''' 
        try:
            self.u_steady_state = df.Function(self.W)
            self.u_steady_state.assign(self.u)

            
            df.solve(0 == self.b_form(self.u_steady_state), self.u_steady_state,
             solver_parameters=
             {"newton_solver":{'maximum_iterations':15}})

            self.converged = True
            self.u.assign(self.u_steady_state)
            self.t = np.Inf # If converged, this is infinity

        except RuntimeError:
            print('Newton Solver did not converge.')
        return None

    def refine_mesh(self):
        '''
        Refines the mesh
        '''
        self.mesh = df.refine(self.mesh)
        self._on_mesh_update()
        return None
    
    def refine_p(self):
        '''
        p refine
        '''
        self.dg_degree += 1
        self._on_mesh_update()
        return None	

    def simulate(self,strategy=[50]):
        '''
        Performs a simulation with chosen strategy list, which may include
        * integers: number of time steps
        * function: execute arbitrary function on self (e.g. postproc functions)
        * 'steady': solve for steady state
        * 'refine': refine mesh
        * 'p_refine': increase one order in the polinomial base
        * 'double': double the dt
        * 'halve': halve the dt
        * 'clearfigpath': clear fig directories
        * 'clearsavepath': clear save directories
        '''
        print('Initiating simulation...')
        for s in strategy:
            print('Next strategy: ' + str(s))
            if type(s) == int:
                for i in range(1,s+1):
                    print('Initiating step ' + str(i) + ' of ' + str(s))
                    self.step()
            elif callable(s):
                s(self)
            elif s == 'steady':
                self.steady_state()
            elif s == 'refine':
                self.refine_mesh()
                self.refinement += 1
                self.Propagator.initialize_solver(self.du, self.a_form, self.b_form, self.u, self.dt)
                self.converged = False
                self.t = 0
            elif s == 'p_refine':
                self.refine_p()
                self.Propagator.initialize_solver(self.du, self.a_form, self.b_form, self.u, self.dt)
                self.converged = False
                self.t = 0
            elif s == 'double':
                self.dt = self.dt * 2
                self.Propagator.initialize_solver(self.du, self.a_form, self.b_form, self.u, self.dt)
                print('dt doubled to: ' + str(self.dt))            
            elif s == 'halve':
                self.dt = self.dt / 2
                self.Propagator.initialize_solver(self.du, self.a_form, self.b_form, self.u, self.dt)
                print('dt halved to: ' + str(self.dt))
            elif s == 'clearfigpath':
                shutil.rmtree(self.figpath, True) 
            elif s == 'clearsavepath':
                shutil.rmtree(self.savepath, True)
            else:
                raise ValueError("Invalitegy: "+str(s))
        return None

    def tstr(self):
        '''
        Returns a string with the current time, rounded to 0.00001
        '''
        return str(np.round(100000*self.t)/100000)

    def save_u(self,filename=None):
        '''
        Save the solution to file TODO: change this to a dolfin save, to save df.Function completely, not only the vector! Perhaps also save mesh, t, and others
        ''' 
        os.makedirs(self.savepath, exist_ok=True)
        filename = self.tstr() if filename is None else filename
        np.save(self.savepath+filename, self.u.vector()[:])
        return None

    def read_u_from_file(self, path):
        '''
        Read an existing vector state from file and set u.
        '''

        u_read = df.Function(self.W)

        df.XDMFFile(path).read_checkpoint(u_read, "Solution Vector", -1)
        self.u.assign(u_read)

        return None

    def run_to_convergence(self, cfg, log):
        tol     = cfg["convergence"]["plasma_tol"]
        n_steps = cfg["convergence"]["steps_per_check"]
        max_iter = cfg["convergence"].get("max_iter", None)   
        
        it = 0
        while True:
            it += 1
            self.simulate([n_steps])
            dq_dt = self.time_convergence()
            log.info("dq/dt = %.3e  (tol = %.1e)", dq_dt, tol)
            src.save_xdmf(self)
            if dq_dt < tol:
                log.info("Converged.")
                break
            if max_iter is not None and it >= max_iter:
                log.warning("Max iterations (%d) reached before convergence", max_iter)
                break

    def time_convergence(self):
        q_vec_0 = self.u.vector()[:]
        self.simulate([20])
        q_vec_1 = self.u.vector()[:]
        dq_dt = np.linalg.norm(q_vec_1-q_vec_0)/(20*self.dt)
        return dq_dt

    def solve_time_evol(self, plot, back_dens = 0, tol=1e-6, its = 500, max_IT = 100):
        iter = 0
        while not self.converged:
            print(f'This is running simulation number: {self.test_num} with refinemet {self.refinement} and order p = {self.dg_degree} (IT: {iter+1})\n')
            self.simulate([its])
            print(f'Elapsed time is : {self.t} (since beginning) \n')
            #plot(self)
            dq_dt = self.time_convergence()
            print(f'Total time derivative in the last 20 timesteps: {dq_dt}')
            plot(self)
            src.save_xdmf(self)

            if back_dens > 1e-6:
                if dq_dt < tol:
                    self.converged = True
                    self.t = np.Inf
                    print('Simulation has converged (dq_dt < tol)')
            else:
                if dq_dt < tol:
                    self.converged = True
                    self.t = np.Inf
                    print('Simulation has converged (dq_dt < tol)')
                #self.simulate(['steady'])
            
            if self.converged:
                plot(self)
                src.save_xdmf(self)
                print('Simulation saved in ' + self.XDMFpath)
            
            iter += 1
            
            if iter >= max_IT:
                print('Maximum number of iterations reached')
                plot(self)
                print('Simulation stopped without convergence and dq_dt = ' + str(dq_dt))
                src.save_xdmf(self)
                print('Simulation saved in ' + self.XDMFpath)
                break
        return None
