'''
Time propagators for dolfin problems of the form a(v,du) = b(v,u) * dt
'''
import dolfin as df
import ufl_legacy as ufl 
import numpy as np

class Propagator:
    '''
    Abstract propagator class for dolfin problems
    ''' 
    def __init__(self,tol=1e-4,maxiter=10):
        self.memory = None # Memory of the Propagator 
        self.solver_parameters = {'newton_solver': { 'absolute_tolerance': tol, 'relative_tolerance': tol, 'maximum_iterations': maxiter} } 
    def update_du(self, du, a_form, b_form, u, dt):
        '''
        Updates the solution du in the problem a_form(du) = b_form(u) * dt
        WARNING: overwrites du

        Parameters:
        -----------
        du: Update solution (INTENT INOUT)
        a_form: Left hand side of the update equation
        b_form: Right hand side of the update equation (except for the dt)
        t: Current time
        u: Current solution
        dt: Time step
        ''' 
        raise NotImplementedError("Abstract method") 
    def reset_memory(self):
        '''
        Resets the Jacobian memory, forcing the Jacobian to be recomputed on next call to update_du
        '''
        self.memory = None
        return None

class ForwardEuler(Propagator):
    '''
    Forward Euler propagator for dolfin problems
    '''  
    def update_du(self, du, a_form, b_form, u, dt): 
        if self.memory is None: 
            self._du = df.TrialFunction(du.function_space())    
            self.a_form = a_form(self._du)
            self.b_form = b_form(u)
            self.memory = 1  
        df.solve(self.a_form == dt * self.b_form, du)
        # df.solve(self.assembled_mass_matrix, du.vector(), df.assemble(b_form(t,u)) * dt)
        return None

class BackwardEuler(Propagator):
    '''
    Backward Euler propagator for dolfin problems
    ''' 
    def update_du(self, du, a_form, b_form, u, dt): 
        if self.memory is None: 
            self.a_form = a_form(du)
            self.b_form = b_form(u + du)
            self.Ja = df.derivative(self.a_form, du) 
            self.Jb = df.derivative(self.b_form, du) 
            self.memory = 1 
        df.solve(self.a_form - dt * self.b_form == 0, du, J=self.Ja - dt * self.Jb, solver_parameters=self.solver_parameters)
        return None

class RK(Propagator):
    '''
    Strong stability preserving Runge Kutta propagator
    '''
    def __init__(self, lumped_mass = False, order=3):
        super().__init__()
        self.lumped_mass = lumped_mass
        self.order = order
        if order not in [3]:
            raise ValueError("Only orders 3 is implemented")
    def initialize_solver(self, du, a_form, b_form, u, dt):
        if self.memory is None:
            self._du = df.TrialFunction(du.function_space())
            self.a_form = a_form(self._du)
            self.b_form0 = dt*b_form(u) #*dt
            self.u_1 = df.Function(u.function_space())
            self.u_2 = df.Function(u.function_space())
            self.b_form_2 = ufl.replace(self.b_form0, {u: self.u_1})
            self.b_form_3 = ufl.replace(self.b_form0, {u: self.u_2})

            self.dtc = df.Constant(dt)

            self._b_vec = df.Vector(du.vector())

            self.memory = 1

        if not self.lumped_mass:      
            
            self.prob1 = df.LinearVariationalProblem(self.a_form, self.dtc*self.b_form0, du)
            self.solv1 = df.LinearVariationalSolver(self.prob1)
            self.solv1.parameters['preconditioner'] = 'ilu'
            self.solv1.parameters['linear_solver'] = 'gmres'
        
            self.prob2 = df.LinearVariationalProblem(self.a_form, self.dtc*self.b_form_2, du)
            self.solv2 = df.LinearVariationalSolver(self.prob2)
            self.solv2.parameters['preconditioner'] = 'ilu'
            self.solv2.parameters['linear_solver'] = 'gmres'
            

            self.prob3 = df.LinearVariationalProblem(self.a_form, self.dtc*self.b_form_3, du)
            self.solv3 = df.LinearVariationalSolver(self.prob3)
            self.solv3.parameters['preconditioner'] = 'ilu'
            self.solv3.parameters['linear_solver'] = 'gmres'
            

        else:

            print("[RK] Using explicit mode with row-sum mass lumping.")

            M = df.assemble(self.a_form)
            M_petsc = df.as_backend_type(M).mat()

            # Compute conservative row-sum lumped mass
            M_lumped_vec = M_petsc.createVecLeft()
            M_petsc.getRowSum(M_lumped_vec)

            # Store as numpy array
            self.M_lumped_vec = M_lumped_vec.getArray().copy()

            eps = 1e-14
            self.M_lumped_vec[self.M_lumped_vec < eps] = eps

        #from petsc4py import PETSc
        # Set PETSc options globally
        #opts = PETSc.Options()
        #opts["ksp_rtol"] = 1e-5
        #opts["ksp_atol"] = 1e-5
        #opts["ksp_max_it"] = 10000
        #opts["ksp_monitor"] = None

        return None

    def update_du(self, du, a_form, b_form, u, dt):
        # if self.memory is None:
        #     self._du = df.TrialFunction(du.function_space())
        #     self.a_form = a_form(self._du)
        #     self.b_form = dt*b_form(u)
        #     self.u_1 = df.Function(u.function_space())
        #     self.u_2 = df.Function(u.function_space())
        #     self.b_form_2 = ufl.replace(self.b_form, {u: self.u_1})
        #     self.b_form_3 = ufl.replace(self.b_form, {u: self.u_2})
        #     self.memory = 1

        #self.solv1.solve()
        #print("After solv1: |du| =", np.abs(du.vector().get_local()))
        #self.u_1.assign(u + du)

        #self.solv2.solve()
        #print("After solv2: |du| =", np.abs(du.vector().get_local()))
        #self.u_2.assign(0.75 * u + 0.25 * (self.u_1 + du))

        #self.solv3.solve()
        #print("After solv3: |du| =", np.abs(du.vector().get_local()))

        if not self.lumped_mass:
            self.solv1.solve()
            #df.solve(self.a_form ==  dt*self.b_form, du)
            self.u_1.assign(u + du)

            self.solv2.solve()
            #df.solve(self.a_form == dt*self.b_form_2, du)
            self.u_2.assign(0.75*u + 0.25*(self.u_1 + du))

            self.solv3.solve()
            #df.solve(self.a_form == dt*self.b_form_3, du)
            #u.assign((1./3.)*u + (2./3.)*(self.u_2 + du))
        
        else:

            df.assemble(self.dtc * self.b_form0, tensor=self._b_vec)
            b_local = self._b_vec.get_local()
            # Safety check: finite numbers
            if not np.isfinite(b_local).all():
                raise FloatingPointError("Non-finite entries in assembled RHS (stage1).")
            du.vector()[:] = b_local / self.M_lumped_vec
            if not np.isfinite(du.vector().get_local()).all():
                raise FloatingPointError("Non-finite entries in du after stage1 division.")
            self.u_1.assign(u + du)

            # Stage 2
            df.assemble(self.dtc * self.b_form_2, tensor=self._b_vec)
            b_local = self._b_vec.get_local()
            if not np.isfinite(b_local).all():
                raise FloatingPointError("Non-finite entries in assembled RHS (stage2).")
            du.vector()[:] = b_local / self.M_lumped_vec
            if not np.isfinite(du.vector().get_local()).all():
                raise FloatingPointError("Non-finite entries in du after stage2 division.")
            self.u_2.assign(0.75 * u + 0.25 * (self.u_1 + du))

            # Stage 3
            df.assemble(self.dtc * self.b_form_3, tensor=self._b_vec)
            b_local = self._b_vec.get_local()
            if not np.isfinite(b_local).all():
                raise FloatingPointError("Non-finite entries in assembled RHS (stage3).")
            du.vector()[:] = b_local / self.M_lumped_vec
            if not np.isfinite(du.vector().get_local()).all():
                raise FloatingPointError("Non-finite entries in du after stage3 division.")
            # Final u update is normally performed in Simulation.step() after update_du()
            
        return None




