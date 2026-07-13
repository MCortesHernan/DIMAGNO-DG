#%%
# Take a simulation object and calculate self induced B for that flow configuration

import numpy as np
import dolfin as df

def self_B(P, dHe_dpsi, h, mu_r, z, x, norm_flag = False, layers_flag = True):

    '''Induced Magnetic field normalised to 1 at position z, x if norm_flag = True'''

    #Function Spaces for vector potential and B-Field
    V = df.FunctionSpace(P.mesh, 'CG', 2)
    V_mu = df.TensorFunctionSpace(P.mesh, 'DG', 0, (2,2))
    Vdim = df.VectorFunctionSpace(P.mesh, 'CG', 2)

    #Sizes
    zL = np.min(P.mesh.coordinates()[:,0])
    zR = np.max(P.mesh.coordinates()[:,0])
    xD = np.min(P.mesh.coordinates()[:,1])
    xU = np.max(P.mesh.coordinates()[:,1])

    #def boundary(x, on_boundary):
    #    if on_boundary:
    #        if df.near(x[0], zR) or df.near(x[1], xU):
    #            return True

    #        else:
    #            True #False originalmente

    def boundary(x, on_boundary):
        return on_boundary and  (df.near(x[1], xU))

    bc = df.DirichletBC(V, df.Constant(0), boundary)


    # Define integration measure
    dx = df.Measure('dx', domain=P.mesh)

    if type(mu_r) == list:
        N_layers = len(mu_r)

    else:
        N_layers = 1


    #Assign permeability to each domain
    class inv_Permeability(df.UserExpression):

        def eval(self, values, x):

            values[0] = 1
            values[1] = 0
            values[2] = 0
            values[3] = 1
            
            for i in range(1, N_layers+1):

                # if (zR - i*h) < x[0] <= (zR - (i-1)*h) and x[1] <= xU - h:
                #     values[0] = 1/mu_r[-i]
                #     values[1] = 0
                #     values[2] = 0
                #     values[3] = mu_r[-i]

                if (xU-i*h) < x[1] <= (xU - (i-1)*h):
                    values[0] = mu_r[-i]
                    values[1] = 0
                    values[2] = 0
                    values[3] = 1/mu_r[-i]


        def value_shape(self):
            return(2,2)

    if layers_flag:
        inv_mu = df.interpolate(inv_Permeability(), V_mu)

    else: 
        inv_mu = df.Constant(1.0)

    #Currents from solution
    J_y_e = P.u[0]*dHe_dpsi

    J_y_i = P.u[3]

    J_y = J_y_e + J_y_i

    J_y = df.project(J_y, V)

    #Erase Currents in the material layer

    class Total_Current(df.UserExpression):

        def eval(self, values, x):
            if zR - N_layers*h < x[0] <= zR or xU - N_layers*h < x[1] <= xU and layers_flag:
                values[0] = 0

            else:
                values[0] = J_y(x[0],x[1])

        def value_shape(self):
            return ()

    J_total = df.project(Total_Current(), V)

    psi_p = df.TrialFunction(V)

    v = df.TestFunction(V)

    a =  df.dot(inv_mu*df.grad(psi_p), df.grad(v))*dx

    L = J_total*v*dx

    psi_p = df.Function(V)

    df.solve(a == L, psi_p, bc)

    B_p_x = -psi_p.dx(0)

    B_p_z = psi_p.dx(1)

    B_p = df.project(df.as_vector((B_p_z, B_p_x)), Vdim)

    if norm_flag:
        norm = np.linalg.norm(B_p(z,x))
    else:
        norm = 1

    psi_p = df.project(psi_p/norm, V)

    B_p = df.project(df.as_vector((B_p_z/norm, B_p_x/norm)), Vdim)

    return B_p, psi_p
 