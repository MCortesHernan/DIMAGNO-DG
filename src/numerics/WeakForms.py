'''
Prepares ufl weakforms
'''
import ufl_legacy as ufl
import dolfin as df
 
i = ufl.Index()
j = ufl.Index()


def a_form(v,du,dx,u, PhysicalModel):
    '''
    Mass matrix form

    Parameters:
    -----------
    v: Test function
    dx: Cell measure
    ''' 
    
    axisym = PhysicalModel.axisymmetry
    x_dim = PhysicalModel.x_dim()
    mesh = PhysicalModel.mesh
    xx = ufl.SpatialCoordinate(mesh)[0] 
    x = xx + 1e-1
    ax = int(axisym)*x + (1-int(axisym))

    if x_dim < 3:
        q_degree = 10
    else:
        q_degree = 5
    
    return du[i]*v[i]*dx(metadata={"quadrature_degree": q_degree})
    
        
def flux_volume_form(v,F,dx,u, PhysicalModel):
    '''
    Volume integral of -F_ij(u) * d_j(v_i)

    Parameters:
    -----------
    v: Test function
    F: Flux tensor
    dx: Cell measure
    ''' 

    axisym = PhysicalModel.axisymmetry
    x_dim = PhysicalModel.x_dim()
    mesh = PhysicalModel.mesh
    xx = ufl.SpatialCoordinate(mesh)[0] 
    x = xx + 1e-1
    ax = int(axisym)*x + (1-int(axisym))

    if x_dim < 3:
        q_degree = 10
    else:
        q_degree = 5
    
    return -F[i,j]*v[i].dx(j)*dx(metadata={"quadrature_degree": q_degree})


def flux_volume_ax_form(v,dx,u, ElectronModel,PhysicalModel):
    '''
    Axisymmetric terms of the flux volume integral
    Volume integral of -F_ij(u) * d_j(v_i)

    Parameters:
    -----------
    v: Test function
    F: Flux tensor
    dx: Cell measure
    ''' 
    axisym = PhysicalModel.axisymmetry
    mesh = PhysicalModel.mesh
    xx = ufl.SpatialCoordinate(mesh)[1] 
    x = ufl.conditional(xx > 1e-6, xx, 1e-6)
    ax = 0*int(axisym)

    #axi_term = ax/x*(-v[3]*u[2]*u[3]/u[0] + v[3]*(u[3]*u[3]/u[0] + 0.66667*u[4] + ElectronModel.pe(u[0])))*x*dx(metadata={"quadrature_degree": 10})
    axi_term = 0
    try:
        axi_term += 0 #-ax/x*(0.66667*u[8])*v[7]*x*dx(metadata={"quadrature_degree": 10})
    except:
        ValueError

    return 0*axi_term

   
def flux_surface_form(v,Fn,ds,u, PhysicalModel):
    '''
    Surface integral of normal * Fn_i(u) * (v_i)

    Parameters:
    -----------
    v: Test function
    Fn: Normal flux vector
    ds: Surface measure (either ds or dS)
    '''
    axisym = PhysicalModel.axisymmetry
    x_dim = PhysicalModel.x_dim()
    mesh = PhysicalModel.mesh
    xx = ufl.SpatialCoordinate(mesh)[0] 
    x = xx + 1e-1
    ax = int(axisym)*x + (1-int(axisym))

    if x_dim < 3:
        q_degree = 10
    else:
        q_degree = 5

    return Fn[i]*v[i]*ds(metadata={"quadrature_degree": q_degree}) #(ufl.ln(ufl.sqrt(ufl.avg(u[0]))+0.1)**2+1)

def rhs_volume_form(v,rhs,dx,u, PhysicalModel):
    '''
    Volume integral of rhs_i * v_i * dx

    Parameters:
    -----------
    v: Test function
    rhs: Right hand side
    dx: Cell measure
    ''' 
    axisym = PhysicalModel.axisymmetry
    x_dim = PhysicalModel.x_dim()
    mesh = PhysicalModel.mesh
    xx = ufl.SpatialCoordinate(mesh)[0] 
    x = xx + 1e-1
    ax = int(axisym)*x + (1-int(axisym))

    if x_dim < 3:
        q_degree = 10
    else:
        q_degree = 5

    return rhs[i]*v[i]*dx(metadata={"quadrature_degree": q_degree})


