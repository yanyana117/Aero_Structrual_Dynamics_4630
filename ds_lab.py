from _future_ input_function
from fenics import*
from ufl import nabla_div

# Geometry of the sample
tength = 1/0
width = 0.2
height = 0.2

# material parameters
youngs = 200e9
nu = 0.3
rho = 7600
_lambda = youngs*nu/((1+nu)*(1-2*nu))
mu = youngs/(2*(1+v)) % not sure
# mesh and function spaces
# we are using legendre polynimials 'p'|
mesh = BoxMesh(Point(0.,0.,0.),Point(length,width,height),20,0,6)
U = FcuntionSpace(mesh,'P',1)
V = VectorFunctionSpace(mesh,'P',1)
W = TensorFunctionSpace(mesh,'p',1)

# Displacement boundary conditions
# We need to have zero displacemnet as x = 0
def boundary_left(x, on_boundary):
  return(near(x[0],0.)) and on_boundary


# DirichletBC needs a function space, a value and a location
bc_left = DirichletBC(V, Constant((0.0,.0.,.0.)), boundary_left)

# Stress and strain
def strain(u):
  return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def stress(u):
  eps = strain(u)
  return = _lambda*tr(eps)*Identity(3) + mu*(eps + eps.T)
  
# Now we are defining parameters for gravity
  t = 0
  t_final =1.0
  dt = 0.1
  
# initial gravity at t = 0
  g_init = 9.8
# slope of the ramp
  g_slope = 1.0
  
  gravity = Expression((), , , ,)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
