# Geometry of the sample
Length = 1.0
width = 0.2
height = 0.2

# Material parameters
youngs = 200e9
nu=0.3
rho=7800
_lambda=youngs*nu/((1.0+nu)*(1.0-2.0*nu))
mu = youngs/(2.0*(1.0+nu))

# Mesh
mesh = BoxMesh(Point(0.,0.,0.), Point(length,width,height),20,6,6)
U = FunctionSpace(mesh,'P',1)
V = VectorFunctionSpace(mesh,'P',1)
W = TensorFunctionSpace(mesh,'P',1)

# Displacement boundary conditions
# We need to have zero displacement at x = 0
def boundary_left(x, on_boundary):
    return near (x[0],0.) and on_boundary

# DirichletBC needs a function space, a value and a location. 
bc_left = DirichletBC(V, Constant((0.0,0.0,0.0)), boundary_left)

# Stress and strain
# u is a displacement vector
# It is a computing du/dx in 3D. 
def strain(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u). T)
def stress(u):
    eps = strain(u)
    return _lambda*tr(eps)*Identity(3)+mu*(eps+eps.T)

t=0
t_final = 1.0
dt = 0.1

# initial gravity at t = 0
g_init = 9.8
# slope of the ramp. 
g_slope = 1.0

gravity = Expression(('g1 + time*g2','0.0','0.0'), g1=g_init , g2=g_slope , time=t , degree=1)

#=========================================================
# Now we are creating output files..
output_stress = XDMFFile('stress.xdmf')
output_strain = XDMFFile('strain.xdmf')

output_stress.parameters["flush_output"] = True
output_strain.parameters["flush_output"] = True

output_stress.parameters["rewrite_function_mesh"] = False
output_strain.parameters["rewrite_function_mesh"] = False

stress_proj = Function(W)
strain_proj = Function(W)
#==========================================================

# Steps of defining a variational problem
# 1. Define a trail function
u = TrialFunction(V)
# 2. Define a test function
v = TestFunction(V)

# 2.5 define a body force
f = rho*gravity
T=constant((0,0,0))

# 3. Define the left hand side of the variational problem
# integral of sigma(u) epsilon(V) dx
a = inner(sigma(u),epsilon(v))*dx
L = dot(f,v)*dx + dot (T,v)*ds

# 4. Make u from TrailFunction to Function
u = Function(V)

# Now we are turning the knob
