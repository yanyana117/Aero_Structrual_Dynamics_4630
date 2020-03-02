from __future__ import print_function
from fenics import *
from ufl import nabla_div
import math

# 3b_Aluminum
length = 50 * 0.001     # 0.05
width = 5 * 0.001       # 0.005
height = 5 * 0.001      # 0.005

mesh = BoxMesh(Point(0.,0.,0.), Point(length,width,height), 20, 6, 6)
U = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 1)
W = TensorFunctionSpace(mesh, 'P', 1)

# defining properties of steel, (b) for Aluminum & Cppper
# Material parameters of the sample Aluminum
youngs = 69.e11           # youngs modulus
nu = 0.334                # poisson ratio
rho = 2710            # km/m^3

_lambda = youngs * nu/((1.+nu)*(1.-2.*nu))
mu = youngs/(2.0*(1.0+nu))


def boundary_left(x,on_boundary):
    return on_boundary and near(x[0],0.0)


bc_left = DirichletBC(V,Constant((0.,0.,0.)), boundary_left)


def epsilon(u):
    return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def sigma(u):
    eps = epsilon(u)
    return _lambda*tr(eps)*Identity(3) + mu*(eps + eps.T)

#==========================================================#
# Static problem
#  d/dx (EA du/dx) + f = 0
#==========================================================

f = Constant((0.0,0.0,0.0))

force_applied = 1.0e3
# External tractions (force / area)
traction = 20000.0         # N/m^2

T = Expression(('near(x[0],length) ? force_applied/(width*height) : 0.0 ', '0.0', '0.0'),
    degree=1 , length=length , width=width, height=height, force_applied=force_applied)

T = Expression(('0.0' ,' near(x[1],width) && x[0] >= 0.9*length && x[0] <= length ?  - traction : 0.0 ' , '0.0'), degree=1,length=length,width=width, traction=traction)


u_init = TrialFunction(V)
v = TestFunction(V)

a = inner(sigma(u_init),epsilon(v))*dx  # Variational problem (Energy balance)
L = dot(f,v)*dx + dot(T,v)*ds
u_init = Function(V)
solve(a==L, u_init, bc_left)

#==========================================================
#  Dynamic problem
#  d/dx (EA du/dx) + f = m d^2/dt^2 u
#==========================================================

t = 0
t_final = 3e-3  # 1x10^-3
# copper t_final = 4e-3
dt = 1.7e-7   	   # 5x10^-5

# dt on notes Feb.19th mesh size, goemeter

T = Constant((0.0,0.0,0.0))
u = TrialFunction(V)
v = TestFunction(V)
u_n_1 = Function(V) # u(x,t-dt)
u_n_2 = Function(V) # u(x,t-2dt)

#=========================================================
# -EA \int du^n/dx dv/dx dx + \int f v dx =  m/(dt^2) \int u^n v dx - 2m/(dt^2) \int u^(n-1) v dx + m/(dt^2) \int u^(n-2) v dx
# -EA (dt^2) \int du^n/dx dv/dx dx + (dt^2) \int f v dx =  m \int u^n v dx - 2m \int u^(n-1) v dx + m \int u^(n-2) v dx
# -EA (dt^2) \int du^n/dx dv/dx dx + (dt^2) \int f v dx  - m \int u^n v dx + 2m \int u^(n-1) v dx- m \int u^(n-2) v dx = 0
# u => displacement
# du/dx => strain
#  E du/dx => stress
#  A*dx = dV (volume integral)
#=========================================================

# variational form
F = -(dt*dt)*inner(sigma(u),epsilon(v))*dx + (dt*dt)*dot(f,v)*dx - rho*dot(u,v)*dx + 2.0*rho*dot(u_n_1,v)*dx - rho*dot(u_n_2,v)*dx + (dt*dt)*dot(T,v)*ds
# dx => integrate over volume, ds => integrate over surface

a, L = lhs(F), rhs(F)   # a = lhs(F) # L = rhs(F)

u = Function(V)
u_n_1.assign(u_init)    # u^(n-1) = u_init (initial displacement is given)
u_n_2.assign(u_init)    # u^(n-2) = u_init (initial velocity is zero)


#=======================================================
# Next, we set out output files
xdmffile_disp = XDMFFile('deflection.xdmf')
# We are setting some parameters to speed up the code
xdmffile_disp.parameters['rewrite_function_mesh'] = False
xdmffile_disp.parameters['flush_output'] = True
#=========================================================

while t <= t_final:
    print('t = %f'%t)
    solve(a==L, u, bc_left)

    # write some output
    xdmffile_disp.write(u,t)
    # xdmffile_disp.write(u.sub(1),t)
    # add in 3c: Accounnece
    # time marching
    u_n_2.assign(u_n_1)     # u_n_2 = u_n_1
    u_n_1.assign(u)         # u_n_1 = u_n
    # increment time

    t = t + dt
print('And we are done!')
