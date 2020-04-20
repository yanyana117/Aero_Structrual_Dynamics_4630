from fenics import *
import numpy as np
from ufl import nabla_grad

parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Initinal condition: L,W,H units in maters
# L = 50mm, W = 5mm and H = 5mm.

length = 0.05
width = 0.005
height =0.005

youngs = 200e9
nu = 0.3
mu = youngs/(2.0*(1.0+nu))
lmbda = youngs*nu/((1.0+nu)*(1.0-2.0*nu))
rho = 7800

mesh = BoxMesh(Point(0.,0.,0.), Point(length,width,height),20,6,6)
U = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh, 'P', 1)
W = TensorFunctionSpace(mesh, 'DG', 0)

def left(x,on_boundary):
	return near(x[0],0.0) and on_boundary

def right(x,on_boundary):
	return near(x[0],length) and on_boundary

bc_left = DirichletBC(V,Constant((0.,0.,0.)), left)

def eps(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)

def sigma(u):
	epsilon = eps(u)
	return lmbda*tr(epsilon)*Identity(3) + mu*(epsilon + epsilon.T)

#==========================
# Static problem
#==========================
f = Constant((0.,0.,0.))

traction = -20000
T = Expression(('0.0' ,' near(x[1],width) && x[0] >= 0.95*length && x[0] <= length ? -traction : 0.0 ' , '0.0'), degree=1, width=width, length=length, traction=traction)

u_st = TrialFunction(V)
du_st = TestFunction(V)

a = inner(sigma(u_st),eps(du_st))*dx
L = dot(f,du_st)*dx + dot (T,du_st)*ds

u_st = Function(V)
solve(a == L, u_st, bc_left)

#==============================
# Dynamic problem
#==============================
alpha_m = Constant(0.2)
alpha_f = Constant(0.4)
gamma   = Constant(0.5 + alpha_f - alpha_m)
beta    = Constant(0.25*((gamma + 0.5)**2))

f = Constant((0.,0.,0.))
T = Constant((0.,0.,0.))

t_final = 1.0
Nsteps  = 1000
dt = Constant(t_final/Nsteps)

u_old = Function(V)
v_old = Function(V)
a_old = Function(V)

u_old.assign(u_st)
v_old.assign(Constant((0.,0.,0.)))
a_old.assign(Constant((0.,0.,0.)))

u = Function(V, name="Displacement")

# Rayleigh damping
eta_m = 0.0
eta_k = 0.0

def m(u, du):
	return rho*inner(u,du)*dx

def k(u, du):
	return inner(sigma(u), eps(du))*dx

def c(u, du):
	return eta_m*m(u,du) + eta_k*k(u,du)

def L(du):
	return rho*inner(f,du)*dx + inner(T,du)*ds

# Updating velocities and accelerations
def update_a (u, u_old, v_old, a_old, ufl=True):
	if ufl:
		dt_ = dt
		beta_ = beta
	else:
		dt_ = float(dt)
		beta_ = float(beta)
	return (u - u_old - dt_*v_old)/(beta_*dt_*dt_) - (1.0 - 2.0*beta_)*a_old/(2.0*beta_)

def update_v (a, u_old, v_old, a_old, ufl=True):
	if ufl:
		dt_ = dt
		gamma_ = gamma
	else:
		dt_ = float(dt)
		gamma_ = float(gamma)
	return v_old + dt_*((1.0-gamma_)*a_old + gamma_*a)

def update_fields(u, u_old, v_old, a_old):
	u_vec = u.vector()
	u0_vec = u_old.vector()
	v0_vec = v_old.vector()
	a0_vec = a_old.vector()

	a_vec = update_a(u_vec,u0_vec,v0_vec,a0_vec,ufl=False)
	v_vec = update_v(a_vec,u0_vec,v0_vec,a0_vec,ufl=False)

	a_old.vector()[:] = a_vec
	v_old.vector()[:] = v_vec
	u_old.vector()[:] = u.vector()

def avg(x_old, x_new, alpha):
	return alpha*x_old + (1.0-alpha)*x_new

# Variational form
du = TrialFunction(V)
u_ = TestFunction(V)

a_new = update_a(du, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

F_form = m(avg(a_old,a_new,alpha_m),u_) + c(avg(v_old,v_new,alpha_f),u_) + k(avg(u_old,du,alpha_f),u_) - L(u_)

a_form, L_form = lhs(F_form), rhs(F_form)

K, res = assemble_system(a_form, L_form, bc_left)
solver = LUSolver(K,"mumps")
solver.parameters["symmetric"] = True

# Output files
xdmf_disp = XDMFFile("Displacement.xdmf")
#xdmf_disp.parameters["flush_output"] = True


# time marching
t = 0
time = np.linspace(0, t_final, Nsteps+1)

for (i,dt) in enumerate(np.diff(time)):
	t = time[i+1]
	print ("Time = ",t)

	res = assemble(L_form)
	bc_left.apply(res)
	solver.solve(K,u.vector(),res)

	update_fields(u, u_old, v_old, a_old)

	xdmf_disp.write(u,t)
	print('And we are done!')
