from __future__ import print_function
from ufl import nabla_grad
from fenics import *
from dolfin import *
from mshr import *

import numpy as np


parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Parameters in maters:
length = 1.8288
width = 0.254
height = 0.4572

youngs = 200e9
nu = 0.3
mu = youngs/(2.0*(1.0+nu))
lmbda = youngs*nu/((1.0+nu)*(1.0-2.0*nu))
rho = 7800

# Assume gravity applied on z-axis:
volume = 0.05215
gravity = -9.8
Force = rho*volume*gravity

# create geometry:
body = Box(Point(0,0,0), Point(1.8288, 0.254, 0.4572))
b1 = Box(Point(0,0,0.0508), Point(1.8288,0.12319,0.4064))
b2 = Box(Point(0,0.254,0.0508), Point(1.8288, 0.13081, 0.4064))

geometry = body-b1-b2


# Create mesh:
mesh = generate_mesh(geometry, 64)		# Meshpoint: 64
meshfile = XDMFFile("meshfile.xdmf")
meshfile.write(mesh)				# Check meshfile

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
f = Constant((0.,0.,Force))
traction = 581251.1625
T = Expression(('0.0' ,' near(x[2],height) && x[0] >= 0.97222222*length && x[0] <= length ? -traction : 0.0 ' , '0.0'), degree=1, height=height, length=length, traction=traction)


u_st = TrialFunction(V)
du_st = TestFunction(V)

a = inner(sigma(u_st),eps(du_st))*dx
L = dot(f,du_st)*dx + dot (T,du_st)*ds

u_st = Function(V)
solve(a == L, u_st, bc_left)


# Output files
xdmf_disp = XDMFFile("Displacement.xdmf")
xdmf_disp.write(u)
