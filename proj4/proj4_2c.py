from __future__ import print_function
from fenics import *
from dolfin import *
from mshr import *
from ufl import nabla_grad

import numpy as np
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

# Initial condition: L,W,H units in maters
length = 1.8288
width = 0.254
height = 0.4572

youngs = 200e9
nu = 0.3
mu = youngs/(2.0*(1.0 + nu))
lmbda = youngs*nu/((1.0 + nu)*(1.0 - 2.0*nu))
rho = 7800

gravity = -9.8
volume = 0.0288585
Froce = rho * volume * gravity

# create geometry
body = Box(Point(0,0,0), Point(1.8288, 0.254, 0.4572))
b1 = Box(Point(0,0,0.0254), Point(1.8288,0.12319,0.4318))
b2 = Box(Point(0,0.254,0.0254), Point(1.8288, 0.13081, 0.4318))

# Cylinder
c1 = Cylinder(Point(0.1524,0.12319,0.2286), Point(0.1524,0.13081,0.2286), 0.0635, 0.0635)
c2 = Cylinder(Point(0.5334,0.12319,0.2286), Point(0.5334,0.13081,0.2286), 0.0635, 0.0635)
c3 = Cylinder(Point(0.9144,0.12319,0.2286), Point(0.9144,0.13081,0.2286), 0.0635, 0.0635)
c4 = Cylinder(Point(1.2954,0.12319,0.2286), Point(1.2954,0.13081,0.2286), 0.0635, 0.0635)
c5 = Cylinder(Point(1.6764,0.12319,0.2286), Point(1.6764,0.13081,0.2286), 0.0635, 0.0635)

geometry = body-b1-b2-c1-c2-c3-c4-c5

# Create mesh
mesh = generate_mesh(geometry, 64)
meshfile = XDMFFile("meshfile.xdmf")
meshfile.write(mesh)

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
f = Constant((0.,0.,Froce))
traction = 581251.1625			# N/m

# Assume gravity toward z-axis
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
