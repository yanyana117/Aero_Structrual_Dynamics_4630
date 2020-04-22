# Importing modules needed for the simulation
from __future__ import print_function
from fenics import *
from ufl import nabla_div
import numpy as np 	# importing numpy
import matplotlib.pylab as plt 	# Import pylab
from dolfin import*
from mshr import*


# pip3 install matplotlib
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

youngs = 72e9
nu = 0.33
mu = youngs/(2.0*(1.0+nu))
height = 103
lmbda = youngs*nu/((1.0+nu)*(1.0-2.0*nu))
rho = 2810

# Creat 3D geometry unit in maters
c1 = Cylinder(dolfin.Point(0,0,0), dolfin.Point(0,0,100), 10,10)
c2 = Cylinder(dolfin.Point(0,0,0), dolfin.Point(0,0,100), 11,11)
cone = Cylinder(dolfin.Point(0, 0, 100),dolfin.Point(0, 0, 103), 11, 0)

geometry = c2-c1+cone
mesh = generate_mesh(geometry, 64)
meshfile = XDMFFile("meshfile.xdmf")
meshfile.write(mesh)

U = FunctionSpace(mesh, 'P', 1)
V = VectorFunctionSpace(mesh,'P',1)
W = TensorFunctionSpace(mesh, 'DG', 0)
#=====================================================================

def clamped_boundary(x, on_boundary):
	return on_boundary and near(x[2],0.)


bc_left = DirichletBC(V, Constant((0,0,0)), clamped_boundary)

def epsilon(u):
	return 0.5*(nabla_grad(u) + nabla_grad(u).T)


def sigma(u):
	eps = epsilon(u)
	return lambda_*tr(eps)*Identity(d) + mu_*(eps + eps.T)

u = TrialFunction(V)
d = u.geometric_dimension()
v = TestFunction(V)
#=====================================================================

# Here we have ignored all gravity.
f = Constant((0,0,0))

# If you want to include gravity, uncomment these lines
gravity = -9.8	# negative for -z direction.
f = Constant((0,0,gravity))

# Initinal condition:
v_ref = 8048.72 # 5 miles/hr to mater/hr
z_ref = 3.048 # 10 ft to mater
z0 = 0.0024 # m

###########
def v_wind(z,z_ref,z0):
	z = np.linspace(1,103,500)
	vw = np.log(z/z0) / np.log(z_ref / z0)
	return vw


def drag_traction(Cd,rho,vw):
	
	td = 1/2 * Cd * rho * vw**2
	return td 


zzz = np.linspace(1,103,500)
vw = v_wind(zzz,z_ref,z0)
traction_applied = drag_traction(0.5, 1.2, vw)

###############################
# T = Expression(('x[2]>=0*height && x[2]<=0.97*height ? traction: 0.0','0.0','0.0'), traction=traction, height=height, degree = 2)

T = Expression(( 'x[2]>=0*height && x[2]<= 0.97*height ? traction_applied : 0.0', '0.0','0.0'), traction=traction_applied, height=height, degree = 1)
# Note on Expressions:
# 	Expressions is a  very powerful tool. It is primarily used to define
#	complex conditions in a statement using C++ like syntax.
#
# Let's analyze the above defiition of T.
# The definition of T looks like Expression( (expression) , arguments, degree = <>)
# 	1. (expression): Here it is 3D. It has '0.0', '....', '0.0'
#			Think of it as x,y,z components of T.
#			It says that x and y component of T is 0. 
#			T[0] = 0.0, T[1] = 0.0
#			The z component T[2] is more complicated. 
# 			Let's unpack that.
#
#			'near(x[0],L) && near(x[2],H) ? force : 0.0'
#			a. < condition > ? < outcome 1 > : < outcome 2 >
#				is a conditional statement that says 
#				if condition == true, then execute outcome 1
#				else if condition == false, then execute outcome 2.
#			b. near(x[0],L) && near(x[2],H)
#				is x[0] near L AND x[2] near H ?
#			c. if above evaluates to yes, then T[2] = force, else T[2] = 0.0
#
#			This basically tells you that if the location is at
#			x = L and z = H, then apply a force. Otherwise don't.
#			Note: there is no condition on y. So the code will apply
#			force on the entire line (y = 0) to (y = W)
#
#	2. arguments: It defines the variables you used in (expression).
#			Here, we used length, height and force in the (expression).
#			We need to communicate to the program what exactly is
#			length, height and force.
#	3. degree: Specfies the degree of the expression. Here the (expression)
#			was a simple check.. and force was a constant. So the degree should
#			be zero. Here we set degree =1 just to be safe.
#
# Let's define a different expression for T.
# This time we are applying a Gaussian profile (or a bell curve) of force
# along on the length and width of the beam.
# The peak of the bell curve is set at x[0] = L.
# The spread of the bell curve is set to L/100.
# 
# Uncomment the lines below to enable that force function
# spread = length/100.0
# peak = length
# T = Expression(('0.0','0.0','near(x[2],height) ? (1.0/(spread*sqrt(2.0*pi)))*exp(-pow((x[0]-peak)/spread,2)/2.0) : 0.0'), height=height, spread=spread, peak=peak, degree=2)
#=====================================================================

#=====================================================================
# Next, we set out output files
xdmffile_disp = XDMFFile('deflection.xdmf')
xdmffile_stress = XDMFFile('stress.xdmf')
xdmffile_vonmises = XDMFFile('vonMises.xdmf')
#=====================================================================

#=====================================================================
# This sets up the variational problem. For now, you can leave it as is.
a = inner(sigma(u),epsilon(v))*dx
L = dot(f,v)*dx + dot(T,v)*ds
#=====================================================================

#=====================================================================
# Finally we solve the problem. 
# First we define that we expect. We expect a vector output. 
u = Function(V)

# Next we actually solve the problem and specify the dirichlet BC we 
# defined earlier. We also tell the program that we are solving for u.
# The program solves the equation a - L = 0
# Don't worry about solver_parameters for now
solve(a==L, u, bc_left)#,solver_parameters={"linear_solver": "gmres"})
#=====================================================================

#=====================================================================
# Use numpy and pylab to generate plots.
# Midline : x=0, y=width/2, z=height/2
#	    x=length, y=width/2, z=height/2
point_x = np.linspace(0., length, 51)	# returns an array with 51 equally spaced points.
#point_x = np.arange(0., length, length/100.0) # returns an array with spacing of length/100.0

# u_y = u.sub(1) is the y component of the solution u
# u_y(x,y,z) where x is in point_x, y = width/2, z = height/2

disp_y = np.array([u.sub(2)(point, width/2, height/2) for point in point_x])
# point is an iterator... it iterates over point_x array
# for every value, I compute u.sub(1) at (point, width/2, height/2)
# then I generate an array using numpy array

print('Z-displacement at (length, width/2, height/2) = ',u.sub(2)(length,width/2,height/2))

# Now I generate a plot using matplotlib.pylab as plt
plt.plot(point_x, disp_y)
plt.grid(True)
plt.xlabel('x (m)')
plt.ylabel('Z-displacement (m)')
plt.savefig('DispZMidLine.png')

np.savetxt("DispZMidLine.txt", np.transpose([point_x, disp_y]), delimiter=',')

# plt.triplot()
#=====================================================================

#=====================================================================
# We need to write the output to the file for visualization.
# First, let's write the displacement.
#xdmffile_disp.write(u)

# Next, let's compute the stress based on this displacement
# stress =  lambda_*nabla_div(u)*Identity(d) + mu_*(epsilon(u) + epsilon(u).T)
#xdmffile_stress.write(project(stress,W))#,solver_type='gmres'))

# Finally let's compute the von-mises stress.
#deviatoric_stress = stress - (1./3)*tr(stress)*Identity(d) # deviatoric stress
#von_Mises = sqrt(3./2*inner(deviatoric_stress, deviatoric_stress))
#xdmffile_vonmises.write(project(von_Mises,U))#,solver_type='gmres'))
#=====================================================================


# And we are done!
# It sounds like a lot of works, but in reality, it is not.
# For a different problem, you just have to change a few things.
# Most of the time, the template would be the same.
#
# Welcome to the world of computing. This is so much more fun that
# clicking options and buttons on a software.
# Happy coding!
#=====================================================================
