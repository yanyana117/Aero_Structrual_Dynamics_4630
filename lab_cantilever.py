from _future_ import print function
from fenics im[ort*
from ufl import nabla_div

# step 1: define geometry of the problem
length = 1.0
width = 0.1
height = 0.1

#step 2 : define material parameters
youngs = 2e11 # 200 GPA
nu = 0.3 # poisson ratio

lambda = youngs*nu/((1.+nu)*1.-2.*nu)
mu = youngs/(2.*(1.+nu))

# step 3: define your mesh
mesh = BoxMesh(Point(0.,0.,0.),Point(;ength,width,height,20,6,6))

# step 4: FUnction spaces
U = FunctionSpace(mesh,'P',1)
W = TensorFunctionSPace(,esh,'P',1)

# Boundary conditions
def doundary_left(x,on_boundary):
  return pm_boundary and near(x[0],0.0)
  
  disp_value = Constant((0.,0.,0.))
  
  bc_left = DirichiletBC(V,disp_value,boundary_left)
  
  # stresses and strains
  def epsilon(u):
    return 0.5*(nabla_grad(u)+nabla_grad(u).T)
    
   def sigma(u):
    eps = epsilon(u)
    return_lambda*tr(eps)*Identity(3)+mu*(eps+eps.T)
    
    
    
