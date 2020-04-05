from dolfin import*
from mshr import*


# Geometry
top = Box(Point(0.,0.,80), Point(45,155,88))
leg1 = Box(Point(0.,150,0.), Point(5, 155,80))
leg2 = Box(Point(40,150,0.), Point(45,155,80))
leg3 = Box(Point(0.,0.,80), Point(45,5,80))
c = Cylinder(Point(40,0.,0.), Point(15.,15.,80),3,3)

geometry = top + leg1 + leg2 + leg3 + leg4 - c
mesh = generate_mesh(geometry,64)

meshfile=XDMFFile("meshfile.xdmf")
meshfile.write(mesh)
