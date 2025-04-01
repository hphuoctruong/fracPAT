import sys
sys.path.append('..')
from WaveNewmarkFractional import *
from scipy.integrate import simps
from scipy.linalg import eigh, solve_triangular, inv
import pickle
import gc

from utils import *
import numpy as np
from scipy.optimize import minimize

# Define mesh:
mesh = fn.RectangleMesh(fn.Point(-1.5, -1.5), fn.Point(1.5, 1.5), 60, 60)
V = fn.FunctionSpace(mesh, 'CG', 1)
u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)
M_matrix = fn.assemble(u_trial * u_test * fn.dx)
P_matrix = fn.assemble(u_trial * u_test * fn.dP)
P_solver = Operator2Solver(P_matrix)

# Define boundary observation:
class Boundary(fn.SubDomain):
    def inside(self, x, on_boundary):
        left = x[0] < 0.6 + fn.DOLFIN_EPS and x[0] > -0.6 - fn.DOLFIN_EPS
        right = x[1] < 0.6 + fn.DOLFIN_EPS and x[1] > -0.6 - fn.DOLFIN_EPS
        return left and right

# Create a MeshFunction to mark boundaries
subdomains = fn.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
boundary = Boundary()
boundary.mark(subdomains, 1)  # Mark the left subdomain with the value 1

# Create a measure for the boundary
dx = fn.Measure('dx', domain=mesh, subdomain_data=subdomains)
ds = comp_dS(mesh, subdomains)
M_matrix_sub = fn.assemble(u_trial * u_test * dx(1))

# Define setting of the problem:
c = 15.
# Define simulation times:
T = 0.1
dt = 0.0005
simulation_times = np.arange(0., T + 0.5 * dt, dt)

# Define the ground truth:
class RectangleExpression(fn.UserExpression):
    def eval(self, value, x):
        # Define the boundaries of the smaller L-shape within the square [-0.8, 0.8] x [-0.8, 0.8]
        if x[0] >= -0.5 - fn.DOLFIN_EPS and x[0] <= 0.25 + fn.DOLFIN_EPS and x[1] >= -0.4 - fn.DOLFIN_EPS and x[1] <= 0.125 + fn.DOLFIN_EPS:
            value[0] = 0.03  # Value inside the smaller L-shape
        # Define a circle with radius 0.5 and center (0.1, -0.1)
        # elif (x[0] - 0.25)**2 + (x[1] + 0.125)**2 <= 0.3**2:
        #     value[0] = 3.0  # Value inside the circle
        else:
            value[0] = 0.0  # Value outside the smaller L-shape

    def value_shape(self):
        return ()

ic_expr = RectangleExpression(element=V.ufl_element())
u_init = fn.interpolate(ic_expr, V).vector()

# Define boundary condition:
bc = fn.DirichletBC(V, 0., "on_boundary")
bc.apply(u_init)


