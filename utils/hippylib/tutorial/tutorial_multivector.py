import fenics as fn
import ufl
import math
import numpy as np
import matplotlib.pyplot as plt
from utils.hippylib import *
from scipy.linalg import inv, eigh

mesh = fn.UnitSquareMesh(60, 60)
V = fn.FunctionSpace(mesh, "Lagrange", 1)
fn.plot(mesh)
plt.show()
print("Number of dofs: {0}".format(V.dim()))
u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)
M = fn.assemble(u_trial*u_test*fn.dx)
M_matrix = M.array()
print(M_matrix.shape)


