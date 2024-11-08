import numpy as np
from scipy.linalg import sqrtm, eigh
import fenics as fn
from scipy.linalg import solve_triangular
from utils import *
from utils.hippylib import *


def comp_covariance(XX, func):
    # Input a grid off XX and YY, return matrix of values.
    X_mesh_1, Y_mesh_1 = np.meshgrid(XX[:, 0], XX[:, 0], indexing='xy')
    X_mesh_2, Y_mesh_2 = np.meshgrid(XX[:, 1], XX[:, 1], indexing='xy')
    result = func([X_mesh_1, X_mesh_2], [Y_mesh_1, Y_mesh_2])
    return result

def comp_sqrt(covariance):
    # Compute the square root of the covariance matrix.
    # Check if covariance matrix is symmetric:
    assert np.allclose(covariance, covariance.T), 'Covariance matrix is not symmetric.'
    eigvals, eigvecs = eigh(covariance)

    # Revert the order of eigenvalues and eigenvectors:
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]

    # Check if eigenvalues are positive:
    assert np.all(eigvals > 0), 'Eigenvalues are not positive.'
    sqrt_eigvals = np.sqrt(eigvals)
    sqrt_covariance = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
    sqrt_covariance = 0.5 * (sqrt_covariance + sqrt_covariance.T)
    return sqrt_covariance, eigvals, eigvecs

def comp_mass(V):
    u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)
    M_matrix = fn.assemble(u_trial * u_test * fn.dx)
    M_array = M_matrix.array()
    sqrt_M, eigvals, eigvecs = comp_sqrt(M_array)
    assert np.all(eigvals > 0)
    sqrt_inv_eigvals = 1/np.sqrt(eigvals)
    sqrt_M_inv = eigvecs @ np.diag(sqrt_inv_eigvals) @ eigvecs.T
    return M_array, sqrt_M, sqrt_M_inv

class GaussianPrior(Prior):

    def __init__(self, V, covariance, mean=None):

        self.V = V
        self.covariance = covariance
        u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)
        self.M = fn.assemble(fn.inner(u_trial, u_test) * fn.dx)

        if mean:
            self.mean = mean
        else:
            self.mean = init_vector(self.M, 0)

        # Cholesky decomposition:

        self.chol = np.linalg.cholesky(self.covariance)
        self.chol_inv = solve_triangular(self.chol, np.identity(V.dim()), lower=True)
        precision = np.dot(self.chol_inv.T, self.chol_inv)
        self.precision = (precision + precision.T) / 2

        self.M_precision = self.precision @ self.M.array()

        self.R = Array2Matrix(self.M_precision, self.M.mpi_comm())
        self.Rsolver = Matrix2Solver(self.covariance, self.M, self.M.mpi_comm())

        u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)
        self.M = fn.assemble(fn.inner(u_trial, u_test) * fn.dx)
        self.Msolver = PETScKrylovSolver(self.V.mesh().mpi_comm(), "cg", "jacobi")
        self.Msolver.set_operator(self.M)

    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)


