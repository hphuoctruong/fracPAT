import numpy as np
import fenics as fn
import scipy.linalg as la


class Array2Matrix:
    def __init__(self, array, mpi):
        self.array = array
        self.mpi = mpi

    def mult(self, x, y):
        y.zero()
        y[:] = self.array @ x[:]

    def solve(self, x, y):
        x[:] = la.solve(self.array, y[:])

    def init_vector(self, x: fn.PETScVector, dim):
        x.init(self.array.shape[dim])

    def mpi_comm(self):
        return self.mpi

class Matrix2Solver:
    def __init__(self, array, M, mpi):
        self.array = array
        self.M = M
        self.mpi = mpi

    def solve(self, x, y):
        x.zero()
        temp = self.array @ y[:]
        M_array = self.M.array()
        result = la.solve(M_array, temp)
        x[:] = result[:]

    def init_vector(self, x, dim):
        self.M.init_vector(x, dim)

    def mpi_comm(self):
        return self.mpi


class ProductMatrix:

    # Return A @ B
    def __init__(self, A: np.array, B, init_vector=None):
        self.A = A
        self.B = B

        if init_vector is None:
            if hasattr(self.A, "init_vector"):
                self.init_vector = self.A.init_vector
            elif hasattr(self.B, "init_vector"):
                self.init_vector = self.B.init_vector
            else:
                raise NotImplementedError("init_vector")
        else:
            self.init_vector = init_vector

        # Cholesky factorization of A:
        self.chol = la.cholesky(A, lower=True)
        self.chol_inv = la.solve_triangular(self.chol, np.eye(A.shape[0]), lower=True)
        self.A_inv = self.chol_inv.T @ self.chol_inv

    def mult(self, x, y):
        Bx = self.B * x
        y[:] = np.dot(self.A, Bx[:])

    def init_vector(self, x, dim):
        self.init_vector(x, dim)

    def inner(self, x, y):
        Cy = fn.Vector(self.B.mpi_comm())
        self.init_vector(Cy, 0)
        self.mult(y, Cy)
        return x.inner(Cy)

    def solve(self, x , y):
        # Solve ABx = y
        # Assume that A is invertible, we solve Bx = A^-1 y.
        A_inv_y = fn.Vector(self.B.mpi_comm())
        self.init_vector(A_inv_y, 0)
        A_inv_y[:] = self.A_inv @ y[:]
        fn.solve(self.B, x, A_inv_y)

class J_action:
    """
    Action of BA^-1C where A is positive definite.
    """

    def __init__(self, B, Asolver, C):
        self.Asolver = Asolver
        self.B = B
        self.C = C
        self.temp0 = fn.Vector(self.C.mpi_comm())
        self.temp1 = fn.Vector(self.C.mpi_comm())
        self.temp1_help = fn.Vector(self.C.mpi_comm())
        self.B.init_vector(self.temp0, 0)
        self.B.init_vector(self.temp1, 1)
        self.B.init_vector(self.temp1_help, 1)

    def init_vector(self, x, dim):
        if dim == 1:
            self.C.init_vector(x, 1)
        elif dim == 0:
            self.B.init_vector(x, 0)

    def mpi_comm(self):
        return self.B.mpi_comm()

    def mult(self, x, y):
        self.C.mult(x, self.temp1)
        self.Asolver.solve(self.temp1_help, self.temp1)
        self.B.mult(self.temp1_help, y)

    def transpmult(self, x, y):
        self.B.transpmult(x, self.temp1)
        self.Asolver.solve(self.temp1_help, self.temp1)
        self.C.transpmult(self.temp1_help, y)

