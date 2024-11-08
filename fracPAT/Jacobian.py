from WaveNewmarkFractional import *
from utils import *
from scipy.special import gamma as Gamma
from WaveNewmarkFractional import sum_u
from WaveSolver import *
class Hessian_PAT:
    def __init__(self, model: WaveInverseDamped, design):
        self.model = model
        self.design = design
        self.variance = model.misfit.noise_variance

    def init_vector(self, x, dim):
        self.model.init_parameter(x)

    def mult(self, x, Hx):
        Hx.zero()
        X = self.model.generate_vector()
        X[PARAMETER][:] = x[:]
        obs = self.model.computeFwd(X, self.design)
        X_adj = self.model.computeAdj(obs, self.design)
        MX_adj = self.model.M_matrix * X_adj / self.variance
        Hx[:] = MX_adj[:]

    def inner(self, x, y):
        Hy = self.model.generate_vector(PARAMETER)
        Hy.zero()
        self.mult(y, Hy)
        return x.inner(Hy)


class Jacobian_PAT:
    def __init__(self, model, design):
        """
        Construct the Jacobian Operator
        """
        self.model = model
        self.design = design
        self.n_calls = 0

        self.RHS = model.generate_vector(STATE)
        self.u = model.generate_vector(STATE)
        self.p = model.generate_vector(ADJOINT)
        self.y_help = model.generate_vector(PARAMETER)
        self.M_matrix = self.model.M_matrix

        # Get RHS:
        X_zero = self.model.generate_vector()
        self.M_b = self.model.generate_vector(PARAMETER)
        self.model.solveAdj(X_zero[ADJOINT], X_zero)
        self.model.evalGradientParameter(X_zero, self.M_b, design)
        self.M_b *= -1

    def init_vector(self, x, dim):
        self.model.init_parameter(x)

    def mult(self, x, Jx):
        """
        Apply the Jacobian operator to the vector :code:`x`. Return the result in :code:`Jx`.
        """
        Jx.zero()
        x_state = self.model.generate_vector(STATE)
        x_adj = self.model.generate_vector(ADJOINT)
        X = [x_state, x, x_adj]
        self.model.solveFwd(x_state, X, self.design)
        self.model.solveAdj(x_adj, X)
        adj = self.model.computeAdj_old(X, self.design)
        M_adj = self.M_matrix * adj
        Jx[:] = M_adj[:]
        Jx.axpy(1., self.M_b)

        # Regularization term:
        self.model.applyR(x, self.y_help)
        Jx.axpy(1., self.y_help)

    def inner(self, x, y):
        """
        Return the inner product of x and y.
        """
        Jy = self.model.generate_vector(PARAMETER)
        Jy.zero()
        self.mult(y, Jy)
        return x.inner(Jy)


# Add some text here: