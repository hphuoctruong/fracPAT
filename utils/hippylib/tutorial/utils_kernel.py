import numpy as np

# Kernel classes:
class GaussianKernel:
    def __init__(self, sigma):
        self.sigma = sigma
        self.value = lambda x, y: np.exp(-((x[0] - y[0]) ** 2 + (x[1] - y[1]) ** 2) / (2 * self.sigma ** 2))
        self.grad  = lambda x, y: (1 / self.sigma ** 2 * np.block([x[0] - y[0], x[1] - y[1]]) * np.block([self.value(x, y), self.value(x, y)]))

# Kernel function:
def K_matrix_2D(kernel, X, Y, indexing='xy', reshape=True):
    value, grad = kernel.value, kernel.grad
    # Input: A kernel k and list of coordinates V.
    # Output: K: K_ij = k(X_i, Y_j), grad_K: grad_K_ij = grad_X k(X_i, Y_j).
    X_mesh_1, Y_mesh_1 = np.meshgrid(X[:, 0], Y[:, 0], indexing=indexing)
    X_mesh_2, Y_mesh_2 = np.meshgrid(X[:, 1], Y[:, 1], indexing=indexing)
    # Compute function:
    value_F = value([X_mesh_1, X_mesh_2], [Y_mesh_1, Y_mesh_2])
    # Compute gradient:
    grad_F = grad([X_mesh_1, X_mesh_2], [Y_mesh_1, Y_mesh_2])
    if reshape:
        value_F = value_F.T.reshape(X.shape[0], -1, order='F')
        grad_F = grad_F.T.reshape(X.shape[0], -1, order='F')
        result = [value_F, grad_F]
    else:
        result = [value_F.T, grad_F.T]
    return result
