from __future__ import annotations
import sys
import fenics as fn
from mshr import *
from utils.hippylib import *
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from mpi4py import MPI
# Define auxiliary functions:

def generate_variables(V, bc=None):
    u_trial = fn.TrialFunction(V)
    u_test = fn.TestFunction(V)
    M_form = fn.inner(u_trial, u_test) * fn.dx
    K_form = fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx
    temp_form = fn.inner(fn.Function(V), u_test) * fn.dx
    if bc == None:
        M_matrix = fn.assemble(M_form)
        return u_trial, u_test, M_matrix
    else:
        M_matrix_bc, _ = fn.assemble_system(M_form, temp_form, bc)
        K_matrix_bc, _ = fn.assemble_system(K_form, temp_form, bc)
        return u_trial, u_test, M_matrix_bc, K_matrix_bc

def generate_trial_test(V):
    u_trial = fn.TrialFunction(V)
    u_test = fn.TestFunction(V)
    return u_trial, u_test

def generate_matrices(V, bc=None):
    u_trial, u_test = generate_trial_test(V)
    M_form = fn.inner(u_trial, u_test) * fn.dx
    K_form = fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx
    temp_form = fn.inner(fn.Function(V), u_test) * fn.dx
    if bc == None:
        M_matrix = fn.assemble(M_form)
        K_matrix = fn.assemble(K_form)
        return M_matrix, K_matrix
    else:
        M_matrix_bc, _ = fn.assemble_system(M_form, temp_form, bc)
        K_matrix_bc, _ = fn.assemble_system(K_form, temp_form, bc)
        return M_matrix_bc, K_matrix_bc
def init_vector(matrix, dim):
    # Initialize a vector that is compatible with the matrix:
    # dim = 0: Image, dim = 1: Domain.
    vec = fn.Vector()
    matrix.init_vector(vec, dim)
    return vec

def init_time_vector(matrix, times, dim):
    # Initialize a time-dependent vector that is compatible with the matrix for given times:
    # dim = 0: Image, dim = 1: Domain.
    time_vec = TimeDependentVector(times)
    time_vec.initialize(matrix, dim)
    return time_vec

def vector_like(vector):
    # Initialize a vector that is compatible with the input vector:
    new_vec = fn.Vector(MPI.COMM_WORLD, vector.size())
    return new_vec

def time_vector_like(time_vector):
    # Initialize a time-dependent vector that is compatible with the input time-dependent vector:
    new_time_vec = TimeDependentVector(time_vector.times, size=time_vector.data[0].size())
    return new_time_vec

def get_index(value, array_value, tol=10e-8):
    array_value_temp = array_value - value + 0.01 * (array_value[1] - array_value[0])
    index = np.argmax(array_value_temp > 0)
    assert abs(value - array_value[index]) < tol, "Value not found."
    return index

# def diff_time_vectors(time_vec_1, time_vec_2):
#     # Sum of two time-dependent vectors:
#     diff_temp = time_vec_1.copy()
#     diff_temp.axpy(-1., time_vec_2)
#     return diff_temp

# def sum_time_vectors(time_vec_1, time_vec_2):
#     sum_temp = time_vec_1.copy()
#     sum_temp.axpy(1., time_vec_2)
#     return sum_temp

# Trapezoidal rule for time-dependent vector:
def time_integrate(time_vector: TimeDependentVector, int_time=None):
    times = time_vector.times
    step = times[1] - times[0]
    if int_time is not None:
        start_index = get_index(int_time[0], times)
        end_index = get_index(int_time[-1], times)
    else:
        start_index = 0
        end_index = len(times)

    integrated_vector = init_vector(time_vector.M, 0)
    temp_vector = init_vector(time_vector.M, 0)
    for i in range(start_index, end_index - 1):
        temp_vector.zero()
        temp_vector.axpy(1, time_vector.data[i])
        temp_vector.axpy(1, time_vector.data[i + 1])
        integrated_vector.axpy(0.5 * step, temp_vector)
    return integrated_vector


def integrate_time_vectors(u_time: TimeDependentVector, v_time, B_matrix, observation_times, method='simpson'):
    assert u_time.nsteps == len(observation_times), \
        'Number of time steps must be equal to number of observation times.'
    store_vector = np.ones_like(observation_times)

    if B_matrix is not None:
        for i in range(len(observation_times)):
            store_vector[i] = u_time.data[i].inner(B_matrix * v_time.data[i])
    else:
        for i in range(len(observation_times)):
            store_vector[i] = u_time.data[i].inner(v_time.data[i])

    if method == 'trapezoid':
        # Use trapezoid rule in scipy:
        result = integrate.trapz(store_vector, observation_times)
    elif method == 'simpson':
        # Use Simpson's rule in scipy:
        result = integrate.simps(store_vector, observation_times)
    else:
        raise ValueError('Method not supported.')
    return result

def space_time_mult(time_vector, space_vector, M_matrix, simulation_times):
    result_vector = init_time_vector(M_matrix, simulation_times, 0)
    if isinstance(space_vector, fn.PETScVector) or isinstance(space_vector, fn.Vector):
        # Init a time-dependent vector:
        for i in range(len(simulation_times)):
            result_vector.data[i] = time_vector[i] * space_vector

    elif isinstance(space_vector, TimeDependentVector):
        for i in range(len(simulation_times)):
            result_vector.data[i] = space_vector.data[i] * time_vector[i]

    else:
        raise ValueError('Space vector not supported.')

    return result_vector

# Debugging functions:
def plot_time_vector(time_vector: TimeDependentVector, V):
    for i in range(time_vector.nsteps):
        u_snapshot = fn.Function(V, time_vector.data[i])
        nb.plot(u_snapshot)
        plt.show()

def shift_time_vector(time_vector: TimeDependentVector, k):
    # Shift the time-dependent vector by k steps.
    assert k < time_vector.nsteps, "k is too large."
    for i in range(time_vector.nsteps - k)[::-1]:
        time_vector.data[i + k] = time_vector.data[i]
    for i in range(k):
        time_vector.data[i].zero()
def shape_time_vector(time_vector: TimeDependentVector):
    print('Shape: ', np.shape(time_vector.data[0]))
    print('Times: ', np.shape(time_vector.times))

def mult(M_matrix: fn.Matrix | fn.PETScMatrix, time_vector: TimeDependentVector):
    # Multiply the matrix M_matrix to every frame of the time-dependent vector.
    M_time_vector = init_time_vector(M_matrix, time_vector.times, 0)
    for i in range(time_vector.nsteps):
        M_matrix.mult(time_vector.data[i], M_time_vector.data[i])
    return M_time_vector

def observation_points(V: fn.FunctionSpace, x_coordinates, y_coordinates):
    V_coordinates = V.tabulate_dof_coordinates()
    mask = np.logical_and(np.isin(V_coordinates[:, 0], x_coordinates),
                          np.isin(V_coordinates[:, 1], y_coordinates))
    result = V_coordinates[mask]
    return result

def zero_out(u_vector, tol=1e-12):
    u_array = u_vector[:]
    u_array[np.abs(u_array) < tol] = 0.
    u_vector[:] = u_array[:]

def check(WaveSolverEuler, simulation_times, V):

    _, _, M_matrix = generate_variables(V)
    bc = fn.DirichletBC(V, 0., "on_boundary")

    u_true = init_time_vector(M_matrix, simulation_times, 0)
    f_true = init_time_vector(M_matrix, simulation_times, 0)

    source_true_expr = fn.Expression('sin(pi * x[0]) * sin(pi * x[1])', degree=3)
    source_true = fn.interpolate(source_true_expr, V)
    f_true_time_func = lambda t: np.ones_like(t)
    u_true_time_func = lambda t: np.sin((np.pi * t) / np.sqrt(2)) ** 2 / np.pi ** 2
    f_true_time = f_true_time_func(simulation_times)
    u_true_time = u_true_time_func(simulation_times)
    space_time_mult(u_true_time, source_true.vector(), u_true)
    space_time_mult(f_true_time, source_true.vector(), f_true)

    u_init = fn.Function(V).vector()
    u_dot_init = fn.Function(V).vector()
    data = [u_init.copy(), u_dot_init.copy(), f_true.copy()]
    u_time = WaveSolverEuler(V, simulation_times, data, bc)
    diff = diff_time_vectors(u_time, u_true)
    # result_1 = assemble_time_vectors(u_time, u_time, fn.dx, V)
    # result_2 = assemble_time_vectors(u_true, u_true, fn.dx, V)
    # result = assemble_time_vectors(diff, diff, fn.dx, V)

    return 2


def checkFowardEnd(WaveSolver, V, dt):
    simulation_times = np.arange(0., 1. + 0.5 * dt, dt)
    kappa = 1.
    a = np.pi * kappa
    # Inputs for wave solver: WaveSolverForward(V, simulation_times, data, bc).
    _, _, M_matrix = generate_variables(V)
    # Compute ground truth:
    f_true = init_time_vector(M_matrix, simulation_times, 0)
    source_true_expr = fn.Expression('sin(pi * x[0]) * sin(pi * x[1])', degree=3)
    u_0 = fn.interpolate(source_true_expr, V).vector()
    u_dot_0 = init_vector(M_matrix, 0)
    u_true_time_func = lambda t: np.cos(a * t * np.sqrt(2))
    # u_init = u_0.copy()
    u_init = fn.Function(V).vector()
    u_dot_init = fn.Function(V).vector()

    data = [u_init, u_dot_init, f_true]
    bc = fn.DirichletBC(V, fn.Constant(0.), 'on_boundary')
    u_time = WaveSolver(V, simulation_times, data, bc, kappa)
    u_time_end = u_time[simulation_times[-1]]
    u_true_end = u_true_time_func(1) * u_0
    diff = u_time_end - u_true_end
    result_1 = u_time_end.inner(M_matrix * u_time_end)
    result_2 = u_true_end.inner(M_matrix * u_true_end)
    result = diff.inner(M_matrix * diff)
    return result_1, result_2, result


# Math functions:
def check_multiple(x, a, tol=1e-7):
    # Check if x is a multiple of a.
    return np.abs(x - a * np.round(x / a)) < tol


# Assemble matrices:
# def M(u_trial, u_test):
#     return fn.inner(u_trial, u_test) * fn.dx
#
# def K(u_trial, u_test):
#     return fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx


def get_mesh(rectangle, circle, res=50):
    domain = rectangle
    domain.set_subdomain(1, circle)
    domain.set_subdomain(2, rectangle - circle)
    mesh = generate_mesh(domain, res)
    return mesh


## Integration functions:
def integrate_simp(time_vector, simulation_times):
    dt = simulation_times[1] - simulation_times[0]
    result = time_vector[simulation_times[0]] + time_vector[simulation_times[-1]]
    for i in range(1, time_vector.nsteps -1):
        weight = 2 + 2 * (i % 2)
        result += weight * time_vector[simulation_times[i]]
    result *= dt / 3
    return result

def clone_vec(vector):
    return vector.copy()


def get_observation(mesh, subdomains, boundaries, type):
    # subdomains = fn.MeshFunction("size_t", mesh, 2, mesh.domains())
    # boundaries = fn.MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
    for f in fn.facets(mesh):
        domains = []
        for c in fn.cells(f):
            domains.append(subdomains[c])
        domains = list(set(domains))  # remove duplicates
        if len(domains) > 1:  # it is on the interface
            boundaries[f] = 1
    if type == 'boundary':
        dX = fn.dS(subdomain_data=boundaries)
    if type == 'interior':
        dX = fn.dx(subdomain_data=subdomains)
    return dX

def init_vector_numpy(np_vector):
    vec = fn.Vector(fn.MPI.comm_world, len(np_vector))
    vec.set_local(np_vector)
    return vec

# def comp_covariance(XX, func):
#     # Input a grid off XX and YY, return matrix of values.
#     X_mesh_1, Y_mesh_1 = np.meshgrid(XX[:, 0], XX[:, 0], indexing='xy')
#     X_mesh_2, Y_mesh_2 = np.meshgrid(XX[:, 1], XX[:, 1], indexing='xy')
#     result = func([X_mesh_1, X_mesh_2], [Y_mesh_1, Y_mesh_2])
#     return result
#
# def comp_sqrt(covariance):
#     # Compute the square root of the covariance matrix.
#     # Check if covariance matrix is symmetric:
#     assert np.allclose(covariance, covariance.T), 'Covariance matrix is not symmetric.'
#     eigvals, eigvecs = np.linalg.eigh(covariance)
#
#     # Revert the order of eigenvalues and eigenvectors:
#     sorted_indices = np.argsort(eigvals)[::-1]
#     eigvals = eigvals[sorted_indices]
#     eigvecs = eigvecs[:, sorted_indices]
#
#     # Check if eigenvalues are positive:
#     assert np.all(eigvals > 0), 'Eigenvalues are not positive.'
#     sqrt_eigvals = np.sqrt(eigvals)
#     sqrt_covariance = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T
#     sqrt_covariance = 0.5 * (sqrt_covariance + sqrt_covariance.T)
#     return sqrt_covariance, eigvals, eigvecs

