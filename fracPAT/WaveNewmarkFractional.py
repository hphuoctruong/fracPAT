from utils.functions import *
from scipy.special import gamma as Gamma
import fenics as fn

# Define forms:
def M(u_trial, u_test):
    return fn.inner(u_trial, u_test) * fn.dx
def M_c(u_trial, u_test, c):
    return fn.inner(c ** (-2) * u_trial, u_test) * fn.dx

def K(u_trial, u_test):
    return fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx

def C(u_trial, u_test, c):
    return fn.inner(c * fn.grad(u_trial), fn.grad(u_test)) * fn.dx

# Define predictors and correctors:
def pred(u, u_dot, u_ddot, dt, gamma=0.5, beta=0.25):
    # u_(n + 1)_hat = u_n + (dt) * u_dot_n + (1 - 2 * beta) * (dt)^2/2 * u_ddot_n.
    # u_dot_(n+1)_hat = u_dot_n + (dt) * (1 - gamma) * u_ddot_n.
    u_hat = u + dt * u_dot + (dt ** 2 / 2) * (1 - 2 * beta) * u_ddot
    u_dot_hat = u_dot + dt * (1 - gamma) * u_ddot
    return u_hat, u_dot_hat

def correct(u_new, u_hat, u_dot_hat, dt, gamma=0.5, beta=0.25):
    u_ddot_new = (u_new - u_hat) / (beta * dt ** 2)
    u_dot_new = u_dot_hat + gamma / (beta * dt) * (u_new - u_hat)
    return u_ddot_new, u_dot_new

def b_coeffs(N, alpha):
    b_vector = np.ones(N + 1)
    index = np.arange(1, N + 1)
    b_vector[index] = pow(index + 1, 2 - alpha) - 2 * pow(index, 2 - alpha) + pow(index - 1, 2 - alpha)
    if N > 1:
        b_vector[N] = pow(N, 1 - alpha) * (2 - alpha - N) + pow(N - 1, 2 - alpha)
    result = 1 / Gamma(3 - alpha) * b_vector[::-1]
    return result


def sum_u_forward(u_dot_time: TimeDependentVector, t, simulation_times, alpha):
    # Assume that we have u_dot, which is a TimeDependentVector.
    N = get_index(t, simulation_times)
    b_vec = b_coeffs(N, alpha)
    sum_u_dot = u_dot_time[simulation_times[0]].copy()
    sum_u_dot.zero()
    for i in range(N + 1):
        sum_u_dot[:] += b_vec[i] * u_dot_time[simulation_times[i]][:]
    return sum_u_dot

def sum_u_backward(u_dot_time: TimeDependentVector, t, simulation_times, alpha):
    N = get_index(simulation_times[-1] - t, simulation_times)
    b_vec = b_coeffs(N, alpha)
    sum_u_dot = u_dot_time[simulation_times[-1]].copy()
    sum_u_dot.zero()
    for i in range(N + 1):
        sum_u_dot[:] += b_vec[i] * u_dot_time[simulation_times[-1-i]][:]
    return sum_u_dot

def sum_u(u_dot_time, t, simulation_times, alpha, forward=True):
    if forward:
        return sum_u_forward(u_dot_time, t, simulation_times, alpha)
    else:
        return sum_u_backward(u_dot_time, t, simulation_times, alpha)

def frac_diff_u(u_dot_time, t, simulation_times, alpha):
    dt = simulation_times[1] - simulation_times[0]
    temp_result = sum_u_forward(u_dot_time, t, simulation_times, alpha)
    result = temp_result * dt ** (1 - alpha)
    return result

def FractionalWaveSolverNewmark(V, simulation_times, data, parameter, gamma=0.5, beta=0.25):
    # Parameter = [kappa, c, alpha]
    c, b, alpha = parameter
    dt = simulation_times[1] - simulation_times[0]
    dt_expr = fn.Constant(dt)
    b_0 = 1. / Gamma(3 - alpha)
    # Prepare matrices:
    u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)

    M_form = M(u_trial, u_test)
    M_c_form = M_c(u_trial, u_test, c)
    K_form = K(u_trial, u_test)
    C_form = C(u_trial, u_test, b)
    M_matrix = fn.assemble(M_form)
    M_c_matrix = fn.assemble(M_c_form)
    K_matrix = fn.assemble(K_form)
    C_matrix = fn.assemble(C_form)
    K_star_form = (K_form + 1. / (beta * dt_expr ** 2) * M_c_form + 1. / (beta * dt_expr ** alpha) * b_0 * gamma * C_form)
    RHS_temp = fn.inner(fn.Constant(0.), u_test) * fn.dx
    bc = fn.DirichletBC(V, 0., "on_boundary")
    K_matrix_star, _ = fn.assemble_system(K_star_form, RHS_temp, bc)

    # Solve problem:
    u_old = data[0].copy()
    u_dot_old = data[1].copy()
    u_ddot_old = init_vector(M_matrix, 0)
    f_source = data[2].copy()
    u_temp = init_vector(M_matrix, 0)

    # Initialize time-dependent solution vectors:
    u_time = time_vector_like(f_source)
    u_dot_time = time_vector_like(f_source)

    u_time.store(u_old, simulation_times[0])
    u_dot_time.store(u_dot_old, simulation_times[0])

    # Solve for t = t_0:
    print('Solve for t =', np.round(simulation_times[0], 3))
    RHS_0 = M_matrix * f_source[simulation_times[0]] - K_matrix * u_old
    fn.solve(M_c_matrix, u_ddot_old, RHS_0)
    # Solve for t > t_0:
    for t in simulation_times[1::]:
        if check_multiple(t, 0.01):
            print('Solve for t =', np.round(t, 3))

        # Predict and store u_dot_hat to u_dot_time at time t:
        u_hat, u_dot_hat = pred(u_old, u_dot_old, u_ddot_old, dt, gamma, beta)
        # Temporarily store u_dot_hat at time t:
        u_dot_time.store(u_dot_hat, t)
        # Solve:
        RHS = (M_matrix * f_source[t]
               + 1. / (beta * dt ** 2) * M_c_matrix * u_hat
               + 1. / (beta * dt ** alpha) * b_0 * gamma * C_matrix * u_hat
               - C_matrix * dt ** (1 - alpha) * sum_u(u_dot_time, t, simulation_times, alpha))
        bc.apply(RHS)
        fn.solve(K_matrix_star, u_temp, RHS)
        # Correct:
        u_ddot_temp, u_dot_temp = correct(u_temp, u_hat, u_dot_hat, dt, gamma, beta)
        # Update:
        u_time.store(u_temp, t)
        u_dot_time.store(u_dot_temp, t)
        u_old[:] = u_temp[:]
        u_dot_old[:] = u_dot_temp[:]
        u_ddot_old[:] = u_ddot_temp[:]
    print(' ')
    return u_time, u_dot_time

def MultipleFractionalWaveSolver(V, simulation_times, list_of_data, parameter, i_func, gamma=0.5, beta=0.25):
    # Parameter = [c, b, alpha]
    c, b, alpha = parameter
    dt = simulation_times[1] - simulation_times[0]
    dt_expr = fn.Constant(dt)
    b_0 = 1. / Gamma(3 - alpha)
    # Prepare matrices:
    u_trial, u_test = fn.TrialFunction(V), fn.TestFunction(V)

    # Assemble matrices:
    M_matrix = fn.assemble(M(u_trial, u_test))
    M_c_matrix = fn.assemble(M_c(u_trial, u_test, c))
    C_form = C(u_trial, u_test, b)
    C_matrix = fn.assemble(C_form)
    K_star_form = (K(u_trial, u_test)
                   + 1. / (beta * dt_expr ** 2) * M_c(u_trial, u_test, c)
                   + 1. / (beta * dt_expr ** alpha) * b_0 * gamma * C_form)
    RHS_temp = fn.inner(fn.Constant(0.), u_test) * fn.dx
    bc = fn.DirichletBC(V, 0., "on_boundary")
    K_matrix_star, _ = fn.assemble_system(K_star_form, RHS_temp, bc)
    M_solver = fn.KrylovSolver('cg', 'hypre_amg')
    M_solver.set_operator(M_matrix)

    M_c_solver = fn.KrylovSolver('cg', 'hypre_amg')
    M_c_solver.set_operator(M_c_matrix)

    K_matrix_star_solver = fn.KrylovSolver('cg', 'hypre_amg')
    K_matrix_star_solver.set_operator(K_matrix_star)


    # Initialize time-dependent solution vectors:
    u_time_list = [init_time_vector(M_matrix, simulation_times, 0) for _ in list_of_data]
    u_dot_time_list = [init_time_vector(M_matrix, simulation_times, 0) for _ in list_of_data]

    # Initialize vectors:
    u_old = [init_vector(M_matrix, 0) for _ in list_of_data]
    u_dot_old = [init_vector(M_matrix, 0) for _ in list_of_data]
    u_ddot_old = [init_vector(M_matrix, 0) for _ in list_of_data]
    u_temp = init_vector(M_matrix, 0)
    # Solve for t = t_0:
    print('Solve for t =', np.round(simulation_times[0], 3))
    for j in range(len(list_of_data)):
        RHS_0 = M_matrix * i_func[0] * list_of_data[j]
        M_c_solver.solve(u_ddot_old[j], RHS_0)

    # Solve for t > t_0:
    t_index = 0
    for t in simulation_times[1::]:
        t_index += 1
        if check_multiple(t, 0.01):
            print('Solve for t =', np.round(t, 3))
        # Predict and store u_dot_hat to u_dot_time at time t:
        for j in range(len(list_of_data)):
            u_hat, u_dot_hat = pred(u_old[j], u_dot_old[j], u_ddot_old[j], dt, gamma, beta)
            u_dot_time_list[j].store(u_dot_hat, t)
            # Solve:
            RHS = (M_matrix * i_func[t_index] * list_of_data[j]
                   + 1. / (beta * dt ** 2) * M_c_matrix * u_hat
                   + 1. / (beta * dt ** alpha) * b_0 * gamma * C_matrix * u_hat
                   - C_matrix * dt ** (1 - alpha) * sum_u(u_dot_time_list[j], t, simulation_times, alpha))

            bc.apply(RHS)
            u_temp.zero()
            K_matrix_star_solver.solve(u_temp, RHS)

            # Correct:
            u_ddot_temp, u_dot_temp = correct(u_temp, u_hat, u_dot_hat, dt, gamma, beta)

            # Update:
            u_time_list[j].store(u_temp, t)
            u_dot_time_list[j].store(u_dot_temp, t)
            u_old[j][:] = u_temp[:]
            u_dot_old[j][:] = u_dot_temp[:]
            u_ddot_old[j][:] = u_ddot_temp[:]
    print(' ')
    return u_time_list

