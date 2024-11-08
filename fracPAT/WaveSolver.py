# Define WaveSolver:
from utils import *
from scipy.special import gamma as Gamma
from WaveNewmarkFractional import *

class WaveInverseUndamped:
    def __init__(self, V, simulation_times, misfit, prior, parameter):
        self.V = V
        self.simulation_times = simulation_times
        self.misfit = misfit
        self.prior = prior

        self.dt = simulation_times[1] - simulation_times[0]
        self.bc = fn.DirichletBC(self.V, 0., "on_boundary")

        # Parameter:
        self.c = parameter
        self.c_expr = fn.Constant(self.c)
        self.beta = 0.25
        self.gamma = 0.5

        # Help functions:
        u_trial, u_test = fn.TrialFunction(self.V), fn.TestFunction(self.V)
        self.M_matrix = fn.assemble(fn.inner(u_trial, u_test) * fn.dx)
        self.M_c_matrix = fn.assemble(fn.inner(self.c_expr ** (-2) * u_trial, u_test) * fn.dx)
        self.B_matrix = misfit.B_matrix
        K_star_form = K(u_trial, u_test) + 1. / (self.beta * self.dt ** 2) * M_c(u_trial, u_test, self.c_expr)

        RHS_temp = fn.inner(fn.Constant(0.), u_test) * fn.dx
        self.bc = fn.DirichletBC(V, 0., "on_boundary")
        K_matrix_star, _ = fn.assemble_system(K_star_form, RHS_temp, self.bc)

        # Set solvers:
        self.M_c_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.M_c_solver.set_operator(self.M_c_matrix)
        self.M_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.M_solver.set_operator(self.M_matrix)
        self.K_star_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.K_star_solver.set_operator(K_matrix_star)

    def solveFwd(self, out, X, design):
        """
        Solve forward problem.
        """
        print("Solve forward problem.")

        out.zero()

        u_old = init_vector(self.M_matrix, 0)
        u_dot_old = init_vector(self.M_matrix, 0)
        u_ddot_old = init_vector(self.M_matrix, 0)
        u_temp = init_vector(self.M_matrix, 0)

        u_time = init_time_vector(self.M_matrix, self.simulation_times, 0)
        u_dot_time = init_time_vector(self.M_matrix, self.simulation_times, 0)

        # Solve for t = 0:
        print('Solve for t =', np.round(self.simulation_times[0], 3))
        RHS_0 = self.M_matrix * design[0] * X[PARAMETER]
        self.M_c_solver.solve(u_ddot_old, RHS_0)

        t_index = 0
        for t in self.simulation_times[1::]:
            if check_multiple(t, 0.1):
                print('Solve for t =', np.round(t, 3))
            t_index = t_index + 1
            u_temp.zero()

            # Predict:
            u_hat, u_dot_hat = pred(u_old, u_dot_old, u_ddot_old, self.dt, self.gamma, self.beta)

            # Solve:
            RHS = (self.M_matrix * design[t_index] * X[PARAMETER]
                   + 1. / (self.beta * self.dt ** 2) * self.M_c_matrix * u_hat)
            self.bc.apply(RHS)
            self.K_star_solver.solve(u_temp, RHS)

            # Correct:
            u_ddot_temp, u_dot_temp = correct(u_temp, u_hat, u_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            out.store(u_temp, t)
            u_time.store(u_temp, t)
            u_dot_time.store(u_dot_temp, t)
            u_old[:] = u_temp[:]
            u_dot_old[:] = u_dot_temp[:]
            u_ddot_old[:] = u_ddot_temp[:]

        return u_time, u_dot_time

    def solveAdj(self, out, X):
        """
        Solve adjoint problem.
        """
        print("Solve adjoint problem.")
        out.zero()
        grad_state = init_time_vector(self.M_matrix, self.simulation_times, 0)
        self.misfit.grad(STATE, X, grad_state)

        p_old = init_vector(self.M_matrix, 0)
        p_dot_old = init_vector(self.M_matrix, 0)
        p_ddot_old = init_vector(self.M_matrix, 0)
        p_temp = init_vector(self.M_matrix, 0)

        # Solve for t = T:
        # print('Solve for t =', np.round(self.simulation_times[-1], 3))
        RHS_T = (-1) * self.B_matrix * grad_state[self.simulation_times[-1]]
        self.M_c_solver.solve(p_ddot_old, RHS_T)

        for t in self.simulation_times[-2::-1]:
            p_temp.zero()

            # Predict:
            p_hat, p_dot_hat = pred(p_old, p_dot_old, p_ddot_old, self.dt, self.gamma, self.beta)

            # Solve:
            RHS_adjoint = 1. / (self.beta * self.dt ** 2) * self.M_c_matrix * p_hat - self.B_matrix * grad_state[t]
            self.bc.apply(RHS_adjoint)

            self.K_star_solver.solve(p_temp, RHS_adjoint)

            # Correct:
            p_ddot_temp, p_dot_temp = correct(p_temp, p_hat, p_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            out.store(p_temp, t)
            p_old[:] = p_temp[:]
            p_dot_old[:] = p_dot_temp[:]
            p_ddot_old[:] = p_ddot_temp[:]

    def computeFwd(self, X, design):

        self.solveFwd(X[STATE], X, design)
        p_obs = self.misfit.observe(X[STATE])
        return p_obs

    def computeAdj_old(self, X, design):
        p_adj_integrand = space_time_mult(design, X[ADJOINT], self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times)
        return p_adj

    def computeAdj(self, p_obs, design):
        # p_obs as an input, return adjoint variable:
        print("Solve adjoint problem.")
        p_old = init_vector(self.M_matrix, 0)
        p_dot_old = init_vector(self.M_matrix, 0)
        p_ddot_old = init_vector(self.M_matrix, 0)
        p_temp = init_vector(self.M_matrix, 0)
        p_time = init_time_vector(self.M_matrix, self.simulation_times, 0)

        # Solve for t = T:
        RHS_T = self.B_matrix * p_obs[self.simulation_times[-1]] * (-1)
        self.M_c_solver.solve(p_ddot_old, RHS_T)

        for t in self.simulation_times[-2::-1]:
            # Predict:
            p_temp.zero()
            p_hat, p_dot_hat = pred(p_old, p_dot_old, p_ddot_old, self.dt, self.gamma, self.beta)

            # Solve:
            RHS_adjoint = 1. / (self.beta * self.dt ** 2) * self.M_c_matrix * p_hat - self.B_matrix * p_obs[t]
            self.bc.apply(RHS_adjoint)
            self.K_star_solver.solve(p_temp, RHS_adjoint)

            # Correct:
            p_ddot_temp, p_dot_temp = correct(p_temp, p_hat, p_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            p_time.store(p_temp, t)
            p_old[:] = p_temp[:]
            p_dot_old[:] = p_dot_temp[:]
            p_ddot_old[:] = p_ddot_temp[:]

        # Compute adjoint:
        p_adj_integrand = space_time_mult(design, p_time, self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times) * (-1)
        return p_adj


    def cost(self, X):
        """
        Compute cost functional.
        """
        R_dx = init_vector(self.M_matrix, 0)
        dx = X[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, R_dx)

        cost_reg = 0.5 * dx.inner(R_dx)
        cost_misfit = self.misfit.cost(X)

        return [cost_reg + cost_misfit, cost_reg, cost_misfit]

    def evalGradientParameter(self, X, mg, design, misfit_only=False):
        self.prior.init_vector(mg, 1)

        if misfit_only == False:
            dm = X[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()

        p_adj_integrand = space_time_mult(design, X[ADJOINT], self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times)
        mg.axpy(1., self.M_matrix * p_adj)
        g = init_vector(self.M_matrix, 0)
        self.M_solver.solve(g, mg)
        grad_norm = g.inner(mg)

        return grad_norm

    def evalGradientAD(self, X, dm, design):
        print("# Compute gradient with adjoint method. #")
        # Gradient of the regularization part:
        delta_m = X[PARAMETER] - self.prior.mean
        R_delta_m = self.generate_vector(PARAMETER)
        self.prior.R.mult(delta_m, R_delta_m)
        grad_reg = R_delta_m.inner(dm)

        # Gradient of the misfit part:
        self.solveFwd(X[STATE], X, design)
        self.solveAdj(X[ADJOINT], X)
        adj = self.computeAdj_old(X, design)
        M_adj = self.M_matrix * adj
        grad_misfit = M_adj.inner(dm)
        return grad_misfit, grad_reg

    def evalGradientADForward(self, X, dm, design):
        print("# Compute gradient with a direct method. #")
        # Gradient of the regularization part:
        delta_m = X[PARAMETER] - self.prior.mean
        R_delta_m = self.generate_vector(PARAMETER)
        self.prior.R.mult(delta_m, R_delta_m)
        grad_reg = R_delta_m.inner(dm)

        # Gradient of the misfit part:
        X_per = self.generate_vector()
        X_per[PARAMETER][:] = dm[:]
        self.solveFwd(X_per[STATE], X_per, design)
        self.solveFwd(X[STATE], X, design)
        u_obs_misfit = self.misfit.observe(X[STATE]) - self.misfit.p_obs
        u_obs_per = self.misfit.observe(X_per[STATE])
        # grad_misfit is the integration:
        grad_misfit = integrate_time_vectors(u_obs_misfit, u_obs_per, self.B_matrix,
                                             self.simulation_times) / self.misfit.noise_variance
        return grad_misfit, grad_reg

    def evalGradientFD(self, X, dm, design, eps=1e-8):
        print("# Compute gradient with finite difference method. #")
        # Define perturbed variables:
        X_per = self.generate_vector()
        X_per[PARAMETER][:] = X[PARAMETER][:] + eps * dm[:]

        # Solve initial problem and perturbed problem:
        self.solveFwd(X[STATE], X, design)
        self.solveFwd(X_per[STATE], X_per, design)

        # Compute cost:
        _, cost_reg, cost_misfit = self.cost(X)
        _, cost_reg_per, cost_misfit_per = self.cost(X_per)

        grad_misfit = (cost_misfit_per - cost_misfit) / eps
        grad_reg = (cost_reg_per - cost_reg) / eps
        return grad_misfit, grad_reg

    def init_parameter(self, m):
        self.prior.init_vector(m, 0)

    # TODO: Check sign of applyC and applyCt.
    # Check self.M_matrix
    def applyC(self, dm, design, out):
        # out = design * m
        out.zero()
        out_temp = init_vector(self.M_matrix, 0)
        t_index = 0
        for t in self.simulation_times:
            out_temp[:] = (-1) * design[t_index] * self.M_matrix * dm
            out.store(out_temp, t)

    def applyCt(self, dp, design, out):
        # TODO: Check if -1 or 1.
        # out = -int_0^T design(t) * dp(t,.) dt.
        out.zero()
        temp_integrand = space_time_mult(design, dp, self.M_matrix, self.simulation_times)
        out_temp = self.M_matrix * integrate_simp(temp_integrand, self.simulation_times)
        out.axpy(-1., out_temp)

    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    def applyWum(self, dm, out):
        out.zero()

    def applyWmu(self, du, out):
        out.zero()

    def applyR(self, dm, out):
        self.prior.R.mult(dm, out)

    def applyWmm(self, dm, out):
        out.zero()


    def generate_vector(self, component="ALL"):
        if component == "ALL":
            u = init_time_vector(self.M_matrix, self.simulation_times, 0)
            m = init_vector(self.M_matrix, 0)
            p = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return [u, m, p]
        elif component == STATE:
            u = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return u
        elif component == PARAMETER:
            m = init_vector(self.M_matrix, 0)
            return m
        elif component == ADJOINT:
            p = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return p

    def solveFwdIncremental(self, sol, my_RHS):
        """
        Solve incremental forward problem.
        """
        print("Solve incremental forward problem.")
        sol.zero()

        u_old = init_vector(self.M_matrix, 0)
        u_dot_old = init_vector(self.M_matrix, 0)
        u_ddot_old = init_vector(self.M_matrix, 0)
        u_temp = init_vector(self.M_matrix, 0)

        # Solve for t = 0:
        # print('Solve for t =', np.round(self.simulation_times[0], 3))
        RHS_0 = self.M_matrix * my_RHS[self.simulation_times[0]]
        self.M_solver.solve(u_ddot_old, RHS_0)

        for t in self.simulation_times[1::]:
            # if check_multiple(t, 0.01):
            #     print('Solve for t =', np.round(t, 3))

            # Predict:
            u_hat, u_dot_hat = pred(u_old, u_dot_old, u_ddot_old, self.dt, self.gamma, self.beta)

            # Solve:
            RHS_state = my_RHS[t] + 1. / (self.beta * self.dt ** 2) * self.M_matrix * u_hat
            self.bc.apply(RHS_state)
            self.K_star_solver.solve(u_temp, RHS_state)

            # Correct:
            u_ddot_temp, u_dot_temp = correct(u_temp, u_hat, u_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            sol.store(u_temp, t)
            u_old[:] = u_temp[:]
            u_dot_old[:] = u_dot_temp[:]
            u_ddot_old[:] = u_ddot_temp[:]

    def solveAdjIncremental(self, sol, my_RHS):
        """
        Solve incremental adjoint problem.
        """
        # Basically solve adjoint equation with a source term RHS.

        print("Solve incremental adjoint problem.")
        sol.zero()

        p_old = init_vector(self.M_matrix, 0)
        p_dot_old = init_vector(self.M_matrix, 0)
        p_ddot_old = init_vector(self.M_matrix, 0)
        p_temp = init_vector(self.M_matrix, 0)

        # Solve for t = T:
        RHS_T = my_RHS[self.simulation_times[-1]]
        # RHS_T = f_source[simulation_times[-1]] - K_matrix * p_old
        self.M_solver.solve(p_ddot_old, RHS_T)

        for t in self.simulation_times[-2::-1]:
            # if check_multiple(t, 0.01):
            #     print('Solve for t =', np.round(t, 3))

            # Predict:
            p_hat, p_dot_hat = pred(p_old, p_dot_old, p_ddot_old, self.dt, self.gamma, self.beta)

            # Solve:
            RHS = my_RHS[t] + 1. / (self.beta * self.dt ** 2) * self.M_matrix * p_hat
            self.bc.apply(RHS)
            self.K_star_solver.solve(p_temp, RHS)

            # Correct:
            p_ddot_temp, p_dot_temp = correct(p_temp, p_hat, p_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            sol.store(p_temp, t)
            p_old[:] = p_temp[:]
            p_dot_old[:] = p_dot_temp[:]
            p_ddot_old[:] = p_ddot_temp


class WaveInverseDamped:
    def __init__(self, V, simulation_times, misfit, prior, parameter):
        self.V = V
        self.simulation_times = simulation_times
        self.misfit = misfit
        self.prior = prior

        self.dt = simulation_times[1] - simulation_times[0]
        self.bc = fn.DirichletBC(self.V, 0., "on_boundary")

        # Parameter:
        [self.c, self.b, self.alpha] = parameter
        self.c_expr = fn.Constant(self.c)
        self.b_0 = 1. / Gamma(3 - self.alpha)
        self.beta = 0.25
        self.gamma = 0.5

        # Help functions:
        u_trial, u_test = fn.TrialFunction(self.V), fn.TestFunction(self.V)
        self.M_form = fn.inner(u_trial, u_test) * fn.dx
        self.M_c_form = fn.inner(self.c_expr ** (-2) * u_trial, u_test) * fn.dx
        self.K_form = fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx
        self.C_form = self.b * fn.inner(fn.grad(u_trial), fn.grad(u_test)) * fn.dx

        self.M_matrix = fn.assemble(fn.inner(u_trial, u_test) * fn.dx)
        self.M_c_matrix = fn.assemble(self.M_c_form)
        self.C_matrix = fn.assemble(self.C_form)

        K_star_form = (self.K_form
                       + 1. / (self.beta * self.dt ** 2) * self.M_c_form
                       + 1. / (self.beta * self.dt ** self.alpha) * self.b_0 * self.gamma * self.C_form)
        self.B_matrix = misfit.B_matrix

        RHS_temp = fn.inner(fn.Constant(0.), u_test) * fn.dx
        self.bc = fn.DirichletBC(V, 0., "on_boundary")
        K_matrix_star, _ = fn.assemble_system(K_star_form, RHS_temp, self.bc)

        # Set solvers:
        self.M_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.M_solver.set_operator(self.M_matrix)
        self.M_c_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.M_c_solver.set_operator(self.M_c_matrix)
        self.K_star_solver = fn.PETScKrylovSolver("cg", "hypre_amg")
        self.K_star_solver.set_operator(K_matrix_star)

    def solveFwd(self, out, X, design):
        """
        Solve forward problem.
        """
        # print("Solve forward problem.")

        out.zero()

        u_old = init_vector(self.M_matrix, 0)
        u_dot_old = init_vector(self.M_matrix, 0)
        u_ddot_old = init_vector(self.M_matrix, 0)
        u_temp = init_vector(self.M_matrix, 0)

        u_time = time_vector_like(out)
        u_dot_time = time_vector_like(out)

        # Solve for t = 0:
        #print('Solve for t =', np.round(self.simulation_times[0], 3))
        RHS_0 = self.M_matrix * design[0] * X[PARAMETER]
        fn.solve(self.M_c_matrix, u_ddot_old, RHS_0)
        # Solve for t > t_0:
        t_index = 0
        for t in self.simulation_times[1::]:
            # if check_multiple(t, 0.1):
                # print('Solve for t =', np.round(t, 3))

            t_index = t_index + 1
            u_temp.zero()

            # Predict:
            u_hat, u_dot_hat = pred(u_old, u_dot_old, u_ddot_old, self.dt, self.gamma, self.beta)
            u_dot_time.store(u_dot_hat, t)

            # Solve:
            RHS = (self.M_matrix * design[t_index] * X[PARAMETER]
                   + 1. / (self.beta * self.dt ** 2) * self.M_c_matrix * u_hat
                   + 1. / (self.beta * self.dt ** self.alpha) * self.b_0 * self.gamma * self.C_matrix * u_hat
                   - self.C_matrix * self.dt ** (1 - self.alpha) * sum_u(u_dot_time, t, self.simulation_times, self.alpha))
            self.bc.apply(RHS)
            self.K_star_solver.solve(u_temp, RHS)

            # Correct:
            u_ddot_temp, u_dot_temp = correct(u_temp, u_hat, u_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            out.store(u_temp, t)
            u_time.store(u_temp, t)
            u_dot_time.store(u_dot_temp, t)
            u_old[:] = u_temp[:]
            u_dot_old[:] = u_dot_temp[:]
            u_ddot_old[:] = u_ddot_temp[:]

        return u_time, u_dot_time

    def solveAdj(self, out, X):
        """
        Solve adjoint problem.
        """
        # print("Solve adjoint problem.")
        out.zero()
        grad_state = time_vector_like(out)
        self.misfit.grad(STATE, X, grad_state)

        p_old = init_vector(self.M_matrix, 0)
        p_dot_old = init_vector(self.M_matrix, 0)
        p_ddot_old = init_vector(self.M_matrix, 0)
        p_temp = init_vector(self.M_matrix, 0)

        p_time = time_vector_like(out)
        p_dot_time = time_vector_like(out)

        # Solve for t = T:
        # print('Solve for t =', np.round(self.simulation_times[-1], 3))
        RHS_T = (-1) * self.B_matrix * grad_state[self.simulation_times[-1]]
        self.M_c_solver.solve(p_ddot_old, RHS_T)

        for t in self.simulation_times[-2::-1]:
            p_temp.zero()
            # if check_multiple(t, 0.1):
                # print('Solve for t =', np.round(t, 3))

            # Predict:
            p_hat, p_dot_hat = pred(p_old, p_dot_old, p_ddot_old, self.dt, self.gamma, self.beta)
            p_dot_time.store(p_dot_hat, t)
            # Solve:
            RHS_adjoint = (1. / (self.beta * self.dt ** 2) * self.M_c_matrix * p_hat
                           + 1. / (self.beta * self.dt ** self.alpha) * self.b_0 * self.gamma * self.C_matrix * p_hat
                           - self.C_matrix * self.dt ** (1 - self.alpha) * sum_u(p_dot_time, t, self.simulation_times, self.alpha, False)
                           - self.B_matrix * grad_state[t])

            self.bc.apply(RHS_adjoint)
            self.K_star_solver.solve(p_temp, RHS_adjoint)

            # Correct:
            p_ddot_temp, p_dot_temp = correct(p_temp, p_hat, p_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            out.store(p_temp, t)
            p_time.store(p_temp, t)
            p_dot_time.store(p_dot_temp, t)
            p_old[:] = p_temp[:]
            p_dot_old[:] = p_dot_temp[:]
            p_ddot_old[:] = p_ddot_temp[:]

    def computeFwd(self, X, design):
        print("Solve forward problem.")
        self.solveFwd(X[STATE], X, design)
        p_obs = self.misfit.observe(X[STATE])
        return p_obs

    def computeAdj(self, p_obs, design):
        # p_obs as an input, return adjoint variable:
        print("Solve adjoint problem.")
        p_old = init_vector(self.M_matrix, 0)
        p_dot_old = init_vector(self.M_matrix, 0)
        p_ddot_old = init_vector(self.M_matrix, 0)
        p_temp = init_vector(self.M_matrix, 0)
        p_time = init_time_vector(self.M_matrix, self.simulation_times, 0)
        p_dot_time = time_vector_like(p_obs)
        # Solve for t = T:
        RHS_T = self.B_matrix * p_obs[self.simulation_times[-1]] * (-1)
        self.M_c_solver.solve(p_ddot_old, RHS_T)

        for t in self.simulation_times[-2::-1]:
            # Predict:
            p_temp.zero()
            p_hat, p_dot_hat = pred(p_old, p_dot_old, p_ddot_old, self.dt, self.gamma, self.beta)
            p_dot_time.store(p_dot_hat, t)

            # Solve:
            RHS_adjoint = (1. / (self.beta * self.dt ** 2) * self.M_c_matrix * p_hat
                           + 1. / (self.beta * self.dt ** self.alpha) * self.b_0 * self.gamma * self.C_matrix * p_hat
                           - self.C_matrix * self.dt ** (1 - self.alpha) * sum_u(p_dot_time, t, self.simulation_times, self.alpha, False)
                           - self.B_matrix * p_obs[t])
            self.bc.apply(RHS_adjoint)
            self.K_star_solver.solve(p_temp, RHS_adjoint)

            # Correct:
            p_ddot_temp, p_dot_temp = correct(p_temp, p_hat, p_dot_hat, self.dt, self.gamma, self.beta)

            # Update:
            p_time.store(p_temp, t)
            p_dot_time.store(p_dot_temp, t)
            p_old[:] = p_temp[:]
            p_dot_old[:] = p_dot_temp[:]
            p_ddot_old[:] = p_ddot_temp[:]

        # Compute adjoint:
        p_adj_integrand = space_time_mult(design, p_time, self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times) * (-1)
        return p_adj

    def computeAdj_old(self, X, design):
        p_adj_integrand = space_time_mult(design, X[ADJOINT], self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times)
        return p_adj

    def cost(self, X):
        """
        Compute cost functional.
        """
        R_dx = init_vector(self.M_matrix, 0)
        dx = X[PARAMETER] - self.prior.mean
        self.prior.R.mult(dx, R_dx)

        cost_reg = 0.5 * dx.inner(R_dx)
        cost_misfit = self.misfit.cost(X)

        return [cost_reg + cost_misfit, cost_reg, cost_misfit]

    def evalGradientParameter(self, X, mg, design, misfit_only=False):
        self.prior.init_vector(mg, 1)

        if misfit_only == False:
            dm = X[PARAMETER] - self.prior.mean
            self.prior.R.mult(dm, mg)
        else:
            mg.zero()

        p_adj_integrand = space_time_mult(design, X[ADJOINT], self.M_matrix, self.simulation_times)
        p_adj = integrate_simp(p_adj_integrand, self.simulation_times)
        mg.axpy(1., self.M_matrix * p_adj)
        g = init_vector(self.M_matrix, 0)
        self.M_solver.solve(g, mg)
        grad_norm = g.inner(mg)

        return grad_norm

    def evalGradientAD(self, X, dm, design):
        print("# Compute gradient with adjoint method. #")
        # Gradient of the regularization part:
        delta_m = X[PARAMETER] - self.prior.mean
        R_delta_m = self.generate_vector(PARAMETER)
        self.prior.R.mult(delta_m, R_delta_m)
        grad_reg = R_delta_m.inner(dm)

        # Gradient of the misfit part:
        self.solveFwd(X[STATE], X, design)
        self.solveAdj(X[ADJOINT], X)
        adj = self.computeAdj_old(X, design)
        M_adj = self.M_matrix * adj
        grad_misfit = M_adj.inner(dm)
        return grad_misfit, grad_reg

    def evalGradientADForward(self, X, dm, design):
        print("# Compute gradient with a direct method. #")
        # Gradient of the regularization part:
        delta_m = X[PARAMETER] - self.prior.mean
        R_delta_m = self.generate_vector(PARAMETER)
        self.prior.R.mult(delta_m, R_delta_m)
        grad_reg = R_delta_m.inner(dm)

        # Gradient of the misfit part:
        X_per = self.generate_vector()
        X_per[PARAMETER][:] = dm[:]
        self.solveFwd(X_per[STATE], X_per, design)
        self.solveFwd(X[STATE], X, design)
        u_obs_misfit = self.misfit.observe(X[STATE]) - self.misfit.p_obs
        u_obs_per = self.misfit.observe(X_per[STATE])
        # grad_misfit is the integration:
        grad_misfit = integrate_time_vectors(u_obs_misfit, u_obs_per, self.B_matrix,
                                             self.simulation_times) / self.misfit.noise_variance
        return grad_misfit, grad_reg

    def evalGradientFD(self, X, dm, design, eps=1e-8):
        print("# Compute gradient with finite difference method. #")
        # Define perturbed variables:
        X_per = self.generate_vector()
        X_per[PARAMETER][:] = X[PARAMETER][:] + eps * dm[:]

        # Solve initial problem and perturbed problem:
        self.solveFwd(X[STATE], X, design)
        self.solveFwd(X_per[STATE], X_per, design)

        # Compute cost:
        _, cost_reg, cost_misfit = self.cost(X)
        _, cost_reg_per, cost_misfit_per = self.cost(X_per)

        grad_misfit = (cost_misfit_per - cost_misfit) / eps
        grad_reg = (cost_reg_per - cost_reg) / eps
        return grad_misfit, grad_reg

    def init_parameter(self, m):
        self.prior.init_vector(m, 0)

    # TODO: Check sign of applyC and applyCt.
    # Check self.M_matrix
    def applyC(self, dm, design, out):
        # out = design * m
        out.zero()
        out_temp = init_vector(self.M_matrix, 0)
        t_index = 0
        for t in self.simulation_times:
            out_temp[:] = (-1) * design[t_index] * self.M_matrix * dm
            out.store(out_temp, t)

    def applyCt(self, dp, design, out):
        # TODO: Check if -1 or 1.
        # out = -int_0^T design(t) * dp(t,.) dt.
        out.zero()
        temp_integrand = space_time_mult(design, dp, self.M_matrix, self.simulation_times)
        out_temp = self.M_matrix * integrate_simp(temp_integrand, self.simulation_times)
        out.axpy(-1., out_temp)

    def applyWuu(self, du, out):
        out.zero()
        self.misfit.apply_ij(STATE, STATE, du, out)

    def applyWum(self, dm, out):
        out.zero()

    def applyWmu(self, du, out):
        out.zero()

    def applyR(self, dm, out):
        self.prior.R.mult(dm, out)

    def applyWmm(self, dm, out):
        out.zero()

    def generate_vector(self, component="ALL"):
        if component == "ALL":
            u = init_time_vector(self.M_matrix, self.simulation_times, 0)
            m = init_vector(self.M_matrix, 0)
            p = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return [u, m, p]
        elif component == STATE:
            u = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return u
        elif component == PARAMETER:
            m = init_vector(self.M_matrix, 0)
            return m
        elif component == ADJOINT:
            p = init_time_vector(self.M_matrix, self.simulation_times, 0)
            return p
