from utils.functions import *


class WavePointObservation(Misfit):

    """
    Point-wise observation in space and time: F: L^2(Ω) -> R^(m_S x m_T).
    m_S:    Number of spatial points.
    m_T:    Number of temporal points.
    """

    def __init__(self, V, observation_times, targets, p_obs=None, noise_variance=None):

        self.V = V
        self.observation_times = observation_times

        self.B_matrix = assemblePointwiseObservation(self.V, targets)
        # B: (R^m, M) -> R^m_S
        self.n_targets = targets

        if p_obs is None:
            self.p_obs = init_time_vector(self.B_matrix, self.observation_times, 0)
        else:
            self.p_obs = p_obs

        self.noise_variance = noise_variance

        # Temporary variables:
        self.u_snapshot = init_vector(self.B_matrix, 1)
        self.Bu_snapshot = init_vector(self.B_matrix, 0)
        self.d_snapshot = init_vector(self.B_matrix, 0)

    def observe(self, u_time):
        obs = init_time_vector(self.B_matrix, self.observation_times, 0)
        for t in self.observation_times:
            obs[t] = self.B_matrix * u_time[t]
        return obs

    def observe_transpose(self, Bu_time):
        obs_transp = init_time_vector(self.B_matrix, self.observation_times, 1)
        for i in range(len(self.observation_times)):
            self.B_matrix.transpmult(Bu_time.data[i], obs_transp.data[i])
        obs_transp.zero_out()
        return obs_transp

    def cost(self, X):
        """
        Compute the cost 1/2 * ||Bu - d||^2 / (2 * noise_variance).
        """
        u_obs = self.observe(X[STATE])
        misfit_temp = u_obs - self.p_obs
        c = misfit_temp.inner(misfit_temp)
        return c / (2. * self.noise_variance)

    def grad(self, i, X, out):
        """
        Compute the gradient of the cost function with respect to the state, stored in out.
        B*(Bu - d) / noise_variance
        """
        out.zero()
        B_obs = self.observe(X[STATE])
        if i == STATE:
            for t in self.observation_times:
                self.B_matrix.transpmult(B_obs[t] / self.noise_variance, self.u_snapshot)
                out.store(self.u_snapshot, t)
        else:
            pass

    def setLinearizationPoint(self, u_time, gauss_newton_approx=False):
        return 2

    def apply_ij(self, i, j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B_matrix.mult(self.u_snapshot, self.Bu_snapshot)
                self.Bu_snapshot *= 1. / self.noise_variance
                self.B_matrix.transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, t)
        else:
            pass


class OldWaveSpaceTimeStateObservation(Misfit):
    """
    Space-time observation: F: L^2(Ω) -> L^2(Σ x (0,T)).
    Σ:      Observation set.
    """
    def __init__(self, V, observation_times, simulation_times, B_matrix, p_obs=None, noise_variance=None):

        self.V = V
        self.observation_times = observation_times
        self.simulation_times = simulation_times
        self.B_matrix = B_matrix

        if p_obs is None:
            self.p_obs = init_time_vector(self.B_matrix, self.observation_times, 0)
        else:
            self.p_obs = p_obs

        self.noise_variance = noise_variance

        # Temporary variables:
        self.u_snapshot = init_vector(self.B_matrix, 1)
        self.Bu_snapshot = init_vector(self.B_matrix, 0)
        self.d_snapshot = init_vector(self.B_matrix, 0)

    def observe(self, u_time):
        obs = init_time_vector(self.B_matrix, self.simulation_times, 0)
        for t in self.observation_times:
            obs[t] = u_time[t]
        return obs

    def cost(self, X):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified.")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0, set to 1.0 for deterministic inverse problems")
        obs = self.observe(X[STATE])
        res = self.d.copy() - obs
        result = integrate_time_vectors(res, res, self.B_matrix, self.simulation_times) / (2. * self.noise_variance)
        return result

    def grad(self, i, X):
        result = init_time_vector(self.B_matrix, self.simulation_times, 0)
        if i == STATE:
            obs = self.observe(X[STATE])
            res = self.d.copy() - obs
            for t in self.observation_times:
                # self.B_matrix.transpmult(res[t] * 1. / self.noise_variance, self.u_snapshot)
                # result.store(self.u_snapshot, t)
                # u_temp = self.B_matrix * res[t] * 1. / (self.noise_variance)
                u_temp = self.B_matrix.transpmult(res[t] * 1. / self.noise_variance)
                result.store(u_temp, t)
            return result
        else:
            return result

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass

    # TODO: Implement apply_ij.
    def apply_ij(self, i, j, direction, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                direction.retrieve(self.u_snapshot, t)
                self.B.mult(self.u_snapshot, self.Bu_snapshot, )
                self.Bu_snapshot *= 1. / self.noise_variance
                self.B.transpmult(self.Bu_snapshot, self.u_snapshot)
                out.store(self.u_snapshot, t)
        else:
            pass


class WaveSpaceTimeStateObservation(Misfit):
    """
    Space-time observation: F: L^2(Ω) -> L^2(Σ x (0,T)).
    Σ:      Observation set.
    """
    def __init__(self, V, simulation_times, observation_times, B_matrix, p_obs=None, noise_variance=None):

        self.V = V
        self.observation_times = observation_times
        self.simulation_times = simulation_times
        self.B_matrix = B_matrix

        # Save observation points:
        u_temp = init_vector(self.B_matrix, 0)
        u_temp[:] = 1.
        Bu_temp = self.B_matrix * u_temp
        self.targets = np.nonzero(Bu_temp[:])[0]

        if p_obs is None:
            self.p_obs = init_time_vector(self.B_matrix, self.simulation_times, 0)
        else:
            self.p_obs = p_obs

        self.noise_variance = noise_variance

        # Temporary variables:
        self.u_snapshot = init_vector(self.B_matrix, 1)
        self.Bu_snapshot = init_vector(self.B_matrix, 0)
        self.d_snapshot = init_vector(self.B_matrix, 0)
    def observe(self, u_time):
        obs = init_time_vector(self.B_matrix, self.simulation_times, 0)
        obs_temp = init_vector(self.B_matrix, 0)
        u_temp = init_vector(self.B_matrix, 0)
        for t in self.observation_times:
            u_temp.zero()
            u_time.retrieve(u_temp, t)
            obs_temp[self.targets] = u_temp[self.targets]
            obs.store(obs_temp, t)
        return obs

    def cost(self, X):
        if self.noise_variance is None:
            raise ValueError("Noise Variance must be specified.")
        elif self.noise_variance == 0:
            raise ZeroDivisionError("Noise Variance must not be 0.0."
                                    "Set to 1.0 for deterministic inverse problems")
        obs = self.observe(X[STATE])
        res = self.p_obs - obs
        result = obs_int(res, res, self.B_matrix, self.simulation_times) / (2. * self.noise_variance)
        return result

    def grad(self, i, X, out):
        """
        Compute the gradient of the cost function with respect to the state, stored in out.
        :param i:   STATE or PARAMETER
        :param X:
        :param out:
        """
        out.zero()
        if i == STATE:
            obs = self.observe(X[STATE])
            residual = self.p_obs - obs
            for t in self.observation_times:
                u_temp = residual[t]  / self.noise_variance
                out.store(u_temp, t)

    def setLinearizationPoint(self, x, gauss_newton_approx=False):
        pass

    # TODO: Implement apply_ij.
    def apply_ij(self, i, j, dir, out):
        out.zero()
        if i == STATE and j == STATE:
            for t in self.observation_times:
                dir.retrieve(self.u_snapshot, t)
                self.u_snapshot *= 1. / self.noise_variance
                obs_temp = self.B_matrix * self.u_snapshot
                out.store(obs_temp, t)
        else:
            pass


## Help functions:
def obs_int(u_time: TimeDependentVector, v_time: TimeDependentVector,
            B_matrix, observation_times, method='simpson'):

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



