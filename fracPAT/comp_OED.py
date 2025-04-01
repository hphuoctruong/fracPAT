import numpy as np
from scipy.integrate import simps
from scipy.linalg import eigh, solve_triangular, inv
from utils import *


def comp_FIM(multi_solution_1, multi_solution_2, B_matrix, simulation_times):
    n = len(multi_solution_1)
    FIM = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            FIM[i, j] = integrate_time_vectors(multi_solution_1[i], multi_solution_2[j], B_matrix, simulation_times)
            FIM[j, i] = FIM[i, j]
    return FIM

def comp_1(observations, B_matrix, simulation_times):
    result = []
    for i in range(len(observations)):
        for j in range(len(observations)):
            print('Compute FIM:', i, j)
            temp_result = comp_FIM(observations[i], observations[j], B_matrix, simulation_times)
            result.append(temp_result)
    return result
