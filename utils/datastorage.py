import sys

import numpy as np
from utils.functions import init_time_vector
import pickle

def export_list_time_vectors(file_name, list_time_vectors):
    shape = list_time_vectors[0].shape()
    with open(file_name, 'wb') as f:
        for X in list_time_vectors:
            for j in range(shape[1]):
                pickle.dump(X.data[j][:], f)

def import_list_time_vectors(file_name, size, simulation_times, M_matrix):
    num_time_vector = size
    size_vector = np.size(simulation_times)
    list_vectors_import = [init_time_vector(M_matrix, simulation_times, 0) for i in range(num_time_vector)]
    with open(file_name, 'rb') as f:
        for i in range(num_time_vector):
            for j in range(size_vector):
                temp_vector = pickle.load(f)
                list_vectors_import[i].data[j][:] = temp_vector[:]
    return list_vectors_import




