# Copyright (c) 2016-2018, The University of Texas at Austin 
# & University of California--Merced.
# Copyright (c) 2019-2020, The University of Texas at Austin 
# University of California--Merced, Washington University in St. Louis.
#
# All Rights reserved.
# See file COPYRIGHT for details.
#
# This file is part of the hIPPYlib library. For more information and source code
# availability see https://hippylib.github.io.
#
# hIPPYlib is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License (as published by the Free
# Software Foundation) version 2.0 dated June 1991.

import dolfin as dl
import numpy as np

class TimeDependentVector(object):
    """
    A class to store time dependent vectors.
    Snapshots are stored/retrieved by specifying the time of the snapshot.
    Times at which the snapshot are taken must be specified in the constructor.
    """
    
    def __init__(self, times, tol=1e-10, mpi_comm=dl.MPI.comm_world, size=None):
        """
        Constructor:

        - :code:`times`: time frame at which snapshots are stored.
        - :code:`tol`  : tolerance to identify the frame of the snapshot.
        """
        self.nsteps = len(times)
        self.data = []
        if size is None:
            for i in range(self.nsteps):
                self.data.append(dl.Vector(mpi_comm))
        else:
            for i in range(self.nsteps):
                self.data.append(dl.Vector(mpi_comm, size))
             
        self.times = times
        self.tol = tol
        self.mpi_comm = mpi_comm

    def __imul__(self, other):
        """ Multiply each snapshot by :code:`other`. """
        for d in self.data:
            d *= other
        return self

    def __mul__(self, other):
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []
        for v in self.data:
            v_temp = v * other
            res.data.append(v_temp)
        return res

    def __rmul__(self, other):
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []
        for v in self.data:
            v_temp = other * v
            res.data.append(v_temp)
        return res

    def __add__(self, other):
        """ Add two TimeDependentVectors. """
        time_vec_sum = self.copy()
        for i in range(self.nsteps):
            time_vec_sum.data[i] += other.data[i]
        return time_vec_sum

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        """ Subtract two TimeDependentVectors. """
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []
        for i in range(self.nsteps):
            v_temp = self.data[i] - other.data[i]
            res.data.append(v_temp)
        return res

    # TODO: Check if the vector is copied or overwritten.
    def __getitem__(self, key):
        vec_temp = dl.Vector(self.mpi_comm, self.data[0].size())
        self.retrieve(vec_temp, key)
        return vec_temp.copy()

    def __setitem__(self, key, value):
        if isinstance(value, dl.Vector):
            self.store(value, key)
        elif isinstance(value, dl.PETScVector):
            self.store(value, key)
        elif isinstance(value, np.ndarray):
            vec_temp = dl.Vector(self.mpi_comm, self.data[0].size())
            self.store(vec_temp, key)

    def init_vector(self):
        """
        Initialize the vector of the same size.
        """
        vec_temp = dl.Vector(self.mpi_comm, self.data[0].size())
        return vec_temp

    def copy(self):
        """
        Return a copy of all the time frames and snapshots.
        """        
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []

        for v in self.data:
            res.data.append(v.copy())

        return res

    def clone(self):
        """
        Return a TimeDependentVector with the same time frames but zero snapshots.
        """
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []
        for i in range(self.nsteps):
            vec_temp = self.init_vector()
            res.data.append(vec_temp)
        return res

    def shape(self):
        """
        Return the shape of the snapshot and the number of time frames.
        """
        shape_space = self.data[0].size()
        shape_time = self.nsteps
        return [shape_space, shape_time]
        
    def initialize(self, M, dim):
        """
        Initialize all the snapshot to be compatible with the range/domain of an operator :code:`M`.
        """
        if isinstance(M, dl.Matrix):
            for d in self.data:
                M.init_vector(d, dim)
                d.zero()
        elif isinstance(M, int):
            for d in self.data:
                d.init(M)
                d.zero()
        else:
            raise NotImplementedError
            
    def axpy(self, a, other):
        """
        Compute :math:`x = x + \\mbox{a*other}` snapshot per snapshot.
        """
        for i in range(self.nsteps):
            self.data[i].axpy(a,other.data[i])
        
    def zero(self):
        """
        Zero out each snapshot.
        """
        for d in self.data:
            d.zero()
            
    def store(self, u, t):
        """
        Store snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        i = 0
        while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
            i += 1
            
        assert abs(t - self.times[i]) < self.tol, 'Time not found.'
        
        self.data[i].zero()
        self.data[i].axpy(1., u)
        
    def retrieve(self, u, t):
        """
        Retrieve snapshot :code:`u` relative to time :code:`t`.
        If :code:`t` does not belong to the list of time frame an error is raised.
        """
        # i = 0
        # while i < self.nsteps-1 and 2*t > self.times[i] + self.times[i+1]:
        #     i += 1
        # assert abs(t - self.times[i]) < self.tol, 'Time not found.'

        list_index = np.abs(t - self.times)
        assert np.min(list_index) < self.tol, 'Time not found.'
        i = np.argmin(list_index)
        
        u.zero()
        u.axpy(1., self.data[i])
        
    def norm(self, time_norm, space_norm):
        """
        Compute the space-time norm of the snapshot.
        """
        assert time_norm == "linf"
        s_norm = 0
        for i in range(self.nsteps):
            tmp = self.data[i].norm(space_norm)
            if tmp > s_norm:
                s_norm = tmp
        return s_norm
        
    def inner(self, other):
        """
        Compute the inner products: :math:`a+= (\\mbox{self[i]},\\mbox{other[i]})` for each snapshot.
        """
        assert self.nsteps == other.nsteps, 'TimeDependentVector have different number of snapshots.'
        a = 0.
        for i in range(self.nsteps):
            a += self.data[i].inner(other.data[i])
        return a

    def zero_out(self):
        """
        Zero out all the snapshots that are smaller than :code:`tol`.
        """
        for i in range(self.nsteps):
            u_temp = self.data[i][:]
            u_temp[np.abs(u_temp) < self.tol] = 0.
            self.data[i][:] = u_temp[:]

    def set_value(self, other):
        """
        Set the value of all the snapshots to be equal to :code:`other`.
        """
        for i in range(self.nsteps):
            self.data[i][:] = other.data[i][:]

    def shift(self, k):
        time_vector_clone = self.clone()
        if k < 0:
            for i in range(self.nsteps + k):
                time_vector_clone.data[i][:] = self.data[i-k][:]
        elif k > 0:
            for i in range(self.nsteps - k):
                time_vector_clone.data[i+k][:] = self.data[i][:]
        return time_vector_clone

    def flip(self):
        """
        Flip the order of the snapshots.
        """
        res = TimeDependentVector(self.times, tol=self.tol, mpi_comm=self.mpi_comm)
        res.data = []

        for v in self.data[::-1]:
            res.data.append(v.copy())
        return res
