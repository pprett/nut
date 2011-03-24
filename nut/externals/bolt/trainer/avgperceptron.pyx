# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: avgperceptron.pyx
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time

__authors__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


# ----------------------------------------
# C functions for fast sparse-dense vector operations
# ----------------------------------------

cdef struct Pair:
    np.uint32_t idx
    np.float32_t val

cdef double dot(double *w, Pair *x, int nnz):
    """Dot product of weight vector w and example x. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    cdef int i
    for i from 0 <= i < nnz:
        pair = x[i]
        sum += w[pair.idx] * pair.val
    return sum

cdef void add(double *w, int stride, Pair *x, int nnz, int y,
              double c, double *wbar, double u):
    cdef Pair pair
    cdef int i
    cdef int offset = (y*stride)
    for i from 0 <= i < nnz:
        pair = x[i]
        w[offset + pair.idx] += (pair.val * c)
        wbar[offset + pair.idx] += (c*u*pair.val)

cdef int argmax(double *w, int wstride, Pair *x, int xnnz, int y, int k):
    cdef Pair pair
    cdef int j
    cdef double *wk
    cdef double max_score = -1.0
    cdef double p = 0.0
    cdef int max_j = 0
    for j from 0 <= j < k:
        wk = w + (wstride*j)
        p = dot(wk, x, xnnz)
        if p >= max_score:
            max_j = j
            max_score = p
    return max_j

cdef class AveragedPerceptron(object):
    """Averaged Perceptron learning algorithm. 

**References**:
   * [Freund1998]_.
          
**Parameters**:
   * *epochs* - The number of iterations through the dataset. Default `epochs=5`.

    """
    cdef int epochs
    
    def __init__(self, epochs = 5):
        """        
        :arg epochs: The number of iterations through the dataset.
        :type epochs: int
        """
        self.epochs = epochs
        

    def train(self, model, dataset, verbose = 0, shuffle = False):
        """Train `model` on the `dataset` using SGD.

        :arg model: The model that is going to be trained.
        Either :class:`bolt.model.GeneralizedLinearModel` or
        :class:`bolt.model.LinearModel`.
        :arg dataset: The :class:`bolt.io.Dataset`. 
        :arg verbose: The verbosity level. If 0 no output to stdout.
        :arg shuffle: Whether or not the training data should be
        shuffled after each epoch. 
        """
        self._train_multi(model, dataset, verbose, shuffle)

    cdef void _train_multi(self,model, dataset, verbose, shuffle):
        cdef int m = model.m
        cdef int k = model.k
        cdef int n = dataset.n
        cdef int length = k*m

        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] w = model.W
        # maintain a averaged w
        cdef np.ndarray[np.float64_t, ndim=2, mode="c"] wbar = np.zeros((k,m),
                                                                        dtype=np.float64)
        cdef double *wdata = <double *>w.data
        cdef double *wbardata = <double *>wbar.data
        cdef int wstride0 = w.strides[0]
        cdef int wstride1 = w.strides[1]
        cdef int wstride = <int> (wstride0 / wstride1)

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        cdef float y = 0
        cdef int z = 0
        cdef int xnnz = 0
        cdef int nadds = 0
        cdef int i = 0
        cdef int E = self.epochs
        cdef double u = 0.0
        t1=time()
        for e from 0 <= e < E:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            if shuffle:
                dataset.shuffle()
            nadds = 0
            i = 0
            for x,y in dataset:
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                z = argmax(wdata, wstride, xdata, xnnz, <int>y, k)
                u = <double>(E*n - (n*e+i+1))
                if z != y:
                    add(wdata, wstride, xdata, xnnz, <int>y, 1, wbardata, u)
                    add(wdata, wstride, xdata, xnnz, z, -1, wbardata, u)
                    nadds += 1
                    
                i += 1
            # report epoche information
            if verbose > 0:
                print("NADDs: %d; NNZs: %d. " % (nadds, w.nonzero()[0].shape[0]))
                print("Total training time: %.2f seconds." % (time()-t1))
        
        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w)):
            raise ValueError("floating-point under-/overflow occured.")
        
        model.W = wbar * (1.0 / (n*E))
