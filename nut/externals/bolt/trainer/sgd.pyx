# encoding: utf-8
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# filename: sgd.pyx
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.


"""
The :mod:`bolt.trainer.sgd` module is the core of bolt. Its an extension
module written in cython containing efficient implementations of
Stochastic Gradient Descent and PEGASOS.

The module contains two `Trainer` classes:

  * :class:`SGD`: A plain stochastic gradient descent implementation,
  supporting various :class:`LossFunction` and different penalties,
  including L1, L2, and Elastic-net penalty.
  * :class:`PEGASOS`: Similar to SGD, however, after each update it
  projects the current weight vector onto the L2 ball of
  radius 1/sqrt(lambda). Currently, only supports hinge loss and L2 penalty. 

The module contains a number of concrete `LossFunction` implementations
that can be plugged into the `SGD` trainer. Bolt provides `LossFunctions`
for `Classification` and `Regression`.

The following :class:`Classification` loss functions are supported:

  * :class:`ModifiedHuber`: A quadratical smoothed version of the hinge loss. 
  * :class:`Hinge`: The loss function employed by the Support Vector Machine
  classifier.
  * :class:`Log`: The loss function of Logistic Regression.

The following :class:`Regression` loss functions are supported:

  * :class:`SquaredError`: Standard squared error loss function.
  
  * :class:`Huber`: Huber robust regression loss.

The module also contains a number of utility function:

  * :func:`predict`: computes the dot product between a sparse and a
  dense feature vector. 

"""
from __future__ import division

import numpy as np
import sys

cimport numpy as np
cimport cython

from time import time
from itertools import izip

__authors__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"

cdef extern from "math.h":
    cdef extern double exp(double x)
    cdef extern double log(double x)
    cdef extern double sqrt(double x)
    
# ----------------------------------------
# Extension Types for Loss Functions
# ----------------------------------------

cdef class LossFunction:
    """Base class for convex loss functions"""
    cpdef double loss(self, double p, double y):
        """Evaluate the loss function.
        
        :arg p: The prediction.
        :type p: double
        :arg y: The true value.
        :type y: double
        :returns: double"""
        raise NotImplementedError()
    cpdef double dloss(self, double p, double y):
        """Evaluate the derivative of the loss function.
        
        :arg p: The prediction.
        :type p: double
        :arg y: The true value.
        :type y: double
        :returns: double"""
        raise NotImplementedError()

cdef class Regression(LossFunction):
    """Base class for loss functions for regression."""
    cpdef double loss(self,double p,double y):
        raise NotImplementedError()
    cpdef double dloss(self,double p,double y):
        raise NotImplementedError()


cdef class Classification(LossFunction):
    """Base class for loss functions for classification."""
    cpdef double loss(self,double p,double y):
        raise NotImplementedError()
    cpdef double dloss(self,double p,double y):
        raise NotImplementedError()

cdef class ModifiedHuber(Classification):
    """Modified Huber loss function for binary
    classification tasks with y in {-1,1}.
    Its equivalent to quadratically smoothed SVM
    with gamma = 2.

    See T. Zhang 'Solving
    Large Scale Linear Prediction Problems Using
    Stochastic Gradient Descent', ICML'04.
    """
    cpdef double loss(self,double p,double y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return (1-z) * (1-z) 
        else:
            return -4*z

    cpdef double dloss(self,double p,double y):
        cdef double z = p*y
        if z >= 1:
            return 0
        elif z >= -1:
            return 2*(1-z)*y
        else:
            return 4*y

    def __reduce__(self):
        return ModifiedHuber,()

cdef class Hinge(Classification):
    """SVM classification loss for binary
    classification tasks with y in {-1,1}.
    """
    cpdef  double loss(self,double p,double y):
        cdef double z = p*y
        if z < 1.0:
            return (1 - z)
        return 0
    cpdef  double dloss(self,double p,double y):
        cdef double z = p*y
        if z < 1.0:
            return y
        return 0

    def __reduce__(self):
        return Hinge,()


cdef class Log(Classification):
    """Logistic regression loss for binary classification
    tasks with y in {-1,1}.
    """
    cpdef double loss(self,double p,double y):
        cdef double z = p*y
        if z > 18:
            return exp(-z)
        if z < -18:
            return -z * y
        return log(1.0+exp(-z)) 

    cpdef  double dloss(self,double p,double y):
        cdef double z = p*y
        if z > 18:
            return exp(-z) * y
        if z < -18:
            return y
        return y / (exp(z) + 1.0)

    def __reduce__(self):
        return Log,()

cdef class SquaredError(Regression):
    """
    """
    cpdef  double loss(self,double p,double y):
        return 0.5 * (p-y) * (p-y)
    cpdef  double dloss(self,double p,double y):
        return y - p

    def __reduce__(self):
        return SquaredError,()

cdef class Huber(Regression):
    """
    """
    cdef double c
    def __init__(self,c):
        self.c = c
    cpdef  double loss(self,double p,double y):
        cdef double r = p-y
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return 0.5 * r * r
        else:
            return self.c * abs_r - (0.5*self.c*self.c)

    cpdef  double dloss(self,double p,double y):
        cdef double r = y - p 
        cdef double abs_r = abs(r)
        if abs_r <= self.c:
            return r
        elif r > 0:
            return self.c
        else:
            return -self.c

    def __reduce__(self):
        return Huber,(self.c,)

# ----------------------------------------
# Python function for external prediction
# ----------------------------------------
def predict(np.ndarray x, np.ndarray w,
            double bias):
    """Computes x*w + b efficiently.

    :arg x: the instance represented as a sparse vector. 
    :type x: np.ndarray(dtype=bolt.sparsedtype)
    :arg w: the weight vector represented as a dense vector.
    :type w: np.ndarray(dtype=bolt.densedtype)
    :arg b: the bias term (aka offset or intercept).
    :type b: float
    :returns: A double representing `x*w + b`.
    """
    cdef int xnnz = x.shape[0]
    cdef int wdim = w.shape[0]
    cdef double y = 0.0
    if xnnz == 0:
        y = bias
    else:
        y = dot_checked(<double *>w.data, <Pair *>x.data, xnnz, wdim) + bias
    return y
  
 # ----------------------------------------
 # C functions for fast sparse-dense vector operations
 # ----------------------------------------

cdef struct Pair:
    np.uint32_t idx
    np.float32_t val
    
cdef inline double max(double a, double b):
    return a if a >= b else b

cdef inline double min(double a, double b):
    return a if a <= b else b

cdef double dot(double *w, Pair *x, int nnz, int *mask_ptr):
    """Dot product of weight vector w and example x. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    cdef int i
    for i from 0 <= i < nnz:
        pair = x[i]
        sum += w[pair.idx] * pair.val * mask_ptr[pair.idx]
    return sum

cdef double dot_checked(double *w, Pair *x, int nnz, int wdim):
    """ Checked version of dot product. Ignores features in x
    with a higher index than dimension of w. 
    """
    cdef double sum = 0.0
    cdef Pair pair
    cdef int i
    for i from 0 <= i < nnz:
        pair = x[i]
        if pair.idx < wdim:
            sum +=w[pair.idx]*pair.val
    return sum

cdef double add(double *w, double wscale, Pair *x, int nnz, double c, int *mask_ptr):
    """Scales example x by constant c and adds it to the weight vector w. 
    """
    cdef Pair pair
    cdef int i
    cdef double innerprod = 0.0
    cdef double xsqnorm = 0.0
    cdef double val = 0.0
    for i from 0 <= i < nnz:
        pair = x[i]
        val = pair.val * mask_ptr[pair.idx]
        innerprod += w[pair.idx] * val
        xsqnorm += val * val
        w[pair.idx] += val * (c / wscale)
        
    return (xsqnorm * c * c) + (2.0 * innerprod * wscale * c)

# ----------------------------------------
# Extension type for Stochastic Gradient Descent
# ----------------------------------------

cdef class SGD:
    """Plain stochastic gradient descent solver. The solver supports
    various :class:`LossFunction` and different penalties (L1, L2, and Elastic-Net). 

**References**:
   * SGD implementation inspired by Leon Buttuo's sgd and [Zhang2004]_.
   * L1 penalty via truncation [Tsuruoka2009]_.
   * Elastic-net penalty [Zou2005]_.
          
**Parameters**:
   * *loss* - The :class:`LossFunction`.
   * *reg* -  The regularization parameter lambda.
   * *epochs* - The number of iterations through the dataset. Default `epochs=5`. 
   * *norm* - Whether to minimize the L1, L2 norm or the Elastic Net
   (either 1,2, or 3; default 2).
   * *alpha* - The elastic net penality parameter (0<=`alpha`<=1).
   A value of 1 amounts to L2 regularization whereas a value of 0
   gives L1 penalty (requires `norm=3`). Default `alpha=0.85`.
    """
    cdef int epochs
    cdef double reg
    cdef LossFunction loss
    cdef int norm
    cdef double alpha
    
    def __init__(self, loss, reg, epochs=5, norm=2, alpha=0.85):
        """

        :arg loss: The :class:`LossFunction` (default ModifiedHuber) .
        :arg reg: The regularization parameter lambda (>0).
        :type reg: float.
        :arg epochs: The number of iterations through the dataset.
        :type epochs: int
        :arg norm: Whether to minimize the L1, L2 norm or the Elastic Net.
        :type norm: 1 or 2 or 3
        :arg alpha: The elastic net penality parameter.
        A value of 1 amounts to L2 regularization whereas a value of 0
        gives L1 penalty. 
        :type alpha: float (0 <= alpha <= 1)
        """
        if loss == None:
            raise ValueError("Loss function must not be None.")
        if reg < 0.0:
            raise ValueError("reg must be larger than 0. ")
        if norm not in [1,2,3]:
            raise ValueError("norm must be in {1,2,3}. ")
        if alpha > 1.0 or alpha < 0.0:
            raise ValueError("alpha must be in [0,1]. ")
        self.loss = loss
        self.reg = reg
        self.epochs = epochs
        self.norm = norm
        self.alpha = alpha

    def __reduce__(self):
        return SGD,(self.loss,self.reg, self.epochs, self.norm, self.alpha)

    def train(self, model, dataset, verbose=0, shuffle=False, mask=None):
        """Train `model` on the `dataset` using SGD.

        :arg model: The :class:`bolt.model.LinearModel` that is going to be trained. 
        :arg dataset: The :class:`bolt.io.Dataset`. 
        :arg verbose: The verbosity level. If 0 no output to stdout.
        :arg shuffle: Whether or not the training data should be shuffled
        after each epoch.
        :arg mask: The feature mask, a binary array that specificies which features
        should be considered.
        """
        if mask == None:
            mask = np.ones((model.m,), dtype=np.int32, order="C")
        else:
            mask = np.asarray(mask, dtype=np.int32, order="C")
            assert mask.shape == (model.m,)
        self._train(model, dataset, verbose, shuffle, mask)

    cdef void _train(self, model, dataset, verbose, shuffle, mask_arr):
        
        cdef LossFunction loss = self.loss
        cdef int m = model.m
        cdef int n = dataset.n
        cdef double reg = self.reg

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] w = model.w
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # the scale of w
        cdef double wscale = 1.0

        # the feature mask
        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] mask = mask_arr
        cdef int *mask_ptr = <int *>mask.data

        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL

        cdef double y = 0.0
        
        # Variables for penalty term
        cdef int norm = self.norm
        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] q = None
        cdef double *qdata = NULL
        cdef double u = 0.0
        if norm == 1 or norm == 3:
            q = np.zeros((m,), dtype = np.float64, order = "c" )
            qdata = <double *>q.data
            
        cdef double alpha = 1.0
        if norm == 1:
            alpha = 0.0
        elif norm == 3:
            alpha = self.alpha

        # bias term (aka offset or intercept)
        cdef int usebias = 1
        if model.biasterm == False:
            usebias = 0

        cdef double b = 0.0
        cdef double p = 0.0
        cdef double wnorm = 0.0
        cdef double t = 0.0
        cdef double update = 0.0
        cdef double sumloss = 0.0
        cdef double eta = 0.0
        cdef int xnnz = 0
        cdef int count = 0
        cdef int i = 0
        cdef int e = 0
        
        # computing eta0
        cdef double typw = sqrt(1.0 / sqrt(reg))
        cdef double eta0 = typw /max(1.0, loss.dloss(-typw,1.0))
        t = 1.0 / (eta0 * reg)
        t1=time()
        for e from 0 <= e < self.epochs:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            if shuffle:
                dataset.shuffle()
            for x,y in dataset:
                eta = 1.0 / (reg * t)
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                p = (dot(wdata, xdata, xnnz, mask_ptr) * wscale) + b
                sumloss += loss.loss(p,y)
                update = eta * loss.dloss(p,y)
                if update != 0:
                    add(wdata, wscale, xdata,
                        xnnz, update, mask_ptr)
                    if usebias == 1:
                        b += update * 0.01

                if norm != 1:
                    wscale *= (1 - alpha * eta * reg)
                    if wscale < 1e-9:
                        w*=wscale
                        wscale = 1
                if norm == 1 or norm == 3:
                    u += ((1-alpha) * eta * reg)
                    l1penalty(wscale, wdata, qdata, xdata, xnnz, u, mask_ptr)
                
                t += 1
                count += 1

            # report epoche information
            if verbose > 0:
                wnorm = sqrt(np.dot(w,w) * wscale * wscale)
                print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, " \
                      "Avg. loss: %.6f" % (wnorm, w.nonzero()[0].shape[0],
                                           b, count, sumloss / count))
                print("Total training time: %.2f seconds." % (time()-t1))

        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w)) \
               or np.isnan(b) or np.isinf(b):
            raise ValueError("floating-point under-/overflow occured.")
        if norm == 3:
            # FIXME rescale naive elastic net coefficient?
            model.w = w * wscale #* (1.0 + alpha)
        else:
            model.w = w * wscale
        model.bias = b

cdef void l1penalty(double wscale, double *w, double *q,
                    Pair *x, int nnz, double u, int *mask_ptr):
    cdef double z = 0.0
    cdef Pair pair
    cdef int i,j
    cdef double val = 0.0
    for i from 0 <= i < nnz:
        pair = x[i]
        j = pair.idx
        val = pair.val * mask_ptr[j]
        if val == 0.0:
            continue
        z = w[j]
        if (wscale * w[j]) > 0:
            w[j] = max(0,w[j] - ((u + q[j])/wscale) )
        elif (wscale * w[j]) < 0:
            w[j] = min(0,w[j] + ((u - q[j])/wscale) )
        q[j] += (wscale * (w[j] - z))

########################################
#
# PEGASOS
#
########################################
        
cdef class PEGASOS:
    """Primal estimated sub-gradient solver for svm [Shwartz2007]_.
    
**Parameters**:
   * *reg* -  The regularization parameter lambda (> 0).
   * *epochs* - The number of iterations through the dataset.
    """
    cdef int epochs
    cdef double reg
    
    def __init__(self, reg, epochs):
        if reg < 0.0:
            raise ValueError("`reg` must be larger than 0. ")
        self.epochs = epochs
        self.reg = reg

    def __reduce__(self):
        return PEGASOS,(self.reg, self.epochs)

    def train(self, model, dataset, verbose=0, shuffle=False, mask=None):
        """Train `model` on the `dataset` using PEGASOS.

        :arg model: The :class:`LinearModel` that is going to be trained. 
        :arg dataset: The :class:`Dataset`. 
        :arg verbose: The verbosity level. If 0 no output to stdout.
        :arg shuffle: Whether or not the training data should be shuffled
        after each epoch. 
        """
        if mask == None:
            mask = np.ones((model.m,), dtype=np.int32, order="C")
        else:
            mask = np.asarray(mask, dtype=np.int32, order="C")
            assert mask.shape == (model.m,)
        self._train(model, dataset, verbose, shuffle, mask)

    cdef void _train(self, model, dataset, verbose, shuffle, mask_arr):
        cdef int m = model.m
        cdef int n = dataset.n
        cdef double reg = self.reg
        cdef double invsqrtreg = 1.0 / np.sqrt(reg)

        cdef np.ndarray[np.float64_t, ndim=1, mode="c"] w = model.w
        # weight vector w as c array
        cdef double *wdata = <double *>w.data
        # norm of w
        cdef double wscale = 1.0
        cdef double wnorm = 0.0

        cdef np.ndarray[np.int32_t, ndim=1, mode="c"] mask = mask_arr
        cdef int *mask_ptr = <int *>mask.data
        
        # training instance
        cdef np.ndarray x = None
        cdef Pair *xdata = NULL
        
        cdef double b = 0.0,p = 0.0,update = 0.0, z = 0.0
        cdef double y = 0.0
        cdef double eta = 0.0
        
        cdef int xnnz=0
        
        # bias term (aka offset or intercept)
        cdef int usebias = 1
        if model.biasterm == False:
            usebias = 0
        
        cdef double sumloss = 0.0
        cdef int t = 1, i = 0

        t1=time()
        for e from 0 <= e < self.epochs:
            if verbose > 0:
                print("-- Epoch %d" % (e+1))
            
            if shuffle:
                dataset.shuffle()
            for x,y in dataset:
                eta = 1.0 / (reg * t)
                xnnz = x.shape[0]
                xdata = <Pair *>x.data
                p = (dot(wdata, xdata, xnnz, mask_ptr) * wscale) + b
                z = p*y
                if z < 1:
                    wnorm += add(wdata, wscale, xdata,
                                 xnnz, (eta*y), mask_ptr)
                    if usebias == 1:
                        b += (eta*y) * 0.01
                    sumloss += (1-z)
                scale(&wscale, &wnorm, 1 - (eta * reg))
                project(wdata, &wscale, &wnorm, reg)
                if wscale < 1e-11:
                    w *= wscale
                    wscale = 1.0
                    
                t += 1

            if verbose > 0:
                print("Norm: %.2f, NNZs: %d, Bias: %.6f, T: %d, " \
                      "Avg. loss: %.6f" % (sqrt(wnorm), w.nonzero()[0].shape[0],
                                           b, t+1, sumloss / (t+1)))
                print("Total training time: %.2f seconds." % (time()-t1))
                
        # floating-point under-/overflow check.
        if np.any(np.isinf(w)) or np.any(np.isnan(w))or np.isnan(b) or np.isinf(b):
            raise ValueError("floating-point under-/overflow occured.")
        model.w = w * wscale
        model.bias = b

cdef inline void project(double *wdata, double *wscale, double *wnorm, double reg):
    """Project w onto L2 ball.
    """
    cdef double val = 1.0 
    if (wnorm[0]) != 0:
        val = 1.0 / sqrt(reg *  wnorm[0])
        if val < 1.0:
            scale(wscale,wnorm,val)    

cdef inline void scale(double *wscale, double *wnorm, double factor):
    """Scale w by constant factor. Update wnorm too.
    """
    wscale[0] *= factor
    wnorm[0] *= (factor*factor)

