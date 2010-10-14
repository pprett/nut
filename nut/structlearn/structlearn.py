#!/usr/bin/python2.6
#
# Copyright (C) 2010 Peter Prettenhofer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
structlearn
===========

"""

import numpy as np
import bolt
import sparsesvd
import math
from ..structlearn import util

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from scipy import sparse

from ..util import timeit

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"


T = 10**6
"""The minimum number of observed training examples for SGD. 
"""

def to_sparse_bolt(X):
    """Convert n x dim numpy array to sequence of bolt instances. 
    """
    res = np.empty((len(X),), dtype=np.object)
    for i,x in enumerate(X):
	res[i] = bolt.dense2sparse(x)
    return bolt.fromlist(res, np.object)

class Error(Exception):
    pass

class StructLearner(object):
    """
    Structural Learner.

    This class learns the structural parameter theta from a seq of auxiliary tasks and provides functionality to project instances into the new feature space induced by theta. 
    """

    def __init__(self, k, ds, auxtasks, classifier_trainer, training_strategy):
	if k < 1 or k > len(auxtasks):
	    raise Error("0 < k < m")
	self.ds = ds
	self.auxtasks = auxtasks
	self.n = ds.n
	self.dim = ds.dim
	self.k = k
	self.classifier_trainer = classifier_trainer
	self.training_strategy = training_strategy
	self.W = sparse.dok_matrix((self.dim, self.n), dtype=np.float64)
	
    @timeit
    def learn(self):
	"""
	Learns the structural parameter theta from the auxiliary tasks.
	"""
	self.training_strategy.train_aux_classifiers(self)
	self.W = self.W.tocsc()
	Ut, s, Vt = sparsesvd.sparsesvd(self.W, self.k)
	print "Ut.shape = (%d,%d)" % Ut.shape
	self.theta = Ut.T	

    @timeit
    def project(self, ds, dense = True):
	"""Project dataset `ds` using structural parameter theta.

	:arg ds: A `bolt.Dataset` instance.
	:arg dense: Whether the output should be a numpy array or a seq of bolt instances.
	:returns: The projected dataset as a numpy array or a seq of bolt instances. 
	"""
	if not self.theta:
	    raise Error("Structural learner not parametrized - make sure you run `learn()`.")
	k, dim = self.theta.shape
	ds_prime = np.empty((ds.n, k), dtype = np.float64)
	for i, x in enumerate(ds.instances):
	    ds_prime[i] = self._project_instance_dense(x)
	if not dense:
	    ds.instances = to_sparse_bolt(ds_prime)
	    ds.dim = k
	    ds_prime = ds
	return ds_prime

    def _project_instance_dense(self, x):
	theta = self.theta
	tmp = np.zeros((self.k,), dtype = np.float64)
	for j,v in x:
	    tmp += v * theta[j]
	return tmp

    def project_instance(self, x, dense = True):
	tmp = self._project_instance_dense(x)
	if dense:
	    return tmp
	else:
	    return bolt.dense2sparse(tmp)


class TrainingStrategy(object):
    """An interface of different training strategies for the auxiliary classifiers. 

    Use this to implement various parallel or distributed training strategies.
    Delegates the training of a single classifier to a concrete `AuxTrainer`.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_aux_classifiers(self, struct_learner):
	"""Abstract method to train auxiliary classifiers, i.e. to fill `struct_learner.W`.
	"""
	return 0

class SerialTrainingStrategy(TrainingStrategy):
    """A serial training strategy.

    Trains one auxiliary classifier after another. Does not exploit multi core architectures. 
    """

    @timeit
    def _train_aux_classifier(self, i, auxtask, original_instances,
			      classifier_trainer, W, dim):
	instances = deepcopy(original_instances)
	labels = util.autolabel(instances, auxtask)
	util.mask(instances, auxtask)
	ds = bolt.MemoryDataset(dim, instances, labels)
	w = classifier_trainer.train_classifier(ds)
	for j in w.nonzero()[0]:
	    W[j,i] = w[j]
	if i % 10 == 0:
	    print "%d classifiers trained..." % i

    def train_aux_classifiers(self, struct_learner):
	dim = struct_learner.ds.dim
	auxtasks = struct_learner.auxtasks
	struct_learner.ds.shuffle(seed = 13)
	original_instances = struct_learner.ds.instances[struct_learner.ds._idx]
	W = struct_learner.W
	classifier_trainer = struct_learner.classifier_trainer

	for i, auxtask in enumerate(auxtasks):
	    self._train_aux_classifier(i, auxtask, original_instances,
				       classifier_trainer, W, dim)

class AuxTrainer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_classifier(self, ds):
	return None


class ElasticNetTrainer(AuxTrainer):
    """Trains the auxiliary classifiers using Elastic Net regularization.

    See [Prettenhofer and Stein, 2010b].
    """

    def __init__(self, reg, alpha, epochs = -1):
	self.reg = reg
	self.alpha = alpha
	self.epochs = epochs
	
    def train_classifier(self, ds):
	epochs = self.epochs
	if epochs <= 0:
	    epochs = int(math.ceil(T / ds.n))
	model = bolt.LinearModel(ds.dim, biasterm = False)
	loss = bolt.ModifiedHuber()
	sgd = bolt.SGD(loss, self.reg, epochs = epochs, norm = 3, alpha = self.alpha)
	sgd.train(model, ds, verbose = 0, shuffle = False)
	return model.w

class L2Trainer(AuxTrainer):
    """Trains the auxiliary classifiers using L2 regularization.

    If `truncation` is True, negative weights are set to zero.
    See [Ando and Zhang, 2005] or [Blitzer et al, 2006]. 
    """

    def __init__(self, reg, epochs = -1, truncation = False):
	self.reg = reg
	self.epochs = epochs
	self.truncation = truncation
	

    def train_classifier(self, ds):
	epochs = self.epochs
	if epochs <= 0:
	    epochs = int(math.ceil(T / ds.n))
	model = bolt.LinearModel(ds.dim, biasterm = False)
	loss = bolt.ModifiedHuber()
	sgd = bolt.SGD(loss, self.reg, epochs = epochs, norm = 2)
	sgd.train(model, ds, verbose = 0, shuffle = False)
	w = model.w
	if self.truncation:
	    w[w<0.0] = 0.0
	return w
	
