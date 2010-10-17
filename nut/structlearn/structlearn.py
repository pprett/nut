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
from __future__ import division

import numpy as np
import bolt
import sparsesvd
from collections import defaultdict

from ..util import timeit, trace

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"


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
	#self.create_inverted_index()

    @timeit
    def create_inverted_index(self):
	iidx = defaultdict(list)
	fid_task_map = defaultdict(list)
	for i, task in enumerate(self.auxtasks):
	    for fx in task:
		fid_task_map[fx].append(i)
		
	for i, x in enumerate(self.ds.iterinstances()):
	    for fid, fval in x:
		if fid in fid_task_map:
		    for task_id in fid_task_map[fid]:
			iidx[task_id].append(i)

	iidx = dict((task_id, np.unique(np.array(occurances))) for task_id, occurances in iidx.iteritems())
	self.inverted_index = iidx
	
    @timeit
    def learn(self):
	"""
	Learns the structural parameter theta from the auxiliary tasks.
	"""
	W = self.training_strategy.train_aux_classifiers(self.ds, self.auxtasks, self.classifier_trainer, inverted_index = None)
	density = W.nnz / float(W.shape[0]*W.shape[1])
	print "density: %.4f" % density
	Ut, s, Vt = sparsesvd.sparsesvd(W, self.k)
	print "Ut.shape = (%d,%d)" % Ut.shape
	self.thetat = Ut.T	

def project_instance_dense(x, thetat):
    tmp = np.zeros((thetat.shape[1],), dtype = np.float64)
    for j, v in x:
	tmp += v * thetat[j]
    return tmp

def project(ds, thetat, dense = True):
    """Projects the `bolt.io.Dataset` onto the feature space
    induced by `thetat`.

    If `dense` is True it returns a new numpy array (a design matrix);
    else it returns a new `bolt.io.MemoryDataset`. 
    """
    dim, k = thetat.shape
    ds_prime = np.empty((ds.n, k), dtype = np.float64)
    for i, x in enumerate(ds.instances):
	ds_prime[i] = project_instance_dense(x, thetat)
    if not dense:
	instances = to_sparse_bolt(ds_prime)
	dim = k
	ds_prime = bolt.io.MemoryDataset(dim, instances, ds.labels)
	ds_prime._idx = ds._idx
    return ds_prime


