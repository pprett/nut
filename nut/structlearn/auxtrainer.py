#!/usr/bin/python
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
auxtrainer
==========

A module containing different trainers for the auxiliary tasks. 

"""
from __future__ import division

import math
import bolt
from abc import ABCMeta, abstractmethod

class AuxTrainer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_classifier(self, ds):
	return None


class ElasticNetTrainer(AuxTrainer):
    """Trains the auxiliary classifiers using Elastic Net regularization.

    See [Prettenhofer and Stein, 2010b].
    """

    def __init__(self, reg, alpha, num_iterations):
	self.reg = reg
	self.alpha = alpha
	self.num_iterations = num_iterations
	
    def train_classifier(self, ds):
	epochs = int(math.ceil(float(self.num_iterations) / float(ds.n)))
	model = bolt.LinearModel(ds.dim, biasterm = False)
	loss = bolt.ModifiedHuber()
	sgd = bolt.SGD(loss, self.reg, epochs = epochs, norm = 3,
		       alpha = self.alpha)
	sgd.train(model, ds, verbose = 0, shuffle = False)
	return model.w

class L2Trainer(AuxTrainer):
    """Trains the auxiliary classifiers using L2 regularization.

    If `truncation` is True, negative weights are set to zero.
    See [Ando and Zhang, 2005] or [Blitzer et al, 2006]. 
    """

    def __init__(self, reg, num_iterations, truncation = False):
	self.reg = reg
	self.num_iterations = num_iterations
	self.truncation = truncation

    def train_classifier(self, ds):
	epochs = int(math.ceil(self.num_iterations / ds.n))
	model = bolt.LinearModel(ds.dim, biasterm = False)
	loss = bolt.ModifiedHuber()
	sgd = bolt.SGD(loss, self.reg, epochs = epochs, norm = 2)
	sgd.train(model, ds, verbose = 0, shuffle = False)
	w = model.w
	if self.truncation:
	    w[w<0.0] = 0.0
	return w
	
