#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
auxtrainer
==========

A module containing different trainers for the auxiliary tasks.c

"""
from __future__ import division

import math

from ..externals import bolt


class AuxTrainer(object):

    def train_classifier(self, ds, mask):
        raise NotImplementedError("AuxTrainer is an abstract class")

    def __repr__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
            class_name,
            ", ".join(["%s=%s" % (key, repr(val)) for
                       key, val in self.__dict__.items()])
            )

    def __str__(self):
        class_name = self.__class__.__name__
        return '%s(%s)' % (
            class_name,
            ", ".join(["%s=%s" % (key, repr(val)) for
                       key, val in self.__dict__.items()])
            )


class ElasticNetTrainer(AuxTrainer):
    """Trains the auxiliary classifiers using Elastic Net regularization.

    See [Prettenhofer and Stein, 2010b].
    """

    def __init__(self, reg, alpha, num_iterations):
        self.reg = reg
        self.alpha = alpha
        self.num_iterations = num_iterations

    def train_classifier(self, ds, mask):
        epochs = int(math.ceil(self.num_iterations / ds.n))
        model = bolt.LinearModel(ds.dim, biasterm=False)
        loss = bolt.ModifiedHuber()
        sgd = bolt.SGD(loss, self.reg, epochs=epochs, norm=3,
                       alpha=self.alpha)
        sgd.train(model, ds, verbose=0, shuffle=False, mask=mask)
        return model.w


class L2Trainer(AuxTrainer):
    """Trains the auxiliary classifiers using L2 regularization.

    If `truncate` is True, negative weights are set to zero.
    See [Ando and Zhang, 2005] or [Blitzer et al, 2006].
    """

    def __init__(self, reg, num_iterations, truncate=False):
        self.reg = reg
        self.num_iterations = num_iterations
        self.truncate = truncate

    def train_classifier(self, ds, mask):
        epochs = int(math.ceil(self.num_iterations / ds.n))
        model = bolt.LinearModel(ds.dim, biasterm=False)
        loss = bolt.ModifiedHuber()
        sgd = bolt.SGD(loss, self.reg, epochs=epochs, norm=2)
        sgd.train(model, ds, verbose=0, shuffle=False, mask=mask)
        w = model.w
        if self.truncate:
            w[w < 0.0] = 0.0
        return w


class L1Trainer(AuxTrainer):
    """Trains the auxiliary classifiers using L1 regularization.
    """

    def __init__(self, reg, num_iterations):
        self.reg = reg
        self.num_iterations = num_iterations

    def train_classifier(self, ds, mask):
        epochs = int(math.ceil(self.num_iterations / ds.n))
        model = bolt.LinearModel(ds.dim, biasterm=False)
        loss = bolt.ModifiedHuber()
        sgd = bolt.SGD(loss, self.reg, epochs=epochs, norm=1)
        sgd.train(model, ds, verbose=0, shuffle=False, mask=mask)
        return model.w
