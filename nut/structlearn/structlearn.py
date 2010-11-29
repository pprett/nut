#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

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

    This class learns the structural parameter theta from a seq of
    auxiliary tasks and provides functionality to project instances
    into the new feature space induced by theta.

    Parameters
    ----------
    k : int
        The dimensionality of the shared representation.
    ds : bolt.MemoryDataset
        The unlabeled data set.
    auxtasks : list
        A list of features (or tuples of features);
        each representing one task.
    classifier_trainer : AuxTrainer
        The trainer for the auxiliary classifiers.
    training_strategy : TrainingStrategy
        The strategy how to invoke the `classifier_trainer`.

    Attributes
    ----------
    `dataset` : bolt.MemoryDataset
        The unlabeled data set.
    `auxtask` : list
        A list of tuples; each tuple contains a set of features
        which comprise the task.
    """

    def __init__(self, k, dataset, auxtasks, classifier_trainer,
                 training_strategy):
        if k < 1 or k > len(auxtasks):
            raise Error("0 < k < m")
        self.dataset = dataset
        self.auxtasks = [task if isinstance(task, tuple) else (task,)
                         for task in auxtasks]
        self.n = dataset.n
        self.dim = dataset.dim
        self.k = k
        self.classifier_trainer = classifier_trainer
        self.training_strategy = training_strategy
        self.create_inverted_index()

    @timeit
    def create_inverted_index(self):
        iidx = defaultdict(list)
        fid_task_map = defaultdict(list)
        for i, task in enumerate(self.auxtasks):
            for fx in task:
                fid_task_map[fx].append(i)

        for i, x in enumerate(self.dataset.iterinstances()):
            for fid, fval in x:
                if fid in fid_task_map:
                    for task_id in fid_task_map[fid]:
                        iidx[task_id].append(i)

        iidx = dict((task_id, np.unique(np.array(occurances))) for task_id,
                    occurances in iidx.iteritems())
        self.inverted_index = iidx

    @timeit
    def learn(self):
        """
        Learns the structural parameter theta from the auxiliary tasks.
        """
        W = self.training_strategy.train_aux_classifiers(self.dataset, self.auxtasks,
                                                         self.classifier_trainer,
                                                         inverted_index=self.inverted_index)
        density = W.nnz / float(W.shape[0] * W.shape[1])
        print "density: %.4f" % density
        Ut, s, Vt = sparsesvd.sparsesvd(W, self.k)
        print "Ut.shape = (%d,%d)" % Ut.shape
        self.thetat = Ut.T


def project_instance_dense(x, thetat):
    tmp = np.zeros((thetat.shape[1],), dtype=np.float64)
    for j, v in x:
        tmp += v * thetat[j]
    return tmp


def project(dataset, thetat, dense=True):
    """Projects the `bolt.io.Dataset` onto the feature space
    induced by `thetat`.

    If `dense` is True it returns a new numpy array (a design matrix);
    else it returns a new `bolt.io.MemoryDataset`.
    """
    dim, k = thetat.shape
    dataset_prime = np.empty((dataset.n, k), dtype=np.float64)
    for i, x in enumerate(dataset.instances):
        dataset_prime[i] = project_instance_dense(x, thetat)
    if not dense:
        instances = to_sparse_bolt(dataset_prime)
        dim = k
        dataset_prime = bolt.io.MemoryDataset(dim, instances, dataset.labels)
        dataset_prime._idx = dataset._idx
    return dataset_prime
