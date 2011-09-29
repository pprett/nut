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
import sparsesvd
from collections import defaultdict
from itertools import chain, izip

from ..util import timeit, trace
from .auxstrategy import HadoopTrainingStrategy
from ..externals import bolt

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


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

    dataset : bolt.MemoryDataset
        The unlabeled data set.

    auxtasks : list
        A list of features (or tuples of features);
        each representing one task.

    classifier_trainer : AuxTrainer
        The trainer for the auxiliary classifiers.

    training_strategy : TrainingStrategy
        The strategy how to invoke the `classifier_trainer`.

    task_masks : list
        The list of feature masks (parallel to `auxtasks`).
        If None use auxtasks.

    useinvertedindex : bool, True
        Whether to create an inverted index for efficient autolabeling
        and masking (only if training strategy not hadoop).

    feature_types : dict, None
        A dict mapping each feature type to the minimum and maximum
        feature index in the vocabulary (i.e. feature_type: (min, max)).
        It uses inclusive semantics (i.e. max is part of the span).
        Feature types are assumed to be contignouse blocks in the vocabulary.

    Attributes
    ----------
    `dataset` : bolt.MemoryDataset
        The unlabeled data set.
    `auxtask` : list
        A list of tuples; each tuple contains a set of features
        which comprise the task.
    `inverted_index` : defaultdict(list)
        An index which holds the ids of the positive instances for each task.
    """

    inverted_index = None

    def __init__(self, k, dataset, auxtasks, classifier_trainer,
                 training_strategy, task_masks=None,
                 useinvertedindex=True, feature_types=None):
        if k < 1 or k > len(auxtasks):
            raise Error("0 < k < m")
        self.dataset = dataset
        self.auxtasks = [np.atleast_1d(task).ravel() for task in auxtasks]
        if task_masks == None:
            self.task_masks = self.auxtasks
        else:
            self.task_masks = task_masks
        self.n = dataset.n
        self.dim = dataset.dim
        self.k = k
        self.classifier_trainer = classifier_trainer
        self.training_strategy = training_strategy
        if useinvertedindex and \
               not isinstance(training_strategy, HadoopTrainingStrategy):
            self.create_inverted_index()
        if feature_types == None:
            feature_type_split = [(0, dataset.dim - 1)]
        else:
            feature_type_split = sorted(feature_types.values())
        self.feature_type_split = np.array(feature_type_split)

    @timeit
    def create_inverted_index(self):
        iidx = defaultdict(list)
        fid_task_map = defaultdict(list)
        for i, task in enumerate(self.task_masks):
            for fx in task:
                fid_task_map[fx].append(i)

        for i, x in enumerate(self.dataset.instances):
            for fid, fval in x:
                if fid in fid_task_map:
                    for task_id in fid_task_map[fid]:
                        iidx[task_id].append(i)

        iidx = dict((task_id, np.unique(np.array(occurances))) for task_id,
                    occurances in iidx.iteritems())
        self.inverted_index = iidx

    @timeit
    def learn(self, verbose=0, compute_svd=True, store_W=False):
        """
        Learns the structural parameter theta from the auxiliary tasks.
        """
        W = self.training_strategy.train_aux_classifiers(self.dataset,
                                                         self.auxtasks,
                                                         self.task_masks,
                                                         self.classifier_trainer,
                                                         inverted_index=self.inverted_index)
        density = W.nnz / float(W.shape[0] * W.shape[1])
        print "density of W: %.8f" % density
        if store_W:
            self.W = W
        if compute_svd:
            self.thetat = self.compute_svd(W)

    def print_W_cols(self, task_idx, vocabulary, n_terms=10, n_cols=10):
        if not hasattr(self, "W"):
            raise AttributeError("learn() has to be run with argument " \
                                 "`store_W=True`.")
        print "_" * 80
        print "Print significant cols of W"
        print
        Wt = self.W.T
        for i in task_idx:
            task = self.auxtasks[i]
            w = Wt[i]
            idx = w.data.argsort()[:n_terms:-1][:n_terms]
            if not hasattr(task, "__iter__"):
                task = [task]
            task = [vocabulary[term] for term in task]
            corr_terms = ["%s (%.2f)" % (vocabulary[term], weight) for term, weight in
                          izip(w.indices[idx], w.data[idx])]
            print "%s: %s" % (str(task), ", ".join(corr_terms))

    def compute_svd(self, W):
        """Compute the sparse SVD of W.

        Perform SVD for each `feature_type_split` and concatenate the
        resulting matrices.

        Parameters
        ----------

        W : array, shape = [n_features, n_auxtasks]
            The weight matrix, each column vector represents
            one auxiliary classifier.

        Returns
        -------
        array, shape = [n_features, k]
            Theta transposed.
        """
        k = self.k

        if self.feature_type_split.shape[0] == 1:
            print("Compute SVD w/o feature type splits")
            print "W.nnz:", W.nnz
            print "W.shape:", W.shape
            Ut, s, _ = sparsesvd.sparsesvd(W, k)
            print "Ut.shape", Ut.shape
            print("Spectrum: %.4f - %.4f" % (s.min(), s.max()))
            thetat = Ut.T
        else:

            # create theta^t
            thetat = np.zeros((W.shape[0], k),
                              dtype=np.float64)
            #col_offset = 0
            for f_min, f_max in self.feature_type_split:
                print "_" * 40
                print "block (%d, %d)" % (f_min, f_max)
                A = W[f_min:f_max + 1]
                print "A.nnz:", A.nnz
                print "A.shape:", A.shape
                Ut, s, Vt = sparsesvd.sparsesvd(A, k)
                print "Ut.shape", Ut.shape
                if s.shape[0] == 0 or np.all(s == 0.0):
                    print "skip block (%d, %d)" % (f_min, f_max)
                    continue
                print "Spectrum: %.4f - %.4f" % (s.min(), s.max())

                # check feature span of Ut
                span = (f_max + 1) - f_min
                assert Ut.shape[1] == span

                # If Ut.shape[0] != k the missing cols of thetat are padded with zeros.
                thetat[f_min:f_max + 1, :Ut.shape[0]] = Ut.T

        if thetat == None:
            raise Exception("Error in compute_svd; spectrum is too small. "\
                            "It seems that W is too sparse?")
        print "_" * 80
        print "thetat.shape", thetat.shape
        print "dim of embedding: %d" % (thetat.shape[1] * \
                                        self.feature_type_split.shape[0])
        return thetat

    @timeit
    def project(self, dataset, dense=True):
        """Project `dataset` onto subspace induced by `self.thetat`.

        Parameters
        ----------
        dataset : bolt.io.MemoryDataset
            The dataset.
        dense : bool
            Whether a numpy array should be returned or a MemoryDataset.

        Returns
        -------
        np.ndarray or bolt.io.MemoryDataset
        """
        dim, k = self.thetat.shape
        n_splits = self.feature_type_split.shape[0]
        dataset_prime = np.zeros((dataset.n, k * n_splits),
                                 dtype=np.float32)
        if n_splits == 1:
            projector = self.project_instance_dense_nosplit
        else:
            projector = self.project_instance_dense
        for i, x in enumerate(dataset.instances):
            dataset_prime[i] = projector(x)
        if not dense:
            instances = to_sparse_bolt(dataset_prime)
            dim = k
            dataset_prime = bolt.io.MemoryDataset(dim, instances,
                                                  dataset.labels)
            dataset_prime._idx = dataset._idx
        return dataset_prime

    def project_instance_dense(self, x):
        """Project dense instance `x` onto subspace induced by `thetat`.
        Takes into account the feature splits. """
        feature_type_split = self.feature_type_split
        dim, k = self.thetat.shape
        res = np.zeros((k * feature_type_split.shape[0],), dtype=np.float32)
        assert x.dtype == bolt.sparsedtype
        type_indices = np.searchsorted(feature_type_split[:, 0], x['f0'],
                                       side="right")
        for (j, v), idx in izip(x, type_indices):
            assert j < dim
            res[(idx - 1) * k:idx * k] += (v * self.thetat[j])
        return res

    def project_instance_dense_nosplit(self, x):
        """Project dense instance `x` onto subspace induced by `thetat`."""
        res = np.zeros((self.thetat.shape[1],), dtype=np.float32)
        for j, v in x:
            res += v * self.thetat[j]
        return res


## @timeit
## def project(dataset, thetat, dense=True):
##     """Projects the `bolt.io.MemoryDataset` onto the feature space
##     induced by `thetat`.

##     If `dense` is True it returns a new numpy array (a design matrix);
##     else it returns a new `bolt.io.MemoryDataset`.
##     """
##     dim, k = thetat.shape
##     dataset_prime = np.empty((dataset.n, k), dtype=np.float32)
##     for i, x in enumerate(dataset.instances):
##         dataset_prime[i] = project_instance_dense(x, thetat)
##     if not dense:
##         instances = to_sparse_bolt(dataset_prime)
##         dim = k
##         dataset_prime = bolt.io.MemoryDataset(dim, instances, dataset.labels)
##         dataset_prime._idx = dataset._idx
##     return dataset_prime


@timeit
def concat_datasets(a, b):
    """Concat two bolt.io.MemoryDatasets as two views.

    WARNING: this method does not take _idx into account and
    resets c._idx to `arange(c.n)`.

    Paramters
    ---------
    a : bolt.io.MemoryDataset
        The first view
    b : bolt.io.MemoryDataset
        The second view

    Returns
    -------
    c : bolt.io.MemoryDataset
        The concatenation of a and b. The features of b
        are shifted by a.dim.

    Precondition: a.n == b.n
    """
    assert a.n == b.n
    dim_a = a.dim
    res = np.empty((a.n,), dtype=np.object)
    for i in xrange(a.n):
        instance_a = a.instances[i]
        instance_b = b.instances[i]
        res[i] = concat_instances(instance_a, instance_b,
                                  offset=dim_a)
    c = bolt.io.MemoryDataset(a.dim + b.dim, res, a.labels)
    return c


def concat_instances(instance_a, instance_b, offset):
    """Concats two sparse instances; shifts the feature idx
    of the second instance by `offset`.

    Parameters
    ----------
    instance_a : array, dtype=bolt.sparsedtype
        The first instance.
    instance_b : array, dtype=bolt.sparsedtype
        The second instance.
    offset : int
        The feature index offset for the second
        instance.

    Returns
    -------
    instance_c : array, dtype=bolt.sparsedtype
        The concatenation of a and b.
    """
    instance_b['f0'] += offset
    return np.fromiter(chain(instance_a, instance_b),
                             bolt.sparsedtype)


def standardize(docterms, mean, std, beta=1.0):
    """Standardize document-term matrix `docterms`
    to 0 mean and variance 1. `beta` is an optional
    scaling factor.
    """
    docterms -= mean
    docterms /= std
    docterms *= beta


def to_sparse_bolt(X):
    """Convert n x dim numpy array to sequence of bolt instances.

    Parameters
    ----------
    X : ndarray, shape = [n, dim]

    Returns
    -------
    ndarray, dtype=np.object
        An object array containing n recarrays.
    """
    res = np.empty((X.shape[0],), dtype=np.object)
    for i, x in enumerate(X):
        res[i] = bolt.dense2sparse(x)
    return bolt.fromlist(res, np.object)
