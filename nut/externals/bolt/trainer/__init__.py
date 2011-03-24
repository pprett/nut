#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""
The :mod:`trainer` package contains concrete
`Trainer` classes which are used
to train a `Model` on a `Dataset`.

"""

__authors__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"

import sgd
import avgperceptron
import maxent

from time import time

from ..model import LinearModel
from ..io import BinaryDataset

from copy import deepcopy


class OVA(object):
    """A One-versus-All trainer for multi-class models.

    It trains one binary classifier for each class `c`
    that predicts the class or all-other classes.
    """

    trainer = None
    """The concrete trainer for the binary classifiers. """

    def __init__(self, trainer):
        """
        :arg trainer: A concrete `Trainer` implementation which is used to
        train `k` `LinearModel` classifiers that try to predict one
        class versus all others.
        """
        self.trainer = trainer
        """:member trainer: the trainer... """

    def train(self, glm, dataset, verbose=1, shuffle=False, ncpus=1):
        """Train the `glm` using `k` binary `LinearModel` classifiers by
        applying the One-versus-All multi-class strategy.

        :arg glm: A :class:`bolt.model.GeneralizedLinearModel`.
        :arg dataset: A `Dataset`.
        :arg verbose: Verbose output.
        :type verbose: int
        :arg shuffle: Whether to shuffle the training data; argument
        is passed to `OVA.trainer`.
        :type shuffle: bool
        :arg ncpus: The number of CPUs used for parallel training.
        If 1 don't use serialization, if -1 determine automatically.
        :type ncpus: int
        """
        assert glm.k == len(dataset.classes)
        t1 = time()
        if ncpus == 1:
            self.serialtrain(glm, dataset, verbose, shuffle)
        else:
            self.paralleltrain(glm, dataset, verbose, shuffle, ncpus)
        if verbose > 0:
            print("%d models trained in %.2f seconds. " % (len(dataset.classes),
                                                           time() - t1))

    def serialtrain(self, glm, dataset, verbose, shuffle):
        classes = dataset.classes
        t1 = time()
        for i, c in enumerate(classes):
            bmodel = LinearModel(glm.m, biasterm=glm.biasterm)
            dtmp = BinaryDataset(dataset, c)
            self.trainer.train(bmodel, dtmp, verbose=0,
                               shuffle=shuffle)
            glm.W[i] = bmodel.w.T
            glm.b[i] = bmodel.bias
            if verbose > 1:
                print("Model %d trained. \n" \
                      "Total training time %.2f seconds." % (i, time() - t1))

    def paralleltrain(self, glm, dataset, verbose, shuffle, ncpus):
        import multiprocessing
        if ncpus == None or ncpus <= 0:
            ncpus = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(ncpus)
        protomodel = LinearModel(glm.m, biasterm=glm.biasterm)
        prototrainer = self.trainer
        t1 = time()
        tasks = [(i, c, deepcopy(protomodel), prototrainer,
                  BinaryDataset(dataset, c), verbose, shuffle)
                 for i, c in enumerate(dataset.classes)]
        bmodels = pool.map(paralleltrain_impl, tasks)
        for i, c, model in bmodels:
            glm.W[i] = model.w.T
            glm.b[i] = model.bias


def paralleltrain_impl(args):
    i, c, model, trainer, ds, verbose, shuffle = args
    t1 = time()
    trainer.train(model, ds, verbose=0, shuffle=shuffle)
    if verbose > 1:
        print("Model %d trained.\n" \
              "Training time %.2f seconds." % (i, time() - t1))
    return (i, c, model)
