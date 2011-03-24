#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""
The :mod:`eval` module contains various routines for model evaluation.

The following evaluation metrics are currently supported:

  :func:`errorrate`: the error rate of the binary classifier.
  
  :func:`rmse`: the root mean squared error of a regressor.
  
  :func:`cost`: the cost of a model w.r.t. a given loss function.

"""
from __future__ import division

from itertools import izip

import numpy as np
from trainer.sgd import LossFunction, Classification, Regression

def errorrate(model, ds):
    """Compute the misclassification rate of the model.
    Assumes that labels are coded as 1 or -1. 

    zero/one loss: if p*y > 0 then 0 else 1

    :arg model: A :class:`bolt.model.LinearModel`.
    :arg ds: A :class:`bolt.io.Dataset`.
    :returns: `(100.0 / n) * sum( p*y > 0 ? 0 : 1 for p,y in ds)`.
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(ds.iterinstances()),ds.iterlabels()):
        if isinstance(p, float):
            p = int(np.sign(p))
        if p != y:
            err += 1
        n += 1
    errrate = err / n
    return errrate * 100.0

def rmse(model, ds):
    """Compute the root mean squared error of the model.

    :arg model: A :class:`bolt.model.LinearModel`.
    :arg ds: A :class:`bolt.io.Dataset`.
    :returns: `sum([(model(x)-y)**2.0 for x,y in ds])`.
    """
    n = 0
    err = 0
    for p,y in izip(model.predict(ds.iterinstances()),ds.iterlabels()):
        err += (p-y)**2.0
        n += 1
    err /= n
    return np.sqrt(err)

def cost(model, ds, loss):
    """The cost of the loss function.

    :arg model: A :class:`bolt.model.LinearModel`.
    :arg ds: A :class:`bolt.io.Dataset`.
    :returns: `sum([loss.(model(x),y) for x,y in ds])`
    """
    cost = 0
    for p,y in izip(model.predict(ds.iterinstances()), ds.iterlabels()):
        cost += loss.loss(p, y)
    return cost

def error(model, ds, loss):
    """Report the error of the model on the
    test examples. If the loss function of the model
    is :class:`bolt.trainer.sgd.Classification` then :func:`errorrate`
    is computes, else :func:`rmse` is computed if loss function inherits
    from :class:`bolt.trainer.sgd.Regression`.

    :arg model: A :class:`bolt.model.LinearModel`.
    :arg ds: A :class:`bolt.io.Dataset`.
    :arg loss: A :class:`bolt.trainer.sgd.LossFunction`.
    :returns: Either :func:`errorrate` or :func:`rmse`; depending
    on the `loss` function.
      
    """
    err = 0.0
    if isinstance(loss, Classification):
        err = errorrate(model,ds)
    elif isinstance(loss, Regression):
        err = rmse(model, ds)
    else:
        raise ValueError("lm.loss: either Regression or " \
                         "Classification loss expected")
    return err
