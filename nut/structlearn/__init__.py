#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style


"""
structlearn
===========

Structural Learning package.

This package provides functionality for

 - Alternating Structural Optimization (ASO)
 - Structural Correspondence Learning (SCL)
 - Cross-Language Structural Correspondence Learning (CL-SCL)


"""

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


from .structlearn import StructLearner, concat_datasets, \
     concat_instances, Error, standardize, \
     to_sparse_bolt

from . import auxtrainer
from . import auxstrategy
