#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.

"""
Bolt
====

Bolt online learning toolbox.

Documentation is available in the docstrings. 

Subpackages
-----------

model
   Model specifications. 

trainer
   Extension module containing various model trainers.

io
   Input/Output routines; reading datasets, writing predictions.

eval
   Evaluation metrics. 

parse
   Command line parsing.

see http://github.com/pprett/bolt

"""

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"

import eval

from trainer import OVA
from trainer.sgd import predict, SGD, LossFunction, Classification, \
     Regression, Hinge, ModifiedHuber, Log, SquaredError, Huber, PEGASOS
from trainer.maxent import MaxentSGD
from trainer.avgperceptron import AveragedPerceptron
from io import MemoryDataset, sparsedtype, dense2sparse, fromlist
from model import LinearModel, GeneralizedLinearModel
from eval import errorrate

__version__ = "1.4"

