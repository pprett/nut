#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
from __future__ import division

import copy
import numpy as np
import math

import eval
import parse
import cli

from time import time
from io import MemoryDataset
from model import LinearModel, GeneralizedLinearModel

def crossvalidation(ds, trainer, model, nfolds=10, verbose=1, shuffle=False,
                    error=eval.errorrate, seed=None):
    n = ds.n
    ds.shuffle(seed = seed)
    folds = ds.split(nfolds)
    err = []
    for foldidx in range(nfolds):
        if verbose > 1:
            print("--------------------")
            print("Fold-%d" % (foldidx+1))
            print("--------------------")
        lm = copy.deepcopy(model)
        t1 = time()
        dtest = folds[foldidx]
        trainidxs = range(nfolds)
        del trainidxs[foldidx]
        dtrain = MemoryDataset.merge(folds[trainidxs])
        trainer.train(lm, dtrain,
                      verbose = (verbose-1),
                      shuffle = shuffle)
        e = error(lm,dtest)
        if verbose > 0:
            fid = ("%d" % (foldidx+1)).ljust(5)
            print("%s %s" % (fid , ("%.2f"%e).rjust(5)))
        err.append(e)
        if verbose > 1:
            print "Total time for fold-%d: %f" % (foldidx+1, time()-t1)
    return np.array(err)
    
def main():
    try:
        parser  = parse.parseCV(cli.__version__)
        options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
            parser.error("Incorrect number of arguments. ")
        
        verbose = options.verbose
        fname = args[0]
        ds = MemoryDataset.load(fname,verbose = verbose)

        if len(ds.classes) > 2:
            model = GeneralizedLinearModel(ds.dim,len(ds.classes), 
                                           biasterm = options.biasterm)
        else:
            model = LinearModel(ds.dim,
                                biasterm = options.biasterm)
        if options.epochs == -1:
            options.epochs = math.ceil(10**6 / (
                (options.nfolds - 1) * (ds.n / options.nfolds)))
            print "epochs: ", options.epochs
            
        trainer = cli.create_trainer(options)
        print("%s %s" % ("Fold".ljust(5), "Error"))
        err = crossvalidation(ds, trainer, model,
                              nfolds = options.nfolds,
                              shuffle = options.shuffle,
                              error = eval.errorrate,
                              verbose = options.verbose,
                              seed = options.seed)
        print("%s %s (%.2f)" % ("avg".ljust(5),
                                ("%.2f"%np.mean(err)).rjust(5),
                                np.std(err)))

    except Exception, exc:
        print "[ERROR] ", exc


if __name__ == "__main__":
    main() 
