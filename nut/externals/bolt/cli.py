#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.

"""
Command Line Interface

"""
from __future__ import division

import sys
import numpy as np
import pickle

from itertools import izip
from time import time

import parse
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

loss_functions = {0:Hinge, 1:ModifiedHuber, 2:Log, 5:SquaredError, 6:Huber}

def write_predictions(lm, ds, pfile):
    """Write model predictions to file.
    The prediction file has as many lines as len(examples).
    The i-th line contains the prediction for the i-th example, encoded as
    a floating point number

    Parameters:
    lm: Either a `LinearModel` or a `GeneralizedLinearModel`
    ds: A `Dataset`
    pfile: The filename to which predictions are written
    """
    f = pfile
    out = sys.stdout if f == "-" else open(f,"w+")
    try:
        for p,c in lm.predict(ds.iterinstances(), confidence=True):
            out.write("%d\t%.6f\n" % (p,c))
    finally:
        out.close()

def create_trainer(options):
    """Create the trainer for the given options. 
    """
    loss_class = loss_functions[options.loss]
    loss = None
    if options.epsilon:
        loss = loss_class(options.epsilon)
    else:
        loss = loss_class()

    
    if options.clstype == "sgd":
        trainer = SGD(loss, options.regularizer,
                      norm=options.norm,
                      epochs=options.epochs,
                      alpha=options.alpha)

    elif options.clstype == "pegasos":
        trainer = PEGASOS(options.regularizer,
                          epochs=options.epochs)
    elif options.clstype == "ova":
        subtrainer = SGD(loss, options.regularizer,
                         norm=options.norm,
                         epochs=options.epochs,
                         alpha=options.alpha)
        trainer = OVA(subtrainer)
    elif options.clstype == "maxent":
        trainer = MaxentSGD(options.regularizer,
                            epochs=options.epochs)
    elif options.clstype == "avgperc":
        trainer = AveragedPerceptron(epochs = options.epochs)
    else:
        parser.error("classifier type \"%s\" not supported." % options.clstype)
    return trainer

def test_model(model, dataset, text="Test error:"):
    """Tests the `model` on the `dataset` and reports `eval.errorrate`.
    """
    print("-" * len(text))
    print(text)
    print("-" * len(text))
    t1 = time()
    err = eval.errorrate(model, dataset)
    print("error: %.4f" % err)
    print("Total prediction time: %.2f seconds." % (time() - t1))

def main():
    try: 
        parser  = parse.parseSB(__version__)
        options, args = parser.parse_args()
        if len(args) < 1 or len(args) > 1:
            parser.error("incorrect number of arguments (`--help` for help).")

        if options.test_only and not options.model_file:
            parser.error("option -m is required for --test-only.")

        if options.test_only and options.test_file:
            parser.error("options --test-only and -t are mutually exclusive.")

        verbose = options.verbose
        data_file = args[0]
        dtrain = MemoryDataset.load(data_file, verbose = verbose)
        
        if not options.test_only:
            if verbose > 0:
                print("---------")
                print("Training:")
                print("---------")

            if len(dtrain.classes) > 2:
                model = GeneralizedLinearModel(dtrain.dim,len(dtrain.classes), 
                                               biasterm = options.biasterm)
            else:
                model = LinearModel(dtrain.dim,
                                    biasterm = options.biasterm)

            trainer = create_trainer(options)
            
            if isinstance(trainer, (OVA,MaxentSGD,AveragedPerceptron)):
                if not isinstance(model, GeneralizedLinearModel):
                    raise ValueError("Multi-class classifiers "\
                                     "require > 2 classes. ")
            else:
                if isinstance(model, GeneralizedLinearModel):
                    raise ValueError("%s cannot be used "\
                                     "for multi-class problems." % str(trainer))
            trainer.train(model,dtrain,verbose = verbose,
                      shuffle = options.shuffle)

            if options.computetrainerror:
                test_model(model, dtrain, text="Training error")
            if options.model_file:
                f = open(options.model_file, 'w+')
                try:
                    pickle.dump(model,f)
                finally:
                    f.close()
                
            if options.test_file:
                dtest = MemoryDataset.load(options.test_file,
                                           verbose = verbose)
                if options.prediction_file:
                    write_predictions(model, dtest, options.prediction_file)
                else:
                    test_model(model, dtest)
        else:
            model = None
            f = open(options.model_file, 'r')
            try:
                model = pickle.load(f)
            finally:
                f.close()
            if not model:
                raise Exception("cannot deserialize "\
                                "model in '%s'. " % options.model_file)
            if options.prediction_file:
                write_predictions(model, dtrain, options.prediction_file)
            else:
                test_model(model, dtrain)

    except Exception, exc:
        print "[ERROR] ", exc

if __name__ == "__main__":
    main() 


