#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import sys
import optparse
import numpy as np
import cPickle as pickle

from time import time
from itertools import islice
from ..io import conll, compressed_dump, compressed_load
from ..tagger import tagger


__version__ = "0.1"


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """%s    """ % str(__file__)
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file model_file",
                                   version="%prog " + __version__,
                                   description = description)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")
    parser.add_option("-f", "--feature-module",
                      dest="feature_module",
                      help="The module in the features package containing the `fd` and `hd` functions. [Default: %default].",
                      default="rr09",
                      metavar="str",
                      type="str")
    parser.add_option("-r", "--reg",
                      dest="reg",
                      help="regularization parameter. ",
                      default=0.00001,
                      metavar="float",
                      type="float")
    parser.add_option("-E", "--epochs",
                      dest="epochs",
                      help="Number of training epochs. ",
                      default=100,
                      metavar="int",
                      type="int")
    parser.add_option("--min-count",
                      dest="minc",
                      help="min number of occurances.",
                      default=1,
                      metavar="int",
                      type="int")
    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                      "-1 for unlimited.",
                      default=-1,
                      metavar="int",
                      type="int")
    parser.add_option("-l", "--lang",
                      dest="lang",
                      help="The language (`en` or `de`).",
                      default="en",
                      metavar="str",
                      type="str")
    parser.add_option("--shuffle",
                      action="store_true",
                      dest="shuffle",
                      default=False,
                      help="Shuffle the training data after each epoche.")
    parser.add_option("--stats",
                      action="store_true",
                      dest="stats",
                      default=False,
                      help="Print model statistics.")
    parser.add_option("--eph",
                      action="store_true",
                      dest="use_eph",
                      default=False,
                      help="Use Extended Prediction History.")
    parser.add_option("--aso",
                      action="store_true",
                      dest="aso",
                      default=False,
                      help="Use Alternating Structural Optimization.")

    return parser


def train():
    """Training script for Named Entity Recognizer.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 2:
        parser.error("incorrect number of arguments (use `--help` for help).")
    if options.aso:
        if not options.funlabeled:
            raise parser.error("specify unlabeled data with --unlabeled. ")
    # get filenames
    f_train = argv[0]
    f_model = argv[1]
    # get feature extraction module
    try:
        import_path = "nut.ner.features.%s" % options.feature_module
        mod = __import__(import_path, fromlist=[options.feature_module])
        fd = mod.fd
        hd = mod.hd
    except ImportError:
        print "Error: cannot import feature extractors " \
              "from %s" % options.feature_module
        sys.exit(-2)

    train_reader = conll.Conll03Reader(f_train, options.lang)

    if options.aso:
        print "ASO not yet implemented."
        sys.exit(-1)

    
    #model = tagger.AvgPerceptronTagger(fd, hd, verbose=options.verbose)
    model = tagger.GreedySVMTagger(fd, hd, lang=options.lang,
                                   verbose=options.verbose)
    model.feature_extraction(train_reader, minc=options.minc,
                             use_eph=options.use_eph)
    
    model.train(train_reader, reg=options.reg, epochs=options.epochs,
                shuffle=options.shuffle)
    if options.stats:
        print "------------------------------"
        print " Stats\n"
        model.describe(k=40)
        nnz = 0
        for instance in model.dataset.iterinstances():
            nnz += instance.shape[0]
        print "avg. nnz: %.4f" % (float(nnz) / model.dataset.n)

    compressed_dump(f_model, model)


def predict():
    """Test script for Named Entity Recognizer.
    """
    def usage():
        print """Usage: %s [OPTIONS] MODEL_FILE TEST_FILE PRED_FILE
        Load NER model from MODEL_FILE, test the model on
        TEST_FILE and write predictions to PRED_FILE.
        The predictions are appended to the test file.
        Options:
          -h, --help\tprint this help.

        """ % sys.argv[0]
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        usage()
        sys.exit(-2)
    if len(argv) != 3:
        print "Error: wrong number of arguments. "
        usage()
        sys.exit(-2)

    print >> sys.stderr, "loading tagger...",
    sys.stderr.flush()
    model = compressed_load(argv[0])
    print >> sys.stderr, "[done]"
    print >> sys.stderr, "use_eph: ", model.use_eph
    test_reader = conll.Conll03Reader(argv[1], model.lang)
    if argv[2] != "-":
        f = open(argv[2], "w+")
    else:
        f = sys.stdout
    t0 = time()
    test_reader.write_sent_predictions(model, f, raw=False)
    f.close()
    print >> sys.stderr, "processed input in %.4fs sec." % (time() - t0)
