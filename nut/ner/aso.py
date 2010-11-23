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
from ..io import conll
from ..tagger import tagger
from ..structlearn.pivotselection import FreqSelector


__version__ = "0.1"


class ASO(object):
    """Alternating structural optimization for Named Entity Recognition.
    """
    def __init__(self, fd, hd, vocabulary, tags, use_eph=False):
        self.fd = fd
        self.hd = hd
        self.vocabulary = vocabulary
        self.tags = tags
        self.use_eph = use_eph
        self.fidx_map = dict([(fname, i) for i, fname in enumerate(vocabulary)])
        self.tidx_map = dict([(tag, i) for i, tag in enumerate(tags)])
        self.tag_map = dict([(i, t) for i, t in enumerate(tags)])

    def preselect_tasks(self):
        preselection = set()
        for fx in self.fidx_map:
            if fx.startswith("w=") or fx.startswith("pre_w=") \
               or fx.startswith("post_w="):
                preselection.add(self.fidx_map[fx])
        return preselection

    def create_aux_tasks(self, m):
        preselection = self.preselect_tasks()
        aux_tasks = list(islice(
            FreqSelector(0).select(self.dataset, preselection), m))
        return aux_tasks

    def print_tasks(self, tasks):
        for task in tasks:
            print self.vocabulary[task]

    def learn(self, reader):
        """Learns the ASO embedding theta.

        Parameters
        ----------
        reader : ConllReader
            The unlabeled data reader.
        Returns
        -------
        self
        """
        print "run ASO.learn..."
        self.dataset = tagger.build_examples(reader, self.fd, self.hd,
                                             self.fidx_map, self.tags,
                                             pos_prefixes=["NN", "JJ"])
        print "num examples: %d" % self.dataset.n
        #self.filter_noun_adjectives()
        aux_tasks = self.create_aux_tasks(100)
        self.print_tasks(aux_tasks)
##      ds.shuffle(9)
        ## struct_learner = structlearn.StructLearner(k, ds, pivots,
##                                                 self.trainer,
##                                                 self.strategy)
##      struct_learner.learn()
        return self


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """%s    """ % str(__file__)
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file unlabeled_file model_file",
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
                      default=0,
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
    parser.add_option("--eph",
                      action="store_true",
                      dest="use_eph",
                      default=False,
                      help="Use Extended Prediction History.")
    return parser


def train():
    """Training script for Named Entity Recognizer.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 3:
        parser.error("incorrect number of arguments (use `--help` for help).")

    # get filenames
    f_train = argv[0]
    f_unlabeled = argv[1]
    f_model = argv[2]
    
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
    unlabeled_reader = conll.Conll03Reader(f_unlabeled,
                                           options.lang)
    readers = [train_reader, unlabeled_reader]
    print "build vocabulary..."
    V, T = tagger.build_vocabulary(readers, fd, hd, minc=options.minc,
                                       use_eph=options.use_eph)
    model = ASO(fd, hd, V, T, use_eph=options.use_eph)
    model.learn(unlabeled_reader)
    
    print "|V|:", len(V)
    print "|T|:", len(T)
    print "T:", T
    #aso = ASO(model, unlabeled_reader)
    #aso.learn()
    sys.exit()

    ## TODO implement ASO code here

    # dump the model
    if f_model.endswith(".gz"):
        import gzip
        f = gzip.open(f_model, mode="wb")
    elif f_model.endswith(".bz2"):
        import bz2
        f = bz2.BZ2File(f_model, mode="w")
    else:
        f = open(f_model)
    pickle.dump(model, f)
    f.close()
