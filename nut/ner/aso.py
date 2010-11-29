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
from ..structlearn import StructLearner
from ..structlearn import auxtrainer, auxstrategy


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
        self.fidx_map = dict([(fname, i) for i, fname
                              in enumerate(vocabulary)])
        self.tidx_map = dict([(tag, i) for i, tag in enumerate(tags)])
        self.tag_map = dict([(i, t) for i, t in enumerate(tags)])

    def preselect_tasks(self):
        preselection = set()
        for fx in self.fidx_map:
            if fx.startswith("w=") or fx.startswith("pre_w=") \
               or fx.startswith("post_w="):
                preselection.add(self.fidx_map[fx])
        return preselection

    def create_aux_tasks(self, dataset, m):
        """Select the m most frequent left, right, and current words.

        TODO: mask all features derived from the left, right, and current word.

        Parameters
        ----------
        m : int
            The number of tasks to generate.

        Returns
        -------
        aux_tasks : list
            The list of auxiliary tasks. These correspond to features
            which will subsequently be used to autolabel the unlabeled data.
        task_masks : list
            A list of feature tuples. The i-th tuple holds the masked features
            for the i-th auxiliary task.
        """
        preselection = self.preselect_tasks()
        aux_tasks = list(islice(
            FreqSelector(0).select(dataset, preselection), m))
        return aux_tasks, aux_tasks

    def print_tasks(self, tasks):
        for task in tasks:
            print self.vocabulary[task]

    def learn(self, reader, m, k):
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
        dataset = tagger.build_examples(reader, self.fd, self.hd,
                                        self.fidx_map, self.tags,
                                        pos_prefixes=["NN", "JJ"])
        print "num examples: %d" % dataset.n
        aux_tasks, masks = self.create_aux_tasks(dataset, m)
        print "selected %d auxiliary tasks. " % len(aux_tasks)
        # self.print_tasks(aux_tasks)
        trainer = auxtrainer.ElasticNetTrainer(0.00001, 0.85,
                                               10**6)
        strategy = auxstrategy.HadoopTrainingStrategy()
        struct_learner = StructLearner(k, dataset,
                                       aux_tasks,
                                       trainer,
                                       strategy)
        # learn the embedding
        # TODO apply SVD by feature type.
        struct_learner.learn()
        
        # TODO post-process embedding
        # - project unlabeled data
        # - compute and store mean and std

        # store data in model
        print
        print "size of theta: %.2f MB" % (struct_learner.thetat.nbytes / 1024.0 / 1024.0)
        print
        self.thetat = struct_learner.thetat
        self.m = m
        self.k = k
        return self


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """%s    """ % str(__file__)
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file unlabeled_file model_file",
                                   version="%prog " + __version__,
                                   description=description)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")
    parser.add_option("-f", "--feature-module",
                      dest="feature_module",
                      help="The module in the features package containing the" \
                      " `fd` and `hd` functions. [Default: %default].",
                      default="rr09",
                      metavar="str",
                      type="str")
    parser.add_option("-m",
                      dest="m",
                      help="Number of auxiliary tasks from left, right, " \
                      "and current word to create. ",
                      default=1000,
                      metavar="int",
                      type="int")
    parser.add_option("-k",
                      dest="k",
                      help="Dimensionality of the shared representation.",
                      default=50,
                      metavar="int",
                      type="int")
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
    print "|V|:", len(V)
    print "|T|:", len(T)
    print
    sys.exit(-1)
    model = ASO(fd, hd, V, T, use_eph=options.use_eph)
    model.learn(unlabeled_reader, options.m, options.k)

    # dump the model
    compressed_dump(f_model, model)
