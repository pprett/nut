#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""Alternating Structural Optimization for Named Entity Recognition.

This module implements Alternating Structural Optimization (ASO), a semi-
supervised learning technique proposed by Ando and Zhang (2005) for
named entity recognition (NER).
"""

import sys
import optparse
import numpy as np
import re
import bolt

from collections import defaultdict
from time import time
from itertools import islice

from ..io import conll, compressed_dump, compressed_load
from ..tagger import tagger
from ..structlearn.pivotselection import FreqSelector

from ..nut import structlearn


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
            if fx.startswith("word_cur=") or fx.startswith("word_pre=") \
               or fx.startswith("word_post="):
                preselection.add(self.fidx_map[fx])
        return preselection

    def create_masks(self, tokens=["pre", "cur", "post"]):
        pattern = re.compile("_|=")
        masks = {}
        for token in tokens:
            mask = set((i for i, fx in enumerate(self.fidx_map.iterkeys())
                                if token in pattern.split(fx)[1]))
            masks[token] = np.array(mask)
        return masks

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

    def create_feature_type_splits(self):
        """compute begin and end idx for each feature type.
        """
        feature_types = defaultdict(list)
        for i, fid in enumerate(self.vocabulary):
            ftype = fid.split("=")[0]
            feature_types[ftype].append(i)
        for key, value in feature_types.iteritems():
            value = np.array(value)
            feature_types[key] = (value.min(), value.max())
        return feature_types

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
        feature_type_splits = self.create_feature_type_splits()
        trainer = structlearn.auxtrainer.ElasticNetTrainer(0.00001, 0.85,
                                               10**6)
        strategy = structlearn.auxstrategy.HadoopTrainingStrategy()
        struct_learner = structlearn.StructLearner(k, dataset,
                                                   aux_tasks,
                                                   trainer,
                                                   strategy)
        # learn the embedding
        # TODO apply SVD by feature type.
        struct_learner.learn()

        # TODO post-process embedding
        # - project unlabeled data
        # - compute and store mean and std
        # FIXME Blitzer does this on training data only
        # This could be done in ner.py!

        # store data in model
        print
        print "size of theta: %.2f MB" % (struct_learner.thetat.nbytes
                                          / 1024.0 / 1024.0)
        print
        self.thetat = struct_learner.thetat
        self.m = m
        self.k = k
        return self

    def project_dataset(self, dataset, usestandardize=True, beta=1.0):
        """Project dataset onto the subspace induced by theta
        and concatenate the new features with the original
        features.

        Parameters
        ----------
        dataset : bolt.io.MemoryDataset
            The dataset to project.
        usestandardize : bool, True
            Whether or not the projected features should be standardized.
            If so mean and std are stored.
        beta : float, 1.0
            Scaling factor for the projected features.

        Returns
        -------
        dataset : bolt.io.MemoryDataset
            The new dataset with dataset.dim = dataset.dim
            + thetat.shape[1]
        """
        assert dataset.dim == self.thetat.shape[0]
        # get projection as dense array
        proj_dataset = structlearn.project(dataset,
                                           self.thetat,
                                           dense=True)

        if usestandardize:
            mean = proj_dataset.mean(axis=0)
            std = proj_dataset.std(axis=0)
            self.mean, self.std = mean, std
            self.beta = beta
            structlearn.standardize(proj_dataset, mean, std, beta)

        # from dense array to MemoryDataset
        proj_dataset = structlearn.to_sparse_bolt(proj_dataset)
        dim = self.thetat.shape[1]
        labels = dataset.labels
        proj_dataset = bolt.io.MemoryDataset(dim, proj_dataset, labels)
        proj_dataset._idx = dataset._idx

        # concat both MemoryDatasets
        new_dataset = structlearn.concat_datasets(dataset, proj_dataset)

        assert new_dataset.dim == dataset.dim + self.thetat.shape[1]
        return new_dataset

    def project_instance(self, instance, usestandardize=True):
        """Project instance onto the subspace induced by theta
        and concatenate the new features with the original
        features.

        Returns
        -------
        new_instance : array, dtype=bolt.sparsedtype
            The new instance with size instance.shape[0] + thetat.shape[1]
        """
        proj_instance = structlearn.project_instance_dense(instance,
                                                           self.thetat)
        if usestandardize:
            structlearn.standardize(proj_instance, self.mean, self.std,
                                    self.beta)
        proj_instance = bolt.dense2sparse(proj_instance)
        new_instance = structlearn.concat_instances(instance,
                                                    proj_instance,
                                                    self.thetat.shape[0])
        assert new_instance.shape[0] == instance.shape[0] + self.thetat.shape[1]
        return new_instance


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
    parser.add_option("--model",
                      dest="model",
                      default=False,
                      help="Re-use an existing model for feature extraction.",
                      metavar="str",
                      type="str")
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

    if options.model:
        model = compressed_load(options.model)
    else:
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
        # FIXME use minc+2 for unlabeled data.
        V, T = tagger.build_vocabulary(readers, fd, hd, minc=[options.minc,
                                                              options.minc + 2],
                                           use_eph=options.use_eph)
        print "|V|:", len(V)
        print "|T|:", len(T)
        print
        model = ASO(fd, hd, V, T, use_eph=options.use_eph)

    print "run model.learn"
    print "m:", options.m
    print "k:", options.k
    print
    model.learn(unlabeled_reader, options.m, options.k)

    # Store meta data in model
    model.lang = options.lang
    model.minc = options.minc

    # dump the model
    compressed_dump(f_model, model)
