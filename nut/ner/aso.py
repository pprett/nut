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

from collections import defaultdict
from itertools import islice
from pprint import pprint

from ..io import conll, compressed_dump, compressed_load
from ..tagger import tagger
from ..structlearn.pivotselection import FreqSelector
from ..nut import structlearn
from ..util import timeit
from ..structlearn.util import count
from ..externals import bolt

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

    def preselect_tasks(self, prefix):
        preselection = set()
        for fx in self.fidx_map:
            if fx.startswith(prefix):
                preselection.add(self.fidx_map[fx])
        return preselection

    @timeit
    def create_masks(self, tokens=["pre", "cur", "post"]):
        """Create the masks for the three auxiliary task types:
        pre, cur, and post.

        each feature name is assumed to adher to the following structure:
        '<type>_<info>_<loc>+=<val>', where <loc> indicate
        the token locations from which the feature is derived.

        To mask all features which are derived from location <loc> we simply
        split the feature names by underscore ('_') and than check whether
        the task type occurs in the <loc> list.

        Parameters
        ----------
        tokens : list, ['pre', 'cur', 'post']
            The token locations to mask.

        Returns
        -------
        masks : dict
            A dict mapping a token (e.g. 'pre') to an array containing the
            masked feature indizes.
        """
        pattern = re.compile("_|=")
        masks = {}
        print "_"*80
        print "Masked features stats"
        print 
        for token in tokens:
            masked_features = set((idx for fname, idx in self.fidx_map.iteritems()
                                   if token in pattern.split(fname)[2:]))
            print "%s has %d masked features." % (token, len(masked_features))
            masks[token] = np.unique(masked_features)

        return masks

    def create_aux_tasks(self, dataset, m):
        """Select the m most frequent left, right, and current words.

        TODO: mask all features derived from the left, right, and
        current word.

        Parameters
        ----------
        m : int
            The number of tasks for left, right, and current
            word to generate.

        Returns
        -------
        aux_tasks : list, len(list)==3*m
            The list of auxiliary tasks. These correspond to features
            which will subsequently be used to autolabel the unlabeled data.
        task_masks : list, len(list)==3*m
            A list of feature tuples. The i-th tuple holds the masked features
            for the i-th auxiliary task.
        """
        # get m most frequent current, left, and right words
        aux_tasks = []
        counts = count(dataset)
        print "_"*80
        print "Auxiliary problem stats"
        print
        for prefix in ["word_unigram_cur=", "word_unigram_pre=",
                       "word_unigram_post="]:
            preselection = self.preselect_tasks(prefix)
            tmp = list(islice(FreqSelector(0).select(dataset, preselection), m))
            print ", ".join(["%s (%d)" % (self.vocabulary[tmp[i]], counts[tmp[i]])
                             for i in range(10)])
            print
            aux_tasks.extend(tmp)

        masks = self.create_masks()
        task_masks = [masks["cur"]]*m + [masks["pre"]]*m + [masks["post"]]*m
        assert len(task_masks) == len(aux_tasks)
        return aux_tasks, task_masks

    def print_tasks(self, tasks):
        for task in tasks:
            print self.vocabulary[task]

    def create_feature_type_splits(self):
        """compute begin and end idx for each feature type.
        """
        pattern = re.compile("_|=")
        feature_types = defaultdict(list)
        for i, fid in enumerate(self.vocabulary):
            ftype = pattern.split(fid, 1)[0]
            feature_types[ftype].append(i)
        for key, value in feature_types.iteritems():
            value = np.array(value)
            feature_types[key] = (value.min(), value.max())

        indices = sorted(feature_types.values())
        for i in range(len(indices)-1):
            assert indices[i][1]+1 == indices[i+1][0]
        return dict(feature_types)

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
        aux_tasks, task_masks = self.create_aux_tasks(dataset, m)
        print "_"*80
        print "|examples|: %d" % dataset.n
        print "|aux_tasks|: %d" % len(aux_tasks)
        print "|task_masks|: %d" % len(task_masks)
        
        # self.print_tasks(aux_tasks)
        feature_types = self.create_feature_type_splits()
        print "_"*80
        print "Feature types:"
        pprint(feature_types)
        print
        self.feature_types = feature_types

        #trainer = structlearn.auxtrainer.ElasticNetTrainer(0.00001, 0.85,
        #                                                   10**7)

        trainer = structlearn.auxtrainer.L2Trainer(0.00001, 10**6,
                                                   truncate=True)

        strategy = structlearn.auxstrategy.HadoopTrainingStrategy()
        #strategy = structlearn.auxstrategy.ParallelTrainingStrategy(n_jobs=3)

        dataset.shuffle(9)
        struct_learner = structlearn.StructLearner(k, dataset,
                                                   aux_tasks,
                                                   trainer,
                                                   strategy,
                                                   task_masks=task_masks,
                                                   useinvertedindex=False,
                                                   feature_types=self.feature_types)
        # learn the embedding
        struct_learner.learn()

        # store data in model
        print
        print "size of theta: %.2f MB" % (struct_learner.thetat.nbytes
                                          / 1024.0 / 1024.0)
        print
        self.thetat = struct_learner.thetat
        self.m = m
        self.k = k
        return self

    def project_dataset(self, dataset, usestandardize=True, useavgnorm=True,
                        beta=1.0):
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
        useavgnorm : bool, True
            Whether to scale the projected features such that their L2 norm
            equals the L2 norm of the sparse features.
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
            self.mean = mean
            self.std = std
            structlearn.standardize(proj_dataset, mean, std, 1.0)

        if useavgnorm:
            sparse_norm = sum((np.linalg.norm(x['f1'])
                            for x in dataset.iterinstances()))
            sparse_norm /= dataset.n
            print "sparse_norm:", sparse_norm
            norms = np.sqrt((proj_dataset * proj_dataset).sum(axis=1))
            dense_norm = np.mean(norms)
            print "dense_norm:", dense_norm
            scaling_factor = sparse_norm / dense_norm

            # scale dense features and store scaling factor for future use
            proj_dataset *= scaling_factor
            self.scaling_factor = scaling_factor

            ## recompute dense norm and assert if equal to sparse norm
            #norms = np.sqrt((proj_dataset * proj_dataset).sum(axis=1))
            #dense_norm = np.mean(norms)
            #np.testing.assert_almost_equal(dense_norm, sparse_norm, decimal=4)

        # we have to scale by beta at last because otherwise
        # the above assert will break.
        self.beta = beta
        proj_dataset *= beta

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

    def project_instance(self, instance, usestandardize=True, useavgnorm=True):
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
            # save to scale by beta before average norm scaling factor.
            structlearn.standardize(proj_instance, self.mean, self.std,
                                    self.beta)

        if useavgnorm:
            proj_instance *= self.scaling_factor

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
                                                              options.minc + 4],
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
