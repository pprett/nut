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
from __future__ import division

import sys
import optparse
import numpy as np
import re
import gc

from itertools import islice
from pprint import pprint

from ..io import conll, compressed_dump, compressed_load
from ..tagger import tagger
from ..structlearn.pivotselection import FreqSelector
from ..nut import structlearn
from ..util import timeit, sizeof
from ..structlearn.util import count
from ..externals import bolt

__version__ = "0.1"


class ASO(object):
    """Alternating structural optimization for named entity recognition.
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

        print "sizeof(vocabulary): %.1f MB" % sizeof(self.vocabulary)
        print "sizeof(fidx_map):   %.1f MB" % sizeof(self.fidx_map)

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
        print "_" * 80
        print "Masked features stats"
        print
        print "check:",
        try:
            for fname in self.fidx_map.iterkeys():
                fname.index("=")
        except ValueError:
            print "failed!"
            print fname
            assert False
        else:
            print "succeeded!"

        for token in tokens:
            masked_features = set((idx for fname, idx in self.fidx_map.iteritems()
                                   if token in pattern.split(fname)[1:-1]))
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
        print "_" * 80
        print "Auxiliary problem stats"
        print
        for prefix in ["uword_cur=", "uword_pre=",
                       "uword_post="]:
            preselection = self.preselect_tasks(prefix)
            tmp = list(islice(FreqSelector(0).select(dataset, preselection), m))
            print ", ".join(["%s (%d)" % (self.vocabulary[e], counts[e])
                             for e in tmp[:10]])
            print
            aux_tasks.extend(tmp)

        print "|auxtasks| = %d" % len(aux_tasks)
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
        feature_types = {}
        min_i = -1
        current_ftype = "DUMMY"

        for i, fid in enumerate(self.vocabulary):
            ftype = fid[:fid.index("=")]
            if ftype != current_ftype:
                feature_types[current_ftype] = (min_i, i - 1)
                current_ftype = ftype
                min_i = i

        feature_types[current_ftype] = (min_i, i)
        del feature_types["DUMMY"]

        indices = sorted(feature_types.values())
        for i in xrange(len(indices) - 1):
            assert indices[i][1] + 1 == indices[i+1][0]
        return dict(feature_types)

    def build_examples(self, reader, verbose=0):
        dataset = tagger.build_examples(reader, self.fd, self.hd,
                                        self.fidx_map, self.tags,
                                        pos_prefixes=["NN", "JJ"],
                                        verbose=verbose)
        return dataset

    def learn(self, m, k, verbose=0, n_jobs=1):
        """Learns the ASO embedding theta.

        The learned embedding is stored in `self.struct_learner.thetat`.
        To project new data use :meth:`self.struct_learner.project`.

        Parameters
        ----------
        m : int
            The number of auxiliary tasks (per task type).
        k : int
            The dimensionaly of the shared representation.
        verbose : int
            Verbose output.
        n_jobs : int
            The number of processes to fork (via joblib).
        Returns
        -------
        self
        """
        print "run ASO.learn..."

        dataset = self.dataset
        print "|examples|: %d" % dataset.n
        print "|features|: %d" % dataset.dim
        print "Size of dataset: %.4f MB" % sizeof(dataset)
        aux_tasks, task_masks = self.create_aux_tasks(dataset, m)
        print "_" * 80

        # self.print_tasks(aux_tasks)
        feature_types = self.create_feature_type_splits()
        print "_" * 80
        print "Feature types:"
        pprint(feature_types)
        print "total number: %d" % len(feature_types)
        print
        self.feature_types = feature_types

        #trainer = structlearn.auxtrainer.ElasticNetTrainer(0.00001, 0.85,
        #                                                   10**7)

        trainer = structlearn.auxtrainer.L2Trainer(0.00001, 10**6,
                                                   truncate=True)

        #strategy = structlearn.auxstrategy.HadoopTrainingStrategy()
        strategy = structlearn.auxstrategy.ParallelTrainingStrategy(n_jobs=n_jobs)

        dataset.shuffle(9)
        struct_learner = structlearn.StructLearner(k, dataset,
                                                   aux_tasks,
                                                   trainer,
                                                   strategy,
                                                   task_masks=task_masks,
                                                   useinvertedindex=False,
                                                   feature_types=self.feature_types)

        # FIXME store W
        store_W = False  ## True
        # learn the embedding
        struct_learner.learn(store_W=store_W)

        ##struct_learner.print_W_cols(range(len(aux_tasks)), self.vocabulary)

        # store data in model
        print
        print "size of theta: %.2f MB" % sizeof(struct_learner.thetat)
        print
        del struct_learner.dataset
        self.struct_learner = struct_learner
        self.m = m
        self.k = k
        self.input_dim = struct_learner.thetat.shape[0]
        self.embedding_dim = struct_learner.thetat.shape[1] * \
                             struct_learner.feature_type_split.shape[0]
        return self

    def post_process(self, dataset):
        """Post process the model.

        Computes mean and std of new spectral features by first projecting
        the unlabeled dataset and then computing mean and std deviation.
        The resulting values are stored in the model.

        Parameters
        ----------
        dataset : bolt.io.MemoryDataset
            The unlabeled data.

        Returns
        -------
        self
        """
        print "_" * 80
        print "Post process embedding"
        print
        proj_dataset = self.struct_learner.project(dataset, dense=True)
        mean = proj_dataset.mean(axis=0)
        std = proj_dataset.std(axis=0)
        self.mean = mean
        self.std = std
        return self

    def set_scaling_factor(self, dataset, proj_dataset):
        print "compute scaling factor"
        sparse_norm = sum((np.linalg.norm(x['f1'])
                           for x in dataset.iterinstances()))
        sparse_norm /= dataset.n
        print "sparse_norm:", sparse_norm

        norms = [np.linalg.norm(x) for x in proj_dataset]
        dense_norm = np.mean(norms)
        print "dense_norm:", dense_norm
        scaling_factor = sparse_norm / dense_norm
        
        scaling_factor *= 10.0  # FIXME
        print "scaling factor: %.4f" % scaling_factor

        # store scaling factor for future use
        self.scaling_factor = scaling_factor

    @timeit
    def project_dataset(self, dataset):
        """Project dataset onto the subspace induced by theta
        and concatenate the new features with the original
        features.

        Parameters
        ----------
        dataset : bolt.io.MemoryDataset
            The dataset to project.
        
        Returns
        -------
        dataset : bolt.io.MemoryDataset
            The new dataset with dataset.dim = dataset.dim
            + thetat.shape[1]
        """
        assert dataset.dim == self.input_dim
        # get projection as dense array
        proj_dataset = self.struct_learner.project(dataset,
                                                   dense=True)

        dim = proj_dataset.shape[1]
        assert proj_dataset.shape[0] == dataset.n
        assert proj_dataset.shape[1] == self.embedding_dim

        if not (hasattr(self, "mean") or hasattr(self, "std")):
            self.mean = proj_dataset.mean(axis=0)
            std = proj_dataset.std(axis=0)
            std[std == 0.0] = 1.0
            self.std = std
            
        structlearn.standardize(proj_dataset, self.mean, self.std, 1.0)

        ## if not hasattr(self, "scaling_factor"):
##             self.set_scaling_factor(dataset, proj_dataset)

##         proj_dataset *= self.scaling_factor

        # from dense array to MemoryDataset
        proj_dataset = structlearn.to_sparse_bolt(proj_dataset)
        
        proj_dataset = bolt.io.MemoryDataset(dim, proj_dataset, dataset.labels)
        proj_dataset._idx = dataset._idx
        
        # concat both MemoryDatasets
        new_dataset = structlearn.concat_datasets(dataset, proj_dataset)

        assert new_dataset.dim == (dataset.dim + self.embedding_dim)
        
        return new_dataset

    def project_instance(self, instance):
        """Project instance onto the subspace induced by theta
        and concatenate the new features with the original
        features.

        Returns
        -------
        new_instance : array, dtype=bolt.sparsedtype
            The new instance.
        """
        proj_instance = self.struct_learner.project_instance_dense(instance)
        if hasattr(self, "mean") and hasattr(self, "std"):
            structlearn.standardize(proj_instance, self.mean, self.std)

        if hasattr(self, "scaling_factor"):
            proj_instance *= self.scaling_factor

        proj_instance = bolt.dense2sparse(proj_instance)
        return structlearn.concat_instances(instance,
                                            proj_instance,
                                            self.input_dim)


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = "Training script for Alternating Structural Optimization."
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
                      " `fd` and `hd` functions. [default %default].",
                      default="rr09_aso",
                      metavar="str",
                      type="str")
    parser.add_option("-m",
                      dest="m",
                      help="Number of auxiliary tasks from left, right, " \
                      "and current word to create [default %default]. ",
                      default=1000,
                      metavar="int",
                      type="int")
    parser.add_option("-k",
                      dest="k",
                      help="Dimensionality of the shared representation " \
                      "[default %default].",
                      default=50,
                      metavar="int",
                      type="int")
    parser.add_option("-r", "--reg",
                      dest="reg",
                      help="regularization parameter [default %default]. ",
                      default=0.00001,
                      metavar="float",
                      type="float")
    parser.add_option("-E", "--epochs",
                      dest="epochs",
                      help="Number of training epochs [default %default]. ",
                      default=100,
                      metavar="int",
                      type="int")
    parser.add_option("--n-jobs",
                      dest="n_jobs",
                      help="Number of cpus to use [default %default].",
                      default=3,
                      metavar="int",
                      type="int")
    parser.add_option("--min-count",
                      dest="minc",
                      help="min number of occurances [default %default].",
                      default=1,
                      metavar="int",
                      type="int")
    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                      "-1 for unlimited [default %default].",
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
    parser.add_option("--no-learn",
                      action="store_true",
                      dest="no_learn",
                      default=False,
                      help="Just run build_vocabulary and store model.")
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

    train_reader = conll.Conll03Reader(f_train, options.lang)
    unlabeled_reader = conll.Conll03Reader(f_unlabeled, options.lang,
                                           raw=True)

    if options.model:
        print "_" * 80
        print "Load model"
        model = compressed_load(options.model)
    else:
        print "_" * 80
        print "Build vocabulary"
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

        # FIXME use minc+2 for unlabeled data.
        V, T = tagger.build_vocabulary([train_reader, unlabeled_reader],
                                       fd, hd, minc=[options.minc,
                                                     options.minc + 4],
                                       use_eph=options.use_eph,
                                       verbose=options.verbose,
                                       pos_prefixes=[[], ["NN", "JJ"]])
        print "|V|:", len(V)
        print "|T|:", len(T)
        print
        model = ASO(fd, hd, V, T, use_eph=options.use_eph)

    gc.collect()
    if not hasattr(model, "dataset"):
        print "_" * 80
        print "Build examples"
        print
        dataset = model.build_examples(unlabeled_reader,
                                       verbose=options.verbose)
        print "Size of dataset: %.2f MB" % sizeof(dataset)
        model.dataset = dataset

    gc.collect()
    if not options.no_learn:
        print "_" * 80
        print "Learn theta"
        print "m:", options.m
        print "k:", options.k
        print
        model.learn(options.m, options.k, verbose=options.verbose,
                    n_jobs=options.n_jobs)
        # model.post_process(model.dataset)
        del model.dataset

    # Store meta data in model
    model.lang = options.lang
    model.minc = options.minc

    gc.collect()
    # dump the model
    print "_" * 80
    print "Dump model"
    raw_input("hit any key to continue.")
    compressed_dump(f_model, model)
    print "model dumped to %s." % f_model
