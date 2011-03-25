#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
clscl
=====

"""
from __future__ import division

import sys
import numpy as np
import math
import optparse


from itertools import islice, ifilter
from functools import partial

from ..structlearn import pivotselection
from ..structlearn import util
from ..structlearn import structlearn
from ..structlearn import auxtrainer
from ..structlearn import auxstrategy
from ..io import compressed_dump, compressed_load
from ..bow import vocabulary, disjoint_voc, load
from ..util import timeit
from ..structlearn import standardize
from ..externals import bolt

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__version__ = "0.1"


class CLSCLModel(object):
    """

    Parameters
    ----------
    thetat : array, shape = [|voc|, k]
        Theta transposed.
    mean : array, shape = [|voc|]
        Mean value of each feature.
    std : array, shape = [|voc|]
        Standard deviation of each feature.
        Used to post-process the projection.
    avg_norm : float
        The average L2 norm of the training data.
        Used to post-process the projection.

    Attributes
    ----------
    `thetat` : array, shape = [|voc|, k]
        Theta transposed.
    `mean` : array, shape = [|voc|]
        Mean value of each feature.
    `std` : array, shape = [|voc|]
        Standard deviation of each feature.
        Used to post-process the projection.
    `avg_norm` : float
        The average L2 norm of the training data.
        Used to post-process the projection.
    `s_voc` : dict
        Source vocabulary.
    `t_voc` : dict
        Target vocabulary.

    """
    def __init__(self, thetat, mean=None, std=None, avg_norm=None):
        self.thetat = thetat
        self.s_voc = None
        self.t_voc = None
        self.mean = mean
        self.std = std
        self.avg_norm = avg_norm

    @timeit
    def project(self, ds):
        """Projects the given dataset onto the space induces by `self.thetat`
        and postprocesses the projection using `mean`, `std`, and `avg_norm`.

        Parameters
        ----------
        ds : bolt.io.MemoryDataset
            The dataset to be projected.

        Returns
        -------
        bolt.io.MemoryDataset
            A new bolt.io.MemoryDataset equal to `ds`
            but contains projected feature vectors.
        """
        dense_instances = structlearn.project(ds, self.thetat, dense=True)

        if self.mean != None and self.std != None:
            standardize(dense_instances, self.mean, self.std)
        if self.avg_norm != None:
            dense_instances /= self.avg_norm

        instances = structlearn.to_sparse_bolt(dense_instances)
        dim = self.thetat.shape[1]
        labels = ds.labels
        new_ds = bolt.io.MemoryDataset(dim, instances, labels)
        new_ds._idx = ds._idx
        return new_ds


class CLSCLTrainer(object):
    """Trainer class that creates CLSCLModel objects.

    Parameters
    ----------
    s_train : bolt.io.MemoryDataset
        Labeled training data in the source language.
    s_unlabeled : bolt.io.MemoryDataset
        Unlabeled data in the source language.
    t_unlabeled : bolt.io.MemoryDataset
        Unlabeled data in the target language.
    pivotselector : PivotSelector
        Pivot selector to select words from the source vocabulary
        for potential pivots.
    pivotselector : PivotTranslator
        Translates words from the source to the target vocabulary (1-1).
    trainer : AuxTrainer
        Trainer for the pivot classifiers.
    strategy : AuxStrategy
        Processing strategy for the pivot classifier training.

    Attributes
    ----------
    `s_train` : bolt.io.MemoryDataset
        Labeled training data in the source language.
    `s_unlabeled` : bolt.io.MemoryDataset
        Unlabeled data in the source language.
    `t_unlabeled` : bolt.io.MemoryDataset
        Unlabeled data in the target language.
    `pivotselector` : PivotSelector
        Pivot selector to select words from the source vocabulary
        for potential pivots.
    `pivotselector` : PivotTranslator
        Translates words from the source to the target vocabulary (1-1).
    `trainer` : AuxTrainer
        Trainer for the pivot classifiers.
    `strategy` : AuxStrategy
        Processing strategy for the pivot classifier training.
    """

    def __init__(self, s_train, s_unlabeled, t_unlabeled,
                 pivotselector, pivottranslator, trainer, strategy):
        self.s_train = s_train
        self.s_unlabeled = s_unlabeled
        self.t_unlabeled = t_unlabeled
        self.pivotselector = pivotselector
        self.pivottranslator = pivottranslator
        self.trainer = trainer
        self.strategy = strategy

    @timeit
    def select_pivots(self, m, phi):
        """Selects the pivots.
        First, it selects words from the source vocabulary using
        the `pivotselector` member. Than, the source words are translated
        using the `pivottranslator` member. Finally, the support condition
        is enforced by eliminating those pivot candidates which occur less
        then `phi` times in the unlabeled data. At most `m` pivots are
        selected.

        Parameter
        ---------
        m : int
            The desired number of pivots.
        phi : int
            The minimum support of a pivot in the unlabeled data.

        Returns
        -------
        list of arrays, len(list) <= m
            A list of arrays array([w_s, w_t]) where w_s is the vocabulary
            index of the source pivot word and w_t is the index of
            the target word.
            The number of pivots might be smaller than `m`.
        """
        vp = self.pivotselector.select(self.s_train)
        candidates = ifilter(lambda x: x[1] != None,
                             ((ws, self.pivottranslator[ws])
                              for ws in vp))
        counts = util.count(self.s_unlabeled, self.t_unlabeled)
        pivots = (np.array([ws, wt]) for ws, wt in candidates \
                         if counts[ws] >= phi and counts[wt] >= phi)
        pivots = [pivot for pivot in islice(pivots, m)]
        #terms = [(self.s_ivoc[ws],self.t_ivoc[wt]) for ws,wt in pivots]
        #for term in terms[:50]:
        #    print term

        return pivots

    def train(self, m, phi, k):
        """Trains the model using parameters `m`, `phi`, and `k`.

        Parameters
        ----------
        m : int
            Number of pivots.
        phi : int
            Minimum support of pivots in unlabeled data.
        k : int
            Dimensionality of the cross-lingual representation.

        Returns
        -------
        model : CLSCLModel
            The trained model.

        """
        pivots = self.select_pivots(m, phi)
        print("|pivots| = %d" % len(pivots))
        ds = bolt.io.MemoryDataset.merge((self.s_unlabeled,
                                          self.t_unlabeled))
        ds.shuffle(13)
        struct_learner = structlearn.StructLearner(k, ds, pivots,
                                                   self.trainer,
                                                   self.strategy)
        struct_learner.learn()
        self.project(struct_learner.thetat, verbose=1)
        return CLSCLModel(struct_learner.thetat, mean=self.mean,
                          std=self.std, avg_norm=self.avg_norm)

    @timeit
    def project(self, thetat, verbose=1):
        """Projects `s_train`, `s_unlabeled` and `t_unlabeled`
        onto the subspace induced by theta transposed, `thetat`,
        and post-processes the projected data.

        Post-processes the projected data by a) standardizing
        (0 mean, unit variance; where mean and variance are estimated from
        labeled and unlabeled data) and b) scaling by a factor beta
        such that the average L2 norm of the training examples
        equals 1.
        """
        s_train = structlearn.project(self.s_train, thetat, dense=True)
        s_unlabeled = structlearn.project(self.s_unlabeled, thetat,
                                          dense=True)
        t_unlabeled = structlearn.project(self.t_unlabeled, thetat,
                                          dense=True)

        data = np.concatenate((s_train, s_unlabeled, t_unlabeled))
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        self.mean, self.std = mean, std
        standardize(s_train, mean, std)

        norms = np.sqrt((s_train * s_train).sum(axis=1))
        avg_norm = np.mean(norms)
        self.avg_norm = avg_norm
        s_train /= avg_norm

        dim = thetat.shape[0]
        self.s_train.instances = structlearn.to_sparse_bolt(s_train)
        self.s_train.dim = dim

        del self.s_unlabeled
        del self.t_unlabeled


class DictTranslator(object):
    """Pivot translation oracle backed by a bilingual
    dictionary represented as a tab separated text file.
    """

    def __init__(self, dictionary, s_ivoc, t_voc):
        self.dictionary = dictionary
        self.s_ivoc = s_ivoc
        self.t_voc = t_voc
        print("debug: DictTranslator contains " \
              "%d translations." % len(dictionary))

    def __getitem__(self, ws):
        try:
            wt = self.normalize(self.dictionary[self.s_ivoc[ws]])
        except KeyError:
            wt = None
        return wt

    def translate(self, ws):
        return self[ws]

    def normalize(self, wt):
        wt = wt.encode("utf-8") if isinstance(wt, unicode) else wt
        wt = wt.split(" ")[0].lower()
        if wt in self.t_voc:
            return self.t_voc[wt]
        else:
            return None

    @classmethod
    def load(cls, fname, s_ivoc, t_voc):
        dictionary = []
        with open(fname) as f:
            for i, line in enumerate(f):
                ws, wt = line.rstrip().split("\t")
                dictionary.append((ws, wt))
        dictionary = dict(dictionary)
        return DictTranslator(dictionary, s_ivoc, t_voc)


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "s_lang t_lang s_train_file " \
                                   "s_unlabeled_file t_unlabeled_file " \
                                   "dict_file model_file",
                                   version="%prog " + __version__,
                                   description=description)

    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")

    parser.add_option("-k",
                      dest="k",
                      help="dimensionality of cross-lingual representation.",
                      default=100,
                      metavar="int",
                      type="int")

    parser.add_option("-m",
                      dest="m",
                      help="number of pivots.",
                      default=450,
                      metavar="int",
                      type="int")

    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                      "-1 for unlimited.",
                      default=-1,
                      metavar="int",
                      type="int")

    parser.add_option("-p", "--phi",
                      dest="phi",
                      help="minimum support of pivots.",
                      default=30,
                      metavar="int",
                      type="int")

    parser.add_option("-r", "--pivot-reg",
                      dest="preg",
                      help="regularization parameter lambda for " \
                      "the pivot classifiers.",
                      default=0.00001,
                      metavar="float",
                      type="float")

    parser.add_option("-a", "--alpha",
                      dest="alpha",
                      help="elastic net hyperparameter alpha.",
                      default=0.85,
                      metavar="float",
                      type="float")

    parser.add_option("--strategy",
                      dest="strategy",
                      help="The strategy to compute the pivot classifiers." \
                      "Either 'serial' or 'parallel' [default] or 'hadoop'.",
                      default="parallel",
                      metavar="str",
                      type="str")

    parser.add_option("--n-jobs",
                      dest="n_jobs",
                      help="The number of processes to fork." \
                      "Only if strategy is 'parallel'.",
                      default=-1,
                      metavar="int",
                      type="int")

    return parser


def train():
    """Training script for CLSCL.

    FIXME: works for binary classification only.
    TODO: multi-class classification.
    TODO: different translators.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 7:
        parser.error("incorrect number of arguments (use `--help` for help).")

    slang = argv[0]
    tlang = argv[1]

    fname_s_train = argv[2]
    fname_s_unlabeled = argv[3]
    fname_t_unlabeled = argv[4]
    fname_dict = argv[5]

    # Create vocabularies
    s_voc = vocabulary(fname_s_train, fname_s_unlabeled,
                       mindf=2,
                       maxlines=options.max_unlabeled)
    t_voc = vocabulary(fname_t_unlabeled,
                       mindf=2,
                       maxlines=options.max_unlabeled)
    s_voc, t_voc, dim = disjoint_voc(s_voc, t_voc)
    s_ivoc = dict([(idx, term) for term, idx in s_voc.items()])
    t_ivoc = dict([(idx, term) for term, idx in t_voc.items()])
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    print("|V| = %d" % dim)

    # Load labeled and unlabeled data
    s_train, classes = load(fname_s_train, s_voc, dim)
    s_unlabeled, _ = load(fname_s_unlabeled, s_voc, dim,
                       maxlines=options.max_unlabeled)
    t_unlabeled, _ = load(fname_t_unlabeled, t_voc, dim,
                       maxlines=options.max_unlabeled)
    print("classes = {%s}" % ",".join(classes))
    print("|s_train| = %d" % s_train.n)
    print("|s_unlabeled| = %d" % s_unlabeled.n)
    print("|t_unlabeled| = %d" % t_unlabeled.n)

    # Load dictionary
    translator = DictTranslator.load(fname_dict, s_ivoc, t_voc)

    pivotselector = pivotselection.MISelector()
    trainer = auxtrainer.ElasticNetTrainer(options.preg, options.alpha,
                                           10**6)
    strategy_factory = {"hadoop": auxstrategy.HadoopTrainingStrategy,
                        "serial": auxstrategy.SerialTrainingStrategy,
                        "parallel": partial(auxstrategy.ParallelTrainingStrategy,
                                            n_jobs=options.n_jobs)}
    clscl_trainer = CLSCLTrainer(s_train, s_unlabeled,
                                 t_unlabeled, pivotselector,
                                 translator, trainer,
                                 strategy_factory[options.strategy]())
    clscl_trainer.s_ivoc = s_ivoc
    clscl_trainer.t_ivoc = t_ivoc
    model = clscl_trainer.train(options.m, options.phi, options.k)

    model.s_voc = s_voc
    model.t_voc = t_voc
    compressed_dump(argv[6], model)


def predict():
    """Prediction script for CLSCL.

    Usage: ./clscl_predict strain model ttest reg
    """
    argv = sys.argv[1:]
    if len(argv) != 4:
        print "Error: wrong number of arguments. "
        print "Usage: %s strain model ttest reg" % sys.argv[0]
        sys.exit(-2)

    fname_s_train = argv[0]
    fname_model = argv[1]
    fname_t_test = argv[2]
    reg = float(argv[3])

    model = compressed_load(fname_model)

    s_voc = model.s_voc
    t_voc = model.t_voc
    dim = len(s_voc) + len(t_voc)
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    print("|V| = %d" % dim)

    s_train, classes = load(fname_s_train, s_voc, dim)
    t_test, _ = load(fname_t_test, t_voc, dim)

    print("classes = {%s}" % ",".join(classes))
    n_classes = len(classes)

    cl_train = model.project(s_train)
    cl_test = model.project(t_test)

    cl_train.shuffle(1)
    epochs = int(math.ceil(10**6 / cl_train.n))
    loss = bolt.ModifiedHuber()
    sgd = bolt.SGD(loss, reg, epochs=epochs, norm=2)
    if n_classes == 2:
        model = bolt.LinearModel(cl_train.dim, biasterm=False)
        trainer = sgd
    else:
        model = bolt.GeneralizedLinearModel(cl_train.dim, n_classes,
                                            biasterm=False)
        trainer = bolt.trainer.OVA(sgd)
        
    trainer.train(model, cl_train, verbose=0, shuffle=False)
    acc = 100.0 - bolt.eval.errorrate(model, cl_test)
    print "ACC: %.2f" % acc
