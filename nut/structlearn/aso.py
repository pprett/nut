
from __future__ import division

import numpy as np
import optparse
import gc

from itertools import islice
from functools import partial

#FIXME
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn import cross_validation
from sklearn import metrics
from sklearn.utils import shuffle

from ..structlearn import pivotselection
from ..structlearn import util
from ..structlearn import structlearn
from ..structlearn import auxtrainer
from ..structlearn import auxstrategy
from ..io import compressed_dump, compressed_load
from ..util import timeit
from ..structlearn import standardize
from ..externals import bolt
from ..externals.joblib import Parallel, delayed

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__version__ = "0.1"


@timeit
def normalize(ds):
    """Lenght normalize the instances in `ds`."""
    for x in ds.instances:
        norm = np.linalg.norm(x['f1']) 
        if norm > 0.0:
            x['f1'] /= norm


class ASOModel(object):
    """

    Parameters
    ----------
    struct_learner : StructLearner
        A trained StructLearner object.
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
    `struct_learner` : StructLearner
        The structLearner object which holds theta
    `mean` : array, shape = [|voc|]
        Mean value of each feature.
    `std` : array, shape = [|voc|]
        Standard deviation of each feature.
        Used to post-process the projection.
    `avg_norm` : float
        The average L2 norm of the training data.
        Used to post-process the projection.

    """
    def __init__(self, struct_learner, mean=None, std=None, avg_norm=None):
        self.struct_learner = struct_learner
        # self.thetat = thetat
        self.mean = mean
        self.std = std
        self.avg_norm = avg_norm

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
        struct_learner = self.struct_learner
        dense_instances = struct_learner.project(ds, dense=True)

        if self.mean != None and self.std != None:
            standardize(dense_instances, self.mean, self.std)
        if self.avg_norm != None:
            dense_instances /= self.avg_norm

        return dense_instances
        

##         instances = structlearn.to_sparse_bolt(dense_instances)
##         dim = struct_learner.thetat.shape[1] * \
##               struct_learner.feature_type_split.shape[0]
##         labels = ds.labels
##         new_ds = bolt.io.MemoryDataset(dim, instances, labels)
##         new_ds._idx = ds._idx
##         return new_ds


class ASOTrainer(object):
    """Trainer class that creates ASOModel objects.

    Parameters
    ----------
    train : bolt.io.MemoryDataset
        Labeled training data
    unlabeled : bolt.io.MemoryDataset
        Unlabeled data
    pivotselector : PivotSelector
        Pivot selector
    trainer : AuxTrainer
        Trainer for the pivot classifiers.
    strategy : AuxStrategy
        Processing strategy for the pivot classifier training.

    Attributes
    ----------
    `train` : bolt.io.MemoryDataset
        Labeled training data
    `unlabeled` : bolt.io.MemoryDataset
        Unlabeled data
    `pivotselector` : PivotSelector
        Pivot selecto
    `trainer` : AuxTrainer
        Trainer for the pivot classifiers.
    `strategy` : AuxStrategy
        Processing strategy for the pivot classifier training.
    """

    def __init__(self, train, unlabeled, pivotselector, trainer, strategy,
                 verbose=0):
        self._train = train
        self.unlabeled = unlabeled
        self.pivotselector = pivotselector

        self.trainer = trainer
        self.strategy = strategy
        self.verbose = verbose

    @timeit
    def select_pivots(self, m, phi):
        """Selects the pivots.
        First, it selects words from the  vocabulary using
        the `pivotselector` member. Then, the support condition
        is enforced by eliminating those  candidates which occur less
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
        list of ints, len(list) <= m
            A list of int values [p_0, ..., p_m-1].
            NOTE: The number of pivots might be smaller than `m`.
        """
        if self.verbose > 1:
            print("Call pivotselector.select...")
        candidates = self.pivotselector.select(self._train)

        ## FIXME some features are really common
        upper_phi = 1000000  #250000

        counts = util.count(self.unlabeled)
        pivots = (p for p in candidates if (counts[p] >= phi and counts[p] < upper_phi))
        pivots = np.array([p for p in islice(pivots, m)], dtype=np.int32)

        #from IPython import embed
        #embed()

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
        #ds = bolt.io.MemoryDataset.merge((self.unlabeled,
        #                                  self._train))

        self.unlabeled.shuffle(13)

        struct_learner = structlearn.StructLearner(k, self.unlabeled, pivots,
                                                   self.trainer,
                                                   self.strategy,
                                                   useinvertedindex=False)
        struct_learner.learn()

        gc.collect()
        self.project(struct_learner, verbose=1)
        del struct_learner.dataset
        
        return ASOModel(struct_learner, mean=self.mean,
                        std=self.std, avg_norm=self.avg_norm)

    @timeit
    def project(self, struct_learner, verbose=1):
        """Projects `train` (and `unlabeled`)
        onto the subspace induced by theta transposed,
        `struct_learner.thetat`, and post-processes the projected data.

        Post-processes the projected data by a) standardizing
        (0 mean, unit variance; where mean and variance are estimated from
        labeled and unlabeled data) and b) scaling by a factor beta
        such that the average L2 norm of the training examples
        equals 1.
        """
        train = struct_learner.project(self._train, dense=True)
        #unlabeled = struct_learner.project(self.unlabeled,
        #                                   dense=True)

        #data = np.concatenate((train, unlabeled))
        data = train  # FIXME rm
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        std[std == 0.0] = 1.0
        self.mean, self.std = mean, std
        standardize(train, mean, std)
        ## FIXME should we use norms?
##         norms = np.sqrt((train * train).sum(axis=1))
##         avg_norm = np.mean(norms)
##         self.avg_norm = avg_norm
##         train /= avg_norm
        self.avg_norm = None

        #dim = struct_learner.thetat.shape[1] * \
        #      struct_learner.feature_type_split.shape[0]
        #self._train.instances = structlearn.to_sparse_bolt(train)
        #self._train.dim = dim


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """train and unlabeled file are expected
    to be svmlight format. """
    
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_data_file "\
                                   "unlabeled_file model_file",
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
                      help="dimensionality of the representation.",
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
                      default=1000,
                      metavar="int",
                      type="int")

    parser.add_option("-r", "--pivot-reg",
                      dest="preg",
                      help="regularization parameter lambda for " \
                      "the pivot classifiers.",
                      default=0.001,
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
                      "Either 'serial' [default] or 'parallel' or 'hadoop'.",
                      default="serial",
                      metavar="str",
                      type="str")

    parser.add_option("--n-jobs",
                      dest="n_jobs",
                      help="The number of processes to fork." \
                      "Only if strategy is 'parallel'.",
                      default=1,
                      metavar="int",
                      type="int")

    return parser


def train():
    """Training script for ASO. """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 3:
        parser.error("incorrect number of arguments (use `--help` for help).")

    fname_train = argv[0]
    fname_unlabeled = argv[1]

    # Load labeled and unlabeled data
    train = bolt.MemoryDataset.load(fname_train)
    unlabeled = bolt.MemoryDataset.load(fname_unlabeled)
    print train.dim, unlabeled.dim
    dim = 1000000
    train.dim = dim
    unlabeled.dim = dim

    # FIXME make sure train and unlabeled are L2 normalized
    #normalize(train)
    #normalize(unlabeled)

    print("|V| = %d" % dim)
    print("|train| = %d" % train.n)
    print("|unlabeled| = %d" % unlabeled.n)

    pivotselector = pivotselection.MISelector()
    trainer = auxtrainer.ElasticNetTrainer(options.preg, options.alpha,
                                           10.0**6)
    strategy_factory = {"hadoop": auxstrategy.HadoopTrainingStrategy,
                        "serial": auxstrategy.SerialTrainingStrategy,
                        "parallel": partial(auxstrategy.ParallelTrainingStrategy,
                                            n_jobs=options.n_jobs)}
    aso_trainer = ASOTrainer(train, unlabeled, pivotselector, trainer,
                             strategy_factory[options.strategy](),
                             verbose=options.verbose)
    
    # now trainer is the only one to hold a ref
    del unlabeled
    model = aso_trainer.train(options.m, options.phi, options.k)
    
    compressed_dump(argv[2], model)


def predict_args_parser():
    """Create argument and option parser for the
    prediction script.
    """
    description = """Prefixes `s_` and `t_` refer to source and target language
    , resp. Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file " \
                                   "model_file " \
                                   "test_file",
                                   version="%prog " + __version__,
                                   description=description)

    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")

    parser.add_option("-R", "--repetition",
                      dest="repetition",
                      help="Repeat training `repetition` times and " \
                      "report avg. error.",
                      default=10,
                      metavar="int",
                      type="int")

    parser.add_option("-r", "--reg",
                      dest="reg",
                      help="regularization parameter lambda. ",
                      default=0.01,
                      metavar="float",
                      type="float")

    parser.add_option("--n-jobs",
                      dest="n_jobs",
                      help="The number of processes to fork.",
                      default=1,
                      metavar="int",
                      type="int")

    return parser


def clone(my_object):
    """Returns a deep copy of `my_object`. """
    return copy.deepcopy(my_object)


def predict():
    """Prediction script for ASO.  """
    parser = predict_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 3:
        parser.error("incorrect number of arguments (use `--help` for help).")

    fname_train = argv[0]
    fname_model = argv[1]
    fname_test = argv[2]
    reg = float(options.reg)

    model = compressed_load(fname_model)
    
    print("theta^T: ", model.struct_learner.thetat.shape)

    train = bolt.MemoryDataset.load(fname_train)
    test = bolt.MemoryDataset.load(fname_test)
    assert train.dim == test.dim

##     model.mean = None
##     model.std = None
    print "mean:", model.mean
    print "std:", model.std
    
    train_dense = model.project(train)
    test_dense = model.project(test)

    ## FIXME should we use norms?
##     norms = np.sqrt((train_dense * train_dense).sum(axis=1))
##     avg_norm = np.mean(norms)
##     print "avg_norm", avg_norm
##     train_dense /= avg_norm
##     test_dense /= avg_norm

    print("train-mean:", train_dense.mean(axis=0))
    print("train-std:", train_dense.std(axis=0))

    print "_" * 80

    # copy original data because we need the precise order later
    X = train_dense.copy()
    y = train.labels.copy()

    X, y = shuffle(X, y, random_state=13)
    
    print "X.shape", X.shape, y.shape
    clf = LinearSVC(loss='l2', penalty='l2', C=1.0,
                    dual=False, tol=1e-4)
    #clf = SVC(kernel='linear', C=1.0, probability=True, shrinking=False)
    #clf = SGDClassifier(loss='modified_huber', penalty='l2',
    #                    n_iter=20, alpha=0.001)

    cv = cross_validation.StratifiedKFold(y, 3)


    print "_" * 80
    print clf
    print
    aucs, zos = [], []
    for train_idx, test_idx in cv:
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf.fit(X_train, y_train)
        #scores = clf.predict_proba(X_test)[:,1].ravel()
        scores = clf.decision_function(X_test)
        y_pred = clf.predict(X_test)
        fpr, tpr, thresholds = metrics.roc_curve(y_test, scores)
        aucs.append(metrics.auc(fpr, tpr))
        zos.append(metrics.zero_one_score(y_test, y_pred))

    print np.mean(aucs), np.std(aucs)

    print "_" * 80

    #from IPython import embed
    #embed()

    #np.savetxt('transformed_public_train.dat', train_dense, delimiter=",", fmt="%.6f")
    #np.savetxt('transformed_public_test.dat', test_dense, delimiter=",", fmt="%.6f")


    # Now prepare the submission
    print("Make submission")
    X_train = train_dense
    y_train = train.labels
    X_test = test_dense
    
    clf = LinearSVC(loss='l2', penalty='l2', C=1.0,
                    dual=False, tol=1e-4)
    clf.fit(X_train, y_train)
    
    #from IPython import embed
    #embed()
    
    scores = clf.decision_function(X_test)
    submission_data = np.hstack((scores, test_dense))
    np.savetxt('submission.csv', submission_data, delimiter=",", fmt="%.6f")
    
    print("fin")
    
##     #test = model.project(test)

##     del model  # free model

##     epochs = int(math.ceil(10.0**6 / train.n))
##     loss = bolt.ModifiedHuber()
##     sgd = bolt.SGD(loss, reg, epochs=epochs, norm=2)
##     if n_classes == 2:
##         model = bolt.LinearModel(train.dim, biasterm=False)
##         trainer = sgd
##     else:
##         model = bolt.GeneralizedLinearModel(train.dim, n_classes,
##                                             biasterm=False)
##         trainer = bolt.trainer.OVA(sgd)

##     scores = Parallel(n_jobs=options.n_jobs, verbose=options.verbose)(
##                 delayed(_predict_score)(i, trainer, clone(model), train, test)
##         for i in range(options.repetition))
##     print "ACC: %.2f (%.2f)" % (np.mean(scores), np.std(scores))


## def _predict_score(i, trainer, model, train, test):
##     train.shuffle(i)
##     trainer.train(model, train, verbose=0, shuffle=False)
##     return 100.0 - bolt.eval.errorrate(model, test)
