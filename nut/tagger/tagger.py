# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""
tagger
======

A module for sequence taggers. Currently, only greedy left to right
taggers are available.

The following classes are provided:

    * AvgPerceptronTagger
    * GreedySVMTagger
"""
import sys
import numpy as np
import gc
from ..externals import bolt

from collections import defaultdict
from itertools import chain, repeat, izip
from ..util import timeit, sizeof
from ..ner.nonlocal import ExtendedPredictionHistory

# FIXME only for ASO
MAX_TOKENS = 2200000


@timeit
def build_vocabulary(readers, detector, minc=1, use_eph=False,
                     verbose=0, pos_prefixes=[], pos_idx=1):
    """
    Arguments
    ---------
    readers : list or ConllReader
        The input data reader(s).
    detector : Detector
        The detector class providing feature and histor detection
        functions.
    minc : int or list
        The minimum feature count to be included in the vocabulary.
        If `minc` is a list, the i-th item is the min count of the i-th
        reader in `readers`.
    use_eph : bool
        Whether or not extended prediction history should be used.
    verbose : int
        Verbose output (report when processed 100k lines).

    Returns
    -------
    vocabulary : list
        A sorted list of feature names.
    tags : list
        The list of tags.
    """
    if not isinstance(readers, list):
        readers = [readers]
    if isinstance(minc, list):
        assert len(readers) == len(minc)
        minc = iter(minc)
    else:
        minc = repeat(minc)

    pp_check = [isinstance(e, list) for e in pos_prefixes]
    if len(pp_check) > 0 and all(pp_check):
        assert len(pos_prefixes) == len(readers)
        pos_prefixes = iter(pos_prefixes)
    else:
        pos_prefixes = repeat(pos_prefixes)

    tags = set()
    i = 0
    all_vocabulary = set()
    for reader, minc, pos_prefixes in izip(readers, minc, pos_prefixes):
        vocabulary = defaultdict(int)
        for sent in reader.sents():
            untagged_sent = [token for (token, tag) in sent]
            tag_seq = [tag for (token, tag) in sent]
            for tag in tag_seq:
                # skip unlabeled examples
                if tag != "Unk":
                    tags.add(tag)
            length = len(untagged_sent)
            for index in xrange(length):
                if pos_prefixes != []:
                    # skip token if not one of the pos prefixes
                    skip = not np.any([sent[index][0][pos_idx].startswith(prefix)
                                       for prefix in pos_prefixes])
                    if skip:
                        continue
                features = detector.fd(untagged_sent, index, length)
                if not reader.raw:
                    pre_tags = tag_seq[:index]
                    history = detector.hd(pre_tags, untagged_sent, index,
                                          length)
                    features.extend(history)
                features = ("%s=%s" % (ftype, fval)
                            for ftype, fval in features if fval)
                for fx in features:
                    vocabulary[fx] += 1
                i += 1
                if i % 100000 == 0:
                    if verbose > 0:
                        print i,
            if i > MAX_TOKENS:  # FIXME
                break
        all_vocabulary.update(set((fx for fx, c in vocabulary.iteritems()
                                   if c >= minc)))

    # If extended prediction history is used add |tags| features.
    if use_eph:
        for t in tags:
            all_vocabulary.add("eph=%s" % t)
    all_vocabulary = sorted(all_vocabulary)
    tags = [t for t in tags]
    return all_vocabulary, tags


def encode_numeric(features):
    """Encodes numeric features."""
    for ftype, fval, fnum in features:
        if fnum != 0.0:
            yield ("%s=%s" % (ftype, fval), fnum)


def encode_indicator(features):
    """Encodes indicator (=binary) features."""
    for ftype, fval in features:
        if fval:
            yield ("%s=%s" % (ftype, fval), 1.0)


def asinstance(enc_features, fidx_map):
    """Convert a list of encoded features
    to the corresponding
    feature vector.

    Arguments
    ---------
    enc_features : list
        list of encoded features
    fidx_map : dict
        A dict mapping features to feature ids.

    Returns
    -------
    array, dtype = bolt.io.sparsedtype
        The feature vector.

    """
    instance = []
    for fid, fval in enc_features:
        if fid in fidx_map:
            instance.append((fidx_map[fid], fval))
    return bolt.fromlist(instance, bolt.sparsedtype)


@timeit
def build_examples(reader, detector, fidx_map, T, use_eph=False,
                   pos_prefixes=[], pos_idx=1, verbose=False):
    """Build feature vectors from the given reader.

    This method creates a feature vector for each token in the reader.
    The feature vector comprises 3 types of features:
    a) indicator features generated by the feature and history detectors.
    b) numeric features generated by the extended prediction history.
    c) spectral features generated by ASO.

    Parameters
    ----------
    reader : ConllReader
        The input data reader.
    detector : Detector
        The detector class providing the feature and history detectors.
    fidx_map : dict
        A mapping from feature ids to indices.
    T : set or list
        The tag set.
    use_eph : bool
        Whether or not to use extended prediction history.
    pos_prefixes : sequence
        A sequence of POS prefixes - examples are only created for
        tokens that match the given prefixes.
    pos_idx : int
        The index of the part-of-speach in the token.
    verbose : int
        Verbose output.

    Returns
    -------
    dataset : bolt.io.MemoryDataset
        A bolt memory dataset.
    """
    tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
    instances = []
    labels = []
    if use_eph:
        eph = ExtendedPredictionHistory(tidx_map)
    i = 0
    for sent in reader.sents():
        untagged_sent, tag_seq = zip(*sent)
        length = len(untagged_sent)
        for index in xrange(length):
            if pos_prefixes != []:
                # skip token if not one of the pos prefixes
                skip = not np.any([sent[index][0][pos_idx].startswith(prefix)
                                   for prefix in pos_prefixes])
                if skip:
                    continue

            # extract node and edge features
            features = detector.fd(untagged_sent, index, length)
            tag = tag_seq[index]
            if not reader.raw:
                pre_tags = tag_seq[:index]
                history = detector.hd(pre_tags, untagged_sent,
                             index, length)
                features.extend(history)

            # encode node and edge features as indicators
            enc_features = encode_indicator(features)
            if not reader.raw and use_eph:
                w = untagged_sent[index][0]
                # add EPH to encoded features
                if w in eph:
                    dist = eph[w]
                    dist = [("eph", T[i], v) for i, v in enumerate(dist)]
                    enc_features = chain(enc_features, encode_numeric(dist))
                # update EPH
                if tag != "O" and tag != "Unk":
                    eph.push(w, tag)

            # Encode unlabeled token as -1
            if tag == "Unk":
                tag = -1
            else:
                tag = tidx_map[tag]

            instance = asinstance(enc_features, fidx_map)
            instances.append(instance)
            labels.append(tag)
            i += 1
            if i % 10000 == 0:
                if verbose > 0:
                    print i,
        if i > MAX_TOKENS:  # FIXME
                break
    instances = bolt.fromlist(instances, np.object)
    labels = bolt.fromlist(labels, np.float64)
    return bolt.io.MemoryDataset(len(fidx_map), instances, labels)


class Tagger(object):
    """Tagger base class.

    Parameters
    ----------
    detector : Detector
        The detector class providing feature and histor detection
        functions.
    lang : str
        The language of the tagger.
    verbose : int
        The verbosity level of the tagger.

    """
    def __init__(self, detector, lang="en", verbose=0):
        self.detector = detector
        self.lang = lang
        self.verbose = verbose

    def tag(self, sent):
        """tag a sentence.
        :returns: a sequence of tags
        """
        pass

    def batch_tag(self, sents):
        """Tags each sentences.
        """
        i = 0
        for sent in sents:
            tagseq = [tag for tag in self.tag(sent)]
            yield tagseq
            i += 1


class GreedyTagger(Tagger):
    """Base class for Taggers with greedy left-to-right decoding.
    """
    use_aso = False

    def __init__(self, *args, **kargs):
        super(GreedyTagger, self).__init__(*args, **kargs)

    @timeit
    def feature_extraction(self, train_reader, minc=1, use_eph=False):
        """Extracts the features from the given training reader.
        Builds up various data structures such as the vocabulary and
        the tag set.
        """
        print "_" * 80
        print "Feature extraction"
        print
        print "min count: ", minc
        print "use eph: ", use_eph
        self.minc = minc
        self.use_eph = use_eph
        V, T = build_vocabulary(train_reader, self.detector, minc=minc,
                                use_eph=self.use_eph)
        self.V, self.T = V, T
        self.fidx_map = dict([(fname, i) for i, fname in enumerate(V)])
        self.tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
        self.tag_map = dict([(i, t) for i, t in enumerate(T)])

    @timeit
    def train(self, train_reader, biasterm=True, **kargs):
        """Trains the tagger on the given `train_reader`.

        This is a template method, it does some housekeeping and
        dispatches to `self._train` implemented by the concrete
        GreedyTagger classes. Keyword arguments are passed on
        to `self._train`.

        Parameters
        ----------
        train_reader : ConllReader
            A reader class for the training data.
        biasterm : bool
            Whether or not the model should use biased hyperplanes.
        kargs : dict
            Keyword arguments passed to `_train`.
        """
        T = self.T
        # Create EPH if necessary
        if self.use_eph:
            self.eph = ExtendedPredictionHistory(self.tidx_map)
        print "creating training examples...",
        sys.stdout.flush()
        dataset = build_examples(train_reader, self.detector,
                                 self.fidx_map, T,
                                 use_eph=self.use_eph)
        print "[done]"

        if self.use_aso:
            from multiprocessing import Pool
            print "_" * 80
            print "use ASO."
            print "projecting %d train examples onto subspace..." % dataset.n

            pool = Pool(processes=1)
            pool.map(_project_process, [(self.aso_model, dataset)])
            pool.close()
            del dataset
            dataset = bolt.io.MemoryDataset.load("/tmp/dataset.npy")

            # Update vocabulary and feature map with ASO features.
            for i in xrange(self.aso_model.embedding_dim):
                self.V.append("aso%d_cur" % i)
                self.fidx_map["aso%d_cur" % i] = self.aso_model.input_dim + i

        gc.collect()
        print "_" * 80
        print
        print "Size of  dataset: %.4f MB" % sizeof(dataset)
        print "_" * 80
        print "Training"
        print
        print "num examples: %d" % dataset.n
        print "num features: %d" % dataset.dim
        print "num classes: %d" % len(T)
        print "classes: ", T
        glm = bolt.GeneralizedLinearModel(dataset.dim, len(T),
                                          biasterm=biasterm)

        #self._train(glm, dataset, epochs=epochs, reg=reg, verbose=self.verbose,
        #            shuffle=shuffle, seed=seed, n_jobs=n_jobs, **kargs)
        self._train(glm, dataset, verbose=self.verbose, **kargs)
        self.glm = glm
        self.tags = train_reader.tags

    def _train(self, glm, dataset, **kargs):
        """Template method; implemented by sub classes.
        """
        raise NotImplementedError

    def tag(self, sent):
        """Tag a sequence of tokens. Returns a generator over the
        output tag sequence.

        Parameters
        ----------
        sent : seq of tuples
            The sentence to tag. A sent is represented by a seq
            of tuples (tag, token). The token is again a tuple,
            usually token = (word, pos, np).

        Example
        -------
        >>> sent = [(('Peter', 'NNP', 'I-NP'), ''),
                    (('Blackburn', 'NNP', 'I-NP'), '')]
        >>> [t for t in tagger.tag(sent)]
        ['B-PER', 'I-PER']
        """
        untagged_sent = [token for (token, tag) in sent]
        length = len(untagged_sent)
        tag_seq = []
        for index in xrange(length):
            w = untagged_sent[index][0]
            features = self.detector.fd(untagged_sent, index, length)
            history = self.detector.hd(tag_seq, untagged_sent, index, length)
            features.extend(history)
            enc_features = encode_indicator(features)
            if self.use_eph:
                # add eph dist as numeric features
                if w in self.eph:
                    dist = self.eph[w]
                    dist = [("eph", self.T[i], v) for i, v in enumerate(dist)]
                    enc_features = chain(enc_features, encode_numeric(dist))
            instance = asinstance(enc_features, self.fidx_map)
            if self.use_aso:
                instance = self.aso_model.project_instance(instance)

            p = self.glm(instance)
            tag = self.tag_map[p]
            tag_seq.append(tag)
            if self.use_eph and tag != "O" and tag != "Unk":
                self.eph.push(w, tag)
            yield tag_seq[-1]

    def describe(self, k=20):
        """Describes the trained model.
        """
        for i, w in enumerate(self.glm.W):
            idxs = w.argsort()
            maxidx = idxs[::-1][:k]
            print self.T[i], ": "
            print ", ".join(["<%s,%.4f>" % (self.V[idx], w[idx])
                             for idx in maxidx])


def _project_process(data):
    aso_model, dataset = data
    dataset = aso_model.project_dataset(dataset)
    dataset.store("/tmp/dataset.npy")


class AvgPerceptronTagger(GreedyTagger):
    """A greedy left-to-right tagger that is based on an
    Averaged Perceptron.
    """

    def _train(self, glm, dataset, **kargs):
        epochs = kargs["epochs"]
        trainer = bolt.trainer.avgperceptron.AveragedPerceptron(epochs=epochs)
        trainer.train(glm, dataset, shuffle=kargs["shuffle"],
                      seed=kargs["seed"],
                      verbose=self.verbose)


class GreedySVMTagger(GreedyTagger):
    """A greedy left-to-right tagger that is based on a one-against-all
    combination  of Support Vector Machines.
    """
    def _train(self, glm, dataset, **kargs):
        sgd = bolt.SGD(bolt.ModifiedHuber(), reg=kargs["reg"],
                       epochs=kargs["epochs"], norm=kargs.get("norm", 2))
        trainer = bolt.OVA(sgd)
        trainer.train(glm, dataset, shuffle=kargs["shuffle"],
                      seed=kargs["seed"],
                      verbose=self.verbose, ncpus=kargs["n_jobs"])
