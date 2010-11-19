# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""
tagger
======

TODO docs
"""

import numpy as np
import cPickle as pickle
import bolt

from collections import defaultdict
from ..util import timeit

@timeit
def build_occurrences(reader):
    """Occurrences seam to hurt more than they do good..
    """
    occurrences = defaultdict(set)
    for sent in reader.sents():
        length = len(sent)
        for index in range(length):
            # add current word to tag occurrences
            w = sent[index][0][0]
            tag = sent[index][1]
            if tag != "O":  # FIXME add and tag != "Unk":
                occurrences[w].add(tag)
    return occurrences

@timeit
def build_vocabulary(reader, fd, hd, minc=1, occurrences={}):
    """
    Arguments
    ---------

    Returns
    -------
    vocabulary : list
        A sorted list of feature names.
    tags : list
        The list of tags.
    """
    tags = set()
    vocabulary = defaultdict(int)
    for sent in reader.sents():
        untagged_sent = [token for (token, tag) in sent]
        tag_seq = [tag for (token, tag) in sent]
        for tag in tag_seq:
            tags.add(tag)
        length = len(untagged_sent)
        for index in range(length):
            features = fd(untagged_sent, index, length)
            pre_tags = tag_seq[:index]
            history = hd(pre_tags, untagged_sent, index, length, occurrences)
            features.extend(history)
            features = ("%s=%s" % (ftype, fval)
                        for ftype, fval in features if fval)
            for fx in features:
                vocabulary[fx] += 1
    vocabulary = [fx for fx, c in vocabulary.items() if c >= minc]
    vocabulary = sorted(vocabulary)
    tags = [t for t in tags]
    return vocabulary, tags


def fs_to_instance(features, fidx_map):
    """Convert a list of features to the corresponding
    feature vector.

    Arguments
    ---------
    features : list of (ftype, fval) tuples.
    fidx_map : dict
        A dict mapping features to feature ids.

    Returns
    -------
    array, dtype = bolt.io.sparsedtype
        The feature vector.

    """
    instance = []
    features = ("%s=%s" % (ftype, fval)
                for ftype, fval in features if fval)
    for fx in features:
        if fx in fidx_map:
            instance.append((fidx_map[fx], 1.0))
    return bolt.fromlist(instance, bolt.sparsedtype)


@timeit
def build_examples(reader, fd, hd, V, T, occurrences={}):
    fidx_map = dict([(fname, i) for i, fname in enumerate(V)])
    tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
    instances = []
    labels = []
    i = 0
    for sent in reader.sents():
        untagged_sent, tag_seq = zip(*sent)
        length = len(untagged_sent)
        for index in range(length):
            features = fd(untagged_sent, index, length)
            pre_tags = tag_seq[:index]
            history = hd(pre_tags, untagged_sent,
                         index, length, occurrences)
            features.extend(history)
            instance = fs_to_instance(features, fidx_map)
            tag = tag_seq[index]
            if tag == "Unk":
                tag = -1
            else:
                tag = tidx_map[tag]
            instances.append(instance)
            labels.append(tag)
    instances = bolt.fromlist(instances, np.object)
    labels = bolt.fromlist(labels, np.float64)
    return bolt.io.MemoryDataset(len(V), instances, labels)


class Tagger(object):
    """Tagger base class."""
    def __init__(self, fd, hd, verbose=0):
        self.fd = fd
        self.hd = hd
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

    def __init__(self, *args, **kargs):
        super(GreedyTagger, self).__init__(*args, **kargs)

    @timeit
    def feature_extraction(self, train_reader, minc=1):
        print "------------------------------"
        print "Feature extraction".center(30)
        print "min count: ", minc
        #occurrences = build_occurrences(train_reader)
        #self.occurrences = occurrences
        #self.print_occurrences()
        V, T = build_vocabulary(train_reader, self.fd, self.hd, minc=minc)
                                #occurrences=occurrences)
        self.V, self.T = V, T
        
        self.fidx_map = dict([(fname, i) for i, fname in enumerate(V)])
        self.tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
        self.tag_map = dict([(i, t) for i, t in enumerate(T)])
        print "building examples..."
        dataset = build_examples(train_reader, self.fd, self.hd, V, T)
                                 #occurrences=occurrences)
        dataset.shuffle(13)
        self.dataset = dataset

    def print_occurrences(self):
        print "Occurrences"
        print "-----------"
        print "Num words with prev. taggings: %d" % len(self.occurrences)
        print "EU :", self.occurrences["EU"]
        print "and :", self.occurrences["and"]
        print

    @timeit
    def train(self, reg=0.0001, epochs=30, shuffle=False):
        dataset = self.dataset
        T = self.T
        print "------------------------------"
        print "Training".center(30)
        print "num examples: %d" % dataset.n
        print "num features: %d" % dataset.dim
        print "num classes: %d" % len(T)
        print "classes: ", T
        print "reg: %.8f" % reg
        print "epochs: %d" % epochs
        glm = bolt.GeneralizedLinearModel(dataset.dim, len(T), biasterm=True)
        self._train(glm, dataset, epochs=epochs, reg=reg, verbose=self.verbose,
                    shuffle=shuffle)
        self.glm = glm

    def _train(self, glm, dataset, **kargs):
        raise NotImplementedError

    def save(self, fname):
        f = open(fname, "wb+")
        pickle.dump(self.glm, f)
        pickle.dump(self.V, f)
        pickle.dump(self.fidx_map, f)
        pickle.dump(self.tidx_map, f)
        pickle.dump(self.tag_map, f)
        f.close()

    def load(self, fname):
        f = open(fname, "rb")
        self.glm = pickle.load(f)
        self.V = pickle.load(f)
        self.fidx_map = pickle.load(f)
        self.tidx_map = pickle.load(f)
        self.tag_map = pickle.load(f)
        f.close()

    def tag(self, sent):
        untagged_sent = [token for (token, tag) in sent]
        length = len(untagged_sent)
        tag_seq = []
        #occurrences = self.occurrences
        for index in range(length):
            features = self.fd(untagged_sent, index, length)
            history = self.hd(tag_seq, untagged_sent, index, length)
                              # occurrences)
            features.extend(history)
            instance = fs_to_instance(features, self.fidx_map)
            p = self.glm(instance)
            tag = self.tag_map[p]
            tag_seq.append(tag)
            # Add assigned tag to occurrences
            w = untagged_sent[index][0]
            #if tag != "O":
                #if w in occurrences:
                #    if tag not in occurrences[w]:
                #        print "adding %s to occurrences of %s" % (w, tag)
            #occurrences[w].add(tag)
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


class AvgPerceptronTagger(GreedyTagger):
    """A greedy left-to-right tagger that is based on an Averaged Perceptron.
    """

    def _train(self, glm, dataset, **kargs):
        epochs = kargs["epochs"]
        trainer = bolt.trainer.avgperceptron.AveragedPerceptron(epochs=epochs)
        trainer.train(glm, dataset, shuffle=kargs["shuffle"],
                      verbose=self.verbose)


class GreedySVMTagger(GreedyTagger):
    """A greedy left-to-right tagger that is based on a one-against-all
    combination  of Support Vector Machines.
    """
    def _train(self, glm, dataset, **kargs):
        sgd = bolt.SGD(bolt.ModifiedHuber(), reg=kargs["reg"],
                       epochs=kargs["epochs"])
        trainer = bolt.OVA(sgd)
        trainer.train(glm, dataset, shuffle=kargs["shuffle"],
                      verbose=self.verbose, ncpus=4)
