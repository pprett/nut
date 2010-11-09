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
def build_vocabulary(reader, fd, hd, minc=1):
    """
    Arguments
    ---------

    Returns
    -------
    tuple, (vocabulary, tags)
       The vocabulary is the set of features; tags is the tag vocabulary.
    """
    tags = set()
    vocabulary = defaultdict(int)
    for sent in reader.sents():
        untagged_sent = [token for (token, tag) in sent]
        tag_seq = [tag for (token, tag) in sent]
        for tag in tag_seq:
            tags.add(tag)
        length = len(untagged_sent)
        for index in range(len(untagged_sent)):
            features = fd(untagged_sent, index, length)
            pre_tags = tag_seq[:index]
            history = hd(pre_tags, untagged_sent, index, length)
            features.extend(history)
            features = ("%s=%s" % (ftype, fval)
                        for ftype, fval in features.iteritems() if fval)
            for fx in features:
                vocabulary[fx] += 1
    vocabulary = [fx for fx, c in vocabulary.items() if c >= minc]
    tags = [t for t in tags]
    return vocabulary, tags


@timeit
def build_examples(reader, fd, hd, V, T):
    fidx_map = dict([(fname, i) for i, fname in enumerate(V)])
    tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
    instances = []
    labels = []
    for sent in reader.sents():
        untagged_sent = [token for (token, tag) in sent]
        tag_seq = [tag for (token, tag) in sent]
        length = len(untagged_sent)
        for index in range(len(untagged_sent)):
            features = fd(untagged_sent, index, length)
            pre_tags = tag_seq[:index]
            history = hd(pre_tags, untagged_sent,
                         index, length)
            features.extend(history)
            instance = []
            features = ("%s=%s" % (ftype, fval)
                        for ftype, fval in features.iteritems() if fval)
            for fx in features:
                if fx in fidx_map:
                    instance.append((fidx_map[fx], 1.0))
            instance = bolt.fromlist(instance, bolt.sparsedtype)
            instances.append(instance)
            tag = sent[index][1]
            labels.append(tidx_map[tag])
    instances = bolt.fromlist(instances, np.object)
    labels = bolt.fromlist(labels, np.float64)
    return bolt.io.MemoryDataset(len(V), instances, labels)


class Tagger(object):
    """Tagger base class."""
    def __init__(self, fd, hd, train_reader, verbose=0):
        self.fd = fd
        self.hd = hd
        self.train_reader = train_reader

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
    def train(self, reg=0.0001, epochs=30, minc=1, shuffle=False, verbose=1):
        V, T = build_vocabulary(self.train_reader, self.fd, self.hd, minc=minc)
        self.V, self.T = V, T
        self.fidx_map = dict([(fname, i) for i, fname in enumerate(V)])
        self.tidx_map = dict([(tag, i) for i, tag in enumerate(T)])
        self.tag_map = dict([(i, t) for i, t in enumerate(T)])
        dataset = build_examples(self.train_reader, self.fd, self.hd, V, T)
        dataset.shuffle(13)
        print "------------------------------"
        print "Training".center(30)
        print "num examples: ", dataset.n
        print "num features: ", dataset.dim
        print "num classes: ", len(T)
        print "classes: ", T
        print "reg: ", reg
        print "epochs: ", epochs
        glm = bolt.GeneralizedLinearModel(dataset.dim, len(T), biasterm=True)
        self._train(glm, dataset, epochs=epochs, reg=reg, verbose=verbose,
                    shuffle=shuffle)
        self.glm = glm

    def _train(self, glm, dataset, **kargs):
        raise NotImplementedError

    def save(self, fname):
        f = open(fname, "w+")
        pickle.dump(self.glm, f)
        pickle.dump(self.V, f)
        pickle.dump(self.fidx_map, f)
        pickle.dump(self.tidx_map, f)
        pickle.dump(self.tag_map, f)
        f.close()

    def load(self, fname):
        f = open(fname, "r")
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
        for index in range(len(untagged_sent)):
            features = self.fd(untagged_sent, index, length)
            history = self.hd(tag_seq, untagged_sent, index, length)
            features.extend(history)
            instance = []
            for (fname, fval) in features:
                fx = "%s(%s)" % (fname, str(fval))
                if fx in self.fidx_map:
                    instance.append((self.fidx_map[fx], 1))
            instance = sorted(instance)
            x = bolt.fromlist(instance, bolt.sparsedtype)
            p = self.glm(x)
            tag_seq.append(self.tag_map[p])
            yield tag_seq[-1]

    def describe(self, k=20):
        """Describes the trained model.
        """
        for i, w in enumerate(self.glm.W):
            idxs = w.argsort()
            maxidx = idxs[::-1][:k]
            #minidx = idxs[:k]
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
                      verbose=kargs["verbose"])


class GreedySVMTagger(GreedyTagger):
    """A greedy left-to-right tagger that is based on a one-against-all
    combination  of Support Vector Machines.
    """
    def _train(self, glm, dataset, **kargs):
        sgd = bolt.SGD(bolt.ModifiedHuber(), reg=kargs["reg"],
                       epochs=kargs["epochs"])
        trainer = bolt.OVA(sgd)
        trainer.train(glm, dataset, shuffle=kargs["shuffle"],
                      verbose=kargs["verbose"], ncpus=4)
