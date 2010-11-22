#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import sys
import re
import optparse
import numpy as np
import cPickle as pickle

from time import time
from itertools import islice
from collections import defaultdict
from ..io import conll
from ..tagger import tagger
from ..structlearn.pivotselection import FreqSelector

__version__ = "0.1"

WORD, POS, NP, LEMMA = 0, 1, 2, 3

date = re.compile(r"\d\d\d\d-\d\d-\d\d")

def numify(s):
    if sum((1 for c in s if not c.isdigit())) > 2:
        return s
    else:
        return "".join(["*d*" if c.isdigit() else c for c in s])

def datify(s):
    if date.match(s):
        s = "*date*"
    return s


def fd_zj03(sent, index, length):
    """Feature detector from ZJ03
    """
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    ## tokens in a 5 token window w_{i-2}..w_{i+2}
    w = datify(context(0, WORD))
    pre_w = datify(context(-1, WORD))
    pre_pre_w = datify(context(-2, WORD))
    post_w = datify(context(1, WORD))
    post_post_w = datify(context(2, WORD))

    ## token bigrams in a 3 token window
    pre_bigram = "/".join([pre_w, w])
    post_bigram = "/".join([w, post_w])

    ## pos in a 5 token window
    pos = context(0, POS)
    pre_pos = context(-1, POS)
    post_pos = context(1, POS)
    pre_pre_pos = context(-2, POS)
    post_post_pos = context(2, POS)

    pre_pos_bigram = "/".join([pre_pos, pos])
    post_pos_bigram = "/".join([pos, post_pos])

    pos_w = "/".join([w, pos])

    ## Word shape features (5 token window)
    istitle = w.istitle()
    isdigit = w.isdigit()
    isupper = w.isupper()
    isalnum = w.isalnum()
    hyphen = "-" in w[1:-1]
    is2digit = isdigit and len(w)==2
    is4digit = isdigit and len(w)==4

    pre_istitle = pre_w.istitle()
    pre_isdigit = pre_w.isdigit()
    pre_isupper = pre_w.isupper()
    pre_hyphen = "-" in pre_w[1:-1]
    pre_isalnum = pre_w.isalnum()
    pre_is2digit = pre_isdigit and len(pre_w)==2
    pre_is4digit = pre_isdigit and len(pre_w)==4

    pre_pre_istitle = pre_pre_w.istitle()
    pre_pre_isdigit = pre_pre_w.isdigit()
    pre_pre_isupper = pre_pre_w.isupper()
    pre_pre_hyphen = "-" in pre_pre_w[1:-1]
    pre_pre_isalnum = pre_pre_w.isalnum()
    pre_pre_is2digit = pre_pre_isdigit and len(pre_pre_w)==2
    pre_pre_is4digit = pre_pre_isdigit and len(pre_pre_w)==4

    post_istitle = post_w.istitle()
    post_isdigit = post_w.isdigit()
    post_isupper = post_w.isupper()
    post_hypen = "-" in post_w[1:-1]
    post_isalnum = post_w.isalnum()
    post_is2digit = post_isdigit and len(post_w)==2
    post_is4digit = post_isdigit and len(post_w)==4

    post_post_istitle = post_post_w.istitle()
    post_post_isdigit = post_post_w.isdigit()
    post_post_isupper = post_post_w.isupper()
    post_post_hypen = "-" in post_post_w[1:-1]
    post_post_isalnum = post_post_w.isalnum()
    post_post_is2digit = post_post_isdigit and len(post_post_w)==2
    post_post_is4digit = post_post_isdigit and len(post_post_w)==4

    ## 2-4 suffixes in a 3 token window
    w_suffix1 = w[-1:]
    w_suffix2 = w[-2:]
    w_suffix3 = w[-3:]
    w_suffix4 = w[-4:]

    pre_w_suffix1 = pre_w[-1:]
    pre_w_suffix2 = pre_w[-2:]
    pre_w_suffix3 = pre_w[-3:]
    pre_w_suffix4 = pre_w[-4:]

    post_w_suffix1 = post_w[-1:]
    post_w_suffix2 = post_w[-2:]
    post_w_suffix3 = post_w[-3:]
    post_w_suffix4 = post_w[-4:]

    ## 3-4 prefixes in a 3 token window
    w_prefix3 = w[:3]
    w_prefix4 = w[:4]

    pre_w_prefix3 = pre_w[:3]
    pre_w_prefix4 = pre_w[:4]

    post_w_prefix3 = post_w[:3]
    post_w_prefix4 = post_w[:4]

    ## Noun phrase in a 3 token window
    np = context(0,NP)
    np_w = "/".join([np, w])
    pre_np = context(-1, NP)
    post_np = context(1, NP)

    ## Extract features from local scope
    features = locals()
    del features["context"]
    del features["sent"]
    del features["index"]
    del features["length"]
    features = features.items()
    return features

def hd_zj03(tags, sent, index, length, occurrences={}):
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    pre_tag = tags[index - 1] if index - 1 >= 0 else "<s>"
    pre_pre_tag = tags[index - 2] if index - 2 >= 0 else "<s>"
    pre_tag_w = "/".join([pre_tag, context(0, WORD)])
    tag_bigram = "/".join([pre_pre_tag, pre_tag])
    
    history = [("pre_tag", pre_tag), ("pre_pre_tag", pre_pre_tag),
               ("pre_tag_w", pre_tag_w), ("tag_bigram", tag_bigram)]
    return history

def fd_az05(sent, index, length):
    """Feature detector from ZJ03
    """
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    def bow(i, j):
        bows = []
        minidx = (index - i if index - i > 0 else 0)
        maxidx = (index + i if index + i < length else length)
        for w in sent[minidx:maxidx]:
            bows.append(('bow%d-%d' % (i, j), w[WORD]))

        return bows

    ## tokens in a 5 token window w_{i-2}..w_{i+2}
    w = context(0, WORD)
    pre_w = context(-1, WORD)
    pre_pre_w = context(-2, WORD)
    post_w = context(1, WORD)
    post_post_w = context(2, WORD)

    ## token bigrams in a 5 token window
    pre_pre_bigram = "/".join([pre_pre_w, pre_w])
    pre_bigram = "/".join([pre_w, w])
    post_bigram = "/".join([w, post_w])
    post_post_bigram = "/".join([post_w, post_post_w])

    ## length of w
    wlen = len(w)

    ## pos in a 5 token window
    pos = context(0, POS)
    pre_pos = context(-1, POS)
    post_pos = context(1, POS)
    pre_pre_pos = context(-2, POS)
    post_post_pos = context(2, POS)

    pre_pos_bigram = "/".join([pre_pos, pos])
    post_pos_bigram = "/".join([pos, post_pos])
    pre_pre_pos_bigram = "/".join([pre_pre_pos, pre_pos])
    post_post_pos_bigram = "/".join([post_pos, post_post_pos])
    pos_w = "/".join([w, pos])

    ## Word shape features (5 token window)
    istitle = w.istitle()
    isdigit = w.isdigit()
    isupper = w.isupper()
    isalnum = w.isalnum()
    hyphen = "-" in w[1:-1]
    is2digit = isdigit and len(w)==2
    is4digit = isdigit and len(w)==4

    pre_istitle = pre_w.istitle()
    pre_isdigit = pre_w.isdigit()
    pre_isupper = pre_w.isupper()
    pre_hyphen = "-" in pre_w[1:-1]
    pre_isalnum = pre_w.isalnum()
    pre_is2digit = pre_isdigit and len(pre_w)==2
    pre_is4digit = pre_isdigit and len(pre_w)==4

    pre_pre_istitle = pre_pre_w.istitle()
    pre_pre_isdigit = pre_pre_w.isdigit()
    pre_pre_isupper = pre_pre_w.isupper()
    pre_pre_hyphen = "-" in pre_pre_w[1:-1]
    pre_pre_isalnum = pre_pre_w.isalnum()
    pre_pre_is2digit = pre_pre_isdigit and len(pre_pre_w)==2
    pre_pre_is4digit = pre_pre_isdigit and len(pre_pre_w)==4

    post_istitle = post_w.istitle()
    post_isdigit = post_w.isdigit()
    post_isupper = post_w.isupper()
    post_hypen = "-" in post_w[1:-1]
    post_isalnum = post_w.isalnum()
    post_is2digit = post_isdigit and len(post_w)==2
    post_is4digit = post_isdigit and len(post_w)==4

    post_post_istitle = post_post_w.istitle()
    post_post_isdigit = post_post_w.isdigit()
    post_post_isupper = post_post_w.isupper()
    post_post_hypen = "-" in post_post_w[1:-1]
    post_post_isalnum = post_post_w.isalnum()
    post_post_is2digit = post_post_isdigit and len(post_post_w)==2
    post_post_is4digit = post_post_isdigit and len(post_post_w)==4

    ## 2-4 suffixes in a 4 token window
    w_suffix1 = w[-1:]
    w_suffix2 = w[-2:]
    w_suffix3 = w[-3:]
    w_suffix4 = w[-4:]

    pre_w_suffix1 = pre_w[-1:]
    pre_w_suffix2 = pre_w[-2:]
    pre_w_suffix3 = pre_w[-3:]
    pre_w_suffix4 = pre_w[-4:]

    post_w_suffix1 = post_w[-1:]
    post_w_suffix2 = post_w[-2:]
    post_w_suffix3 = post_w[-3:]
    post_w_suffix4 = post_w[-4:]

    pre_pre_w_suffix1 = pre_pre_w[-1:]
    pre_pre_w_suffix2 = pre_pre_w[-2:]
    pre_pre_w_suffix3 = pre_pre_w[-3:]
    pre_pre_w_suffix4 = pre_pre_w[-4:]

    post_post_w_suffix1 = post_post_w[-1:]
    post_post_w_suffix2 = post_post_w[-2:]
    post_post_w_suffix3 = post_post_w[-3:]
    post_post_w_suffix4 = post_post_w[-4:]

    ## 2-4 suffixes in a 4 token window
    w_suffix1 = w[:1]
    w_suffix2 = w[:2]
    w_suffix3 = w[:3]
    w_suffix4 = w[:4]

    pre_w_suffix1 = pre_w[:1]
    pre_w_suffix2 = pre_w[:2]
    pre_w_suffix3 = pre_w[:3]
    pre_w_suffix4 = pre_w[:4]

    post_w_suffix1 = post_w[:1]
    post_w_suffix2 = post_w[:2]
    post_w_suffix3 = post_w[:3]
    post_w_suffix4 = post_w[:4]

    pre_pre_w_suffix1 = pre_pre_w[:1]
    pre_pre_w_suffix2 = pre_pre_w[:2]
    pre_pre_w_suffix3 = pre_pre_w[:3]
    pre_pre_w_suffix4 = pre_pre_w[:4]

    post_post_w_suffix1 = post_post_w[:1]
    post_post_w_suffix2 = post_post_w[:2]
    post_post_w_suffix3 = post_post_w[:3]
    post_post_w_suffix4 = post_post_w[:4]

    ## Noun phrase in a 3 token window
    np = context(0,NP)
    np_w = "/".join([np, w])
    pre_np = context(-1, NP)
    post_np = context(1, NP)

    ## Extract features from local scope
    features = locals()
    del features["context"]
    del features['bow']
    del features["sent"]
    del features["index"]
    del features["length"]
    features = features.items()

    features.extend(bow(-4, -1))
    features.extend(bow(1, 4))
    
    return features


def hd_az05(tags, sent, index, length, occurrences={}):
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    pre_tag = tags[index - 1] if index - 1 >= 0 else "<s>"
    pre_pre_tag = tags[index - 2] if index - 2 >= 0 else "<s>"
    pre_tag_w = "/".join([pre_tag, context(0, WORD)])
    tag_bigram = "/".join([pre_pre_tag, pre_tag])
    
    history = [("pre_tag", pre_tag), ("pre_pre_tag", pre_pre_tag),
               ("pre_tag_w", pre_tag_w), ("tag_bigram", tag_bigram)]
    return history


def fd_rr09(sent, index, length):
    """Feature detector for CoNLL 2003 from RR09.
    """
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    ## tokens in a 5 token window x_{i-2}..x_{i+2}
    w = numify(context(0, WORD))
    pre_w = numify(context(-1, WORD))
    pre_pre_w = numify(context(-2, WORD))
    post_w = numify(context(1, WORD))
    post_post_w = numify(context(2, WORD))

    ## token bigrams in a 5 token window
    pre_bigram = "/".join([pre_w, w])
    post_bigram = "/".join([w, post_w])
    #pre_pre_bigram = "/".join([pre_pre_w, pre_w])
    #post_post_bigram = "/".join([post_w, post_post_w])

    ## pos in a 5 token window
    pos = context(0, POS)
    pre_pos = context(-1, POS)
    post_pos = context(1, POS)
    pre_pre_pos = context(-2, POS)
    post_post_pos = context(2, POS)

    ## pos bigrams in a 5 token window
    pre_pos_bigram = "/".join([pre_pos, pos])
    post_pos_bigram = "/".join([pos, post_pos])
    #pre_pre_pos_bigram = "/".join([pre_pre_pos, pre_pos])
    #post_post_pos_bigram = "/".join([post_pos, post_post_pos])

    pos_w = "/".join([w, pos])
    
    ## Word shape features (5 token window)
    istitle = w.istitle()
    isdigit = context(0, WORD).isdigit()
    isupper = w.isupper()
    hyphen = "-" in w[1:-1]
    isalnum = context(0, WORD).isalnum()
    
    pre_istitle = pre_w.istitle()
    pre_isdigit = context(-1, WORD).isdigit()
    pre_isupper = pre_w.isupper()
    pre_hyphen = "-" in pre_w[1:-1]
    pre_isalnum = context(-1, WORD).isalnum()

    pre_pre_istitle = pre_pre_w.istitle()
    pre_pre_isdigit = context(-2, WORD).isdigit()
    pre_pre_isupper = pre_pre_w.isupper()
    pre_pre_hyphen = "-" in pre_pre_w[1:-1]
    pre_pre_isalnum = context(-2, WORD).isalnum()

    post_istitle = post_w.istitle()
    post_isdigit = context(1, WORD).isdigit()
    post_isupper = post_w.isupper()
    post_hypen = "-" in post_w[1:-1]
    post_isalnum = context(1, WORD).isalnum()

    post_post_istitle = post_post_w.istitle()
    post_post_isdigit = context(2, WORD).isdigit()
    post_post_isupper = post_post_w.isupper()
    post_post_hypen = "-" in post_post_w[1:-1]
    post_post_isalnum = context(2, WORD).isalnum()

    ## 2-4 suffixes in a 3 token window
    w_suffix1 = w[-1:]
    w_suffix2 = w[-2:]
    w_suffix3 = w[-3:]
    w_suffix4 = w[-4:]

    pre_w_suffix1 = pre_w[-1:]
    pre_w_suffix2 = pre_w[-2:]
    pre_w_suffix3 = pre_w[-3:]
    pre_w_suffix4 = pre_w[-4:]

    post_w_suffix1 = post_w[-1:]
    post_w_suffix2 = post_w[-2:]
    post_w_suffix3 = post_w[-3:]
    post_w_suffix4 = post_w[-4:]

    ## 3-4 prefixes in a 3 token window
    w_prefix3 = w[:3]
    w_prefix4 = w[:4]

    pre_w_prefix3 = pre_w[:3]
    pre_w_prefix4 = pre_w[:4]

    post_w_prefix3 = post_w[:3]
    post_w_prefix4 = post_w[:4]

    ## Noun phrase in a 3 token window
    np = context(0,NP)
    np_w = "/".join([np, w])
    pre_np = context(-1, NP)
    post_np = context(1, NP)

    ## Extract features from local scope
    features = locals()
    del features["context"]
    del features["sent"]
    del features["index"]
    del features["length"]
    features = features.items()
    return features


def hd_rr09(tags, sent, index, length):
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    pre_tag = tags[index - 1] if index - 1 >= 0 else "<s>"
    pre_pre_tag = tags[index - 2] if index - 2 >= 0 else "<s>"
    pre_tag_w = "/".join([pre_tag, context(0, WORD)])
    pre_tag_w = "/".join([pre_tag, context(-1, WORD)])
    pre_tag_w = "/".join([pre_tag, context(-2, WORD)])
    pre_tag_w = "/".join([pre_tag, context(1, WORD)])
    pre_tag_w = "/".join([pre_tag, context(2, WORD)])
    tag_bigram = "/".join([pre_pre_tag, pre_tag])

    history = locals()
    del history["context"]
    del history["tags"]
    del history["sent"]
    del history["index"]
    del history["length"]
    history = history.items()
    return history

fd = fd_rr09
hd = hd_rr09

class ASO(object):
    def __init__(self, model, reader):
        self.reader = reader
        self.fidx_map = model.fidx_map
        self.vocabulary = model.V
        self.ds = tagger.build_examples(reader, model.fd, model.hd,
                                        model.fidx_map, model.T)

    def instances_by_pos_prefix(self, prefix):
        indices = []
        i = 0
        for sent in self.reader:
            for token in sent:
                if token[POS].startswith(prefix):
                    indices.append(i)
                    i += 1
        return np.array(indices)

    def filter_noun_adjectives(self):
        ds = self.ds
        nn_indices = self.instances_by_pos_prefix("NN")
        jj_indices = self.instances_by_pos_prefix("JJ")
        indices = np.union(nn_indices, jj_indices)
        ds.instances = ds.instances[indices]
        ds.labels = ds.labels[indices]
        ds.n = ds.labels.shape[0]
        self.ds = ds

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
            FreqSelector(0).select(self.ds, preselection), m))
        return aux_tasks

    def print_tasks(self, tasks):
        for task in tasks:
            print self.vocabulary[task]

    def learn(self):
        print "run ASO.learn..."
        self.filter_noun_adjectives()
        aux_tasks = self.create_aux_tasks(1000)
        self.print_tasks(aux_tasks)
##      ds.shuffle(9)
        ## struct_learner = structlearn.StructLearner(k, ds, pivots,
##                                                 self.trainer,
##                                                 self.strategy)
##      struct_learner.learn()


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file model_file",
                                   version="%prog " + __version__,
                                   description = description)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
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
    parser.add_option("--stats",
                      action="store_true",
                      dest="stats",
                      default=False,
                      help="Print model statistics.")
    parser.add_option("--eph",
                      action="store_true",
                      dest="use_eph",
                      default=False,
                      help="Use Extended Prediction History.")
    parser.add_option("--aso",
                      action="store_true",
                      dest="aso",
                      default=False,
                      help="Use Alternating Structural Optimization.")
    parser.add_option("-u", "--unlabeled",
                      dest="funlabeled",
                      help="FILE containing unlabeled data.",
                      metavar="str",
                      type="str")
    return parser


def train():
    """Training script for Named Entity Recognizer.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 2:
        parser.error("incorrect number of arguments (use `--help` for help).")
    if options.aso:
        if not options.funlabeled:
            raise parser.error("specify unlabeled data with --unlabeled. ")
    f_train = argv[0]
    f_model = argv[1]
    train_reader = conll.Conll03Reader(f_train, options.lang)
    #model = tagger.AvgPerceptronTagger(fd, hd, verbose=options.verbose)
    model = tagger.GreedySVMTagger(fd, hd, lang=options.lang,
                                   verbose=options.verbose)
    model.feature_extraction(train_reader, minc=options.minc,
                             use_eph=options.use_eph)
    if options.aso:
        unlabeled_reader = conll.Conll03Reader(options.funlabeled,
                                               options.lang)
        aso = ASO(model, unlabeled_reader)
        aso.learn()
        sys.exit()
    model.train(reg=options.reg, epochs=options.epochs,
                shuffle=options.shuffle)
    if options.stats:
        print "------------------------------"
        print " Stats\n"
        model.describe(k=40)
        nnz = 0
        for instance in model.dataset.iterinstances():
            nnz += instance.shape[0]
        print "avg. nnz: %.4f" % (float(nnz) / model.dataset.n)
    if f_model.endswith(".gz"):
        import gzip
        f = gzip.open(f_model, mode="wb")
    elif f_model.endswith(".bz2"):
        import bz2
        f = bz2.BZ2File(f_model, mode="w")
    else:
        f = open(f_model)
    del model.dataset
    pickle.dump(model, f)
    f.close()


def predict():
    """Test script for Named Entity Recognizer.
    """
    def usage():
        print """Usage: %s [OPTIONS] MODEL_FILE TEST_FILE PRED_FILE
        Load NER model from MODEL_FILE, test the model on
        TEST_FILE and write predictions to PRED_FILE.
        The predictions are appended to the test file.
        Options:
          -h, --help\tprint this help.

        """ % sys.argv[0]
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        usage()
        sys.exit(-2)
    if len(argv) != 3:
        print "Error: wrong number of arguments. "
        usage()
        sys.exit(-2)

    print >> sys.stderr, "loading tagger...",
    sys.stderr.flush()
    f_model = argv[0]
    if f_model.endswith(".gz"):
        import gzip
        f = gzip.open(f_model, mode="rb")
    elif f_model.endswith(".bz2"):
        import bz2
        f = bz2.BZ2File(f_model, mode="r")
    else:
        f = open(f_model)
    model = pickle.load(f)
    f.close()
    print >> sys.stderr, "[done]"
    print >> sys.stderr, "use_eph: ", model.use_eph
    test_reader = conll.Conll03Reader(argv[1], model.lang)
    if argv[2] != "-":
        f = open(argv[2], "w+")
    else:
        f = sys.stdout
    t0 = time()
    test_reader.write_sent_predictions(model, f, raw=False)
    f.close()
    print >> sys.stderr, "processed input in %.4fs sec." % (time() - t0)
