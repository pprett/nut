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
import cPickle as pickle
import optparse
import bolt

from itertools import islice,ifilter

from ..structlearn import pivotselection
from ..structlearn import util
from ..structlearn import structlearn
from ..structlearn import auxtrainer
from ..structlearn import auxstrategy
from ..bow import vocabulary, disjoint_voc, load
from ..util import timeit, trace

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__version__ = "0.1"

class CLSCLModel(object):
    def __init__(self, thetat, mean = None, std = None, avg_norm = None):
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

	:returns: A new bolt.io.MemoryDataset equal to `ds` but contains projected feature vectors. 
	"""
	dense_instances = structlearn.project(ds, self.thetat, dense = True)
	
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
	vp = self.pivotselector.select(self.s_train)
	candidates = ifilter(lambda x: x[1] != None, ((ws, self.pivottranslator[ws]) for ws in vp))
	counts = util.count(self.s_unlabeled, self.t_unlabeled)
	pivots = ((ws,wt) for ws, wt in candidates \
			 if counts[ws] >= phi and counts[wt] >= phi)
	pivots = [pivot for pivot in islice(pivots,m)]
	#terms = [(self.s_ivoc[ws],self.t_ivoc[wt]) for ws,wt in pivots]
	#for term in terms[:50]:
	#    print term
	
	return pivots	

    def train(self, m, phi, k):
	pivots = self.select_pivots(m, phi)
	print("|pivots| = %d" % len(pivots))
	ds = bolt.io.MemoryDataset.merge((self.s_unlabeled,
					  self.t_unlabeled))
	ds.shuffle(9)
	struct_learner = structlearn.StructLearner(k, ds, pivots,
						   self.trainer,
						   self.strategy)
	struct_learner.learn()
	self.project(struct_learner.thetat, verbose = 1)
	return CLSCLModel(struct_learner.thetat, mean = self.mean,
			  std = self.std, avg_norm = self.avg_norm)
	

    @timeit
    def project(self, thetat, verbose = 1):
	s_train = structlearn.project(self.s_train, thetat, dense = True)
	s_unlabeled = structlearn.project(self.s_unlabeled, thetat,
					  dense = True)
	t_unlabeled = structlearn.project(self.t_unlabeled, thetat,
					  dense = True)

	data = np.concatenate((s_train, s_unlabeled, t_unlabeled)) 
	mean = data.mean(axis=0)
	std  = data.std(axis=0)
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

def standardize(docterms, mean, std, alpha = 1.0):
    docterms -= mean
    docterms /= std
    docterms *= alpha

class DictTranslator(object):

    def __init__(self, dictionary, s_ivoc, t_voc):
	self.dictionary = dictionary
	self.s_ivoc = s_ivoc
	self.t_voc = t_voc
	print("debug: DictTranslator contains %d translations." % len(dictionary))

    def __getitem__(self, ws):
	try:
	    wt = self.normalize(self.dictionary[self.s_ivoc[ws]])
	except KeyError:
	    wt = None
	return wt

    def translate(self, ws):
	return self[ws]

    def normalize(self, wt):
	wt = wt.encode("utf-8") if isinstance(wt,unicode) else wt
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
    description = """Prefixes `s_` and `t_` refer to source and target language, resp.
Train and unlabeled files are expected to be in Bag-of-Words format.
    """
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "s_lang t_lang s_train_file " \
                                   "s_unlabeled_file t_unlabeled_file " \
                                   "dict_file model_file",
                                   version="%prog " + __version__,
                                   description = description)
    
    parser.add_option("-v","--verbose",
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
    return parser
    
    
def train():
    """Training script for CLSCL.

    FIXME: works for binary classification only.
    TODO: multi-class classification.
    TODO: different translators. 

    Usage: ./clscl_train slang tlang strain sunlabeled tunlabeled dict out
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
    s_voc, t_voc, dim = disjoint_voc(s_voc,t_voc)
    s_ivoc = dict([(idx,term) for term, idx in s_voc.items()])
    t_ivoc = dict([(idx,term) for term, idx in t_voc.items()])
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    print("  |V| = %d" % dim)

    # Load labeled and unlabeled data
    s_train = load(fname_s_train, s_voc, dim)
    s_unlabeled = load(fname_s_unlabeled, s_voc, dim,
                       maxlines=options.max_unlabeled)
    t_unlabeled = load(fname_t_unlabeled, t_voc, dim,
                       maxlines=options.max_unlabeled)
    print("    |s_train| = %d" % s_train.n)
    print("|s_unlabeled| = %d" % s_unlabeled.n)
    print("|t_unlabeled| = %d" % t_unlabeled.n)

    # Load dictionary
    translator = DictTranslator.load(fname_dict, s_ivoc, t_voc)

    # 
    pivotselector = pivotselection.MISelector()
    trainer = auxtrainer.ElasticNetTrainer(0.00001, 0.85,
                                           10**6)
    strategy = auxstrategy.HadoopTrainingStrategy()
    clscl_trainer = CLSCLTrainer(s_train, s_unlabeled,
				 t_unlabeled, pivotselector,
				 translator, trainer,
				 strategy)
    clscl_trainer.s_ivoc = s_ivoc
    clscl_trainer.t_ivoc = t_ivoc
    model = clscl_trainer.train(options.m, options.phi, options.k)
    
    model.s_voc = s_voc
    model.t_voc = t_voc
    f = open(argv[6], "wb+")
    pickle.dump(model, f)
    f.close()

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

    f = open(fname_model, "rb")
    CLSCLModel = pickle.load(f)
    f.close()

    s_voc = CLSCLModel.s_voc
    t_voc = CLSCLModel.t_voc
    dim = len(s_voc) + len(t_voc)
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    print("|V| = %d" % dim)

    s_train = load(fname_s_train, s_voc, dim)
    t_test  = load(fname_t_test, t_voc, dim)
    
    cl_train = CLSCLModel.project(s_train)
    cl_test  = CLSCLModel.project(t_test)
    
    epochs = int(math.ceil(10**6 / cl_train.n))
    model = bolt.LinearModel(cl_train.dim, biasterm = False)
    loss = bolt.ModifiedHuber()
    sgd = bolt.SGD(loss, reg, epochs = epochs, norm = 2)
    cl_train.shuffle(1)
    sgd.train(model, cl_train, verbose = 0, shuffle = False)
    acc = 100.0 - bolt.eval.errorrate(model, cl_test)
    print "ACC: %.2f" % acc
    
