#!/usr/bin/python
#
# Copyright (C) 2010 Peter Prettenhofer.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
clscl
=====

"""
import sys
import numpy as np
import bolt

from itertools import islice,ifilter

from ..structlearn import pivotselection
from ..structlearn import util
from ..structlearn import structlearn
from ..bow import vocabulary, disjoint_voc, load
from ..util import timeit


__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"

class CLSCLModel(object):
    def __init__(self):
	pass

class CLSCLTrainer(object):

    def __init__(self, s_train, s_unlabeled, t_unlabeled,
		 pivotselector, pivottranslator):
	self.s_train = s_train
	self.s_unlabeled = s_unlabeled
	self.t_unlabeled = t_unlabeled
	self.pivotselector = pivotselector
	self.pivottranslator = pivottranslator

    @timeit
    def select_pivots(self, m, phi):
	vp = self.pivotselector.select(self.s_train)
	
	candidates = ifilter(lambda x: x[1] != None, ((ws, self.pivottranslator[ws]) for ws in vp))
	
	counts = util.count(self.s_unlabeled, self.t_unlabeled)
	pivots = ((ws,wt) for ws, wt in candidates \
			 if counts[ws] >= phi and counts[wt] >= phi)
	pivots = [pivot for pivot in islice(pivots,m)]
	return pivots	

    def train(self, m, phi, k):
	pivots = self.select_pivots(m, phi)
	print("|pivots| = %d" % len(pivots))
	print "create StructLearner"
	ds = bolt.io.MemoryDataset.merge((self.s_unlabeled,
					  self.t_unlabeled))
	struct_learner = structlearn.StructLearner(k, ds, pivots,
				       structlearn.ElasticNetTrainer(0.00001,0.85),
				       structlearn.SerialTrainingStrategy())
	print "struct_learner.learn() "
	struct_learner.learn()
	print "struct_learner.learn() [done]"
	print "projecting domains... ",
	self.project(struct_learner, verbose = 1)
	print "[done]"

    def project(self, struct_learner, verbose = 1):
	s_train = struct_learner.project(self.s_train, dense = True)
	s_unlabeled = struct_learner.project(self.s_unlabeled, dense = True)
	t_unlabeled = struct_learner.project(self.t_unlabeled, dense = True)

	data = np.concatenate((s_train, s_unlabeled, t_unlabeled)) 
	mean = data.mean(axis=0)
	std  = data.std(axis=0)
	self.mean, self.std = mean, std
	standardize(s_train, mean, std)
	
	norms = np.sqrt((s_train * s_train).sum(axis=1))
	avg_norm = np.mean(norms)
	s_train /= avg_norm

	dim = struct_learner.theta.shape[0]
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
	print("DictTranslator contains %d translations." % len(dictionary))

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
	wt = wt.split(" ")[0]
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
    
def main():
    maxlines = 50000
    argv = sys.argv[1:]

    slang = argv[0]
    tlang = argv[1]

    fname_s_train = argv[2]
    fname_s_unlabeled = argv[3]
    fname_t_unlabeled = argv[4]
    fname_dict = argv[5]

    s_voc = vocabulary(fname_s_train, fname_s_unlabeled,
		       mindf = 2, maxlines = maxlines)
    t_voc = vocabulary(fname_t_unlabeled,
		       mindf = 2, maxlines = maxlines)
    s_voc, t_voc, dim = disjoint_voc(s_voc,t_voc)
    s_ivoc = dict([(idx,term) for term, idx in s_voc.items()])
    print("|V_S| = %d\n|V_T| = %d" % (len(s_voc), len(t_voc)))
    print("|V| = %d" % dim)
    
    s_train = load(fname_s_train, s_voc, dim)
    s_unlabeled = load(fname_s_unlabeled, s_voc, dim, maxlines = maxlines)
    t_unlabeled = load(fname_t_unlabeled, t_voc, dim, maxlines = maxlines)
    print("|s_train| = %d" % s_train.n)
    print("|s_unlabeled| = %d" % s_unlabeled.n)
    print("|t_unlabeled| = %d" % t_unlabeled.n)

    
    translator = DictTranslator.load(fname_dict, s_ivoc, t_voc)
    pivotselector = pivotselection.MISelector()
    clscl_trainer = CLSCLTrainer(s_train, s_unlabeled,
				 t_unlabeled, pivotselector,
				 translator)
    
    clscl_trainer.train(450, 30, 100)
    
    


if __name__ == "__main__":
    main()
