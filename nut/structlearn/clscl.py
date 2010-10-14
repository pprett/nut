#!/usr/bin/python2.6
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
import numpy as np
import bolt

import structlearn
import pivotselection
import util


from itertools import islice

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"


class CLSCL(object):

    def __init__(self, S, T):
	self.S = S
	self.T = T
	self.pivotselector = pivotselection.MISelector()

    def select_pivots(self, m, phi):
	vp = self.pivotselector.select(self.S.train)
	candidates = ((ws, self.pivottranslator(ws)) for ws in vp \
		      if self.pivottranslator(ws) != None)
	counts = util.count(self.S.unlabeled, self.T.unlabeled)
	pivots = ((ws,wt) for ws, wt in candidates \
			 if counts[ws] >= phi and counts[wt] >= phi)
	pivots = [pivot for pivot in islice(pivots,m)]
	print "num pivots: %d. " % len(pivots)
	return pivots	

    def learn(self, m, phi, k):
	pivots = self.select_pivots(m, phi)
	print "create StructLearner"
	ds = bolt.io.MemoryDataset.merge((self.S.unlabeled, self.T.unlabeled))
	struct_learner = structlearn.StructLearner(k, ds, pivots,
				       structlearn.ElasticNetTrainer(0.00001,0.85),
				       structlearn.SerialTrainingStrategy())
	print "struct_learner.learn() "
	struct_learner.learn()
	print "struct_learner.learn() [done]"
	print "projecting domains... ",
	_project_domains(struct_learner, S, T, verbose = 1)
	print "[done]"

    def pivottranslator(self, ws):
	term_ws = self.S.idx_to_term[ws]
	term_wt = self._translate(term_ws)
	wt = self.T.voc[term_wt]
	return wt

    def _translate(self, term_ws):
	return "wurst"

def _project_domains(struct_learner, S, T, tostandardize = True,
		     toavgnorm = True, verbose = 1):
    """Project `S` and `T` onto the space induced by `theta`.  """
    nS, nT = util.Domain(), util.Domain()
    for domain, new_domain in ((S,nS),(T,nT)):
	for member in ["train", "test", "unlabeled"]:
	    try:
		ds = domain.__getattribute__[member]
		new_domain.__setattr__(member,struct_learner.project(ds, dense = True))
	    except AttributeError:
		pass
    
    if tostandardize:
	if verbose > 0:
	    print "Standardize features."
	train_dat = np.concatenate((nS.train, nS.unlabeled, nT.unlabeled)) 
	mean = train_dat.mean(axis=0)
	std = train_dat.std(axis=0)
	for new_domain in (nS, nT):
	    for member in ["train", "test", "unlabeled"]:
		try:
		    instances = new_domain.__getattribute__[member]
		    standardize(instances, mean, std)
		except AttributeError:
		    pass
    else:
	if verbose > 0:
	    print "No standardization."

    if toavgnorm:
	norms = np.sqrt((nS.train * nS.train).sum(axis=1))
	avg_norm = np.mean(norms)
	if verbose > 1:
	    print "Old avg norm: ", avg_norm
	for new_domain in (nS, nT):
	    for member in ["train", "test", "unlabeled"]:
		try:
		    instances = new_domain.__getattribute__[member]
		    instances /= avg_norm
		except AttributeError:
		    pass
		
	norms = np.sqrt((nS.train * nS.train).sum(axis=1))
	avg_norm = np.mean(norms)
	if verbose > 0:
	    print "New avg norm: ", avg_norm
    else:
	if verbose > 0:
	    print "No average norm."

    dim = struct_learner.theta.shape[0]
    voc = dict([("fx%d" % i,i) for i in range(struct_learner.theta.shape[0])])
    idx_to_term = dict(((idx, term) for term, idx in voc.items()))
    for domain, new_domain in ((S,nS),(T,nT)):
	domain.voc = voc
	domain.idx_to_term = idx_to_term
	for member in ["train", "test", "unlabeled"]:
	    try:
		instances = new_domain.__getattribute__[member]
		instances = structlearn.to_sparse_bolt(instances)
		ds = domain.__getattribute__[member]
		ds.instances = instances
		ds.dim = dim
	    except AttributeError:
		pass

def standardize(docterms, mean, std, alpha = 1.0):
    docterms -= mean
    docterms /= std
    docterms *= alpha

def parse_bow(line):
    tokens = [tf.split(':') for tf in line.rstrip().split(' ')]
    s,label = tokens[-1]
    assert s == "#label#"
    tokens = tokens[:-1]
    return label,[t for t in tokens if len(t) == 2 and len(t[0]) > 0]

class Experiment(object):
    
    def __init__(self, slang, tlang, sourceDir, targetDir,
		 domain):
	canonical = lambda s: s if s[-1] == "/" else s + "/"
	self.sourceDir = canonical(sourceDir)
	self.targetDir = canonical(targetDir)
	self.slang = slang
	self.tlang = tlang
	self.domain = domain
	self.nunlabeled = 50000

    def createVoc(self, mindf = 2):
	SFD = FreqDist()
	TFD = FreqDist()
	lc = 0
	nunlabeled = self.nunlabeled
	for directory, fd in [(self.sourceDir, SFD),(self.targetDir, TFD)]:
	    for s in ["train","unlabeled"]:
		with open(directory + s + ".processed") as f:
		    for i,line in enumerate(f):
			if i >= nunlabeled and s == "unlabeled":
			    break
			lc += 1
			label, tokens = parse_bow(line)
			for token,freq in tokens:
			    fd.inc(token)
			    
	Svoc = set([t for t,c in SFD.iteritems() if c >= mindf])
	Tvoc = set([t for t,c in TFD.iteritems() if c >= mindf])
	if langs[0] == langs[1]:
	    Svoc, Tvoc, dim = self.createConjunctiveVoc(Svoc, Tvoc)
	else:
	    Svoc, Tvoc, dim = self.createDisjointVoc(Svoc, Tvoc)
	return Svoc, Tvoc, dim

    def createDisjointVoc(self,Svoc,Tvoc):
	if self.verbose > 2:
	    print "createDisjointVoc"
	V_S = len(Svoc)
	V_T = len(Tvoc)
	Svoc = dict(zip(Svoc,range(V_S)))
	Tvoc = dict(zip(Tvoc,range(V_S,V_S+V_T)))
	return Svoc,Tvoc, len(Svoc)+len(Tvoc)

    def createConjunctiveVoc(self,Svoc,Tvoc):
	if self.verbose > 2:
	    print "createConjunctiveVoc"
	uniq_Svoc = sorted(Svoc.difference(Tvoc))
	uniq_Tvoc = sorted(Tvoc.difference(Svoc))
	shared_voc = sorted(Svoc.intersection(Tvoc))
	voc = list(uniq_Svoc+shared_voc+uniq_Tvoc)
	voc = dict(zip(voc,range(len(voc))))
	return voc, voc, len(voc)

    def loadDataText(self):
	Svoc,Tvoc,dim = self.createVoc()
	if self.verbose > 1:
	    print "|Svoc|\t\t|Tvoc|"
	    print "%d\t&\t%d" % (len(Svoc), len(Tvoc))
	S = Domain()
	T = Domain()
	labelMap = {'positive':1,'negative':-1,'unlabeled':0}
	nunlabeled = self.nunlabeled
	for directory, domain, voc in [(self.sourceDir,S,Svoc),(self.targetDir,T,Tvoc)]:
	    lang = directory.split("/")[-3]
	    domain.lang = lang
	    domain.voc = voc
	    domain.idxToTerm = dict(((idx, term) for term, idx in voc.items()))
	    for s in ["train","unlabeled","test"]:
		reviews = []
		labels = []
		with open(directory + s + ".processed") as f:
		    for i, line in enumerate(f):
			if s == "unlabeled" and i >= nunlabeled:
			    break
			label, tokens = bow.parseBow(line)
			review = bow.vectorize(tokens,voc)
			doc = np.array(review,dtype = bolt.sparsedtype)
			norm = np.linalg.norm(doc['f1']) 
			if norm > 0.0:
			    doc['f1'] /= norm
			reviews.append(doc)
			labels.append(labelMap[label])

		instances = bolt.io.fromlist(reviews, np.object)
		labels = bolt.io.fromlist(labels, np.float32)
		examples = bolt.MemoryDataset(dim,instances, labels)
		domain.__setattr__(s,examples)
		
	return S,T
    
def main(argv):
    root = argv[0]
    slang = argv[1]
    tlang = argv[2]
    domain = argv[3]
    "%s/%"
    
    
    clscl_learner = CLSCL(S, T)
    clscl_learner.learn(450, 30, 100)
    
    


if __name__ == "__main__":
    main()
