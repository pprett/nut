#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
pivotselection
==============

"""
from __future__ import division
import numpy as np

from abc import ABCMeta, abstractmethod
from itertools import cycle, izip, count, islice
from operator import itemgetter

import util

from ..externals import bolt
from ..util import trace, timeit

class PivotSelector(object):
    __meta__ = ABCMeta

    @abstractmethod
    def select(self, ds, preselection=None):
	return 0

class FreqSelector(PivotSelector):
    """Select pivots by minimum document frequency.
    """
    def __init__(self, support):
	self.support = support

    def select(self, ds, preselection=None):
	counts = util.count(ds)
	gen = (idx for idx in counts.argsort()[::-1] if counts[idx] >= self.support)
	if preselection != None:
	    preselection = set(preselection)
	    return (idx for idx in gen if idx in preselection)
	else:
	    return gen
	

class RandomSelector(PivotSelector):
    """Selects pivots at random. 
    """
    
    def select(self, ds, preselection = None):
	if preselection == None:
	    indices = np.arange(ds.dim)
	else:
	    indices = preselection
	np.random.shuffle(indices)
	for idx in indices:
	    yield idx

class MISelector(PivotSelector):
    """Selects pivots according to mutual information.

    If the number of classes is larger than 2 than it creates `num classes` binary rankings - one-vs-all. Then, it selects the top ranks in a round robin fashion until a total of `k` pivots are selected. 
    """
    
    def select(self, ds, preselection = None):
	if len(ds.classes) > 2:
	    res = self.select_multi(ds, preselection)
	else:
	    res = self.select_binary(ds, preselection)
	return res

    def select_multi(self, ds, preselection = None):
	fxset = set()
	rankings = []
	for c in ds.classes:
	    bs = bolt.io.BinaryDataset(ds,c)
	    rankings.append(mutualinformation(bs, preselection))

	for idx in roundrobin(*rankings):
	    if idx not in fxset:
		fxset.add(idx)
		yield idx

    def select_binary(self, ds, preselection = None):
	return mutualinformation(ds, preselection)

@timeit
def mutualinformation(bs, preselection = None):
    """Computes mutual information of each column of `docterms` and `labels`.
    Returns the indices of the top `k` columns according to MI.
    """
    N = 0
    N_pos = sum([1 for y in bs.iterlabels() if y == 1.0])
    N_neg = bs.n - N_pos
    N_term = {}
    POS = 1
    NEG = 2
    TOTAL = 0

    for doc,label in bs:
	for term,freq in doc:
	    if freq == 0.0: continue
	    term_stats = N_term.get(term,np.zeros((3,),dtype=int))
	    if label == 1.0:
		term_stats[POS]+=1
	    else:
		term_stats[NEG]+=1
	    term_stats[TOTAL]+=1
	    N_term[term] = term_stats
	N+=1

    mi = {}

    N += 2 # account for pseudo counts
    for term in N_term:
	term_stats = N_term[term]
	N_11 = term_stats[POS] + 1
	N_10 = term_stats[NEG] + 1 
	N_01 = 1 + N_pos - N_11
	N_00 = 1 + N_neg - N_10
	N_1_ = term_stats[TOTAL] + 2
	assert (N_11 + N_10) == N_1_
	N_0_ = N - N_1_
	
	a = ((N_11)/ N) * np.log2((N*N_11) / (N_1_ * N_pos))
 	b = ((N_01) / N) * np.log2((N*N_01) / (N_0_ * N_pos))
 	c = ((N_10) / N) * np.log2((N*N_10) / (N_1_ * N_neg))
 	d = ((N_00) / N) * np.log2((N*N_00) / (N_0_ * N_neg))
	
	mi[term] = a+b+c+d

    mi = sorted(mi.items(),key = itemgetter(1))
    mi.reverse()
    if preselection != None:
	preselection = set(preselection)
	return (i for i,v in mi if i in preselection)
    else:
	return (i for i,v in mi)

def roundrobin(*iterables):
    "roundrobin('ABC', 'D', 'EF') --> A D E B F C"
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))
