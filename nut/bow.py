#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import numpy as np

from collections import defaultdict

from .util import timeit
from .externals import bolt

def parse_bow(line):
    tokens = [tf.split(':') for tf in line.rstrip().split(' ')]
    s,label = tokens[-1]
    assert s == "#label#"
    tokens = tokens[:-1]
    return label,[t for t in tokens if len(t) == 2 and len(t[0]) > 0]

def vectorize(tokens, voc):
    doc = [(voc[term], int(freq)) for term, freq in tokens
	   if term in voc]
    doc = sorted(doc)
    return doc

@timeit
def vocabulary(*bowfnames, **kargs):
    """
    it supports the following kargs:
    - mindf: min document frequency (default 2).
    - maxlines: maximum number of lines to read (default -1).
    """
    mindf = kargs.get("mindf", 2)
    maxlines = kargs.get("maxlines", -1)
    fd = defaultdict(int)
    for fname in bowfnames:
        with open(fname) as f:
	    for i, line in enumerate(f):
		if maxlines != -1 and i >= maxlines:
		    break
		label, tokens = parse_bow(line)
		for token,freq in tokens:
		    fd[token] += 1
    voc = set([t for t,c in fd.iteritems() if c >= mindf])
    return voc

def disjoint_voc(s_voc, t_voc):
    n = len(s_voc)
    m = len(t_voc)
    s_voc = dict(zip(s_voc,range(n)))
    t_voc = dict(zip(t_voc,range(n,n+m)))
    return s_voc, t_voc, len(s_voc) + len(t_voc)

@timeit
def load(fname, voc, dim, maxlines=-1):
    """
    """
    instances = []
    labels = []
    with open(fname) as f:
	for i, line in enumerate(f):
	    if maxlines != -1 and i >= maxlines:
		break
	    label, tokens = parse_bow(line)
	    doc = vectorize(tokens, voc)
	    x = np.array(doc, dtype = bolt.sparsedtype)
	    norm = np.linalg.norm(x['f1']) 
	    if norm > 0.0:
		x['f1'] /= norm
	    instances.append(x)
	    labels.append(label)
    instances = bolt.io.fromlist(instances, np.object)
    labels = np.array(labels)
    classes = np.unique(labels)
    labels = np.searchsorted(classes, labels).astype(np.float32)
    if len(classes) == 2:
        labels[labels == 0] = -1
    return bolt.MemoryDataset(dim, instances, labels), classes
