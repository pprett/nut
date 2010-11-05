#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
auxstrategy
===========

A module containing different trainer strategies for the auxiliary tasks.

"""
from __future__ import division

import sys
import bolt
import numpy as np
import subprocess
import shlex
import inspect
import os
import json
import tempfile
import shutil

from abc import ABCMeta, abstractmethod
from copy import deepcopy
from scipy import sparse
from time import time

from ..structlearn import util
from ..util import timeit, trace

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"

class TrainingStrategy(object):
    """An interface of different training strategies for the auxiliary classifiers. 

    Use this to implement various parallel or distributed training strategies.
    Delegates the training of a single classifier to a concrete `AuxTrainer`.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_aux_classifiers(self, ds, auxtasks, classifier_trainer, inverted_index = None):
	"""Abstract method to train auxiliary classifiers, i.e. to fill `struct_learner.W`.
	"""
	return 0

class SerialTrainingStrategy(TrainingStrategy):
    """A serial training strategy.

    Trains one auxiliary classifier after another. Does not exploit multi core architectures. 
    """	

    @timeit
    def train_aux_classifiers(self, ds, auxtasks, classifier_trainer, inverted_index = None):
	dim = ds.dim
	w_data = []
	row   = []
	col   = []
	original_instances = ds.instances[ds._idx]
	
	for j, auxtask in enumerate(auxtasks):
	    #t0 = time()
	    instances = deepcopy(original_instances)
	    #print "aux task (copy) took %.3g sec." % (time() - t0)
	    if inverted_index is None:
		util.mask(instances, auxtask)
		labels = util.autolabel(instances, auxtask)
	    else:
		occurances = inverted_index[j]
		util.mask(instances[occurances], auxtask)
		labels = np.ones((instances.shape[0],), dtype = np.float32)
		labels *= -1.0
		labels[occurances] = 1.0
	    ds = bolt.io.MemoryDataset(dim, instances, labels)
	    #print "aux task (copy + ds) took %.3g sec." % (time() - t0)
	    w = classifier_trainer.train_classifier(ds)
	    #print "aux task (copy + ds + sgd) took %.3g sec." % (time() - t0)
	    for i in w.nonzero()[0]:
		row.append(i)
		col.append(j)
		w_data.append(w[i])
	    #print "aux task (copy + ds + sgd + set W) took %.3g sec." % (time() - t0)
	    if j % 10 == 0:
		print "%d classifiers trained..." % j

	W = sparse.coo_matrix((w_data,(row,col)),
			      (dim, len(auxtasks)),
			      dtype = np.float64)
	return W.tocsc()
			      
class HadoopTrainingStrategy(TrainingStrategy):
    """A distributed strategy which utilizes Hadoop.

    For each auxiliary task a map task is created. The mapper is implemented as a python script using hadoop streaming. 
    """
    
    @timeit
    def train_aux_classifiers(self, ds, auxtasks, classifier_trainer, inverted_index = None):
	dim = ds.dim
	m = len(auxtasks)
	w_data = []
	row   = []
	col   = []
	tmpdir = tempfile.mkdtemp()
	print "tempdir:", tmpdir
	run_id = os.path.split(tmpdir)[-1]
	try:
	    self._mktasks(tmpdir, auxtasks)
	    ds.store(tmpdir+"/examples.npy")
	    self._mkhdfsdir(run_id)
	    self._send_file_to_hdfs(tmpdir+"/tasks.txt", run_id)
	    self._send_file_to_hdfs(tmpdir+"/examples.npy", run_id)
	    print "processing Hadoop job...",
	    sys.stdout.flush()
	    retcode = self._runmapper(run_id+"/tasks.txt",
				      run_id+"/examples.npy",
				      run_id+"/out.txt")
	    W = self._readoutput(run_id + "/out.txt", (dim, m))
	    self._rm_hdfs_dir(run_id)
    	    return W    
	finally:
	    shutil.rmtree(tmpdir)
	    print("Cleaning local temp dir.")

    def _mktasks(self, tmpdir, auxtasks, alpha = 0.85, norm = 3, reg = 0.00001):
	f = open(tmpdir + "/tasks.txt","w+")
	for i, task in enumerate(auxtasks):
	    params = {"taskid":i, "task":str(task),
		      "alpha":alpha,
		      "norm":norm, "reg":reg}
	    f.write(json.dumps(params))
	    f.write("\n")
	f.close()
	
	

    def _mkhdfsdir(self, hdfspath):
	cmd = "hadoop dfs -mkdir %s" % (hdfspath)
	cmd = shlex.split(cmd)
	retcode = subprocess.call(cmd)
	return retcode

    def _send_file_to_hdfs(self, fname, hdfspath):
	cmd = "hadoop dfs -put %s %s" % (fname,hdfspath)
	cmd = shlex.split(cmd)
	retcode = subprocess.call(cmd)
	return retcode

    def _rm_hdfs_dir(self, hdfspath):
	cmd = "hadoop dfs -rmr %s" % (hdfspath)
	cmd = shlex.split(cmd)
	retcode = subprocess.call(cmd)
	return retcode

    def _runmapper(self, ftasks, fexamples, fout, streaming_jar = "/usr/lib/hadoop/contrib/streaming/hadoop-0.18.3-2cloudera0.3.0-streaming.jar"):
	"""Runs the dumbomapper with input `ftasks` and
	`fexamples`. 
	"""
	import dumbomapper
	import auxtrainer
	
	fmapper = inspect.getsourcefile(dumbomapper)
	fauxtrainer = inspect.getsourcefile(auxtrainer)
	futil = inspect.getsourcefile(util)

	param = {"ftasks":ftasks, "fexamples":fexamples, "fout":fout,
		 "streaming_jar":streaming_jar, "futil":futil, 
		 "fmapper":fmapper, "fauxtrainer":fauxtrainer}
	
	cmd = """hadoop jar %(streaming_jar)s \
	-input %(ftasks)s \
	-output %(fout)s \
	-mapper dumbomapper.py \
	-file %(fmapper)s \
	-file %(fauxtrainer)s \
	-file %(futil)s \
	-cacheFile %(fexamples)s#examples.npy \
	-jobconf mapred.reduce.tasks=0 \
	-jobconf mapred.input.format.class=org.apache.hadoop.mapred.lib.NLineInputFormat \
	-jobconf mapred.line.input.format.linespermap=1
	""" % param
	
	cmd = shlex.split(cmd)
	dn = open("/dev/null")
	retcode = subprocess.call(cmd, stdout = dn, stderr = dn)
	dn.close()
	return retcode

    def _deserialize(self, s):
	return [(int(i),float(v)) for i,v in [f.split(":") for f in s.split(" ")]]

    @timeit
    def _readoutput(self, fout, shape):
	dim, m = shape
	w_data = []
	row   = []
	col   = []
	cmd = "hadoop dfs -cat %s/part*" % (fout)
	cmd = shlex.split(cmd)
	pipe = subprocess.Popen(cmd, bufsize=1, stdout=subprocess.PIPE).stdout
	for line in pipe:
	    fields = line.rstrip().split("\t")
	    pivotid = int(fields[0])
	    w = self._deserialize(fields[1])
	    for fidx,fval in w:
		row.append(fidx)
		col.append(pivotid)
		w_data.append(fval)

	W = sparse.coo_matrix((w_data,(row,col)),
			      (dim, m),
			      dtype = np.float64)
	return W.tocsc()
