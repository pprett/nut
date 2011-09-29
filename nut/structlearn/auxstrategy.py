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
import numpy as np
import subprocess
import shlex
import inspect
import os
import json
import tempfile
import shutil
import cPickle as pickle

from abc import ABCMeta, abstractmethod
from scipy import sparse
from itertools import izip, count
from collections import defaultdict

from ..structlearn import util
from ..util import timeit, trace
from ..externals.joblib import Parallel, delayed


class Error(Exception):
    pass


class TrainingStrategy(object):
    """An interface of different training strategies for the auxiliary
    classifiers.

    Use this to implement various parallel or distributed training strategies.
    Delegates the training of a single classifier to a concrete `AuxTrainer`.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def train_aux_classifiers(self, ds, auxtasks, task_masks,
                              classifier_trainer,
                              inverted_index=None):
        """Abstract method to train auxiliary classifiers."""
        return 0


class SerialTrainingStrategy(TrainingStrategy):
    """A serial training strategy.

    Trains one auxiliary classifier after another.
    Does not exploit multi core architectures.
    """

    @timeit
    def train_aux_classifiers(self, ds, auxtasks, task_masks,
                              classifier_trainer, inverted_index=None):
        dim = ds.dim
        w_data = []
        row = []
        col = []

        for j, auxtask, task_mask in izip(count(), auxtasks, task_masks):

            if inverted_index is None:
                labels = util.autolabel(ds.instances, auxtask)
            else:
                occurances = inverted_index[j]
                labels = np.ones((ds.n,), dtype=np.float32)
                labels *= -1.0
                labels[occurances] = 1.0

            ds.labels = labels

            mask = np.ones((dim,), dtype=np.int32, order="C")
            mask[task_mask] = 0

            w = classifier_trainer.train_classifier(ds, mask)
            for i in w.nonzero()[0]:
                row.append(i)
                col.append(j)
                w_data.append(w[i])
            if j % 10 == 0:
                print "%d classifiers trained..." % j

        W = sparse.coo_matrix((w_data, (row, col)),
                              (dim, len(auxtasks)),
                              dtype=np.float64)
        return W.tocsc()


class ParallelTrainingStrategy(TrainingStrategy):
    """A parallel training strategy.

    Trains the auxiliary classifiers using joblib.Parallel.
    """

    def __init__(self, n_jobs=-1):
        super(ParallelTrainingStrategy, self).__init__()
        self.n_jobs = n_jobs

    @timeit
    def train_aux_classifiers(self, ds, auxtasks, task_masks,
                              classifier_trainer, inverted_index=None):
        dim = ds.dim
        w_data = []
        row = []
        col = []

        if inverted_index == None:
            inverted_index = defaultdict(lambda: None)
        print "Run joblib.Parallel"
        res = Parallel(n_jobs=self.n_jobs, verbose=1)(
                delayed(_train_aux_classifier)(i, auxtask,
                                               task_mask,
                                               ds, classifier_trainer,
                                               inverted_index[i])
            for i, auxtask, task_mask in izip(count(), auxtasks, task_masks))

        for i, (fx_idxs, fx_vals) in res:
            for fx_idx, fx_val in izip(fx_idxs, fx_vals):
                row.append(fx_idx)
                col.append(i)
                w_data.append(fx_val)

        W = sparse.coo_matrix((w_data, (row, col)),
                              (dim, len(auxtasks)),
                              dtype=np.float64)
        return W.tocsc()


def _train_aux_classifier(i, auxtask, task_mask, ds,
                          classifier_trainer, occurrences=None):
    """Trains a single auxiliary classifier.

    Parameters
    ----------
    i : int
        The index of the auxiliary task.
    auxtask : tuple of ints
        The auxiliary task.
    task_mask : set
        The features to mask (=set to zero).
    original_instances : array, dtype=bolt.sparsedtype
        The unlabeled instances.
    dim : int
        The dimensionality of the feature space.
    classifier_trainer : AuxTrainer
        The concrete trainer for the auxiliary classifiers.
    occurrencs : array
        The inverted index posting list for the task - if any.
        The indices in the list are w.r.t. ds.instances and not ds._idx!

    Returns
    -------
    i : int
        The index of the auxtask.
    sparse_w : (array, array)
        The sparse representation of the weight vector; the first
        array holds the indizes of the non zero features and the
        second array holds the values.
    """
    if occurrences is None:
        labels = util.autolabel(ds.instances, auxtask)
    else:
        labels = np.ones((ds.n,), dtype=np.float32)
        labels *= -1.0
        labels[occurrences] = 1.0

    ds.labels = labels
    # create feature mask
    mask = np.ones((ds.dim,), dtype=np.int32, order="C")
    mask[task_mask] = 0
    w = classifier_trainer.train_classifier(ds, mask)
    return i, (w.nonzero()[0], w[w.nonzero()[0]])


class HadoopTrainingStrategy(TrainingStrategy):
    """A distributed strategy which utilizes Hadoop.

    For each auxiliary task a Hadoop map task is created.
    The mapper is implemented as a python script using Hadoop Streaming.
    """
    @timeit
    def train_aux_classifiers(self, ds, auxtasks, task_masks,
                              classifier_trainer, inverted_index=None):
        dim = ds.dim
        m = len(auxtasks)
        tmpdir = tempfile.mkdtemp()
        print "tempdir:", tmpdir
        run_id = os.path.split(tmpdir)[-1]
        try:
            print "use classifier_trainer:", repr(classifier_trainer)
            self._mktasks(tmpdir, auxtasks, classifier_trainer)
            ds.store(tmpdir + "/examples.npy")
            self._mkhdfsdir(run_id)
            self._send_file_to_hdfs(tmpdir + "/tasks.txt", run_id)
            self._send_file_to_hdfs(tmpdir + "/examples.npy", run_id)

            # if auxtasks and masks are not identical (e.g. NER)
            if auxtasks is not task_masks:
                print "Copying task_masks to HDFS."
                f = open(tmpdir + "/task_masks.pkl", "wb")
                pickle.dump(task_masks, f, protocol=-1)
                f.close()
                self._send_file_to_hdfs(tmpdir + "/task_masks.pkl", run_id)
            else:
                print "Don't copy task_masks to HDFS."

            print "processing Hadoop job...",
            sys.stdout.flush()
            retcode = self._runmapper(run_id + "/tasks.txt",
                                      run_id + "/examples.npy",
                                      run_id + "/out.txt")
            W = self._readoutput(run_id + "/out.txt", (dim, m))
            ## FIXME remove below
            #f = open("/tmp/W.pkl", "wb")
            #pickle.dump(W, f, -1)
            #f.close()
            ## FIXME remove above
            self._rm_hdfs_dir(run_id)
            return W
        finally:
            shutil.rmtree(tmpdir)
            print("Cleaning local temp dir.")

    def _mktasks(self, tmpdir, auxtasks, classifier_trainer):
        f = open(tmpdir + "/tasks.txt", "w+")
        for i, task in enumerate(auxtasks):
            params = {"taskid": i, "task": list(task),
                      "trainer": repr(classifier_trainer)}
            f.write(json.dumps(params))
            f.write("\n")
        f.close()

    def _mkhdfsdir(self, hdfspath):
        cmd = "hadoop dfs -mkdir %s" % (hdfspath)
        cmd = shlex.split(cmd)
        retcode = subprocess.call(cmd)
        return retcode

    def _send_file_to_hdfs(self, fname, hdfspath):
        cmd = "hadoop dfs -put %s %s" % (fname, hdfspath)
        cmd = shlex.split(cmd)
        retcode = subprocess.call(cmd)
        return retcode

    def _rm_hdfs_dir(self, hdfspath):
        cmd = "hadoop dfs -rmr %s" % (hdfspath)
        cmd = shlex.split(cmd)
        retcode = subprocess.call(cmd)
        return retcode

    def _runmapper(self, ftasks, fexamples, fout,
                   streaming_jar="/usr/lib/hadoop/contrib/streaming/" \
                   "hadoop-0.18.3-2cloudera0.3.0-streaming.jar"):
        """Runs the dumbomapper with input `ftasks` and
        `fexamples`.
        """
        import dumbomapper
        import auxtrainer

        fmapper = inspect.getsourcefile(dumbomapper)
        fauxtrainer = inspect.getsourcefile(auxtrainer)
        futil = inspect.getsourcefile(util)

        param = {"ftasks": ftasks, "fexamples": fexamples, "fout": fout,
                 "streaming_jar": streaming_jar, "futil": futil,
                 "fmapper": fmapper, "fauxtrainer": fauxtrainer}

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
        retcode = subprocess.call(cmd, stdout=dn, stderr=dn)
        dn.close()
        return retcode

    def _deserialize(self, s):
        return [(int(i), float(v)) for i, v in [f.split(":")
                                                for f in s.split(" ")]]

    @timeit
    def _readoutput(self, fout, shape):
        dim, m = shape
        w_data = []
        row = []
        col = []
        cmd = "hadoop dfs -cat %s/part*" % (fout)
        cmd = shlex.split(cmd)
        pipe = subprocess.Popen(cmd, bufsize=1, stdout=subprocess.PIPE).stdout
        for line in pipe:
            fields = line.rstrip().split("\t")
            pivotid = int(fields[0])
            w = self._deserialize(fields[1])
            for fidx, fval in w:
                row.append(fidx)
                col.append(pivotid)
                w_data.append(fval)

        W = sparse.coo_matrix((w_data, (row, col)),
                              (dim, m),
                              dtype=np.float64)
        return W.tocsc()
