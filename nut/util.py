#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
util
====

"""

import sys
import time

from .externals.bolt.io import MemoryDataset

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


def timeit(func):
    def wrapper(*arg, **kargs):
        t1 = time.time()
        res = func(*arg, **kargs)
        t2 = time.time()
        print '%s took %0.3f sec' % (func.func_name, (t2 - t1))
        return res
    return wrapper


def trace(func):
    def wrapper(*args, **kargs):
        print "calling %s with args %s, %s" % (func.__name__, args, kargs)
        return func(*args, **kargs)
    return wrapper


def sizeof(d):
    """Retuns size of datastructure in MBs. """
    bytes = 0
    if hasattr(d, "nbytes"):
        bytes = d.nbytes
    elif isinstance(d, MemoryDataset):
        for i in d.iterinstances():
            # each examples has 8*nnz + label + idx
            bytes += (i.shape[0] * 8) + 4 + 4
    elif isinstance(d, dict):
        for k, v in d.iteritems():
            bytes += sys.getsizeof(k) + sys.getsizeof(v)
    elif isinstance(d, list):
        for e in d:
            bytes += sys.getsizeof(e)
    else:
        bytes = sys.getsizeof(d)
    return bytes / 1024.0 / 1024.0
