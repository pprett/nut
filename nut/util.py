#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
util
====

"""

import time

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


def timeit(func):
  def wrapper(*arg, **kargs):
      t1 = time.time()
      res = func(*arg, **kargs)
      t2 = time.time()
      print '%s took %0.3f sec' % (func.func_name, (t2-t1))
      return res
  return wrapper 

def trace(func):
    def wrapper(*args, **kargs):
	print "calling %s with args %s, %s" % (func.__name__, args, kargs)
	return func(*args, **kargs)
    return wrapper
