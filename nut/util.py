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
util
====

"""

import time

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"
__copyright__ = "Apache License v2.0"


def timeit(func):
  def wrapper(*arg, **kargs):
      t1 = time.time()
      res = func(*arg, **kargs)
      t2 = time.time()
      print '%s took %0.3f sec' % (func.func_name, (t2-t1))
      return res
  return wrapper 
