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
structlearn.util
================

"""
import operator
import functools
import numpy as np

from itertools import chain

def mask(instances, auxtask):
    count = 0
    for x in instances:
        indices = x['f0']
	for idx in auxtask:
	    if idx in indices:
		p = np.where(indices == idx)[0]
		if len(p) > 0:
		    x['f1'][p] = 0.0
		    count += 1
    return count

def autolabel(instances, auxtask):
    labels = np.ones((instances.shape[0],), dtype = np.float32)
    labels *= -1
    for i, x in enumerate(instances):
	indices = x['f0']
	for idx in auxtask:
	    if idx in indices:
		labels[i] = 1
		break
	
    return labels

def count(*datasets):
    if len(datasets) > 1:
	assert functools.reduce(operator.eq,[ds.dim for ds in datasets])
    counts = np.zeros((ds.dim,),dtype = np.uint16)
    for x, y in chain(*datasets):
	counts[x["f0"]] += 1
    return counts
