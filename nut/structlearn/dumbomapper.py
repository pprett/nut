#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""A mapper for hadoop task parallelism.

To test the mapper locally use:
> cat tasks.txt| ./mapper.py


To send to Hadoop use:
hadoop jar /usr/lib/hadoop/contrib/streaming/hadoop-0.18.3-2cloudera0.3.0-streaming.jar \
    -input tmptasks.txt \
    -output tmpout \
    -mapper dumbo.py \
    -file dumbo.py \
    -file examples.npy \
    -jobconf mapred.reduce.tasks=0 \
    -jobconf mapred.input.format.class=org.apache.hadoop.mapred.lib.NLineInputFormat \
    -jobconf mapred.line.input.format.linespermap=1


"""
 
import sys
import os
import pickle
import numpy as np

try:
    import json
except ImportError:
    import simplejson as json

import bolt
import util
from auxtrainer import *


def serialize(arr):
    return " ".join(["%d:%.20f" %(idx, arr[idx]) for idx in arr.nonzero()[0]])


def main(separator='\t'):
    # input comes from STDIN (standard input)

    ds = bolt.io.MemoryDataset.load("examples.npy", verbose=0)
    instances = ds.instances[ds._idx]
    task_masks = None
    if os.path.exists("task_masks.pkl"):
        f = open("task_masks.pkl", "rb")
        task_masks = pickle.load(f)
        f.close()

    for line in sys.stdin.xreadlines():
        line = line.rstrip()
        rid = "rid" + str(hash(line))  # run id
        line = line.split("\t")[-1]
        params = json.loads(line)
        taskid = params[u"taskid"]
        auxtask = np.array(params[u"task"])
        trainer = eval(params[u"trainer"])

        # label according to auxtask
        labels = util.autolabel(instances, auxtask)

        # mask features (either only auxtask or provided masks)
        mask = np.ones((ds.dim,), dtype=np.int32, order="C")
        if task_masks != None:
            mask[task_masks[taskid]] = 0
        else:
            mask[auxtask] = 0

        new_dataset = bolt.io.MemoryDataset(ds.dim, instances, labels)

        w = trainer.train_classifier(new_dataset, mask)
        sw = serialize(w)
        print >> sys.stdout, "%d\t%s" % (taskid, sw)


if __name__ == "__main__":
    main()
