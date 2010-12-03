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
import copy
import math

try:
    import json
except ImportError:
    import simplejson as json

import bolt
import util


def serialize(arr):
    return " ".join(["%d:%.20f"%(idx,arr[idx]) for idx in arr.nonzero()[0]])


def train(ds, reg=0.00001, alpha=0.85, norm=2, n_iter=10**6):
    epochs = int(math.ceil(float(n_iter) / float(ds.n)))
    loss = bolt.ModifiedHuber()
    model = bolt.LinearModel(ds.dim, biasterm=False)
    sgd = bolt.SGD(loss, reg, epochs=epochs, norm=norm, alpha=alpha)
    sgd.train(model, ds, verbose=0, shuffle=False)
    return model.w

  
def main(separator='\t'):
    # input comes from STDIN (standard input)

    ds = bolt.io.MemoryDataset.load("examples.npy", verbose=0)
    original_instances = ds.instances[ds._idx]

    for line in sys.stdin.xreadlines():
        line = line.rstrip()
        rid = "rid" + str(hash(line))  # run id
        line = line.split("\t")[-1]
        params = json.loads(line)
        taskid = params[u"taskid"]
        auxtask = eval(params[u"task"])
        reg = params[u"reg"]
        alpha = params.get(u"alpha", 0.85)
        norm = params.get(u"norm", 3)
        n_iter = params.get(u"n_iter", 10**6)

        instances = copy.deepcopy(original_instances)
        labels = util.autolabel(instances, auxtask)
        util.mask(instances, auxtask)
        maskedds = bolt.io.MemoryDataset(ds.dim, instances, labels)
        w = train(maskedds, reg=reg, alpha=alpha, norm=norm, n_iter=n_iter)
        if norm == 2:
            w[w<0.0] = 0.0
        
        sw = serialize(w)  
        print >> sys.stdout, "%d\t%s" % (taskid,sw)


if __name__ == "__main__":
    main()
