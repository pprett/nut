#!/usr/bin/python
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style.
"""
The :mod:`bolt.io` module provides :class:`Dataset` specifications and routines for convenient input/output. 

The module provides the following classes:

  :class:`bolt.io.MemoryDataset` : an in-memory dataset. 

"""

import sys
import numpy as np
import gzip

from itertools import izip
from collections import defaultdict

sparsedtype = np.dtype("u4,f4")
"""The data type of sparse vectors.

Example:

>>> x = np.array([(0,1),(4,0.2)], dtype = sparsedtype)
  
"""

densedtype = np.float32

def fromlist(l, dtype):
    """Create a numpy array with data type `dtype` from the given list `l`.

    Arguments
    ---------
    l : list
        The list to be converted.

    dtype : np.dtype
        The numpy data type (e.g. bolt.io.sparsetype or np.float64).

    Returns
    -------
    array, shape=[len(l)]
        The array representation of the list `l`.

    Examples
    --------
    >>> bolt.io.fromlist([(0,0.1),(1,1.0)], bolt.io.sparsedtype)
    array([(0L, 0.10000000149011612), (1L, 1.0)], 
      dtype=[('f0', '<u4'), ('f1', '<f4')])
    """
    length = len(l)
    arr = np.empty((length,), dtype = dtype)
    arr[:] = l
    return arr

def dense2sparse(x):
    """Convert numpy arrays of `bolt.io.densetype` to sparse arrays of `bolt.io.sparsetype`.

    Examples
    --------
    >>> x = bolt.io.fromlist([1,0,0,0,0.2], bolt.io.densedtype)
    >>> bolt.io.dense2sparse(x)
    array([(0L, 1.0), (4L, 0.20000000298023224)], 
      dtype=[('f0', '<u4'), ('f1', '<f4')])
    """
    return fromlist([(nnz, x[nnz]) for nnz in x.nonzero()[0]],sparsedtype)

class Dataset(object):
    """Dataset interface.
    """

    def __iter__(self):
        pass

    def iterinstances(self):
        pass

    def iterlabels(self):
        pass

    def shuffle(self, seed = None):
        pass    

class MemoryDataset(Dataset):
    """An in-memory dataset.
    The instances and labels are stored as two parallel arrays.
    Access to the parallel arrays is via an indexing array which
    allows convenient shuffeling.

    .. todo:
      Implement in Cython if CEP 307 is done.
    """
    def __init__(self, dim, instances, labels, qids=None):
        """
        :arg dim: The dimensionality of the data; the number of features.
        :type dim: integer
        :arg instances: An array of instances.
        :type instances: :class:`numpy.ndarray(dtype=numpy.object)`
        :arg labels: An array of encoded labels associated with the `instances`. 
        :type labels: :class:`numpy.ndarray(dtype=numpy.float64)`
        :arg qids: An optional array of datatype int32 which holds
        the query ids of the associated example.
        :type qids: :class:`numpy.ndarray(dtype=numpy.float64)` or `None`
        """
        assert len(instances) == len(labels)
        assert isinstance(instances, np.ndarray)
        assert all((x.dtype == sparsedtype for x in instances))
        assert instances.dtype == np.object
        assert isinstance(labels, np.ndarray)
        #assert labels.dtype == np.float64
        
        self.dim = dim
        """The dimensionality of the examples in the dataset. """
        self.n = len(instances)
        """The number of instances in the dataset. """
        self.instances = instances
        """The array holding the instances. """
        self.labels = labels
        """The array holding the labels. """
        self._idx = np.arange(self.n)
        """The indexing array. """
        self.classes = np.unique(labels)
        """The classes. """
        self.qids = qids
        
    def __iter__(self):
        """Iterate over training examples."""
        return izip(self.instances[self._idx],self.labels[self._idx])

    def iterinstances(self):
        """Iterate over instances. """
        for i in self._idx:
            yield self.instances[i]

    def iterlabels(self):
        """Iterate over labels. """
        for i in self._idx:
            yield self.labels[i]

    def iterqids(self):
        """Iterate over query ids. """
        for i in self._idx:
            yield self.qids[i]

    def shuffle(self, seed = None):
        """Shuffles the index array using `numpy.random.shuffle`.
        A `seed` for the pseudo random number generator can be provided.
        """
        rs = np.random.RandomState()
        rs.seed(seed)
        rs.shuffle(self._idx)

    def split(self, nfolds):
        """Split the `Dataset` into `nfolds` new `Dataset` objects.
        The split is done according to the index array.
        If `MemoryDataset.n % nfolds` is not 0 the remaining examples
        are discarded. 

        :arg nfolds: The number of folds
        :type nfolds: integer
        :returns: An array of `nfolds` MemoryDataset objects; one for each fold
        """
        r = self.n % nfolds
        num = self.n-r
        folds = np.split(self._idx[:num],nfolds)
        dsets = []
        for fold in folds:
            splitqids = None
            if self.qids != None:
                splitqids = self.qids[fold]
            dsets.append(MemoryDataset(self.dim, self.instances[fold],
                                       self.labels[fold], qids = splitqids))
        return fromlist(dsets, np.object)

    def sample(self, nexamples, seed = None):
        """Samples `nexamples` examples from the dataset.

        :arg nexamples: The number of examples to sample.
        :type nexamples: integer
        :arg seed: The random seed. 
        :returns: A `MemoryDataset` containing the `nexamples` examples. 
        """
        idx = np.arange(self.n)
        rs = np.random.RandomState()
        rs.seed(seed)
        rs.shuffle(idx)
        idx = idx[:nexamples]
        sampleqids = None
        if self.qids != None:
            sampleqids = self.qids[idx]
        return MemoryDataset(self.dim, self.instances[idx], self.labels[idx],
                             qids = sampleqids)

    @classmethod
    def merge(cls, dsets):
        """Merge a sequence of :class:`Dataset` objects.

        :arg dsets: A list of :class:`MemoryDataset`
        :returns: A :class:`MemoryDataset` containing the merged examples. 
        """
        assert len(dsets) > 1
        instances = np.concatenate([ds.instances[ds._idx] for ds in dsets])
        labels = np.concatenate([ds.labels[ds._idx] for ds in dsets])
        qids = None
        if np.all([ds.qids != None for ds in dsets]):
            qids = np.concatenate([ds.qids[ds._idx] for ds in dsets])
        return MemoryDataset(dsets[0].dim, instances, labels, qids = qids)        

    @classmethod
    def load(cls, fname, verbose = 1, qids = False):
        """Loads the dataset from `fname`.

        Currently, two formats are supported:
          a. numpy binary format
          b. SVM^light format

        For binary format the extension of `fname` has to be
        `.npy`, otherwise SVM^light format is assumed.
        For gzipped files thefilename must end with `.gz`.

        :arg fname: The file name of the seralized :class:`Dataset`
        :arg verbose: Verbose output
        :type verbose: integer
        :arg qids: Whether to load qids or not
        :type qids: True or False
        :returns: The :class:`MemoryDataset`

        Examples:

              SVM^light format:

          >>> ds = bolt.io.MemoryDataset.load('train.txt')

          Gzipped SVM^light format:

          >>> ds = bolt.io.MemoryDataset.load('train.txt.gz')

              Binary format:

          >>> ds = bolt.io.MemoryDataset.load('train.npy')
        """
        if verbose > 0:
            print "loading data ...",
        sys.stdout.flush()
        try:
            loader = None
            if fname.endswith('.npy'):
                loader = load_npz
            else:
                loader = load_dat
            data = loader(fname, qids = qids)
        except IOError, (errno, strerror):
            if verbose > 0:
                print(" [fail]")
            raise Exception("cannot open '%s' - %s." % (fname,strerror))
        except Exception, exc:
            if verbose > 0:
                print(" [fail]")
            raise Exception(exc)
        else:
            if verbose > 0:
                print(" [done]")

        if verbose > 1:
            instances = data[1]
            print("%d examples loaded. " % (len(instances)))
        return MemoryDataset(*data)


    def store(self,fname):
        """Store `Dataset` in binary form.
        Uses `numpy.save` for serialization.

        :arg fname: The filename
        """
        f = open(fname,'w+b')
        try:
            # FIXME we should also store self._idx.
            np.save(f,self.instances[self._idx])
            np.save(f,self.labels[self._idx])
            np.save(f,self.dim)
            if self.qids != None:
                np.save(f,self.qids)
        finally:
            f.close()

class BinaryDataset(Dataset):
    """A `Dataset` wrapper which binarizes the class labels.

    Most methods are deligated to the wrapped `Dataset`.
    Only :func:`BinaryDataset.__iter__` and  :func:`BinaryDataset.labels` are wrapped.
    """
    def __init__(self, dataset, c):
        """Creates a binary class wrapper for `dataset`.
        
        :arg dataset: The `Dataset` to wrap.
        :arg c: The positive class. All other classes are treated as negative.
        """
        self._dataset = dataset
        self.c = c
        self.n = dataset.n
        self.dim = dataset.dim
        self.classes = np.array([1,-1],dtype=np.float32)

    def __iter__(self):
        return ((x,self.mask(y)) for x,y in self._dataset)

    def iterinstances(self):
        return self._dataset.iterinstances()

    def iterlabels(self):
        return (self.mask(y) for y in self._dataset.iterlabels())

    def shuffle(self, seed = None):
        self._dataset.shuffle(seed)

    def mask(self, y):
        return 1 if y == self.c else -1


def load_npz(filename, qids=False):
    """Load data from numpy binary format.
    """
    f = open(filename,'rb')
    try:
        instances = np.load(f)
        labels = np.load(f)
        dim = np.load(f)
        res = [dim, instances, labels]
        if qids:
            qids = np.load(f)
            res.append(qids)
        return res
    finally:
        f.close()

def load_dat(filename, qids=False):
    """Load data from svm^light formatted file.
    """
    labels = []
    instances = []
    queries = []
    global_max = -1
    if filename.endswith('gz'):
        f=gzip.open(filename,'r')
    else:
        f=open(filename,'r')
    
    try:
        for i,line in enumerate(f):
            tokens = line.split('#')[0].rstrip().split()
            label = float(tokens[0])
            labels.append(label)
            del tokens[0]
            if len(tokens) > 0 and tokens[0].startswith("qid"):
                queries.append(int(tokens[0].split(":")[1]))
                del tokens[0]
        
            tokens=[(int(t[0]),float(t[1]))
                    for t in (t.split(':')
                              for t in tokens if t != '')]
            tokens = sorted(tokens)
            a = fromlist(tokens, sparsedtype)
            local_max = 0.0
            if a.shape[0]>0:
                local_max = a['f0'].max()
            if local_max > global_max:
                global_max = local_max
            instances.append(a)
        res = [global_max+1, fromlist(instances, np.object),
               np.array(labels)]
        if qids:
            res.append(fromlist(queries, np.int32))
        return res
    
    finally:
        f.close()
        
def svmlToNpy():
    if len(sys.argv) < 3 or "--help" in sys.argv:
        print """Usage: %s in-file out-file

    Converts the svm^light encoded in-file into the binary encoded out-file.
    """ % "svml2npy"
    sys.exit(-2)
    in_filename, out_filename = sys.argv[1:3]
    if "--qids" in sys.argv:
        qids = True
    else:
        qids = False
    ds = MemoryDataset.load(in_filename, qids=qids)
    ds.store(out_filename)
    
