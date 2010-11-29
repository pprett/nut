#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
io
==

This package contains various classes for IO routines;
in particular corpus reader classes.


"""
import bz2
import gzip
import cPickle as pickle

from . import conll


def compressed_dump(fname, model):
    """Pickle the model and write it to `fname`.
    If name ends with '.gz' or '.bz2' use the
    corresponding compressors else it pickles
    in binary format.

    Parameters
    ----------
    fname : str
        Where the model shall be written.
    model : object
        The object to be pickeled.
    """
    if fname.endswith(".gz"):
        f = gzip.open(fname, mode="wb")
    elif fname.endswith(".bz2"):
        f = bz2.BZ2File(fname, mode="w")
    else:
        f = open(fname, "wb")
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
    f.close()


def compressed_load(fname):
    """Unpickle a model from `fname`. If `fname`
    endswith '.bz2' or '.gz' use the corresponding
    decompressor otherwise unpickle binary format.

    Parameters
    ----------
    fname : str
        From where the model shall be read.
    """
    if fname.endswith(".gz"):
        f = gzip.open(fname, mode="rb")
    elif fname.endswith(".bz2"):
        f = bz2.BZ2File(fname, mode="r")
    else:
        f = open(fname, "rb")
    model = pickle.load(f)
    f.close()
    return model
