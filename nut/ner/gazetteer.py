# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""
Gazetteer features for named entity recognition.
"""
from __future__ import division


class Gazetteer(object):
    """An gazetteer (word list) abstraction. Each concept in the
    gazetteer is represented by an encoding.

    E.g. suppose that New York Times is in the gazetteer, a query for
    'New' will return 'B', indicating that 'New' is registered as a
    beginning token of a concept in the gazetteer. 

    Parameters
    ----------
    encoding : str (either 'iob' or '')
        How the concepts in the gazetteer are represented (= encoded).
        

    Attributes
    ----------
    `fifo` : the FIFO datastructure holding the events.
    """

    def __init__(self, fname, encoding="iob"):
        self.fname = fname
        self.encoding = encoding
