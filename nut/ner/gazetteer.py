# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""
Gazetteer features for named entity recognition.
"""
from __future__ import division


def encode_iob(i, length):
    enc = "I"
    if i == 0:
        enc = "B"
    return enc


def encode_bilou(i, length):
    enc = "I"
    if i == 0:
        if length == 1:
            enc = "U"
        else:
            enc = "B"
    elif length > 1 and i == (length - 1):
        enc = "L"
    return enc


class Gazetteer(object):
    """An gazetteer (word list) abstraction. Each concept in the
    gazetteer is represented by an encoding.

    E.g. suppose that New York Times is in the gazetteer, a query for
    'New' will return 'B', indicating that 'New' is registered as a
    beginning token of a concept in the gazetteer.

    NOTE: assumes latin1 encoding.

    Parameters
    ----------
    encoding : str (either 'iob' or '')
        How the concepts in the gazetteer are represented (= encoded).

    Example
    -------
    >>> gaz = Gazetteer('wikipersons.txt', encoding='iob')
    >>> gaz['John']
    'BI'
    """

    def __init__(self, fname, encoding="iob", casesensitive=True):
        self.fname = fname
        self.casesensitive = casesensitive
        encoder = {"iob": encode_iob, "bilou": encode_bilou}[encoding]
        gazetteer = {}
        fd = open(fname)
        try:
            for line in fd:
                if not self.casesensitive:
                    line = line.lower()
                tokens = line.strip().split()
                length = len(tokens)
                for i, token in enumerate(tokens):
                    enc = encoder(i, length)
                    if token not in gazetteer:
                        gazetteer[token] = enc
                    else:
                        if enc not in gazetteer[token]:
                            gazetteer[token] = "".join(sorted(gazetteer[token] + enc))
        finally:
            fd.close()
        self.gazetteer = gazetteer

    def __contains__(self, token):
        if not self.casesensitive:
            token = token.lower()
        return token in self.gazetteer

    def __getitem__(self, token):
        if not self.casesensitive:
            token = token.lower()
        return self.gazetteer[token]

    def get_features(self, name, token):
        """Returns gazetteer features for the given token.
        Features are tuples (name, char). """
        res = []
        if token in self.gazetteer:
            res = [(name, c) for c in self.gazetteer[token]]
        return res


class SimpleGazetteer(object):
    """An simple gazetteer (word list) for lists of single words.
    """

    def __init__(self, fname, encoding="iob", casesensitive=True):
        self.fname = fname
        self.casesensitive = casesensitive
        gazetteer = set()

        fd = open(fname)
        try:
            for token in fd:
                token = token.rstrip()
                if not self.casesensitive:
                    token = token.lower()
                if token not in gazetteer:
                    gazetteer.add(token)
        finally:
            fd.close()
        self.gazetteer = gazetteer

    def __contains__(self, token):
        if not self.casesensitive:
            token = token.lower()
        return token in self.gazetteer
