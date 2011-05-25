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
    """

    def __init__(self, fname, encoding="iob"):
        self.fname = fname
        encoder = {"iob": encode_iob, "bilou": encode_bilou}[encoding]
        gazetteer = {}

        fd = open(fname)
        try:
            for line in fd:
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
        return token in self.gazetteer

    def __getitem__(self, token):
        return self.gazetteer[token]
