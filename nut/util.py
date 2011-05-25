#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

"""
util
====

A collection of utilities for text processing and debugging.
"""

import sys
import time
import re

from .externals.bolt.io import MemoryDataset

__author__ = "Peter Prettenhofer <peter.prettenhofer@gmail.com>"


def timeit(func):
    """A timit decorator. """
    def wrapper(*arg, **kargs):
        t1 = time.time()
        res = func(*arg, **kargs)
        t2 = time.time()
        print '%s took %0.3f sec' % (func.func_name, (t2 - t1))
        return res
    return wrapper


def trace(func):
    """A function trace decorator. """
    def wrapper(*args, **kargs):
        print "calling %s with args %s, %s" % (func.__name__, args, kargs)
        return func(*args, **kargs)
    return wrapper


def sizeof(d):
    """Retuns size of datastructure in MBs. """
    bytes = 0
    if hasattr(d, "nbytes"):
        bytes = d.nbytes
    elif isinstance(d, MemoryDataset):
        for i in d.iterinstances():
            # each examples has 8*nnz + label + idx
            bytes += (i.shape[0] * 8) + 4 + 4
    elif isinstance(d, dict):
        for k, v in d.iteritems():
            bytes += sys.getsizeof(k) + sys.getsizeof(v)
    elif isinstance(d, list):
        for e in d:
            bytes += sys.getsizeof(e)
    else:
        bytes = sys.getsizeof(d)
    return bytes / 1024.0 / 1024.0


class WordTokenizer(object):
    """Word tokenizer adapted from NLTKs WordPunktTokenizer.

    NOTE: splits email adresses.

    Example
    -------
    >>> tokenizer = WordTokenizer()
    >>> tokenizer.tokenize("Here's a url\nwww.bitly.com.")
    ['Here', "'s", 'a', 'url', 'www.bitly.com', '.']
    """

    _re_word_start = r"[^\(\"\`{\[:;&\#\*@\)}\]\-,]"
    """Excludes some characters from starting word tokens"""

    _re_non_word_chars = r"(?:[?!)\";}\]\*:@\'\({\[])"
    """Characters that cannot appear within words"""

    _re_multi_char_punct = r"(?:\-{2,}|\.{2,}|(?:\.\s){2,}\.)"
    """Hyphen and ellipsis are multi-character punctuation"""

    _word_tokenize_fmt = r"""(
        %(MultiChar)s
        |
        (?=%(WordStart)s)\S+?  # Accept word characters until end is found
        (?= # Sequences marking a word's end
            \s|                                 # White-space
            $|                                  # End-of-string
            %(NonWord)s|%(MultiChar)s|          # Punctuation
            ,(?=$|\s|%(NonWord)s|%(MultiChar)s)| # Comma if at end of word
            \.(?=$|\s|%(NonWord)s|%(MultiChar)s) # Dot if at end of word
        )
        |
        \S
        )"""

    def __init__(self):
        """
        """
        self._regex = re.compile(
                self._word_tokenize_fmt %
                {
                    'NonWord':   self._re_non_word_chars,
                    'MultiChar': self._re_multi_char_punct,
                    'WordStart': self._re_word_start,
                },
                re.UNICODE | re.VERBOSE
            )

    def tokenize(self, s):
        """Tokenize a string to split of punctuation."""
        return self._regex.findall(s)
