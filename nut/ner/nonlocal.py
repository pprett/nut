# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""
Non-local features for named entity recognition.
"""
from __future__ import division

import sys
import numpy as np

class Distribution(dict):
    """A frozen defaultdict that returns np.array objects.
    This is used instead of defaultdict due to pickeling support."""

    def __init__(self, n_tags):
        super(Distribution, self).__init__()
        self.n_tags = n_tags

    def __missing__(self, word):
        self[word] = np.zeros((self.n_tags,), dtype=np.uint16)
        return self[word]

class ExtendedPredictionHistory(object):
    """The extended predicition history non-local
    feature presented in [Ratinov and Roth, 2009].

    Parameters
    ----------
    tag_map : dict
        A mapping from the tags to an index in [0, len(tags)].
    capacity : int
        How far back the history should be tracked.

    Attributes
    ----------
    `fifo` : the FIFO datastructure holding the events.
    """

    def __init__(self, tag_map, capacity=1000):
        self.capacity = capacity
        self.tag_map = tag_map
        n_tags = len(tag_map)
        assert np.max(tag_map.values()) < n_tags
        self._fifo = []
        self._distribution = Distribution(n_tags)

    def push(self, word, tag):
        """Pushes a new (word, tag) pair into the history.
        """
        if len(self._fifo) == self.capacity:
            # throw out last item if capacity if full
            old_word, old_tag = self._fifo.pop()
            assert self._distribution[old_word][self.tag_map[old_tag]] != 0
            self._distribution[old_word][self.tag_map[old_tag]] -= 1
            ## XXX kick out word from dict too?
            #print >> sys.stderr, "[eph] throwing (%s,%s) out." % (old_word,
            #                                                      old_tag)
        self._fifo.insert(0, (word, tag))
        self._distribution[word][self.tag_map[tag]] += 1
        #print >> sys.stderr, "[eph] (%s,%s) inserted." % (word, tag)

    def __getitem__(self, word):
        return self._distribution[word]

    def __contains__(self, word):
        return word in self._distribution

    def __len__(self):
        return len(self._fifo)

    def distribution(self, word):
        a = self._distribution[word]
        return a / a.sum()
