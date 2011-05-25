# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""
Wrapper class for word representation features by
Joseph Turian 'Word representations: A simple and general method
for semi-supervised learning', ACL 2010

"""
from __future__ import division


class BrownClusters(object):
    """Brown cluster word representation. Each word is represented as a bit
    string which corresponds to its location in the hierachical brown
    clustering. We use `prefixes` as the concrete word features.
    Thus, for each word `len(prefixes)` features are induced.
    """

    def __init__(self, fname, prefixes=[4, 6, 10, 20]):
        self.prefixes = prefixes
        token_cluster_map = dict()
        fd = open(fname, "r")
        try:
            for line in fd:
                cluster_str, token, count = line.split()
                token_cluster_map[token] = cluster_str
        finally:
            fd.close()
        self._token_cluster_map = token_cluster_map

    def __getitem__(self, token):
        cluster_str = self._token_cluster_map[token]
        return [cluster_str[:prefix] for prefix in self.prefixes]

    def __contains__(self, token):
        return token in self._token_cluster_map

    def __len__(self):
        return len(self._token_cluster_map)
