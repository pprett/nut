"""Feature and history detectors for German NER demo:

It does not use part-of-speech and noun-phrase information provided by
CoNLL03.

It uses the following feature templates:

  * word identity
  * word identity bigrams
  * word shape (title, upper, hyphen, mixedcase)
  * word pre-/suffixes (from 1 to 4 chars)
  * preceeding tags (i-1 and i-2)
  * preceeding tag + current word
  * brown clusters (4, 6, 10, 20 prefixes)
  * gazetteers (first names, places)

Brown clusters from:
J. Turian, et al. (2010). `Word Representations: A Simple and General Method
for Semi-Supervised Learning'. In ACL '10.

"""
import re
from itertools import izip
from string import punctuation

from ..wordembedding import BrownClusters
from ..gazetteer import Gazetteer, SimpleGazetteer


WORD, POS, NP, LEMMA = 0, 1, 2, 3


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------

def numify(s):
    """abstract repr of a token containing digits.

    Example
    -------
    >>> numify('2000-12-12')
    *d**d**d**d*-*d**d*-*d**d*'
    """
    if sum((1 for c in s if not c.isdigit())) > 2:
        return s
    else:
        return "".join(["*d*" if c.isdigit() else c for c in s])

def caseabstract(s):
    """abstracts the characters of a string (upper, lower, digit, else).
    Example
    -------
    >>> caseabstract('B2B')
    'ADA'
    >>> caseabstract('W.')
    'AS'
    """
    def code(c):
        if c.isupper():
            return "A"
        elif c.islower():
            return "a"
        elif c.isdigit():
            return "D"
        else:
            return "S"
    return "".join([code(c) for c in s])


# ------------------------------------------------------------------------------
# The detector
# ------------------------------------------------------------------------------


class Detector(object):

    def __init__(self):
        self.punctuation = set(punctuation)
        self.brown_clusters = BrownClusters("clner/de/resources/brown-clusters/" + 
                                            "brown-clusters.txt",
                                            prefixes=[4, 6, 10, 20])

        # Regular expressions
        self.mixedcase = re.compile(r"^[A-Z]\w+[A-Z]\w*$")
        
        self.ccwp = re.compile(r"^[A-Z]\.$")  # captialzed char with period.
        self.tcsh = re.compile(r"^[A-Z][a-z]+-[A-Z][a-z]+$")  # title cased separated by hyphen.

        # Gazetteers
        self.firstnames = SimpleGazetteer("clner/de/resources/gazetteers/" \
                                          "firstnames.txt")
        self.known_places = Gazetteer("clner/de/resources/gazetteers/" \
                                      "known_places.lst", encoding="bilou")
        self.known_orgs = Gazetteer("clner/de/resources/gazetteers/" \
                                    "known_orgs.lst", encoding="bilou")

        self.name_prefixes = set(["Frau", "Herr", "Fr.", "Hr."])

    def brown_extractor(self, name, token):
        """Creates brown features for token. For each path prefix
        one feature is creates.

        Parameters
        ----------
        name : str
            The name of the features, must contain %d which will
            be set via the concrete prefix.
        token : str
            The token for which the brown clustering is used; case
            sensitive.
        """
        if token in self.brown_clusters:
            clusters = izip(self.brown_clusters.prefixes,
                            self.brown_clusters[token])
            features = [(name % prefix, encoding) for
                        prefix, encoding in clusters]
        else:
            features = []
   
        return features

    def fd(self, sent, index, length):
        """Feature detector for CoNLL 2003 from RR09.
        """
        context = lambda idx, field: sent[index + idx][field] \
                  if index+idx >= 0 and index + idx < length \
                  else "<s>" if index+idx < 0 \
                  else "</s>"

        ## tokens in a 5 token window x_{i-2}..x_{i+2}
        word_unigram_cur = numify(context(0, WORD))
        word_unigram_pre = numify(context(-1, WORD))
        word_unigram_2pre = numify(context(-2, WORD))
        word_unigram_post = numify(context(1, WORD))
        word_unigram_2post = numify(context(2, WORD))

        ## token bigrams in a 5 token window
        word_bigram_pre_cur = "/".join([word_unigram_pre, word_unigram_cur])
        word_bigram_cur_post = "/".join([word_unigram_cur, word_unigram_post])

        ## Word shape features (5 token window)
        shape_islower_cur = word_unigram_cur.islower()
        shape_istitle_cur = word_unigram_cur.istitle()
        shape_isdigit_cur = context(0, WORD).isdigit()
        shape_isupper_cur = word_unigram_cur.isupper()
        shape_hyphen_cur = "-" in word_unigram_cur[1:-1]
        shape_isalnum_cur = context(0, WORD).isalnum()
        shape_mixedcase_cur = self.mixedcase.match(context(0, WORD)) != None
        shape_ccwp_cur = self.ccwp.match(context(0, WORD)) != None
        shape_abstract_cur = caseabstract(context(0, WORD))
        shape_ispunctuation_cur = word_unigram_cur in self.punctuation
        shape_tcsh_cur = self.tcsh.match(context(0, WORD)) != None

        shape_islower_pre = word_unigram_pre.islower()
        shape_istitle_pre = word_unigram_pre.istitle()
        shape_isdigit_pre = context(-1, WORD).isdigit()
        shape_isupper_pre = word_unigram_pre.isupper()
        shape_hyphen_pre = "-" in word_unigram_pre[1:-1]
        shape_isalnum_pre = context(-1, WORD).isalnum()
        shape_mixedcase_pre = self.mixedcase.match(context(-1, WORD)) != None
        shape_ccwp_pre = self.ccwp.match(context(-1, WORD)) != None
        shape_abstract_pre = caseabstract(context(-1, WORD))
        shape_ispunctuation_pre = word_unigram_pre in self.punctuation
        shape_tcsh_pre = self.tcsh.match(context(-1, WORD)) != None

        shape_islower_2pre = word_unigram_2pre.islower()
        shape_istitle_2pre = word_unigram_2pre.istitle()
        shape_isdigit_2pre = context(-2, WORD).isdigit()
        shape_isupper_2pre = word_unigram_2pre.isupper()
        shape_hyphen_2pre = "-" in word_unigram_2pre[1:-1]
        shape_isalnum_2pre = context(-2, WORD).isalnum()
        shape_mixedcase_2pre = self.mixedcase.match(context(-2, WORD)) != None
        shape_ccwp_2pre = self.ccwp.match(context(-2, WORD)) != None
        shape_abstract_2pre = caseabstract(context(-2, WORD))
        shape_ispunctuation_2pre = word_unigram_2pre in self.punctuation
        shape_tcsh_2pre = self.tcsh.match(context(-2, WORD)) != None

        shape_islower_post = word_unigram_post.islower()
        shape_istitle_post = word_unigram_post.istitle()
        shape_isdigit_post = context(1, WORD).isdigit()
        shape_isupper_post = word_unigram_post.isupper()
        shape_hypen_post = "-" in word_unigram_post[1:-1]
        shape_isalnum_post = context(1, WORD).isalnum()
        shape_mixedcase_post = self.mixedcase.match(context(1, WORD)) != None
        shape_ccwp_post = self.ccwp.match(context(1, WORD)) != None
        shape_abstract_post = caseabstract(context(1, WORD))
        shape_ispunctuation_post = word_unigram_post in self.punctuation
        shape_tcsh_post = self.tcsh.match(context(1, WORD)) != None

        shape_islower_2post = word_unigram_2post.islower()
        shape_istitle_2post = word_unigram_2post.istitle()
        shape_isdigit_2post = context(2, WORD).isdigit()
        shape_isupper_2post = word_unigram_2post.isupper()
        shape_hypen_2post = "-" in word_unigram_2post[1:-1]
        shape_isalnum_2post = context(2, WORD).isalnum()
        shape_mixedcase_2post = self.mixedcase.match(context(2, WORD)) != None
        shape_ccwp_2post = self.ccwp.match(context(2, WORD)) != None
        shape_abstract_2post = caseabstract(context(2, WORD))
        shape_ispunctuation_2post = word_unigram_2post in self.punctuation
        shape_tcsh_2post = self.tcsh.match(context(2, WORD)) != None

        ## 2-4 suffixes in a 3 token window
        suffix_1_cur = word_unigram_cur[-1:]
        suffix_2_cur = word_unigram_cur[-2:]
        suffix_3_cur = word_unigram_cur[-3:]
        suffix_4_cur = word_unigram_cur[-4:]

        suffix_1_pre = word_unigram_pre[-1:]
        suffix_2_pre = word_unigram_pre[-2:]
        suffix_3_pre = word_unigram_pre[-3:]
        suffix_4_pre = word_unigram_pre[-4:]

        suffix_1_post = word_unigram_post[-1:]
        suffix_2_post = word_unigram_post[-2:]
        suffix_3_post = word_unigram_post[-3:]
        suffix_4_post = word_unigram_post[-4:]

        ## 3-4 prefixes in a 3 token window
        prefix_3_cur = word_unigram_cur[:3]
        prefix_4_cur = word_unigram_cur[:4]

        prefix_3_pre = word_unigram_pre[:3]
        prefix_4_pre = word_unigram_pre[:4]

        prefix_3_post = word_unigram_post[:3]
        prefix_4_post = word_unigram_post[:4]

        ## Trigger word list
        trg_name_pre = word_unigram_pre in self.name_prefixes and shape_istitle_cur
        trg_name_2pre = word_unigram_2pre in self.name_prefixes and shape_istitle_cur

        ## Gazetteer features
        gaz_firstname_cur = word_unigram_cur in self.firstnames
        gaz_firstname_pre = word_unigram_pre in self.firstnames
        gaz_firstname_post = word_unigram_post in self.firstnames

        ## Extract features from local scope
        features = locals()
        del features["context"]
        del features["sent"]
        del features["index"]
        del features["length"]
        del features["self"]
        features = features.items()

        ## Brown clusters
        features.extend(self.brown_extractor("brown_%d_cur", context(0, WORD)))
        features.extend(self.brown_extractor("brown_%d_pre", context(-1, WORD)))
        features.extend(self.brown_extractor("brown_%d_2pre", context(-2, WORD)))
        features.extend(self.brown_extractor("brown_%d_post", context(1, WORD)))
        features.extend(self.brown_extractor("brown_%d_2post", context(2, WORD)))

        ## Gazetteer features
        features.extend(self.known_places.get_features("gaz_place_cur",
                                                       context(0, WORD)))
        features.extend(self.known_places.get_features("gaz_place_pre",
                                                       context(-1, WORD)))
        features.extend(self.known_places.get_features("gaz_place_post",
                                                       context(1, WORD)))

        features.extend(self.known_orgs.get_features("gaz_org_cur",
                                                     context(0, WORD)))
        features.extend(self.known_orgs.get_features("gaz_org_pre",
                                                     context(-1, WORD)))
        features.extend(self.known_orgs.get_features("gaz_org_post",
                                                     context(1, WORD)))

        return features


    def hd(self, tags, sent, index, length):
        context = lambda idx, field: sent[index + idx][field] \
                  if index+idx >= 0 and index + idx < length \
                  else "<s>" if index+idx < 0 \
                  else "</s>"

        tag_tag_pre = tags[index - 1] if index - 1 >= 0 else "<s>"
        tag_tag_2pre = tags[index - 2] if index - 2 >= 0 else "<s>"
        tag_pretagword_cur = "/".join([tag_tag_pre, numify(context(0, WORD))])
        tag_pretagword_pre = "/".join([tag_tag_pre, numify(context(-1, WORD))])
        tag_pretagword_2pre = "/".join([tag_tag_pre, numify(context(-2, WORD))])
        tag_pretagword_post = "/".join([tag_tag_pre, numify(context(1, WORD))])
        tag_pretagword_2post = "/".join([tag_tag_pre, numify(context(2, WORD))])
        tag_bigram_2pre_pre = "/".join([tag_tag_2pre, tag_tag_pre])

        history = locals()
        del history["context"]
        del history["tags"]
        del history["sent"]
        del history["index"]
        del history["length"]
        del history["self"]
        history = history.items()
        return history
