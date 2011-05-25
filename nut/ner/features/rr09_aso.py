"""Feature and history detectors for CoNLL03 taken from:
`Ratinov, L. and Roth, D. (2009). Design challenges and misconceptions in
named entity recognition. In CoNLL '09`.

This feature extractor is inteded to be used in the NER implementation of ASO.
It uses the following naming convention for feature names:
<parent-type>(_<position>)+

parent-type is either word, pos, shape, etc. In combination with the position
arguments it determines the _feature type_ of a feature. In ASO the SVD is performed
for each feature type separately.

In `fd` there are more than 45 feature types!
"""

WORD, POS, NP, LEMMA = 0, 1, 2, 3


def shape_extractor(feature_name, word):
    features = []
    if word.istitle():
        features.append("istitle")
    if word.isdigit():
        features.append("isdigit")
    if word.isupper():
        features.append("isupper")
    if "-" in word[1:-1]:
        features.append("inhyphen")
    if word.isalnum():
        features.append("isalnum")
    return [(feature_name, f) for f in features]


def numify(s):
    if sum((1 for c in s if not c.isdigit())) > 2:
        return s
    else:
        return "".join(["*d*" if c.isdigit() else c for c in s])

class Detector(object):

    def __init__(self):
        pass


    def fd(self, sent, index, length):
        """Feature detector for CoNLL 2003 from RR09.
        """
        context = lambda idx, field: sent[index + idx][field] \
                  if index+idx >= 0 and index + idx < length \
                  else "<s>" if index+idx < 0 \
                  else "</s>"

        ## tokens in a 5 token window x_{i-2}..x_{i+2}
        uword_cur = numify(context(0, WORD))
        uword_pre = numify(context(-1, WORD))
        uword_2pre = numify(context(-2, WORD))
        uword_post = numify(context(1, WORD))
        uword_2post = numify(context(2, WORD))

        ## token bigrams in a 5 token window
        bword_pre_cur = "/".join([uword_pre, uword_cur])
        bword_cur_post = "/".join([uword_cur, uword_post])

        ## pos in a 5 token window
        upos_cur = context(0, POS)
        upos_pre = context(-1, POS)
        upos_post = context(1, POS)
        upos_2pre = context(-2, POS)
        upos_2post = context(2, POS)

        ## pos bigrams in a 3 token window
        bpos_pre_cur = "/".join([upos_pre, upos_cur])
        bpos_cur_post = "/".join([upos_cur, upos_post])

        bposw_cur = "/".join([uword_cur, upos_cur])

        ## 2-4 suffixes in a 3 token window
        suffix1_cur = uword_cur[-1:]
        suffix2_cur = uword_cur[-2:]
        suffix3_cur = uword_cur[-3:]
        suffix4_cur = uword_cur[-4:]

        suffix1_pre = uword_pre[-1:]
        suffix2_pre = uword_pre[-2:]
        suffix3_pre = uword_pre[-3:]
        suffix4_pre = uword_pre[-4:]

        suffix1_post = uword_post[-1:]
        suffix2_post = uword_post[-2:]
        suffix3_post = uword_post[-3:]
        suffix4_post = uword_post[-4:]

        ## 3-4 prefixes in a 3 token window
        prefix3_cur = uword_cur[:3]
        prefix4_cur = uword_cur[:4]

        prefix3_pre = uword_pre[:3]
        prefix4_pre = uword_pre[:4]

        prefix3_post = uword_post[:3]
        prefix4_post = uword_post[:4]

        ## Noun phrase in a 3 token window
        unp_cur = context(0, NP)
        unp_pre = context(-1, NP)
        unp_post = context(1, NP)

        bnpw_cur = "/".join([unp_cur, uword_cur])

        ## Extract features from local scope
        features = locals()
        del features["context"]
        del features["sent"]
        del features["index"]
        del features["length"]
        del features["self"]
        features = features.items()

        ## Word shape features (5 token window)
        features.extend(shape_extractor("shape_cur", context(0, WORD)))
        features.extend(shape_extractor("shape_pre", context(-1, WORD)))
        features.extend(shape_extractor("shape_2pre", context(-2, WORD)))
        features.extend(shape_extractor("shape_post", context(1, WORD)))
        features.extend(shape_extractor("shape_2post", context(2, WORD)))

        return features


    def hd(self, tags, sent, index, length):
        context = lambda idx, field: sent[index + idx][field] \
                  if index+idx >= 0 and index + idx < length \
                  else "<s>" if index+idx < 0 \
                  else "</s>"

        utag_pre = tags[index - 1] if index - 1 >= 0 else "<s>"
        utag_2pre = tags[index - 2] if index - 2 >= 0 else "<s>"
        btagw_pre_cur = "/".join([utag_pre, numify(context(0, WORD))])
        btagw_pre_pre = "/".join([utag_pre, numify(context(-1, WORD))])
        btagw_pre_2pre = "/".join([utag_pre, numify(context(-2, WORD))])
        btagw_pre_post = "/".join([utag_pre, numify(context(1, WORD))])
        btagw_pre_2post = "/".join([utag_pre, numify(context(2, WORD))])
        btagw_2pre_pre = "/".join([utag_2pre, utag_pre])

        history = locals()
        del history["context"]
        del history["tags"]
        del history["sent"]
        del history["index"]
        del history["length"]
        del history["self"]
        history = history.items()
        return history
