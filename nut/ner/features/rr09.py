"""Feature and history detectors for CoNLL03 taken from:
`Ratinov, L. and Roth, D. (2009). Design challenges and misconceptions in named entity recognition. In CoNLL '00`.

TODO: short description.
"""

WORD, POS, NP, LEMMA = 0, 1, 2, 3


def numify(s):
    if sum((1 for c in s if not c.isdigit())) > 2:
        return s
    else:
        return "".join(["*d*" if c.isdigit() else c for c in s])


def fd(sent, index, length):
    """Feature detector for CoNLL 2003 from RR09.
    """
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    ## tokens in a 5 token window x_{i-2}..x_{i+2}
    w = numify(context(0, WORD))
    pre_w = numify(context(-1, WORD))
    pre_pre_w = numify(context(-2, WORD))
    post_w = numify(context(1, WORD))
    post_post_w = numify(context(2, WORD))

    ## token bigrams in a 5 token window
    pre_bigram = "/".join([pre_w, w])
    post_bigram = "/".join([w, post_w])
    #pre_pre_bigram = "/".join([pre_pre_w, pre_w])
    #post_post_bigram = "/".join([post_w, post_post_w])

    ## pos in a 5 token window
    pos = context(0, POS)
    pre_pos = context(-1, POS)
    post_pos = context(1, POS)
    pre_pre_pos = context(-2, POS)
    post_post_pos = context(2, POS)

    ## pos bigrams in a 5 token window
    pre_pos_bigram = "/".join([pre_pos, pos])
    post_pos_bigram = "/".join([pos, post_pos])
    #pre_pre_pos_bigram = "/".join([pre_pre_pos, pre_pos])
    #post_post_pos_bigram = "/".join([post_pos, post_post_pos])

    pos_w = "/".join([w, pos])
    
    ## Word shape features (5 token window)
    istitle = w.istitle()
    isdigit = context(0, WORD).isdigit()
    isupper = w.isupper()
    hyphen = "-" in w[1:-1]
    isalnum = context(0, WORD).isalnum()
    
    pre_istitle = pre_w.istitle()
    pre_isdigit = context(-1, WORD).isdigit()
    pre_isupper = pre_w.isupper()
    pre_hyphen = "-" in pre_w[1:-1]
    pre_isalnum = context(-1, WORD).isalnum()

    pre_pre_istitle = pre_pre_w.istitle()
    pre_pre_isdigit = context(-2, WORD).isdigit()
    pre_pre_isupper = pre_pre_w.isupper()
    pre_pre_hyphen = "-" in pre_pre_w[1:-1]
    pre_pre_isalnum = context(-2, WORD).isalnum()

    post_istitle = post_w.istitle()
    post_isdigit = context(1, WORD).isdigit()
    post_isupper = post_w.isupper()
    post_hypen = "-" in post_w[1:-1]
    post_isalnum = context(1, WORD).isalnum()

    post_post_istitle = post_post_w.istitle()
    post_post_isdigit = context(2, WORD).isdigit()
    post_post_isupper = post_post_w.isupper()
    post_post_hypen = "-" in post_post_w[1:-1]
    post_post_isalnum = context(2, WORD).isalnum()

    ## 2-4 suffixes in a 3 token window
    w_suffix1 = w[-1:]
    w_suffix2 = w[-2:]
    w_suffix3 = w[-3:]
    w_suffix4 = w[-4:]

    pre_w_suffix1 = pre_w[-1:]
    pre_w_suffix2 = pre_w[-2:]
    pre_w_suffix3 = pre_w[-3:]
    pre_w_suffix4 = pre_w[-4:]

    post_w_suffix1 = post_w[-1:]
    post_w_suffix2 = post_w[-2:]
    post_w_suffix3 = post_w[-3:]
    post_w_suffix4 = post_w[-4:]

    ## 3-4 prefixes in a 3 token window
    w_prefix3 = w[:3]
    w_prefix4 = w[:4]

    pre_w_prefix3 = pre_w[:3]
    pre_w_prefix4 = pre_w[:4]

    post_w_prefix3 = post_w[:3]
    post_w_prefix4 = post_w[:4]

    ## Noun phrase in a 3 token window
    np = context(0,NP)
    np_w = "/".join([np, w])
    pre_np = context(-1, NP)
    post_np = context(1, NP)

    ## Extract features from local scope
    features = locals()
    del features["context"]
    del features["sent"]
    del features["index"]
    del features["length"]
    features = features.items()
    return features


def hd(tags, sent, index, length):
    context = lambda idx, field: sent[index + idx][field] \
              if index+idx >= 0 and index + idx < length \
              else "<s>" if index+idx < 0 \
              else "</s>"

    pre_tag = tags[index - 1] if index - 1 >= 0 else "<s>"
    pre_pre_tag = tags[index - 2] if index - 2 >= 0 else "<s>"
    pre_tag_w = "/".join([pre_tag, context(0, WORD)])
    pre_tag_w = "/".join([pre_tag, context(-1, WORD)])
    pre_tag_w = "/".join([pre_tag, context(-2, WORD)])
    pre_tag_w = "/".join([pre_tag, context(1, WORD)])
    pre_tag_w = "/".join([pre_tag, context(2, WORD)])
    tag_bigram = "/".join([pre_pre_tag, pre_tag])

    history = locals()
    del history["context"]
    del history["tags"]
    del history["sent"]
    del history["index"]
    del history["length"]
    history = history.items()
    return history
