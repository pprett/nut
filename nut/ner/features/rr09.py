"""Feature and history detectors for CoNLL03 taken from:
`Ratinov, L. and Roth, D. (2009). Design challenges and misconceptions in
named entity recognition. In CoNLL '00`.

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
    word_cur = numify(context(0, WORD))
    word_pre = numify(context(-1, WORD))
    word_pre2 = numify(context(-2, WORD))
    word_post = numify(context(1, WORD))
    word_post2 = numify(context(2, WORD))

    ## token bigrams in a 5 token window
    word_precur_bigram = "/".join([word_pre, word_cur])
    word_curpost_bigram = "/".join([word_cur, word_post])
    #pre_pre_bigram = "/".join([pre_pre_w, pre_w])
    #post_post_bigram = "/".join([post_w, post_post_w])

    ## pos in a 5 token window
    pos_cur = context(0, POS)
    pos_pre = context(-1, POS)
    pos_post = context(1, POS)
    pos_pre2 = context(-2, POS)
    pos_post2 = context(2, POS)

    ## pos bigrams in a 5 token window
    pos_precur_bigram = "/".join([pos_pre, pos_cur])
    pos_curpost_bigram = "/".join([pos_cur, pos_post])
    #pre_pre_pos_bigram = "/".join([pre_pre_pos, pre_pos])
    #post_post_pos_bigram = "/".join([post_pos, post_post_pos])

    posw_cur = "/".join([word_cur, pos_cur])

    ## Word shape features (5 token window)
    shape_cur_istitle = word_cur.istitle()
    shape_cur_isdigit = context(0, WORD).isdigit()
    shape_cur_isupper = word_cur.isupper()
    shape_cur_hyphen = "-" in word_cur[1:-1]
    shape_cur_isalnum = context(0, WORD).isalnum()

    shape_pre_istitle = word_pre.istitle()
    shape_pre_isdigit = context(-1, WORD).isdigit()
    shape_pre_isupper = word_pre.isupper()
    shape_pre_hyphen = "-" in word_pre[1:-1]
    shape_pre_isalnum = context(-1, WORD).isalnum()

    shape_pre2_istitle = word_pre2.istitle()
    shape_pre2_isdigit = context(-2, WORD).isdigit()
    shape_pre2_isupper = word_pre2.isupper()
    shape_pre2_hyphen = "-" in word_pre2[1:-1]
    shape_pre2_isalnum = context(-2, WORD).isalnum()

    shape_post_istitle = word_post.istitle()
    shape_post_isdigit = context(1, WORD).isdigit()
    shape_post_isupper = word_post.isupper()
    shape_post_hypen = "-" in word_post[1:-1]
    shape_post_isalnum = context(1, WORD).isalnum()

    shape_post2_istitle = word_post2.istitle()
    shape_post2_isdigit = context(2, WORD).isdigit()
    shape_post2_isupper = word_post2.isupper()
    shape_post2_hypen = "-" in word_post2[1:-1]
    shape_post2_isalnum = context(2, WORD).isalnum()

    ## 2-4 suffixes in a 3 token window
    suffix_cur_1 = word_cur[-1:]
    suffix_cur_2 = word_cur[-2:]
    suffix_cur_3 = word_cur[-3:]
    suffix_cur_4 = word_cur[-4:]

    suffix_pre_1 = word_pre[-1:]
    suffix_pre_2 = word_pre[-2:]
    suffix_pre_3 = word_pre[-3:]
    suffix_pre_4 = word_pre[-4:]

    suffix_post_1 = word_post[-1:]
    suffix_post_2 = word_post[-2:]
    suffix_post_3 = word_post[-3:]
    suffix_post_4 = word_post[-4:]

    ## 3-4 prefixes in a 3 token window
    prefix_cur_3 = word_cur[:3]
    prefix_cur_4 = word_cur[:4]

    prefix_pre_3 = word_pre[:3]
    prefix_pre_4 = word_pre[:4]

    prefix_post_3 = word_post[:3]
    prefix_post_4 = word_post[:4]

    ## Noun phrase in a 3 token window
    np_cur = context(0, NP)
    npw_cur = "/".join([np_cur, word_cur])
    np_pre = context(-1, NP)
    np_post = context(1, NP)

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

    tag_pre = tags[index - 1] if index - 1 >= 0 else "<s>"
    tag_pre2 = tags[index - 2] if index - 2 >= 0 else "<s>"
    pretagword_cur = "/".join([tag_pre, numify(context(0, WORD))])
    pretagword_pre = "/".join([tag_pre, numify(context(-1, WORD))])
    pretagword_pre2 = "/".join([tag_pre, numify(context(-2, WORD))])
    pretagword_post = "/".join([tag_pre, numify(context(1, WORD))])
    pretagword_post2 = "/".join([tag_pre, numify(context(2, WORD))])
    tagbi_pre2pre = "/".join([tag_pre2, tag_pre])

    history = locals()
    del history["context"]
    del history["tags"]
    del history["sent"]
    del history["index"]
    del history["length"]
    history = history.items()
    return history
