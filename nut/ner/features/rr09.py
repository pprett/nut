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

class Detector(object):

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

        ## pos in a 5 token window
        pos_cur = context(0, POS)
        pos_pre = context(-1, POS)
        pos_post = context(1, POS)
        pos_2pre = context(-2, POS)
        pos_2post = context(2, POS)

        ## pos bigrams in a 5 token window
        pos_bigram_pre_cur = "/".join([pos_pre, pos_cur])
        pos_bigram_bigram_cur_post = "/".join([pos_cur, pos_post])
        #pre_pre_pos_bigram = "/".join([pre_pre_pos, pre_pos])
        #post_post_pos_bigram = "/".join([post_pos, post_post_pos])

        pos_posw_cur = "/".join([word_unigram_cur, pos_cur])

        ## Word shape features (5 token window)
        shape_istitle_cur = word_unigram_cur.istitle()
        shape_isdigit_cur = context(0, WORD).isdigit()
        shape_isupper_cur = word_unigram_cur.isupper()
        shape_hyphen_cur = "-" in word_unigram_cur[1:-1]
        shape_isalnum_cur = context(0, WORD).isalnum()

        shape_istitle_pre = word_unigram_pre.istitle()
        shape_isdigit_pre = context(-1, WORD).isdigit()
        shape_isupper_pre = word_unigram_pre.isupper()
        shape_hyphen_pre = "-" in word_unigram_pre[1:-1]
        shape_isalnum_pre = context(-1, WORD).isalnum()

        shape_istitle_2pre = word_unigram_2pre.istitle()
        shape_isdigit_2pre = context(-2, WORD).isdigit()
        shape_isupper_2pre = word_unigram_2pre.isupper()
        shape_hyphen_2pre = "-" in word_unigram_2pre[1:-1]
        shape_isalnum_2pre = context(-2, WORD).isalnum()

        shape_istitle_post = word_unigram_post.istitle()
        shape_isdigit_post = context(1, WORD).isdigit()
        shape_isupper_post = word_unigram_post.isupper()
        shape_hypen_post = "-" in word_unigram_post[1:-1]
        shape_isalnum_post = context(1, WORD).isalnum()

        shape_istitle_2post = word_unigram_2post.istitle()
        shape_isdigit_2post = context(2, WORD).isdigit()
        shape_isupper_2post = word_unigram_2post.isupper()
        shape_hypen_2post = "-" in word_unigram_2post[1:-1]
        shape_isalnum_2post = context(2, WORD).isalnum()

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

        ## Noun phrase in a 3 token window
        syn_np_cur = context(0, NP)
        syn_npw_cur = "/".join([syn_np_cur, word_unigram_cur])
        syn_np_pre = context(-1, NP)
        syn_np_post = context(1, NP)

        ## Extract features from local scope
        features = locals()
        del features["context"]
        del features["sent"]
        del features["index"]
        del features["length"]
        del features["self"]
        features = features.items()
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
