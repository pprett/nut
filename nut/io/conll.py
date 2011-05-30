#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style
"""
conll
=====

This package contains corpus reader for a number
of CoNLL shared tasks.


"""

import codecs
import os
from itertools import izip, count


class Conll03Reader(object):
    """Corpus reader for CoNLL 2003 shared task on
    Named Entity Recognition.

    Parameters
    ----------
    path : str
        The path to the iob file.
    lang : str
        The language of the text (either 'en' or 'de').
    iob : bool
        Whether or not iob encoding should be used (default True).
    raw : bool
        True if file is unlabeled (missing tag column).
        All tokens are given the tag `Unk`.
    outside : str
        The tag for outside tokens (default 'O').
    tags : set
        The set of tags to be considered. For all other tags
        `outside` will be returned.
    """
    def __init__(self, path, lang, iob=True, raw=False, outside="O", tags=[]):
        self._path = os.path.normpath(path)
        if lang == "en":
            self._select = [0, 1, 2]
        elif lang == "de":
            self._select = [0, 2, 3, 1]
        else:
            raise ValueError("lang must be either en or de.")
        self.iob = iob
        self.raw = raw
        self.outside = outside
        self.tags = set(tags)

    def __iter__(self):
        """Iterate over the corpus; i.e. tokens.
        """
        fd = codecs.open(self._path, mode="rt", encoding="latin1")
        try:
            for line in fd:
                line = line.encode("utf8")
                line = line.rstrip()
                if line.startswith("-DOCSTART-"):
                    yield "</doc>"
                    yield "<doc>"
                elif line == "":
                    # new sentence - emit sentence end and begin
                    yield "</s>"
                    yield "<s>"
                else:
                    fields = line.split()
                    if self.raw:
                        fields.append("Unk")
                    # emit (token, tag) tuple
                    tag = fields[-1]

                    # check if tag in tag set otherwise set to outside
                    if tag != self.outside and self.tags:
                        tag_name = tag.split("-")[1] if tag.find("-") != -1 else tag
                        if tag_name not in self.tags:
                            tag = self.outside

                    token = fields[:-1]
                    token = [token[i] for i in self._select]

                    if not self.iob and tag != self.outside:
                        tag = tag.split("-")[1]

                    yield (token, tag)
            yield "</doc>"
            yield "<doc>"
        finally:
            fd.close()

    def sents(self):
        """Iterate over sentences.

        Yields
        ------
        tuple (token, tag) where token is a tuple (term, lemma?, pos, chunk).
        """
        buf = []
        for token in self:
            if token == "</s>":
                if len(buf) > 0:
                    yield buf
            elif token == "<s>":
                buf = []
            else:
                if token != "<doc>" and token != "</doc>":
                    buf.append(token)

    def docs(self):
        """Iterate over documents.

        Yields
        ------
        list of tuples (token, tag)
        where token is a tuple (term, lemma?, pos, chunk).
        """
        buf = []
        for token in self:
            if token == "</doc>":
                if len(buf) != 0:
                    yield buf[1:-1]  # trim leading and trailing <s>.
            elif token == "<doc>":
                buf = []
            else:
                buf.append(token)

    def write_sent_predictions(self, tagger, fd, raw=False):
        """Write predictions of `tagger` to file `fd` in a format
        suited for conlleval.

        Parameters
        ----------
        tagger : Tagger
            The tagger used for prediction.
        fd : file
            A file descriptor to which the predictions shold be written.
        raw : bool
            Raw predictions or iob encoding (see conlleval).
        """
        for i, sent, pred in izip(count(), self.sents(),
                                  tagger.batch_tag(self.sents())):
            fd.write("\n")
            for (token, tag), ptag in izip(sent, pred):
                if raw:
                    tag = tag.split("-")[-1]
                    ptag = ptag.split("-")[-1]
                fd.write("%s %s %s\n" % (" ".join(token), tag, ptag))


def entities(doc):
    entities = []
    buf = []
    notentity = set(["O", "<s>", "</s>"])
    for i, token in enumerate(doc):
        tag = token[-1] if isinstance(token, tuple) else token
        print i, tag
        if tag in notentity:
            if len(buf) > 0:
                entities.append(" ".join(buf))
                buf = []
        else:
            if tag.startswith("B"):
                if len(buf) > 0:
                    entities.append(" ".join(buf))
                    buf = []
            print "appending ", token[0][0]
            buf.append(token[0][0])
    return entities
