#!/usr/bin/python
"""Named entity recognition demo.
"""

import sys

from itertools import izip

from nut.io import conll, compressed_dump, compressed_load
from nut.util import WordTokenizer

def main(argv):
    print __doc__
    print >> sys.stderr, "loading tagger...",
    sys.stderr.flush()
    model = compressed_load(argv[0])
    print >> sys.stderr, "[done]"
    print
    print "_" * 80
    print "Enter sentences to tag"
    print

    tokenizer = WordTokenizer()
    while True:
        sent = raw_input("input> ")

        if sent in ["quite", "bye", "q"]:
            print "Bye!"
            break
        
        tokens = tokenizer.tokenize(sent)
        sent = [((token, "", ""), "") for token in tokens]
        output = ["%s/%s" % (token, tag) for token, tag
                  in izip(tokens, model.tag(sent))]
        
        print "output:", 
        print " ".join(output)
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Usage: %s <model>" % sys.argv[0]
        sys.exit(-2)
    main(sys.argv[1:])
