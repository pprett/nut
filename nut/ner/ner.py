#!/usr/bin/python
#
# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#
# License: BSD Style

import sys
import optparse

from time import time

from ..io import conll, compressed_dump, compressed_load
from ..tagger import tagger


__version__ = "0.1"


class Error(Exception):
    pass


def train_args_parser():
    """Create argument and option parser for the
    training script.
    """
    description = """%s    """ % str(__file__)
    parser = optparse.OptionParser(usage="%prog [options] " \
                                   "train_file model_file",
                                   version="%prog " + __version__,
                                   description=description)
    parser.add_option("-v", "--verbose",
                      dest="verbose",
                      help="verbose output",
                      default=1,
                      metavar="[0,1,2]",
                      type="int")
    parser.add_option("-f", "--feature-module",
                      dest="feature_module",
                      help="The module in the features package containing " \
                      "the `detector` class. [Default: %default].",
                      default="rr09",
                      metavar="str",
                      type="str")
    parser.add_option("-r", "--reg",
                      dest="reg",
                      help="regularization parameter. ",
                      default=0.00001,
                      metavar="float",
                      type="float")
    parser.add_option("-E", "--epochs",
                      dest="epochs",
                      help="Number of training epochs. ",
                      default=100,
                      metavar="int",
                      type="int")
    parser.add_option("--min-count",
                      dest="minc",
                      help="min number of occurances.",
                      default=1,
                      metavar="int",
                      type="int")
    parser.add_option("--max-unlabeled",
                      dest="max_unlabeled",
                      help="max number of unlabeled documents to read;" \
                      "-1 for unlimited.",
                      default=-1,
                      metavar="int",
                      type="int")
    parser.add_option("-l", "--lang",
                      dest="lang",
                      help="The language (`en` or `de`).",
                      default="en",
                      metavar="str",
                      type="str")
    parser.add_option("--shuffle",
                      action="store_true",
                      dest="shuffle",
                      default=False,
                      help="Shuffle the training data after each epoche.")
    parser.add_option("--seed",
                      dest="seed",
                      help="The shuffle seed.",
                      default=13,
                      metavar="int",
                      type="int")
    parser.add_option("--stats",
                      action="store_true",
                      dest="stats",
                      default=False,
                      help="Print model statistics.")
    parser.add_option("--eph",
                      action="store_true",
                      dest="use_eph",
                      default=False,
                      help="Use Extended Prediction History.")
    parser.add_option("--no-biasterm",
                      action="store_false",
                      dest="biasterm",
                      default=True,
                      help="Use unbiased hyperplanes.")
    parser.add_option("--aso",
                      dest="aso",
                      default=False,
                      help="Give an ASO model.",
                      metavar="str",
                      type="str")
    parser.add_option("--n-jobs",
                      dest="n_jobs",
                      help="The number of processes to fork.",
                      default=3,
                      metavar="int",
                      type="int")
    parser.add_option("--tags",
                      dest="tags",
                      default="",
                      help="The tags to consider, " \
                      "tags are separated by comma ('ORG,LOC,PER').",
                      metavar="str",
                      type="str")

    return parser


def train():
    """Training script for Named Entity Recognizer.
    """
    parser = train_args_parser()
    options, argv = parser.parse_args()
    if len(argv) != 2:
        parser.error("incorrect number of arguments (use `--help` for help).")

    # get filenames
    f_train = argv[0]
    f_model = argv[1]

    # get feature extraction module
    try:
        import_path = "nut.ner.features.%s" % options.feature_module
        mod = __import__(import_path, fromlist=[options.feature_module])
        detector = mod.Detector()
    except ImportError:
        print "Error: cannot import feature extractors " \
              "from %s" % options.feature_module
        sys.exit(-2)

    tag_str = options.tags
    tags = [tag.upper() for tag in tag_str.split(",")]

    print "Tags to consider: ", tags
    train_reader = conll.Conll03Reader(f_train, options.lang, tags=tags)

    if options.aso:
        print "Loading ASO model...",
        sys.stdout.flush()
        aso_model = compressed_load(options.aso)
        print "[done]"

        # Create Tagger from ASOModel
        model = tagger.GreedySVMTagger(aso_model.detector,
                                       lang=aso_model.lang,
                                       verbose=options.verbose)
        if options.use_eph != aso_model.use_eph:
            raise Error("options.use_eph != aso_model.use_eph.")
        if options.lang != aso_model.lang:
            raise Error("options.lang != aso_model.lang.")

        model.V = aso_model.vocabulary
        model.T = aso_model.tags
        model.use_eph = aso_model.use_eph
        model.minc = aso_model.minc
        model.fidx_map = aso_model.fidx_map
        model.tidx_map = aso_model.tidx_map
        model.tag_map = aso_model.tag_map

        # turn ASO on and hand over ASO model
        model.use_aso = True
        model.aso_model = aso_model

    else:

        #model = tagger.AvgPerceptronTagger(detector, verbose=options.verbose)
        model = tagger.GreedySVMTagger(detector, lang=options.lang,
                                       verbose=options.verbose)
        model.feature_extraction(train_reader, minc=options.minc,
                                 use_eph=options.use_eph)

    model.train(train_reader, reg=options.reg, epochs=options.epochs,
                shuffle=options.shuffle, seed=options.seed,
                n_jobs=options.n_jobs, biasterm=options.biasterm,
                norm=2)

    if options.stats:
        print "_" * 80
        print "Stats"
        print
        print "Weights: ", model.glm.W.shape
        W_nnz = (model.glm.W != 0.0).sum()
        density = float(W_nnz) / (model.glm.m * model.glm.k)
        print "Density: %.2f" % density
        print "Size (MBs): %.2f" % (float(W_nnz * 8) / 1024.0 / 1024.0)
        print
        model.describe(k=40)
        #nnz = 0
        #for instance in model.dataset.iterinstances():
        #    nnz += instance.shape[0]
        #print "avg. nnz: %.4f" % (float(nnz) / model.dataset.n)

    compressed_dump(f_model, model)


def predict():
    """Test script for Named Entity Recognizer.
    """
    def usage():
        print """Usage: %s [OPTIONS] MODEL_FILE TEST_FILE PRED_FILE
        Load NER model from MODEL_FILE, test the model on
        TEST_FILE and write predictions to PRED_FILE.
        The predictions are appended to the test file.
        Options:
          -h, --help\tprint this help.
          -r, --raw\twrite raw result tags (wo B- and I- prefix).

        """ % sys.argv[0]
    argv = sys.argv[1:]
    if "--help" in argv or "-h" in argv:
        usage()
        sys.exit(-2)

    raw = False
    if "-r" in argv or "--raw" in argv:
        argv = [arg for arg in argv if arg not in ["-r", "--raw"]]
        raw = True

    if len(argv) != 3:
        print "Error: wrong number of arguments. "
        usage()
        sys.exit(-2)

    print >> sys.stderr, "loading tagger...",
    sys.stderr.flush()
    model = compressed_load(argv[0])
    print >> sys.stderr, "[done]"
    print >> sys.stderr, "use_eph: ", model.use_eph
    print >> sys.stderr, "use_aso: ", model.use_aso
    test_reader = conll.Conll03Reader(argv[1], model.lang, tags=model.tags)
    if argv[2] != "-":
        f = open(argv[2], "w+")
    else:
        f = sys.stdout
    t0 = time()
    test_reader.write_sent_predictions(model, f, raw=raw)
    f.close()
    print >> sys.stderr, "processed input in %.4fs sec." % (time() - t0)
