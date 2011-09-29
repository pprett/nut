Natural language Understanding Toolkit
======================================

TOC
---

  * Requirements_
  * Installation_
  * Documentation_
     - CLSCL_
     - NER_
  * References_

.. _Requirements:

Requirements
------------

To install nut you need:

   * Python 2.5 or 2.6
   * Numpy (>= 1.1)
   * Sparsesvd (>= 0.1.4) [#f1]_ (only CLSCL_)

.. _Installation:

Installation
------------

To clone the repository run, 

   git clone git://github.com/pprett/nut.git

To build the extension modules inplace run,

   python setup.py build_ext --inplace

Add project to python path,

   export PYTHONPATH=$PYTHONPATH:$HOME/workspace/nut

.. _Documentation:

Documentation
-------------

.. _CLSCL:

CLSCL
~~~~~

An implementation of Cross-Language Structural Correspondence Learning (CLSCL). 
See [Prettenhofer2010]_ for a detailed description and 
[Prettenhofer2011]_ for more experiments and enhancements.

The data for cross-language sentiment classification that has been used in the above
study can be found here [#f2]_.

clscl_train
???????????

Training script for CLSCL. See `./clscl_train --help` for further details. 

Usage::

    $ ./clscl_train en de cls-acl10-processed/en/books/train.processed cls-acl10-processed/en/books/unlabeled.processed cls-acl10-processed/de/books/unlabeled.processed cls-acl10-processed/dict/en_de_dict.txt model.bz2 --phi 30 --max-unlabeled=50000 -k 100 -m 450 --strategy=parallel

    |V_S| = 64682
    |V_T| = 106024
    |V| = 170706
    |s_train| = 2000
    |s_unlabeled| = 50000
    |t_unlabeled| = 50000
    debug: DictTranslator contains 5012 translations.
    mutualinformation took 5.624 sec
    select_pivots took 7.197 sec
    |pivots| = 450
    create_inverted_index took 59.353 sec
    Run joblib.Parallel
    [Parallel(n_jobs=-1)]: Done   1 out of 450 |elapsed:    9.1s remaining: 67.8min
    [Parallel(n_jobs=-1)]: Done   5 out of 450 |elapsed:   15.2s remaining: 22.6min
    [..]
    [Parallel(n_jobs=-1)]: Done 449 out of 450 |elapsed: 14.5min remaining:    1.9s
    train_aux_classifiers took 881.803 sec
    density: 0.1154
    Ut.shape = (100,170706)
    learn took 903.588 sec
    project took 175.483 sec

.. note:: If you have access to a hadoop cluster, you can use `--strategy=hadoop` to train the pivot classifiers even faster, however, make sure that the hadoop nodes have Bolt (feature-mask branch) [#f3]_ installed. 

clscl_predict
?????????????

Prediction script for CLSCL.

Usage::

    $ ./clscl_predict cls-acl10-processed/en/books/train.processed model.bz2 cls-acl10-processed/de/books/test.processed -r 0.01
    |V_S| = 64682
    |V_T| = 106024
    |V| = 170706
    load took 0.681 sec
    load took 0.659 sec
    classes = {negative,positive}
    project took 2.498 sec
    project took 2.716 sec
    project took 2.275 sec
    project took 2.492 sec
    ACC: 83.05

.. _NER:

Named-Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~

A simple greedy left-to-right sequence labeling approach to named entity recognition (NER). 

pre-trained models
??????????????????

We provide pre-trained named entity recognizers for place, person, and organization names in English and German. To tag a sentence simply use::

    >>> from nut.io import compressed_load
    >>> from nut.util import WordTokenizer

    >>> tagger = compressed_load("model_demo_en.bz2")
    >>> tokenizer = WordTokenizer()
    >>> tokens = tokenizer.tokenize("Peter Prettenhofer lives in Austria .")

    >>> # see tagger.tag.__doc__ for input format
    >>> sent = [((token, "", ""), "") for token in tokens]
    >>> g = tagger.tag(sent)  # returns a generator over tags
    >>> print(" ".join(["/".join(tt) for tt in zip(tokens, g)]))
    Peter/B-PER Prettenhofer/I-PER lives/O in/O Austria/B-LOC ./O

You can also use the convenience demo script `ner_demo.py`::

    $ python ner_demo.py model_en_v1.bz2

The feature detector modules for the pre-trained models are `en_best_v1.py` and `de_best_v1.py` and can be found in the package `nut.ner.features`.
In addition to baseline features (word presence, shape, pre-/suffixes) they use distributional features (brown clusters), non-local features (extended prediction history), and gazetteers (see [Ratinov2009]_). The models have been trained on CoNLL03 [#f4]_. Both models use neither syntactic features (e.g. part-of-speech tags, chunks) nor word lemmas, thus, minimizing the required pre-processing. Both models provide state-of-the-art performance on the CoNLL03 shared task benchmark for English [Ratinov2009]_::

    processed 46435 tokens with 4946 phrases; found: 4864 phrases; correct: 4455.
    accuracy:  98.01%; precision:  91.59%; recall:  90.07%; FB1:  90.83
                  LOC: precision:  91.69%; recall:  90.53%; FB1:  91.11  1648
                  ORG: precision:  87.36%; recall:  85.73%; FB1:  86.54  1630
                  PER: precision:  95.84%; recall:  94.06%; FB1:  94.94  1586

and German [Faruqui2010]_::

    processed 51943 tokens with 2845 phrases; found: 2438 phrases; correct: 2168.
    accuracy:  97.92%; precision:  88.93%; recall:  76.20%; FB1:  82.07
                  LOC: precision:  87.67%; recall:  79.83%; FB1:  83.57  957
                  ORG: precision:  82.62%; recall:  65.92%; FB1:  73.33  466
                  PER: precision:  93.00%; recall:  78.02%; FB1:  84.85  1015


To evaluate the German model on the out-domain data provided by [Faruqui2010]_ use the raw flag (`-r`) to write raw predictions (without B- and I- prefixes)::

    ./ner_predict -r model_de_v1.bz2 clner/de/europarl/test.conll - | clner/scripts/conlleval -r
    loading tagger... [done]
    use_eph:  True
    use_aso:  False
    processed input in 40.9214s sec.
    processed 110405 tokens with 2112 phrases; found: 2930 phrases; correct: 1676.
    accuracy:  98.50%; precision:  57.20%; recall:  79.36%; FB1:  66.48
                  LOC: precision:  91.47%; recall:  71.13%; FB1:  80.03  563
                  ORG: precision:  43.63%; recall:  83.52%; FB1:  57.32  1673
                  PER: precision:  62.10%; recall:  83.85%; FB1:  71.36  694


Note that the above results cannot be compared directly to the resuls of [Faruqui2010]_ since they use a slighly different setting (incl. MISC entity).

ner_train
?????????

Training script for NER. See ./ner_train --help for further details. 

To train a conditional markov model with a greedy left-to-right decoder, the feature 
templates of [Rationov2009]_ and extended prediction history 
(see [Ratinov2009]_) use::

    ./ner_train clner/en/conll03/train.iob2 model_rr09.bz2 -f rr09 -r 0.00001 -E 100 --shuffle --eph
    ________________________________________________________________________________
    Feature extraction
    
    min count:  1
    use eph:  True
    build_vocabulary took 24.662 sec
    feature_extraction took 25.626 sec
    creating training examples... build_examples took 42.998 sec
    [done]
    ________________________________________________________________________________
    Training
    
    num examples: 203621
    num features: 553249
    num classes: 9
    classes:  ['I-LOC', 'B-ORG', 'O', 'B-PER', 'I-PER', 'I-MISC', 'B-MISC', 'I-ORG', 'B-LOC']
    reg: 0.00001000
    epochs: 100
    9 models trained in 239.28 seconds. 
    train took 282.374 sec
    

ner_predict
???????????

You can use the prediction script to tag new sentences formatted in CoNLL format 
and write the output to a file or to stdout. 
You can pipe the output directly to `conlleval` to assess the model performance::

    ./ner_predict model_rr09.bz2 clner/en/conll03/test.iob2 - | clner/scripts/conlleval
    loading tagger... [done]
    use_eph:  True
    use_aso:  False
    processed input in 11.2883s sec.
    processed 46435 tokens with 5648 phrases; found: 5605 phrases; correct: 4799.
    accuracy:  96.78%; precision:  85.62%; recall:  84.97%; FB1:  85.29
                  LOC: precision:  87.29%; recall:  88.91%; FB1:  88.09  1699
                 MISC: precision:  79.85%; recall:  75.64%; FB1:  77.69  665
                  ORG: precision:  82.90%; recall:  78.81%; FB1:  80.80  1579
                  PER: precision:  88.81%; recall:  91.28%; FB1:  90.03  1662

.. _References:
References
----------

.. [#f1] http://pypi.python.org/pypi/sparsesvd/0.1.4
.. [#f2] http://www.uni-weimar.de/medien/webis/research/corpora/webis-cls-10/cls-acl10-processed.tar.gz
.. [#f3] https://github.com/pprett/bolt/tree/feature-mask
.. [#f4] For German we use the updated version of CoNLL03 by Sven Hartrumpf. 

.. [Prettenhofer2010] Prettenhofer, P. and Stein, B., `Cross-language text classification using structural correspondence learning <http://www.aclweb.org/anthology/P/P10/P10-1114.pdf>`_. In Proceedings of ACL '10.

.. [Prettenhofer2011] Prettenhofer, P. and Stein, B., `Cross-lingual adaptation using structural correspondence learning <http://tist.acm.org/papers/TIST-2010-06-0137.R1.html>`_. ACM TIST (to appear). `[preprint] <http://arxiv.org/pdf/1008.0716v2>`_

.. [Ratinov2009] Ratinov, L. and Roth, D., `Design challenges and misconceptions in named entity recognition <http://www.aclweb.org/anthology/W/W09/W09-1119.pdf>`_. In Proceedings of CoNLL '09.

.. [Faruqui2010] Faruqui, M. and Padó S., `Training and Evaluating a German Named Entity Recognizer with Semantic Generalization`. In Proceedings of KONVENS '10

.. _Developer Notes:
Developer Notes
---------------

  * If you copy a new version of `bolt` into the `externals` directory make sure to run cython on the `*.pyx` files. If you fail to do so you will get a `PickleError` in multiprocessing.
