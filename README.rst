Natural language Understanding Toolkit
======================================

Requirements
------------

To install nut you need:

   * Python 2.5 or 2.6
   * Numpy (>= 1.1)
   * Sparsesvd (>= 0.1.4) [#f1]_

Installation
------------

To clone the repository run, 

   git clone git://github.com/pprett/nut.git

To build the extension modules run,

   python setup.py build_ext --inplace

Documentation
-------------

CLSCL
~~~~~

An implementation of Cross-Language Structural Correspondence Learning (CLSCL) 
for cross-language text classification. See [Prettenhofer and Stein, 2010] 
for a detailed description and [Prettenhofer and Stein, 2011] for more 
experiments and enhancements.

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

    $ ./clscl_predict cls-acl10-processed/en/books/train.processed model.bz2 cls-acl10-processed/de/books/test.processed 0.01
    |V_S| = 64682
    |V_T| = 106024
    |V| = 170706
    load took 0.625 sec
    load took 0.579 sec
    project took 3.012 sec
    project took 2.805 sec
    ACC: 82.85

Named-Entity Recognition
~~~~~~~~~~~~~~~~~~~~~~~~

A simple greedy left-to-right sequence labeling approach to named-entity recognition (NER). 

ner_train
?????????

Training script for NER. See ./ner_train --help for further details. 

To train a conditional markov model with a greedy left-to-right decoder, the feature templates 
of [Rationov & Roth, 2009] and extended prediction history (see [Ratinov & Roth, 2009]) use::

    ./ner_train clner/en/conll03/train.iob2 model_rr09.bz2 -f rr09 -r 0.00001 -E 100 --shuffle --eph 

ner_predict
???????????

You can use the prediction script to tag new sentences and write the output to a file or to stdout. 
You can pipe the output directly to `conlleval` to assess the model performance::

    ./ner_predict model_rr09.bz2 clner/en/conll03/test.iob2 - | clner/scripts/conlleval


References
----------

.. [#f1] http://pypi.python.org/pypi/sparsesvd/0.1.4
.. [#f2] http://www.uni-weimar.de/medien/webis/research/corpora/webis-cls-10/cls-acl10-processed.tar.gz
.. [#f3] https://github.com/pprett/bolt/tree/feature-mask

[Prettenhofer, P. and Stein, B., 2010] `Cross-Language Text Classification using Structural Correspondence Learning <http://www.aclweb.org/anthology/P/P10/P10-1114.pdf>`_. In Proceedings of ACL '10.

[Prettenhofer, P. and Stein, B., 2011] `Cross-lingual Adaptation using Structural Correspondence Learning <http://tist.acm.org/papers/TIST-2010-06-0137.R1.html>`_. ACM TIST (to appear). `[preprint] <http://arxiv.org/pdf/1008.0716v2>`_

[Ratinov, L. and Roth, D., 2009] `Design challenges and misconceptions in named entity recognition <http://www.aclweb.org/anthology/W/W09/W09-1119.pdf>`_. In Proceedings of CoNLL '09.


