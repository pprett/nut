Natural language Understanding Toolkit
======================================

Requirements
------------

To install nut you need:

   * Python 2.5 or 2.6
   * Numpy (>= 1.1)
   * Bolt  (>= 1.4) [#f1]_
   * Sparsesvd (>= 0.1.4) [#f2]_

Installation
------------

To clone the repository run, 

   git clone git://github.com/pprett/nut.git

To build nut simply run,

   python setup.py build

To install nut on your system, use

   python setup.py install

Documentation
-------------

CLSCL
~~~~~

An implementation of Cross-Language Structural Correspondence Learning (CLSCL) 
for cross-language text classification. See [Prettenhofer and Stein, 2009] for 
a detailed description. 

The data for cross-language sentiment classification that has been used in the above
study can be found here [#f3]_.

clscl_train
???????????

Training script for CLSCL. See ./clscl_train --help for further details. 

Usage::

    $ ./clscl_train en de cls-acl10-processed/en/books/train.processed cls-acl10-processed/en/books/unlabeled.processed cls-acl10-processed/de/books/unlabeled.processed cls-acl10-processed/dict/en_de_dict.txt model.pkl --phi 30 --max-unlabeled=50000 -k 100 -m 450
    |V_S| = 64682
    |V_T| = 106024
      |V| = 170706
    |s_train| = 2000
    |s_unlabeled| = 50000
    |t_unlabeled| = 50000
    |pivots| = 450
    tempdir: /tmp/tmpI_2Ejw
    processing Hadoop job... 
    Cleaning local temp dir.
    train_aux_classifiers took 259.719 sec
    density: 0.1154
    Ut.shape = (100,170706)
    learn took 281.216 sec
    project took 142.993 sec

clscl_predict
?????????????

Prediction script for CLSCL.

Usage::

    $ ./clscl_predict cls-acl10-processed/en/books/train.processed model.pkl cls-acl10-processed/de/books/test.processed 0.01
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

To train a conditional markov model with a greedy left-to-right decoder, the feature template of 
[Rationov & Roth, 2009] and extended prediction history (see [Ratinov & Roth, 2009]) use::

    ./ner_train clner/en/conll03/train.iob2 model_rr09.bz2 -f rr09 -r 0.00001 -E 100 --shuffle --eph 

ner_predict
???????????

You can use the prediction script to tag new sentences and write the output to a file or to stdout. 
You can pipe the output directly to `conlleval` to assess the model performance::

    ./ner_predict model_rr09.bz2 clner/en/conll03/test.iob2 - | clner/scripts/conlleval


References
----------

.. [#f1] http://github.com/pprett/bolt
.. [#f2] http://pypi.python.org/pypi/sparsesvd/0.1.4
.. [#f3] http://www.uni-weimar.de/medien/webis/research/corpora/webis-cls-10/cls-acl10-processed.tar.gz

[Prettenhofer, P. and Stein, B., 2010] `Cross-Language Text Classification using Structural Correspondence Learning <www.aclweb.org/anthology/P/P10/P10-1114.pdf>`_. In Proceedings of ACL '10.

[Ratinov, L. and Roth, D., 2009] `Design challenges and misconceptions in named entity recognition <www.aclweb.org/anthology/W/W09/W09-1119.pdf>`_. In Proceedings of CoNLL '09.


