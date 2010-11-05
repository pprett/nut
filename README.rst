Natural language Understanding Toolkit
======================================

Requirements
------------

To install nut you need:

   * Python 2.5 or 2.6
   * Numpy (>= 1.1)
   * Bolt  (>= 1.4) [#f1]_
   * Sparsesvd (>= 0.1.4) [#f2]_

.. [#f1] http://github.com/pprett/bolt
.. [#f2] http://pypi.python.org/pypi/sparsesvd/0.1.4

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
study can be found here [2]. 

clscl_train
???????????

Training script for CLSCL. See ./clscl --help for further details. 

Usage:: 

   $ ./clscl_train en de cls-acl10-processed/en/books/train.processed ../../corpora/cls-acl10-processed/en/books/unlabeled.processed cls-acl10-processed/de/books/unlabeled.processed cls-acl10-processed/dict/en_de_dict.txt model.pkl --phi 30 --max-unlabeled=50000 -k 100 -m 450
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

References
----------

[1] http://pprett.github.com/bolt/
[2] http://www.uni-weimar.de/medien/webis/research/corpora/webis-cls-10/cls-acl10-processed.tar.gz

[Prettenhofer, P. and Stein, B., 2010] Cross-Language Text Classification using Structural Correspondence Learning. In Proceedings of ACL '10.


