from distutils.core import setup
from distutils.extension import Extension
from distutils.sysconfig import get_python_inc

import sys
import os.path
import platform
import numpy

np_inc = os.path.join(numpy.__path__[0], 'core', 'include')

# use additional compiler flags: "-ffast-math" "-g"

# FIXME this only works for EPD?
if platform.system() == "Windows":
    lib_path = "C:\Python26\PCbuild"
else:
    lib_path = "/usr/lib"

setup(
    name = "nut.externals.bolt",
    ext_modules = [
                   Extension("nut.externals.bolt.trainer.sgd",
                             ["nut/externals/bolt/trainer/sgd.c"],
                             include_dirs=[np_inc],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3", "-g"]
                             ),
                   Extension("nut.externals.bolt.trainer.avgperceptron",
                             ["nut/externals/bolt/trainer/avgperceptron.c"],
                             include_dirs=[np_inc],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3", "-g"]
                             ),
                   Extension("nut.externals.bolt.trainer.maxent",
                             ["nut/externals/bolt/trainer/maxent.c"],
                             include_dirs=[np_inc],
                             extra_link_args=["-O3"],
                             library_dirs=[lib_path,],
                             extra_compile_args=["-O3", "-g"]
                             ),
                   ],
    version = "0.1",
    description="Natural language Understanding Toolkit (NUT)",
    author='Peter Prettenhofer',
    author_email='peter.prettenhofer@gmail.com',
    url = 'http://www.github.com/pprett/nut/',
    license = 'new BSD',
    classifiers = 
            ['Intended Audience :: Science/Research',
             'Intended Audience :: Developers',
             'License :: OSI Approved',
             'Programming Language :: C',
             'Programming Language :: Python',
             'Topic :: Scientific/Engineering',
             'Operating System :: Microsoft :: Windows',
             'Operating System :: POSIX',
             'Operating System :: Unix',
             'Operating System :: MacOS'
             ]
)

