"""
Helpers for embarassingly parallel code.
"""
# Author: Gael Varoquaux < gael dot varoquaux at normalesup dot org >
# Copyright: 2010, Gael Varoquaux
# License: BSD 3 clause

import sys
import functools
import time
try:
    import cPickle as pickle
except:
    import pickle

try:
    import multiprocessing
except ImportError:
    multiprocessing = None

from .format_stack import format_exc, format_outer_frames
from .logger import Logger, short_format_time
from .my_exceptions import TransportableException, _mk_exception

################################################################################

class SafeFunction(object):
    """ Wraps a function to make it exception with full traceback in
        their representation.
        Useful for parallel computing with multiprocessing, for which 
        exceptions cannot be captured.
    """

    def __init__(self, func):
        self.func = func


    def __call__(self, *args, **kwargs):
        try:
            return self.func(*args, **kwargs)
        except:
            e_type, e_value, e_tb = sys.exc_info()
            text = format_exc(e_type, e_value, e_tb, context=10,
                             tb_offset=1)
            raise TransportableException(text, e_type)

def print_progress(msg, index, total, start_time, n_jobs=1):
    # XXX: Not using the logger framework: need to
    # learn to use logger better.
    if total > 2*n_jobs:
        # Report less often
        if not index % n_jobs == 0:
            return
    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time/(index + 1)*
                (total - index - 1.))
    sys.stderr.write('[%s]: Done %3i out of %3i |elapsed: %s remaining: %s\n'
            % (msg,
                index+1, 
                total, 
                short_format_time(elapsed_time),
                short_format_time(remaining_time),
                ))


################################################################################
def delayed(function):
    """ Decorator used to capture the arguments of a function.
    """
    # Try to pickle the input function, to catch the problems early when
    # using with multiprocessing
    pickle.dumps(function)

    @functools.wraps(function)
    def delayed_function(*args, **kwargs):
        return function, args, kwargs
    return delayed_function


class LazyApply(object):
    """ Lazy version of the apply builtin function.
    """
    def __init__ (self, func, args, kwargs):
        self.func   = func
        self.args   = args
        self.kwargs = kwargs

    def get (self):
        return self.func(*self.args, **self.kwargs)



class Parallel(Logger):
    ''' Helper class for readable parallel mapping.

        Parameters
        -----------
        n_jobs: int
            The number of jobs to use for the computation. If -1 all CPUs
            are used. If 1 is given, no parallel computing code is used
            at all, which is useful for debuging.
        verbose: int, optional
            The verbosity level. If 1 is given, the elapsed time as well
            as the estimated remaining time are displayed.
        
        Notes
        -----

        This object uses the multiprocessing module to compute in
        parallel the application of a function to many different
        arguments. The main functionnality it brings in addition to 
        using the raw multiprocessing API are (see examples for details):

            * More readable code, in particular since it avoids 
              constructing list of arguments.

            * Easier debuging:
                - informative tracebacks even when the error happens on
                  the client side
                - using 'n_jobs=1' enables to turn off parallel computing
                  for debuging without changing the codepath
                - early capture of pickling errors

            * An optional progress meter.

        Examples
        --------


    '''
    def __init__(self, n_jobs=None, verbose=0):
        self.verbose = verbose
        self.n_jobs = n_jobs
        # Not starting the pool in the __init__ is a design decision, to
        # be able to close it ASAP, and not burden the user with closing
        # it.


    def __call__(self, iterable):
        n_jobs = self.n_jobs
        if n_jobs == -1:
            if multiprocessing is None:
                n_jobs = 1
            else:
                n_jobs = multiprocessing.cpu_count()

        if n_jobs is None or multiprocessing is None or n_jobs == 1:
            n_jobs = 1
            apply = LazyApply 
        else:
            pool = multiprocessing.Pool(n_jobs)
            def apply(func, args, kwargs):
                return pool.apply_async(SafeFunction(function), args, kwargs)

        output = list()
        start_time = time.time()
        try:
            for index, (function, args, kwargs) in enumerate(iterable):
                output.append(apply(function, args, kwargs))
                if self.verbose and n_jobs == 1:
                    print '[%s]: Done job %3i | elapsed: %s' % (
                            self, index, 
                            short_format_time(time.time() - start_time)
                        )

            start_time = time.time()
            jobs = output
            output = list()
            for index, job in enumerate(jobs):
                try:
                    output.append(job.get())
                    if self.verbose:
                        print_progress(self, index, len(jobs), start_time,
                                       n_jobs=n_jobs)
                except TransportableException, exception:
                    # Capture exception to add information on 
                    # the local stack in addition to the distant
                    # stack
                    this_report = format_outer_frames(
                                            context=10,
                                            stack_start=1,
                                            )
                    report = """Multiprocessing exception:
%s
---------------------------------------------------------------------------
Sub-process traceback: 
---------------------------------------------------------------------------
%s""" % (
                                this_report,
                                exception.message,
                            )
                    # Convert this to a JoblibException
                    exception_type = _mk_exception(exception.etype)[0]
                    raise exception_type(report)
        finally:
            if n_jobs > 1:
                pool.close()
                pool.join()
        return output


    def __repr__(self):
        return '%s(n_jobs=%s)' % (
                    self.__class__.__name__,
                    self.n_jobs,
                )



