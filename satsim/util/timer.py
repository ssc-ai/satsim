from __future__ import division, print_function, absolute_import

import time
from collections import OrderedDict

# singletons for timing
_satsim_util_tic = OrderedDict()
_satsim_util_toc = OrderedDict()


def tic(name=None, i=None):
    """`tic` starts a stopwatch timer to measure performance. The function
    records the internal time at execution of the tic command. Display the
    elapsed time with the `toc` function. If no `name` is given, the name will
    be assigned to `__default__`.

    Examples::

        tic()
        time.sleep(1)
        print(toc())

    Args:
        name: `str`, name to give timer to reference later
        i: `int`, index if array of times need to be recorded
    """
    global _satsim_util_tic

    if name is None:
        name = '__default__'

    if i is not None:
        _satsim_util_tic['{}_{}'.format(name, i)] = time.time()
    else:
        _satsim_util_tic[name] = time.time()


def toc(name=None, i=None):
    """`toc` reads the elapsed time from the stopwatch timer started by the `tic`
    function. The function reads the internal time at the execution of the `toc`
    command, and returns the elapsed time since the most recent call to the
    `tic` function in seconds.

    Args:
        name: `str`, name to give timer to reference later
        i: `int`, index if array of times need to be recorded

    Returns:
        A `float`, seconds elapsed
    """
    global _satsim_util_tic
    global _satsim_util_toc

    if name is None:
        name = '__default__'

    if i is not None:
        tt = time.time() - _satsim_util_tic['{}_{}'.format(name, i)]
        _satsim_util_toc['{}_{}'.format(name, i)] = tt
    else:
        tt = time.time() - _satsim_util_tic[name]
        _satsim_util_toc[name] = tt

    return tt


def get_timing():
    """Returns all stopwatch recordings from `tic` and `toc`.

    Returns:
        A `dict`, recorded times where the `key` is the `name` given in `tic`
        and `toc`
    """

    global _satsim_util_toc
    return _satsim_util_toc
