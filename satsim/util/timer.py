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


class _NullContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


_NULL_CONTEXT = _NullContext()


class _ProfileScope:
    def __init__(self, profiler, name, accumulate):
        self._profiler = profiler
        self._name = name
        self._accumulate = accumulate
        self._start = None

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._profiler.stop(self._name, self._start, accumulate=self._accumulate)
        return False


class Profiler:
    def __init__(self, enabled=False, logger=None, prefix=None):
        self.enabled = bool(enabled)
        self.logger = logger
        self.prefix = '' if prefix is None else prefix
        self.times = OrderedDict()
        self.metrics = OrderedDict()

    @classmethod
    def from_sim(cls, sim, logger=None, prefix=None):
        enabled = sim.get('enable_profiler', sim.get('profile_objects', False))
        if 'profile_objects' in sim and 'enable_profiler' not in sim and logger is not None:
            if not sim.get('_profile_objects_warned', False):
                logger.warning('sim.profile_objects is deprecated; use sim.enable_profiler instead.')
                sim['_profile_objects_warned'] = True
        return cls(enabled=enabled, logger=logger, prefix=prefix)

    def child(self, prefix):
        return Profiler(enabled=self.enabled, logger=self.logger, prefix=prefix)

    def reset(self):
        self.times.clear()
        self.metrics.clear()

    def start(self, name):
        if not self.enabled:
            return None
        return time.perf_counter()

    def stop(self, name, start, accumulate=False):
        if not self.enabled or start is None:
            return 0.0
        duration = time.perf_counter() - start
        if accumulate and name in self.times:
            self.times[name] += duration
        else:
            self.times[name] = duration
        return duration

    def time(self, name, accumulate=False):
        if not self.enabled:
            return _NULL_CONTEXT
        return _ProfileScope(self, name, accumulate)

    def set_metric(self, name, value):
        if not self.enabled:
            return
        self.metrics[name] = value

    def add_metric(self, name, value):
        if not self.enabled:
            return
        self.metrics[name] = self.metrics.get(name, 0) + value

    def log(self, order_times=None, order_metrics=None, level='debug', prefix=None):
        if not self.enabled or self.logger is None:
            return
        times = self.times
        metrics = self.metrics
        if order_times is None:
            order_times = list(times.keys())
        if order_metrics is None:
            order_metrics = list(metrics.keys())

        parts = []
        for key in order_times:
            if key in times:
                parts.append('{}={:.3f} sec'.format(key, times[key]))
        for key in order_metrics:
            if key in metrics:
                parts.append('{}={}'.format(key, metrics[key]))

        message = ', '.join(parts)
        label = self.prefix if prefix is None else prefix
        if label:
            if message:
                message = '{}: {}'.format(label, message)
            else:
                message = label

        log_fn = getattr(self.logger, level, None)
        if log_fn is not None:
            log_fn(message)
