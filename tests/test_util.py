"""Tests for `satsim.geometry.draw` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import time

import numpy as np

from satsim.util.timer import tic, toc, get_timing, Profiler
from satsim.util.thread import ThreadedTaskQueue, MultithreadedTaskQueue
from satsim.util.system import get_semantic_version
from satsim.util.python import merge_dicts


def test_tic_toc():

    tic()
    time.sleep(0.05)
    a = toc()
    np.testing.assert_almost_equal(a, 0.05, decimal=2)

    tic('name', 7)
    time.sleep(0.1)
    b = toc('name', 7)
    np.testing.assert_almost_equal(b, 0.1, decimal=2)

    d = get_timing()

    assert(d['__default__'] == a)
    assert(d['name_7'] == b)


def test_thread():

    a = []

    def f(i):
        time.sleep(0.1)
        a.append(i)

    def e(i):
        time.sleep(0.1)
        raise Exception('Test Exception')

    q = ThreadedTaskQueue()

    for j in range(10):
        q.task(f, {'i': j})

    q.task(e, {'i': j + 1})

    q.waitUntilEmpty()

    for j in range(10):
        assert j in a


def test_multi_threads():

    a = []

    def f(i):
        time.sleep(0.1)
        a.append(i)

    q = MultithreadedTaskQueue(num_threads=10)

    for j in range(10):
        assert not q.task_queues[j].busy()

    for j in range(10):
        q.task(f, {'i': j})

    time.sleep(0.05)

    for j in range(10):
        assert q.task_queues[j].busy()

    q.waitUntilEmpty()

    for j in range(10):
        assert j in a

    a.clear()
    for j in range(15):
        q.task(f, {'i': j})

    q.waitUntilEmpty()

    for j in range(15):
        assert j in a

    q.stop()

    time.sleep(0.1)

    for j in range(10):
        assert not q.task_queues[j].t.is_alive()


def test_get_semantic_version():

    class TestMod:

        def __init__(self):
            self.__version__ = '2.1.0-beta2'

    t = TestMod()

    ver = get_semantic_version(t)

    assert np.array_equal(ver, [2,1,0])


def test_merge_basic():
    d1 = {'a': 1, 'b': {'x': 10}}
    d2 = {'b': {'y': 20}, 'c': 3}
    expected = {'a': 1, 'b': {'x': 10, 'y': 20}, 'c': 3}
    merge_dicts(d1, d2)
    assert d1 == expected


def test_merge_with_overwrite():
    d1 = {'a': 1, 'b': {'x': 10}}
    d2 = {'b': {'x': 20}, 'c': 3}
    expected = {'a': 1, 'b': {'x': 20}, 'c': 3}
    merge_dicts(d1, d2)
    assert d1 == expected


def test_merge_with_nested_dict():
    d1 = {'a': 1, 'b': {'x': {'y': 10}}}
    d2 = {'b': {'x': {'z': 20}}, 'c': 3}
    expected = {'a': 1, 'b': {'x': {'y': 10, 'z': 20}}, 'c': 3}
    merge_dicts(d1, d2)
    assert d1 == expected


def test_merge_with_non_dict_value():
    d1 = {'a': 1, 'b': {'x': 10}}
    d2 = {'b': 'new_value', 'c': 3}
    expected = {'a': 1, 'b': 'new_value', 'c': 3}
    merge_dicts(d1, d2)
    assert d1 == expected


def test_merge_empty_dict():
    d1 = {'a': 1, 'b': {'x': 10}}
    d2 = {}
    expected = {'a': 1, 'b': {'x': 10}}
    merge_dicts(d1, d2)
    assert d1 == expected


def test_merge_into_empty_dict():
    d1 = {}
    d2 = {'a': 1, 'b': {'x': 10}}
    expected = {'a': 1, 'b': {'x': 10}}
    merge_dicts(d1, d2)
    assert d1 == expected


class _DummyLogger:
    def __init__(self):
        self.debug_messages = []
        self.warning_messages = []

    def debug(self, message):
        self.debug_messages.append(message)

    def warning(self, message):
        self.warning_messages.append(message)


def test_profiler_disabled_noop():
    logger = _DummyLogger()
    profiler = Profiler(enabled=False, logger=logger, prefix='Disabled')

    with profiler.time('noop'):
        time.sleep(0.001)

    assert profiler.start('start') is None
    assert profiler.stop('stop', None) == 0.0
    profiler.set_metric('count', 1)
    profiler.add_metric('count', 2)
    profiler.log(order_times=['noop'], order_metrics=['count'])

    assert profiler.times == {}
    assert profiler.metrics == {}
    assert logger.debug_messages == []


def test_profiler_timing_and_metrics():
    logger = _DummyLogger()
    profiler = Profiler(enabled=True, logger=logger, prefix='Profile')

    with profiler.time('step'):
        time.sleep(0.001)

    start = profiler.start('accum')
    time.sleep(0.001)
    duration1 = profiler.stop('accum', start)

    start = profiler.start('accum')
    time.sleep(0.001)
    duration2 = profiler.stop('accum', start, accumulate=True)

    profiler.set_metric('count', 1)
    profiler.add_metric('count', 2)
    profiler.log(order_times=['step', 'accum'], order_metrics=['count'])

    assert 'step' in profiler.times
    assert 'accum' in profiler.times
    assert profiler.times['accum'] >= duration1
    assert profiler.times['accum'] >= duration2
    assert profiler.metrics['count'] == 3
    assert logger.debug_messages
    assert 'Profile:' in logger.debug_messages[-1]
    assert 'step=' in logger.debug_messages[-1]
    assert 'count=3' in logger.debug_messages[-1]


def test_profiler_from_sim_deprecation():
    logger = _DummyLogger()
    sim = {'profile_objects': True}

    profiler = Profiler.from_sim(sim, logger=logger, prefix='Deprecated')
    assert profiler.enabled is True
    assert sim.get('_profile_objects_warned') is True
    assert len(logger.warning_messages) == 1

    profiler = Profiler.from_sim(sim, logger=logger, prefix='Deprecated')
    assert profiler.enabled is True
    assert len(logger.warning_messages) == 1

    sim = {'profile_objects': True, 'enable_profiler': False}
    profiler = Profiler.from_sim(sim, logger=logger, prefix='Disabled')
    assert profiler.enabled is False

# cannot run these more than once
# def test_configure_eager():

#     configure_eager(True)

#     assert tf.compat.v1.executing_eagerly()

#     physical_devices = tf.config.experimental.list_physical_devices('GPU')

#     for gpu in physical_devices:
#         assert tf.config.experimental.get_memory_growth(gpu)


# def test_configure_single_gpu():

#     gpu = configure_single_gpu(0, 100)

#     assert len(gpu) == 1
