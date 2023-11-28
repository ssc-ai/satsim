"""Tests for `satsim.geometry.draw` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import time

import numpy as np

from satsim.util.timer import tic, toc, get_timing
from satsim.util.thread import ThreadedTaskQueue, MultithreadedTaskQueue
from satsim.util.system import get_semantic_version


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
