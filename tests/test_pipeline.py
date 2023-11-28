"""Tests for `satsim.config`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import math
import numpy as np

from satsim import config
from satsim import pipeline


def test_function_pipeline():

    f = config.parse_function_pipeline([
        {
            "module": "numpy",
            "function": "sin"
        },
        {
            "module": "satsim.pipeline",
            "function": "poly",
            "kwargs": {
                "coef": [2, 10]
            }
        }
    ])

    assert(f(0.0) == 10.0)  # 10 + 2 * sin(0) = 10.0
    assert(f(math.pi / 2.0) == 12.0)  # 10 + 2 * sin(pi/2) = 12.0


def test_glint():

    x = pipeline.glint(np.zeros(10), range(11), period=2, magnitude=5)

    np.testing.assert_equal(x, [0, 5, 0, 5, 0, 5, 0, 5, 0, 5])

    x = pipeline.glint(np.ones(10), range(11), period=0.1, magnitude=5)

    np.testing.assert_equal(x, [5, 5, 5, 5, 5, 5, 5, 5, 5, 5])

    x = pipeline.glint(np.zeros(10), range(11), period=7.5, magnitude=500)

    np.testing.assert_equal(x, [0, 0, 0, 0, 0, 0, 0, 500, 0, 0])


def test_poly():

    x = pipeline.poly(range(10), [], coef=[2, 10])

    np.testing.assert_equal(x, [10., 12., 14., 16., 18., 20., 22., 24., 26., 28.])


def test_polyt():

    x = pipeline.polyt(np.ones(10), list(range(11)), coef=[2, 10])

    np.testing.assert_equal(x, [12., 14., 16., 18., 20., 22., 24., 26., 28., 30.])


def test_sin_add():

    x = pipeline.sin_add([1, 1], [0, 0.5, 1])

    np.testing.assert_almost_equal(x, [2, 1])


def test_sin():

    x = pipeline.sin([1, 1], [0, 0.5, 1])

    np.testing.assert_almost_equal(x, [1, 0])


def test_cos():

    x = pipeline.cos([1, 1], [0, 0.5, 1])

    np.testing.assert_almost_equal(x, [0.5, 0.5])


def test_delta_t():

    t = [0,1,3,6,10,15,21]

    dt = pipeline._delta_t(t)

    np.testing.assert_equal(dt, [1, 2, 3, 4, 5, 6])


def test_constant():

    x = pipeline.constant([0, 0, 0, 0], None, value=5)

    np.testing.assert_equal(x, [5, 5, 5, 5])
