"""Tests for `satsim.geometry.random` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.random import gen_random_points
from satsim.math.random import gen_sample, lognormal


def test_random_points():

    r,c,pe = gen_random_points(500, 1000, 2., 2., [100.,200.,300.], [2.,4.], pad_mult=0.)

    # note 4 calculated from y_fov * x_fov (no pad)
    assert( len(r) == (2 + 4) * 4)
    assert( len(c) == (2 + 4) * 4)
    assert(len(pe) == (2 + 4) * 4)

    assert(np.max(r) < 500)
    assert(np.max(c) > 500)

    r,c,pe = gen_random_points(500, 1000, 2., 2., [100.,200.,300.], [2.,4.], pad_mult=1.)

    # note 36 calculated from y_fov*(2*pad+1) * x_fov*(2*pad+1)
    assert( len(r) == (2 + 4) * 36)
    assert( len(c) == (2 + 4) * 36)
    assert(len(pe) == (2 + 4) * 36)

    assert(np.max(r) >  750 and np.max(r) <= 1000)
    assert(np.max(c) > 1250 and np.max(c) <= 2000)
    assert(np.min(r) < -250 and np.min(r) >= -500)
    assert(np.min(c) < -500 and np.min(c) >= -1000)


def test_random_sample():

    assert(gen_sample(type='uniform', low=0, high=0.0001) <= 0.0001)
    assert(gen_sample(type='uniform', low=0, high=0.0001) >= 0)

    assert(gen_sample(type='uniform', negate=1.0, low=0, high=0.0001) < 0)


def test_lognormal():

    dist = lognormal(mu=1.0, sigma=1.0, size=100000, mu_mode='median')

    assert(len(dist) == 100000)
    np.testing.assert_almost_equal(np.median(dist), 1.0, decimal=1)

    dist = lognormal(mu=1.0, sigma=1.0, size=100000, mu_mode='mean')
    assert(len(dist) == 100000)
    np.testing.assert_almost_equal(np.mean(dist), 1.0, decimal=1)
    np.testing.assert_almost_equal(np.std(dist), 1.0, decimal=1)
