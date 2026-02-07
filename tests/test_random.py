"""Tests for `satsim.geometry.random` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import pytest
import numpy as np

from satsim.geometry.random import gen_random_points
from satsim.math.random import gen_sample, lognormal_mu_sigma, simplex, simplex_stripe


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


def test_seed_is_local():

    np.random.seed(123)
    expected = np.random.uniform()
    np.random.seed(123)
    gen_sample(type='uniform', seed=42, negate=0.5, low=0.0, high=1.0)
    actual = np.random.uniform()

    assert(actual == expected)


def test_lognormal():

    dist = lognormal_mu_sigma(mu=1.0, sigma=1.0, size=100000, mu_mode='median')

    assert(len(dist) == 100000)
    np.testing.assert_almost_equal(np.median(dist), 1.0, decimal=1)

    dist = lognormal_mu_sigma(mu=1.0, sigma=1.0, size=100000, mu_mode='mean')
    assert(len(dist) == 100000)
    np.testing.assert_almost_equal(np.mean(dist), 1.0, decimal=1)
    np.testing.assert_almost_equal(np.std(dist), 1.0, decimal=1)


def test_simplex_seed_deterministic():

    pytest.importorskip("opensimplex")

    a = simplex(size=(32, 32), seed=7, scale=32.0, octaves=3)
    b = simplex(size=(32, 32), seed=7, scale=32.0, octaves=3)

    np.testing.assert_allclose(a, b)


def test_simplex_stripe_axis():

    pytest.importorskip("opensimplex")

    stripe_col = simplex_stripe(size=(8, 6), axis='col', seed=5)
    assert(stripe_col.shape == (8, 6))
    np.testing.assert_allclose(stripe_col[0, :], stripe_col[1, :])

    stripe_row = simplex_stripe(size=(8, 6), axis='row', seed=5)
    np.testing.assert_allclose(stripe_row[:, 0], stripe_row[:, 1])


def test_simplex_center():

    pytest.importorskip("opensimplex")

    img = simplex(size=(64, 64), sigma=0.1, center=1.0, seed=3)
    np.testing.assert_allclose(np.mean(img), 1.0, atol=1e-3)
