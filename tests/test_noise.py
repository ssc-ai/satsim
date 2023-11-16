"""Tests for `satsim.image.noise` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
from scipy.stats import describe
import tensorflow as tf

from satsim.image.noise import add_photon_noise, add_read_noise
from satsim.util import configure_eager

configure_eager()


def test_add_photon_noise():

    sx = 1000
    sy = 1000

    val = 500000
    a = tf.ones([sx, sy]) * val

    b = add_photon_noise(a)

    stat = describe(b.numpy().flatten())

    np.testing.assert_approx_equal(stat.mean, val, significant=2)
    np.testing.assert_approx_equal(stat.mean, stat.variance, significant=2)

    val = 5
    c = tf.ones([sx, sy]) * val

    d = add_photon_noise(c)

    stat = describe(d.numpy().flatten())

    np.testing.assert_approx_equal(stat.mean, val, significant=2)
    np.testing.assert_approx_equal(stat.mean, stat.variance, significant=2)

    c2 = tf.ones([sx, sy]) * val

    d2 = add_photon_noise(c2, samples=30)

    stat2 = describe(d2.numpy().flatten())

    np.testing.assert_approx_equal(stat2.mean, val, significant=2)
    np.testing.assert_approx_equal(stat2.mean / 30.0, stat2.variance, significant=2)


def test_read_noise():

    sx = 1000
    sy = 1000

    val = 1000
    a = tf.ones([sx, sy]) * val

    rn = 5
    en = 5
    b, _ = add_read_noise(a, rn, en)

    stat = describe(b.numpy().flatten())

    np.testing.assert_approx_equal(stat.mean, val, significant=2)
    np.testing.assert_approx_equal(stat.variance, rn * rn + en * en, significant=2)
