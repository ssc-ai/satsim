"""Tests for `satsim.geometry.sgp4` package."""
import numpy as np
from satsim.geometry.sgp4 import load_tle, create_sgp4
from satsim.geometry.astrometric import create_topocentric, get_los
from satsim import time


def test_load_tle():

    sats = load_tle(['./tests/geo.txt'])

    assert(len(sats) == 513)

    sats = load_tle(['./tests/geo.txt', './tests/geo.txt'])

    assert(len(sats) == 513 * 2)


def test_create_sgp4():

    sat = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )

    assert(sat is not None)


def test_get_los():

    topo = create_topocentric("20.746111 N", "156.431667 W")
    sat = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    t = time.utc(2015, 5, 5, 13, 26, 43.288)

    ra, dec, km, az, el, los = get_los(topo, sat, t)

    # print(ra, dec, km, az, el)

    np.testing.assert_almost_equal(ra, 292.3, decimal=1)
    np.testing.assert_almost_equal(dec, -3.4, decimal=1)
    np.testing.assert_almost_equal(km, 36711.69, decimal=2)
    np.testing.assert_almost_equal(az, 132.7, decimal=1)
    np.testing.assert_almost_equal(el, 56.0, decimal=1)
