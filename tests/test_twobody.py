"""Tests for `satsim.geometry.twobody` package."""
import numpy as np

from satsim.geometry.astrometric import create_topocentric, get_los
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.twobody import create_twobody_from_tle
from satsim import time


def test_two_body():

    tt = time.utc(2015, 4, 24, 9, 7, np.arange(0, 5, step=1))

    s0 = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    s1 = create_twobody_from_tle(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866",
        method='vallado'
    )
    s2 = create_twobody_from_tle(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866",
        method='cowell'
    )
    s3 = create_twobody_from_tle(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866",
        method='farnocchia'
    )

    r0, v0, _, _ = s0._at(tt)

    r1, v1, _, _ = s1._at(tt)
    np.testing.assert_almost_equal(r0, r1, decimal=6)
    np.testing.assert_almost_equal(v0, v1, decimal=6)

    r1, v1, _, _ = s2._at(tt)
    np.testing.assert_almost_equal(r0, r1, decimal=6)
    np.testing.assert_almost_equal(v0, v1, decimal=6)

    r1, v1, _, _ = s3._at(tt)
    np.testing.assert_almost_equal(r0, r1, decimal=6)
    np.testing.assert_almost_equal(v0, v1, decimal=6)

    assert('EarthTwoBodySatellite' in str(s1))
    assert('EarthTwoBodySatellite' in str(s2))
    assert('EarthTwoBodySatellite' in str(s3))


def test_get_los():

    topo = create_topocentric("20.746111 N", "156.431667 W")
    sat = create_twobody_from_tle(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    t = time.utc(2015, 5, 5, 13, 26, 43.288)

    ra, dec, km, az, el, los = get_los(topo, sat, t)

    np.testing.assert_almost_equal(ra, 292.1, decimal=1)
    np.testing.assert_almost_equal(dec, -3.4, decimal=1)
    np.testing.assert_almost_equal(km / 10, 36700 / 10, decimal=0)
    np.testing.assert_almost_equal(az, 133.0, decimal=1)
    np.testing.assert_almost_equal(el, 56.3, decimal=1)


def test_propagator_long_term():
    """Ensure the internal propagator matches SGP4 for short intervals."""

    tle1 = "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992"
    tle2 = "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"

    sgp = create_sgp4(tle1, tle2)
    tb = create_twobody_from_tle(tle1, tle2)

    from skyfield.api import EarthSatellite
    base_sat = EarthSatellite(tle1, tle2)
    t1 = base_sat.epoch + 3600.0 / 86400.0  # 1 hour later
    r0, v0, _, _ = sgp._at(t1)
    r1, v1, _, _ = tb._at(t1)

    np.testing.assert_allclose(r0, r1, rtol=0, atol=1e-5)
    np.testing.assert_allclose(v0, v1, rtol=0, atol=1e-5)
