"""Tests for `satsim.geometry.ephemeris` package."""
import numpy as np
from skyfield.constants import AU_KM, DAY_S
from skyfield.api import EarthSatellite

from satsim import time
from satsim.geometry.ephemeris import EphemerisObject, create_ephemeris_object
from satsim.geometry.sgp4 import create_sgp4


def test_ephemeris_object():

    sat1 = EarthSatellite(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    h = 3
    t0 = time.utc(2015, 5, 5, 13, 26, 43.288)
    t1 = time.utc(2015, 5, 5, 13 + h, 26, 43.288)

    # under sample
    n = 30
    x = time.linspace(t0, t1, n)
    y, v, _, _ = sat1._at(x)
    y = np.stack(y * AU_KM, axis=1)
    v = np.stack(v * AU_KM / DAY_S, axis=1)

    # truth samples
    xs = time.linspace(t0, t1, 300)
    ys, vs, _, _ = sat1._at(xs)

    # interpolated samples
    tt = np.linspace(0, h * 60 * 60, n)
    sat2 = EphemerisObject(y, v, tt, t0)
    ysi, vsi, _, _ = sat2._at(xs)

    np.testing.assert_array_less(np.abs((ys - ysi) * AU_KM), 2e-5)
    np.testing.assert_array_less(np.abs((vs - vsi) * AU_KM / DAY_S), 2e-9)

    # test single interpolated point
    t2 = time.utc(2015, 5, 5, 13, 50, 30)
    y2, v2, _, _ = sat1._at(t2)
    y2i, v2i, _, _ = sat2._at(t2)

    np.testing.assert_array_less(np.abs(y2 - y2i) * AU_KM, 2e-5)
    np.testing.assert_array_less(np.abs(v2 - v2i) * AU_KM / DAY_S, 2e-9)

    # test barycenter
    sat3 = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    # test segment boundaries
    sat4 = create_ephemeris_object([y], [v], [tt], t0)

    y3, v3, _, _ = sat3._at(t2)
    y3i, v3i, _, _ = sat4._at(t2)

    np.testing.assert_array_less(np.abs(y3 - y3i) * AU_KM, 2e-5)
    np.testing.assert_array_less(np.abs(v3 - v3i) * AU_KM / DAY_S, 2e-9)

    assert('EphemerisObject' in str(sat4))
