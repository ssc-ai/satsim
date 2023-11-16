"""Tests for `satsim.geometry.twobody` package."""
import numpy as np

from satsim.geometry.astrometric import create_topocentric
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.observation import create_observation
from satsim import time


def test_observation():

    t = time.utc(2023, 6, 20, 8, 49, 5.107)

    sat = create_sgp4("1 52145U 22030A   23170.30195210 +.00000172 +00000+0 +00000+0 0 99995", "2 52145  62.6634 261.0254 7030524 272.8547  15.8426 02.00622996009081")

    topo = create_topocentric("34.96311 N", "-106.497249 E", 0.0)

    p1 = sat.at(t)
    p2 = topo.at(t)

    p3 = p1 - p2
    ra3, dec3, d3 = p3.radec()

    ob = create_observation(ra3._degrees, dec3._degrees, t, topo, None, d3.km)

    p4 = (ob - topo).at(t)
    ra4, dec4, d4 = p4.radec()

    assert(np.abs(ra4._degrees - ra3._degrees) < 1e-6)
    assert(np.abs(dec4._degrees - dec3._degrees) < 1e-6)
    assert(np.abs(d4.km - d3.km) < 1e-6)

    ob2 = create_observation(ra3._degrees, dec3._degrees, t, topo, sat)

    p5 = (ob2 - topo).at(t)
    ra5, dec5, d5 = p5.radec()

    assert(np.abs(ra5._degrees - ra3._degrees) < 1e-6)
    assert(np.abs(dec5._degrees - dec3._degrees) < 1e-6)
    assert(np.abs(d5.km - d3.km) < 1e-6)

    tt = time.utc(2023, 6, 20, 8, 49, [5.107, 5.107])
    p6 = (ob2 - topo).at(tt)
    ra6, dec6, d6 = p6[0].radec()

    assert(np.abs(ra6._degrees - ra3._degrees) < 1e-6)
    assert(np.abs(dec6._degrees - dec3._degrees) < 1e-6)
    assert(np.abs(d6.km - d3.km) < 1e-6)

    ra7, dec7, d7 = p6[1].radec()

    assert(np.abs(ra7._degrees - ra3._degrees) < 1e-6)
    assert(np.abs(dec7._degrees - dec3._degrees) < 1e-6)
    assert(np.abs(d7.km - d3.km) < 1e-6)

    assert('EarthObservation' in str(ob))
    assert('EarthObservation' in str(ob2))
