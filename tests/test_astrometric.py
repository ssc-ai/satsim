"""Tests for `satsim.geometry.transform` package."""
from dateutil import parser

import numpy as np

from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.astrometric import (
    apparent,
    load_earth,
    load_sun,
    load_moon,
    create_topocentric,
    get_los,
    gen_track,
    query_by_los,
    get_los_azel,
    angle_between,
    angle_from_los,
    eci_to_radec,
    radec_to_eci,
    get_analytical_los,
)
from satsim.geometry.greatcircle import GreatCircle
from skyfield.api import Star
from satsim.geometry.photometric import lambertian_sphere_to_mv
from satsim import time


def test_load_earth():

    earth = load_earth()

    assert(earth is not None)


def test_create_topocentric():

    topo = create_topocentric("20.746111 N", "156.431667 W")

    assert(topo is not None)


def test_get_los():

    topo = create_topocentric("20.746111 N", "156.431667 W")
    sat = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )
    t = time.utc(2015, 5, 5, 13, 26, 43.288)

    ra, dec, km, az, el, los = get_los(topo, sat, t)

    np.testing.assert_almost_equal(ra, 292.39212493079657, decimal=5)
    np.testing.assert_almost_equal(dec, -3.4988184979860004, decimal=5)
    np.testing.assert_almost_equal(km, 36711.694341587696, decimal=3)
    np.testing.assert_almost_equal(az, 132.7494647652191, decimal=5)
    np.testing.assert_almost_equal(el, 56.0696088436758, decimal=5)

    ra, dec, km, az, el, los = get_los(topo, sat, t, deflection=True, aberration=False)
    np.testing.assert_almost_equal(ra, 292.3898647283382, decimal=5)
    np.testing.assert_almost_equal(dec, -3.4975034271474295, decimal=5)
    np.testing.assert_almost_equal(km, 36711.694341587696, decimal=3)
    np.testing.assert_almost_equal(az, 132.75078331984074, decimal=5)
    np.testing.assert_almost_equal(el, 56.07211428771807, decimal=5)

    ra2, dec2, km2, az2, el2, los2 = get_los_azel(topo, az, el, t)
    np.testing.assert_almost_equal(ra, ra2)
    np.testing.assert_almost_equal(dec, dec2)
    np.testing.assert_almost_equal(az, az2)
    np.testing.assert_almost_equal(el, el2)


def test_gen_track():

    t0 = time.utc(2015, 5, 5, 13, 26, 43.288)
    t_start = time.utc(2015, 5, 5, 13, 26, 43.288)
    t_end = time.utc(2015, 5, 5, 13, 26, 43.288 + 5.0)

    observer = create_topocentric("20.746111 N", "156.431667 W")
    sat = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    )

    (rr0, rr1), (cc0, cc1), dr, dc, b = gen_track(512, 512, 1, 1, observer, sat, [sat], [1000], t0, [t_start, t_end], rot=0, pad_mult=0, track_type='rate', offset=[0,0])

    assert(len(rr0) == 1)
    np.testing.assert_almost_equal(rr0[0], 256.)
    np.testing.assert_almost_equal(cc0[0], 256.)
    np.testing.assert_almost_equal(rr1[0], 256.)
    np.testing.assert_almost_equal(cc1[0], 256.)
    assert(b[0] == 1000.)

    ra, dec, km, az, el, los = get_los(observer, sat, t_start)
    ra2, dec2, km2, az2, el2, los2 = get_los(observer, sat, t_end)

    (rr0, rr1), (cc0, cc1), dr, dc, b = gen_track(512, 512, 1, 1, observer, sat, [sat], [1000], t0, [t_start, t_end], rot=0, pad_mult=0, track_type='fixed', offset=[0,0], az=[az, az2], el=[el, el2])

    assert(len(rr0) == 1)
    np.testing.assert_almost_equal(rr0[0], 256.)
    np.testing.assert_almost_equal(cc0[0], 256.)
    np.testing.assert_almost_equal(rr1[0], 256.)
    np.testing.assert_almost_equal(cc1[0], 256.)
    assert(b[0] == 1000.)

    (rr0, rr1), (cc0, cc1), dr, dc, b = gen_track(512, 512, 1, 1, observer, sat, [sat], [1000], t0, [t_start, t_end], rot=0, pad_mult=0, track_type='sidereal', offset=[0,0])

    assert(len(rr0) == 1)
    np.testing.assert_almost_equal(rr0[0], 256.)
    np.testing.assert_almost_equal(cc0[0], 256.)
    np.testing.assert_almost_equal(rr1[0], 255.9726454, decimal=4)
    np.testing.assert_almost_equal(cc1[0], 266.67897743, decimal=4)
    assert(b[0] == 1000.)

    (rr0, rr1), (cc0, cc1), dr, dc, b = gen_track(512, 512, 1, 1, observer, sat, [sat], [1000], t0, [t_start, t_end], rot=0, pad_mult=0, track_type='sidereal', offset=[5.5,-10.3])

    assert(len(rr0) == 1)
    np.testing.assert_almost_equal(rr0[0], 256. + 5.5)
    np.testing.assert_almost_equal(cc0[0], 256. - 10.3)
    np.testing.assert_almost_equal(rr1[0], 255.9726454 + 5.5, decimal=4)
    np.testing.assert_almost_equal(cc1[0], 266.67897743 - 10.3, decimal=4)
    assert(b[0] == 1000.)

    (rr0, rr1), (cc0, cc1), dr, dc, b = gen_track(512, 512, 1, 1, observer, sat, [sat], [1000], t0, [t_start, t_end], rot=0, pad_mult=0, track_type='sidereal', offset=[5.5,-10.3], flipud=True, fliplr=True)

    assert(len(rr0) == 1)
    np.testing.assert_almost_equal(rr0[0], 256. - 5.5)
    np.testing.assert_almost_equal(cc0[0], 256. + 10.3)
    np.testing.assert_almost_equal(rr1[0], 256.0273546 - 5.5, decimal=4)
    np.testing.assert_almost_equal(cc1[0], 245.3210226 + 10.3, decimal=4)
    assert(b[0] == 1000.)

    [ra,dec,d,az,el,los] = get_los(observer, sat, t_start)
    visible, idx = query_by_los(512, 512, 1, 1, ra, dec, t_start, observer, [sat, sat], rot=0, pad_mult=0)
    assert(len(visible) == 2)
    assert(idx == [0, 1])

    visible, idx = query_by_los(512, 512, 1, 1, ra, dec, t_start, observer, [sat], rot=0, pad_mult=0)
    assert(len(visible) == 1)
    assert(idx == [0])

    visible, idx = query_by_los(512, 512, 1, 1, ra + 1, dec + 1, t_start, observer, [sat], rot=0, pad_mult=0)
    assert(len(visible) == 0)
    assert(len(idx) == 0)


def test_great_circle():

    t = time.utc(2021, 3, 18, 0, 0, 0.0)
    target = GreatCircle(0, 0, 0, 1.0, t, None)
    observer = create_topocentric("20.746111 N", "156.431667 W")

    for s in range(0, 90):
        t1 = time.utc(2021, 3, 18, 0, 0, s)
        icrf_los = apparent(observer.at(t1).observe(target), False, False)
        ra, dec, d = icrf_los.radec()

        np.testing.assert_almost_equal(360, ra._degrees)
        np.testing.assert_almost_equal(s, dec._degrees)

    for s in range(91, 180):
        t1 = time.utc(2021, 3, 18, 0, 0, s)
        icrf_los = apparent(observer.at(t1).observe(target), False, False)
        ra, dec, d = icrf_los.radec()

        np.testing.assert_almost_equal(180, ra._degrees)
        np.testing.assert_almost_equal(np.abs(180 - s), dec._degrees)

    for s in range(181, 270):
        t1 = time.utc(2021, 3, 18, 0, 0, s)
        icrf_los = apparent(observer.at(t1).observe(target), False, False)
        ra, dec, d = icrf_los.radec()

        np.testing.assert_almost_equal(180, ra._degrees)
        np.testing.assert_almost_equal(180 - s, dec._degrees)

    for s in range(271, 360):
        t1 = time.utc(2021, 3, 18, 0, 0, s)
        icrf_los = apparent(observer.at(t1).observe(target), False, False)
        ra, dec, d = icrf_los.radec()

        np.testing.assert_almost_equal(360, ra._degrees)
        np.testing.assert_almost_equal(s - 360, dec._degrees)

    target = GreatCircle(0, 0, 90, 1.0, t, None)
    for s in range(1, 360):
        t1 = time.utc(2021, 3, 18, 0, 0, s)
        icrf_los = apparent(observer.at(t1).observe(target), False, False)
        ra, dec, d = icrf_los.radec()

        np.testing.assert_almost_equal(s, ra._degrees)
        np.testing.assert_almost_equal(0, dec._degrees)


def test_angle():
    sun = load_sun()
    moon = load_moon()

    epoch = parser.isoparse('2021-09-21T17:40:00Z')
    t_epoch = time.from_datetime(epoch)

    observer = create_topocentric("20.746111 N", "156.431667 W")
    satellite = create_sgp4('1 41788U 16059F   21260.56440839  .00022290  00000-0  23487-2 0  9990', '2 41788  98.2495   9.3816 0022044 246.7613 113.2488 14.89172957267561')

    targ_solar_phase_ang = angle_between(satellite, observer, sun, t_epoch)
    sun_ang = angle_between(observer, satellite, sun, t_epoch)
    moon_solar_phase_ang = angle_between(moon, observer, sun, t_epoch)
    moon_ang = angle_between(observer, satellite, moon, t_epoch)

    np.testing.assert_almost_equal(targ_solar_phase_ang, 31.575062098216204, decimal=5)
    np.testing.assert_almost_equal(sun_ang, 148.4237766197407, decimal=5)
    np.testing.assert_almost_equal(moon_solar_phase_ang, 8.82909120048772, decimal=5)
    np.testing.assert_almost_equal(moon_ang, 40.23102425482865, decimal=5)

    ra, dec, _, az, el, _ = get_los(observer, satellite, t_epoch, True, True, False)

    ang = angle_from_los(observer, satellite, ra, dec, t_epoch)

    np.testing.assert_almost_equal(0, ang, decimal=2)

    ang = angle_from_los(observer, satellite, ra, dec + 1, t_epoch)

    np.testing.assert_almost_equal(1, ang, decimal=2)


def test_get_los_great_circle():
    t0 = time.utc(2021, 3, 18, 0, 0, 0.0)
    target = GreatCircle(0, 0, 90, 1.0, t0, None)
    observer = create_topocentric("20.746111 N", "156.431667 W")
    t1 = time.utc(2021, 3, 18, 0, 0, 10)

    ra, dec, _, _, _, _ = get_los(observer, target, t1, False, False, False)

    np.testing.assert_almost_equal(10, ra)
    np.testing.assert_almost_equal(0, dec)


def test_get_los_star():
    observer = create_topocentric("20.746111 N", "156.431667 W")
    star = Star(ra_hours=0, dec_degrees=0)
    t = time.utc(2021, 3, 18, 0, 0, 0)

    ra, dec, _, _, _, _ = get_los(observer, star, t, False, True, False)
    icrf_los = apparent(observer.at(t).observe(star), False, True)
    ra2, dec2, _ = icrf_los.radec()

    np.testing.assert_almost_equal(ra2._degrees, ra)
    np.testing.assert_almost_equal(dec2._degrees, dec)


def test_get_los_frames():
    topo = create_topocentric("20.746111 N", "156.431667 W")
    sat = create_sgp4(
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866",
    )
    t = time.utc(2015, 5, 5, 13, 26, 43.288)

    ra_b, dec_b, _ = get_analytical_los(topo, sat, t, frame="barycentric")
    ra_g, dec_g, _ = get_analytical_los(topo, sat, t, frame="geocentric")
    ra_o, dec_o, _ = get_analytical_los(topo, sat, t, frame="observer")

    np.testing.assert_almost_equal(ra_b, 292.3898647470882, decimal=5)
    np.testing.assert_almost_equal(dec_b, -3.497503446802312, decimal=5)
    np.testing.assert_almost_equal(ra_g, 292.3920488720271, decimal=5)
    np.testing.assert_almost_equal(dec_g, -3.498820404150905, decimal=5)
    np.testing.assert_almost_equal(ra_o, 292.3921248350655, decimal=5)
    np.testing.assert_almost_equal(dec_o, -3.4988184445922093, decimal=5)


def test_lambertian():

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 180)
    assert(mv > 39)

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 90)
    assert(mv == 16.205977775174308)

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 0)
    assert(mv == 14.96310309343897)


def test_eci_to_radec():

    res = eci_to_radec(6950.8045656, -13978.46949107, 19519.58868464)

    assert(res[0] == 296.4388967291422)
    assert(res[1] == 51.34809573105908)
    assert(res[2] == 24994.512114451572)

    res = radec_to_eci(296.4388967291422, 51.34809573105908, 24994.512114451572)

    np.testing.assert_almost_equal(res[0], 6950.804565599999, decimal=12)
    np.testing.assert_almost_equal(res[1], -13978.469491069998, decimal=11)
    np.testing.assert_almost_equal(res[2], 19519.58868464, decimal=12)
