"""Tests for `satsim.geometry.transform` package."""
import numpy as np

from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.astrometric import distance_between, load_sun, create_topocentric, angle_between
from satsim.geometry.photometric import lambertian_sphere_to_mv, model_to_mv
from satsim import time


def test_lambertian_sphere_to_mv():

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 180)
    assert(mv > 39)

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 90)
    assert(mv == 16.205977775174308)

    mv = lambertian_sphere_to_mv(0.2, 400000000, 5, 0)
    assert(mv == 14.96310309343897)


def test_model_to_mv():

    sun = load_sun()
    t = time.utc(2021, 3, 18, 0, 0, [0, 1, 2, 3, 4])

    observer = create_topocentric("20.746111 N", "156.431667 W")
    satellite = create_sgp4('1 41788U 16059F   21260.56440839  .00022290  00000-0  23487-2 0  9990', '2 41788  98.2495   9.3816 0022044 246.7613 113.2488 14.89172957267561')

    targ_solar_phase_ang = angle_between(satellite, observer, sun, t)

    np.testing.assert_almost_equal(targ_solar_phase_ang, [13.97588734, 14.00264072, 14.02943033, 14.056256, 14.08311756])

    distance = distance_between(satellite, observer, t)
    np.testing.assert_almost_equal(distance, [11207.43860525, 11203.56047005, 11199.67919572, 11195.79478366, 11191.90723549])

    model = {
        'mode': 'lambertian_sphere',
        'albedo': 0.2,
        'diameter': 2
    }
    mv = model_to_mv(observer, satellite, model, t)

    expected = [10.72609951, 10.72546428, 10.72482855, 10.72419232, 10.72355559]

    np.testing.assert_almost_equal(mv, expected)

    model = {
        'mode': 'lambertian_sphere',
        'albedo': 0.2,
        'diameter': 2,
        'distance': distance,
        'phase_angle': targ_solar_phase_ang
    }

    mv = model_to_mv(None, None, model, None)
    np.testing.assert_almost_equal(mv, expected)
