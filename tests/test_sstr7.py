"""Tests for `satsim.geometry.sstr7` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.sstr7 import get_star_mv, query_by_los


def test_star_mv():

    star = np.full((18,), 32.0)

    mv = get_star_mv(star)
    assert(mv == 32.0)

    star[3] = 1.0
    mv = get_star_mv(star)
    assert(mv == 1.0)

    star[4] = 2.0
    mv = get_star_mv(star)
    assert(mv == 2.0)

    star[8] = 3.0
    mv = get_star_mv(star)
    assert(mv == 3.0)

    star[5] = 4.0
    mv = get_star_mv(star)
    assert(mv == 4.0)

    star[0] = 4.0
    mv = get_star_mv(star)
    assert(mv == 4.0)


def test_query_by_los():

    height = 512
    width = 256

    y_fov = 0.2
    x_fov = 0.1

    # vega coordinates
    ra = 279.2358441666667
    dec = 38.78492111111112

    def test_it(expected_num_stars, origin='corner', filter_ob=False, flipud=False, fliplr=False):

        yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin=origin, filter_ob=filter_ob, flipud=flipud, fliplr=fliplr)

        assert(len(yy) == expected_num_stars)
        assert(len(xx) == expected_num_stars)
        assert(len(mm) == expected_num_stars)

        vega = np.where(mm == 0.07)[0]

        assert(len(vega) == 1)

        if origin == 'center':
            np.testing.assert_almost_equal(yy[vega], height / 2.0)
            np.testing.assert_almost_equal(xx[vega], width / 2.0)
            np.testing.assert_almost_equal(mm[vega], 0.07)
        else:
            if flipud:
                np.testing.assert_almost_equal(yy[vega], height)
            else:
                np.testing.assert_almost_equal(yy[vega], 0)

            if fliplr:
                np.testing.assert_almost_equal(xx[vega], width)
            else:
                np.testing.assert_almost_equal(xx[vega], 0)
            np.testing.assert_almost_equal(mm[vega], 0.07)

    rot = 0.0
    test_it(4532)
    test_it(372, 'corner', True)
    test_it(372, 'corner', True, True, True)
    test_it(4426, 'center')
    test_it(285, 'center', True)
    test_it(285, 'center', True, True, True)

    rot = 90.0
    test_it(2852)
    test_it(2980, 'center')

    rot = -45.0
    test_it(4730)
    test_it(4532, 'center')

    # north pole
    ra = 268
    dec = 89.9
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 242)

    # south pole
    ra = 359.9
    dec = -89.99
    rot = 190
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 976)

    # meridian crossing + inside
    ra = 359.9
    dec = -85
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 2434)

    # meridian crossing + minRA
    ra = 1.0
    dec = 0
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, 10, 10, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 83612)
