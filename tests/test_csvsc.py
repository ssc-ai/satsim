"""Tests for `satsim.geometry.sstr7` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.csvsc import query_by_los


def test_query_by_los():

    height = 512
    width = 256

    y_fov = 5
    x_fov = 5

    # vega coordinates
    ra = 279.23410832
    dec = 38.78299311
    mv = 0.03

    def test_it(expected_num_stars, origin='corner', filter_ob=False, flipud=False, fliplr=False):

        yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, rootPath='tests/hip_main.txt', origin=origin, flipud=flipud, fliplr=fliplr)

        assert(len(yy) == expected_num_stars)
        assert(len(xx) == expected_num_stars)
        assert(len(mm) == expected_num_stars)

        vega = np.where(mm == mv)[0]

        assert(len(vega) == 1)

        if origin == 'center':
            np.testing.assert_almost_equal(yy[vega], height / 2.0)
            np.testing.assert_almost_equal(xx[vega], width / 2.0)
            np.testing.assert_almost_equal(mm[vega], mv)
        else:
            if flipud:
                np.testing.assert_almost_equal(yy[vega], height)
            else:
                np.testing.assert_almost_equal(yy[vega], 0)

            if fliplr:
                np.testing.assert_almost_equal(xx[vega], width)
            else:
                np.testing.assert_almost_equal(xx[vega], 0)
            np.testing.assert_almost_equal(mm[vega], mv)

    rot = 0.0
    test_it(3)
    test_it(3, 'corner', True)
    test_it(3, 'corner', True, True, True)
    test_it(3, 'center')
    test_it(3, 'center', True)
    test_it(3, 'center', True, True, True)

    rot = 90.0
    test_it(3)
    test_it(3, 'center')

    rot = -45.0
    test_it(3)
    test_it(3, 'center')
