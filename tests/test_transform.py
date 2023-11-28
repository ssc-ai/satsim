"""Tests for `satsim.geometry.transform` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import math

from satsim.geometry.transform import rotate_and_translate
from satsim.util import configure_eager

configure_eager()


def test_transform():

    # test t=0
    r,c = rotate_and_translate(499, 499, [0,1], [0,1], 0, 1., [1.,1.])

    np.testing.assert_array_equal(r, [0,1])
    np.testing.assert_array_equal(c, [0,1])

    # test translate 1,1
    r,c = rotate_and_translate(499, 499, [0,1], [0,1], 2.0, 0., [1.,1.])

    np.testing.assert_array_equal(r, [2,3])
    np.testing.assert_array_equal(c, [2,3])

    # test translate 1,0
    r,c = rotate_and_translate(499, 499, [0,1], [0,1], 2.0, 0., [1.,0.])

    np.testing.assert_array_equal(r, [2,3])
    np.testing.assert_array_equal(c, [0,1])

    # test translate 0,1
    r,c = rotate_and_translate(499, 499, [0,1], [0,1], 2.0, 0., [0.,1.])

    np.testing.assert_array_equal(r, [0,1])
    np.testing.assert_array_equal(c, [2,3])

    # test translate -1,-1
    r,c = rotate_and_translate(499, 499, [4,5], [4,5], 2.0, 0., [-1.,-1.])

    np.testing.assert_array_equal(r, [2,3])
    np.testing.assert_array_equal(c, [2,3])

    # test rotate (180 deg)
    r,c = rotate_and_translate(499, 499, [0,499,0], [0,499,499], 1.0, math.pi, [0.,0.])

    np.testing.assert_array_equal(r.numpy().astype(int), [499,0,498])
    np.testing.assert_array_equal(c.numpy().astype(int), [499,0,0])

    # test rotate (90 deg clockwise about center)
    r,c = rotate_and_translate(499, 499, [0], [249], 1.0, math.pi * .5, [0.,0.])

    np.testing.assert_array_equal(r.numpy().astype(int), [249])
    np.testing.assert_array_equal(c.numpy().astype(int), [499])

    # test rotate (90 deg counter-clockwise about center)
    r,c = rotate_and_translate(499, 499, [0], [249], 1.0, -math.pi * .5, [0.,0.])

    np.testing.assert_array_equal(r.numpy().astype(int), [250])
    np.testing.assert_array_equal(c.numpy().astype(int), [0])
