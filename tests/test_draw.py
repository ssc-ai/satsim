"""Tests for `satsim.geometry.draw` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.draw import gen_line, gen_line_from_endpoints, gen_curve_from_points


def test_gen_line():

    r,c,p,t = gen_line(500, 500, [.5,.5], [0.,0.], 100, 0, 0.5)

    assert(r[0] == 250)
    assert(c[0] == 250)
    assert(p[0] == 50.0)

    # top left corner of pixel [0,0]
    r,c,p,t = gen_line(500, 500, [0.,0.], [4.,0.], 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,0,0,0,0])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # middle of pixel [0,0]
    r,c,p,t = gen_line(500, 500, [0.001,0.001], [4.,4.], 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,1,2,3,4])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # bottom right corner of pixel [0,0]
    r,c,p,t = gen_line(500, 500, [0.00199,0.00199], [4.,4.], 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,1,2,3,4])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # top left corner of pixel [1,1]
    r,c,p,t = gen_line(500, 500, [0.002,0.002], [2.,2.], 100, 0, 2.0)

    np.testing.assert_array_equal(r, [1,2,3,4,5])
    np.testing.assert_array_equal(c, [1,2,3,4,5])
    np.testing.assert_array_equal(p, [40,40,40,40,40])

    # middle of bottom right corner pixel [499,499]
    r,c,p,t = gen_line(500, 500, [1. - 1. / 500.,1. - 1. / 500.], [-4.,-4.], 100, 0, 1.0)

    np.testing.assert_array_equal(r, [499,498,497,496,495])
    np.testing.assert_array_equal(c, [499,498,497,496,495])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # check row and col and non-square
    r,c,p,t = gen_line(500, 1000, [0.004,0.001], [4.,2.], 90, 0, 2.0)

    np.testing.assert_array_equal(r, [2,3,4,5,6,7,8,9,10])
    np.testing.assert_array_equal(c, [1,2,2,3,3,4,4,5,5])
    np.testing.assert_array_equal(p, [20,20,20,20,20,20,20,20,20])


def test_gen_line_from_endpoints():

    r,c,p,t = gen_line_from_endpoints(250, 250, 250, 250, 100, 0, 0.5)

    assert(r[0] == 250)
    assert(c[0] == 250)
    assert(p[0] == 50.0)

    # top left corner of pixel [0,0]
    r,c,p,t = gen_line_from_endpoints(0., 0., 4., 0., 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,0,0,0,0])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # middle of pixel [0,0]
    r,c,p,t = gen_line_from_endpoints(0.5, 0.5, 4., 4., 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,1,2,3,4])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # bottom right corner of pixel [0,0]
    r,c,p,t = gen_line_from_endpoints(0.99, 0.99, 4., 4., 100, 0, 1.0)

    np.testing.assert_array_equal(r, [0,1,2,3,4])
    np.testing.assert_array_equal(c, [0,1,2,3,4])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # top left corner of pixel [1,1]
    r,c,p,t = gen_line_from_endpoints(1., 1., 5., 5., 100, 0, 2.0)

    np.testing.assert_array_equal(r, [1,2,3,4,5])
    np.testing.assert_array_equal(c, [1,2,3,4,5])
    np.testing.assert_array_equal(p, [40,40,40,40,40])

    # middle of bottom right corner pixel [499,499]
    r,c,p,t = gen_line_from_endpoints(499, 499, 495., 495, 100, 0, 1.0)

    np.testing.assert_array_equal(r, [499,498,497,496,495])
    np.testing.assert_array_equal(c, [499,498,497,496,495])
    np.testing.assert_array_equal(p, [20,20,20,20,20])

    # check row and col and non-square
    r,c,p,t = gen_line_from_endpoints(2, 1, 10, 5, 90, 0, 2.0)

    np.testing.assert_array_equal(r, [2,3,4,5,6,7,8,9,10])
    np.testing.assert_array_equal(c, [1,2,2,3,3,4,4,5,5])
    np.testing.assert_array_equal(p, [20,20,20,20,20,20,20,20,20])


def test_gen_curve_from_points():

    r,c,p,t = gen_curve_from_points(250, 250, 250, 250, 250, 250, 100, 0, 0.5)
    assert(r[0] == 250)
    assert(c[0] == 250)
    assert(p[0] == 50.0)

    r,c,p,t = gen_curve_from_points(250, 250, 251, 251, 252, 252, 90, 0, 0.5)
    np.testing.assert_array_equal(r, [250, 251, 252])
    np.testing.assert_array_equal(c, [250, 251, 252])
    np.testing.assert_array_equal(p, [15, 15, 15])

    r,c,p,t = gen_curve_from_points(0, 0, 2, 3, 0, 6, 80, 0, 0.5)
    np.testing.assert_array_equal(r, [0, 1, 2, 2, 2, 2, 1, 0])
    np.testing.assert_array_equal(c, [0, 1, 2, 3, 3, 4, 5, 6])
    np.testing.assert_array_equal(p, [5] * 8)

    r,c,p,t = gen_curve_from_points(0, 0, 2, 3, 6, 6, 80, 0, 0.5)
    np.testing.assert_array_equal(r, [0, 0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(c, [0, 1, 2, 3, 4, 5, 5, 6])
    np.testing.assert_array_equal(p, [5] * 8)

    r,c,p,t = gen_curve_from_points(250, 250, 255, 255, 250, 260, 90, 0, 0.5)

    np.testing.assert_array_equal(r, [250, 251, 252, 253, 254, 255, 255, 255, 255, 254, 253, 252, 251, 250])
    np.testing.assert_array_equal(c, [250, 251, 251, 252, 253, 254, 255, 255, 256, 257, 258, 259, 260, 260])
    np.testing.assert_array_equal(p, [0.5 * 90 / len(r)] * len(r))

    # for ii in range(100):
    #     r6 = np.random.randint(0, 255, 6)
    #     print(r6)
    #     r,c,p,t = gen_curve_from_points(r6[0], r6[1], r6[2], r6[3], r6[4], r6[5], 90, 0, 0.5)

    r6 = [ 94, 6, 128, 187, 44, 182]
    r,c,p,t = gen_curve_from_points(r6[0], r6[1], r6[2], r6[3], r6[4], r6[5], 90, 0, 0.5)

    assert(r[0] == 94)
    assert(c[0] == 6)
    assert(r[-1] == 44)
    assert(c[-1] == 182)

    r6 = [31, 240, 54, 155, 41, 27]
    r,c,p,t = gen_curve_from_points(r6[0], r6[1], r6[2], r6[3], r6[4], r6[5], 90, 0, 0.5)
