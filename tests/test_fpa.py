"""Tests for `satsim.image.fpa` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.fpa import downsample, crop, analog_to_digital, mv_to_pe, pe_to_mv, add_counts, transform_and_add_counts, add_patch
from satsim.geometry.transform import apply_wrap_around
from satsim.util import configure_eager

configure_eager()


def _centroid(image):
    image = np.asarray(image, dtype=float)
    yy, xx = np.indices(image.shape)
    total = np.sum(image)
    return np.sum(yy * image) / total, np.sum(xx * image) / total


def test_downsample():

    osf = 5
    h = 400
    w = 800
    a = tf.ones([h,w])

    b = downsample(a, osf)

    hds = 80
    wds = 160
    na = osf * osf
    np.testing.assert_array_equal(b.numpy(), np.ones([hds,wds]) * na)

    c = downsample(a, osf, method='pool')
    np.testing.assert_array_equal(c.numpy(), np.ones([hds,wds]) * na)


def test_downsample_block_sum():

    a = tf.reshape(tf.range(16, dtype=tf.float32), [4, 4])

    b = downsample(a, 2, method='block_sum')

    np.testing.assert_array_equal(b.numpy(), np.array([
        [10.0, 18.0],
        [42.0, 50.0],
    ]))

    with pytest.raises(tf.errors.InvalidArgumentError):
        downsample(tf.ones([5, 4], dtype=tf.float32), 2, method='block_sum').numpy()


def test_crop():

    h = 200
    w = 400
    pr = 10
    pc = 20

    a = tf.ones([h, w])
    b = tf.pad(a, [[pr,pr],[pc,pc]])

    c = crop(b, pr, pc, h, w)

    np.testing.assert_array_equal(a.numpy(), c.numpy())


def test_analog_to_digital():

    bits = 14
    max_val = 200000
    gain = max_val / (2 ** bits - 1)

    h = 256
    w = 512
    s = [h,w]

    a = tf.ones(s) * 100000

    # check conversion
    b = analog_to_digital(a, gain, max_val)

    np.testing.assert_array_equal(b, np.floor(np.ones(s) * 100000 / gain))

    # check above max
    c = analog_to_digital(a + 5000000, gain, max_val)

    np.testing.assert_array_equal(c, np.ones(s) * (2 ** bits - 1))

    # check zeros
    d = analog_to_digital(tf.zeros(s), gain, max_val)

    np.testing.assert_array_equal(d, np.zeros(s))

    # check less than zero
    e = analog_to_digital(tf.zeros(s) - 10, gain, max_val)

    np.testing.assert_array_equal(e, np.zeros(s))

    # check bias
    f = analog_to_digital(tf.zeros(s), gain, max_val, 100)

    np.testing.assert_array_equal(f, np.floor(np.ones(s) * 100))


def test_mv_to_pe():

    a = 23.4697

    # check definition of zeropoint
    assert(mv_to_pe(a, a) == 1.0)

    # static check a few values
    b = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]

    c = mv_to_pe(a, b)

    d = [9.7247795e+08, 3.8715075e+08, 1.5412730e+08, 6.1359240e+07, 2.4427548e+07,
         9.7247800e+06, 3.8715078e+06, 1.5412746e+06, 6.1359231e+05, 2.4427547e+05,
         9.7247789e+04, 3.8715078e+04, 1.5412746e+04, 6.1359238e+03, 2.4427546e+03,
         9.7247845e+02, 3.8715054e+02, 1.5412747e+02, 6.1359238e+01]

    np.testing.assert_allclose(c, d, rtol=1e-06)

    e = pe_to_mv(a, c)

    np.testing.assert_allclose(e, b, rtol=1e-06)


def test_add_counts():

    a = tf.Variable(tf.zeros([500,500]))

    r = [5,10,20]
    c = [50,100,200]
    dn = [20.,20.,20.]

    d = add_counts(a, r, c, dn).numpy()

    # check energy conserved
    assert(np.sum(a.numpy().flatten()) == 0.0)
    assert(np.sum(d.flatten()) == np.sum(dn))

    # check correct value in pixels
    for i in range(len(dn)):
        assert(d[r[i],c[i]] == dn[i])

    a = tf.Variable(tf.zeros([500,500], dtype=tf.int32))

    d = add_counts(a, r, c, [20, 20, 20]).numpy()

    # check energy conserved
    assert(np.sum(a.numpy().flatten()) == 0.0)
    assert(np.sum(d.flatten()) == np.sum(dn))

    # check correct value in pixels
    for i in range(len(dn)):
        assert(d[r[i],c[i]] == dn[i])


def test_add_counts_floor_interpolation_preserves_legacy_behavior():

    a = tf.Variable(tf.zeros([8, 8]))

    d = add_counts(a, [2.75], [3.25], [100.0], interpolation='floor').numpy()

    assert(np.sum(d.flatten()) == 100.0)
    assert(d[2, 3] == 100.0)
    assert(np.count_nonzero(d) == 1)


def test_add_counts_bilinear_interpolation():

    a = tf.Variable(tf.zeros([8, 8]))

    d = add_counts(a, [1.25], [2.75], [100.0], interpolation='bilinear').numpy()

    np.testing.assert_allclose(d[1:3, 2:4], np.array([
        [18.75, 56.25],
        [6.25, 18.75],
    ]))
    assert(np.sum(d.flatten()) == 100.0)

    np.testing.assert_allclose(_centroid(d), [1.25, 2.75])


def test_add_counts_bilinear_interpolation_integer_fpa_uses_floor_behavior():

    a = tf.Variable(tf.zeros([8, 8], dtype=tf.int32))

    d = add_counts(a, [1.25], [2.75], [100], interpolation='bilinear').numpy()

    assert(np.sum(d.flatten()) == 100)
    assert(d[1, 2] == 100)
    assert(np.count_nonzero(d) == 1)


def test_add_counts_bilinear_interpolation_multiple_points_preserves_flux_and_centroid():

    a = tf.Variable(tf.zeros([16, 16]))
    r = np.array([2.25, 6.50, 10.75])
    c = np.array([3.75, 7.25, 12.50])
    dn = np.array([100.0, 250.0, 400.0])

    d = add_counts(a, r, c, dn, interpolation='bilinear').numpy()

    np.testing.assert_allclose(np.sum(d), np.sum(dn), atol=1e-5)
    np.testing.assert_allclose(
        _centroid(d),
        [np.average(r, weights=dn), np.average(c, weights=dn)],
        atol=1e-6,
    )


def test_add_counts_bilinear_interpolation_drops_out_of_bounds_weights():

    a = tf.Variable(tf.zeros([4, 4]))

    d = add_counts(a, [-0.25], [0.25], [100.0], interpolation='bilinear').numpy()

    np.testing.assert_allclose(d[0, 0], 56.25)
    np.testing.assert_allclose(d[0, 1], 18.75)
    np.testing.assert_allclose(np.sum(d.flatten()), 75.0)


def test_transform_and_add_counts():

    h = 8
    w = 6
    a = tf.Variable(tf.zeros([h,w]))

    r = [1,5]
    c = [1,5]
    d = [20., 20.]

    a1 = transform_and_add_counts(a, r, c, d, 0., 1., 1, 0., [1.,0.])
    a2 = transform_and_add_counts(a, r, c, d, 0., 1., 1, 0., [1.,0.], filter_out_of_bounds=False)

    # print(a.numpy())
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0. 20.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0. 20.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]]

    # check energy conserved
    assert(np.sum(a1.numpy().flatten()) == np.sum(d))
    assert(np.sum(a2.numpy().flatten()) == np.sum(d))

    # check correct values
    assert(a1.numpy()[1,1] == 20)
    assert(a1.numpy()[2,1] ==  0)
    assert(a1.numpy()[5,5] == 20)
    assert(a1.numpy()[6,5] ==  0)
    assert(a2.numpy()[1,1] == 20)
    assert(a2.numpy()[2,1] ==  0)
    assert(a2.numpy()[5,5] == 20)
    assert(a2.numpy()[6,5] ==  0)

    h = 8
    w = 6
    a = tf.Variable(tf.zeros([h,w]))

    r = [1.5,5]
    c = [1.5,5]
    d = [20., 20.]

    a = transform_and_add_counts(a, r, c, d, 0., 1., 4, 0., [2.,0.])

#     print(a.numpy())
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  5.  0.  0.  0.  0.]
    #  [ 0. 10.  0.  0.  0.  0.]
    #  [ 0.  5.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0. 10.]
    #  [ 0.  0.  0.  0.  0.  5.]
    #  [ 0.  0.  0.  0.  0.  5.]]

    # check energy conserved
    assert(np.sum(a.numpy().flatten()) == np.sum(d))

    # check correct values
    assert(a.numpy()[1,1] ==  5)
    assert(a.numpy()[2,1] == 10)
    assert(a.numpy()[3,1] ==  5)
    assert(a.numpy()[5,5] == 10)
    assert(a.numpy()[6,5] ==  5)
    assert(a.numpy()[7,5] ==  5)


def test_transform_and_add_counts_bilinear_static_centroid_and_flux():

    a = tf.Variable(tf.zeros([16, 16]))
    r = np.array([4.25, 10.75])
    c = np.array([5.50, 12.125])
    dn = np.array([100.0, 300.0])

    d = transform_and_add_counts(
        a,
        r,
        c,
        dn,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        interpolation='bilinear',
    ).numpy()

    np.testing.assert_allclose(np.sum(d), np.sum(dn), atol=1e-5)
    np.testing.assert_allclose(
        _centroid(d),
        [np.average(r, weights=dn), np.average(c, weights=dn)],
        atol=1e-6,
    )


def test_transform_and_add_counts_bilinear_motion_conserves_flux():

    a = tf.Variable(tf.zeros([20, 20]))

    d = transform_and_add_counts(
        a,
        [5.25],
        [6.75],
        [500.0],
        0.0,
        1.0,
        5,
        0.0,
        [2.0, 1.0],
        interpolation='bilinear',
    ).numpy()

    np.testing.assert_allclose(np.sum(d), 500.0, atol=1e-5)


def test_wrap_around():

    h = 8
    w = 6
    a = tf.Variable(tf.zeros([h,w]))
    bounds = [[0, 8], [0, 6], [0, 0]]

    r = [-2.5]
    c = [-0.5]
    d = [20.]

    r, c, new_bounds = apply_wrap_around(h, w, r, c, 0., 1., 0., [2.,0.], wrap_around=bounds)
    a = transform_and_add_counts(a, r, c, d, 0., 1., 4, 0., [2.,0.])

#      print(a.numpy())
    # [[ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  5.]
    #  [ 0.  0.  0.  0.  0. 10.]
    #  [ 0.  0.  0.  0.  0.  5.]]

    # check energy conserved
    assert(np.sum(a.numpy().flatten()) == np.sum(d))

    # check correct values
    assert(a.numpy()[5,5] ==  5)
    assert(a.numpy()[6,5] == 10)
    assert(a.numpy()[7,5] ==  5)

    a = tf.Variable(tf.zeros([h,w]))
    r = [8.5]
    c = [6.5]
    d = [20.]

    r, c, new_bounds = apply_wrap_around(h, w, r, c, 0., 1., 0., [2.,0.], wrap_around=bounds)
    a = transform_and_add_counts(a, r, c, d, 0., 1., 4, 0., [2.,0.])

#      print(a.numpy())
    # [[ 5.  0.  0.  0.  0.  0.]
    #  [10.  0.  0.  0.  0.  0.]
    #  [ 5.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]
    #  [ 0.  0.  0.  0.  0.  0.]]

    # check energy conserved
    assert(np.sum(a.numpy().flatten()) == np.sum(d))

    # check correct values
    assert(a.numpy()[0,0] ==  5)
    assert(a.numpy()[1,0] == 10)
    assert(a.numpy()[2,0] ==  5)


def test_add_patch():

    image = np.zeros((512,512))
    patch = np.ones((3,3)) / 9.0
    expected = np.zeros((512,512))
    expected[9:12,4:7] = 1.0 / 9.0 * 10.0
    expected[19:22,29:32] = 1.0 / 9.0 * 20.0

    output = add_patch(image, [10,20], [5, 30], [10, 20], patch, mode='nofft')

    np.testing.assert_equal(tf.reduce_sum(output).numpy(), 30)
    np.testing.assert_array_equal(output, expected)

    output2 = add_patch(image, [10,20], [5, 30], [10, 20], patch, mode='fft')

    np.testing.assert_almost_equal(tf.reduce_sum(output2).numpy(), 30, decimal=3)
    np.testing.assert_allclose(output2, expected, rtol=1e-4, atol=1e-3)
