"""Tests for `satsim.image.fpa` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import tensorflow as tf

from satsim.image.fpa import downsample, crop, analog_to_digital, mv_to_pe, pe_to_mv, add_counts, transform_and_add_counts, add_patch
from satsim.geometry.transform import apply_wrap_around
from satsim.util import configure_eager

configure_eager()


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
