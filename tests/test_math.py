"""Tests for `satsim.math` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
from scipy import datasets
from scipy import signal
import tensorflow as tf
import pytest

from satsim.math import fftconv2, fftconv2p, fftshift, signal_to_noise_ratio, conv2, mean_degrees, diff_degrees, interp_degrees
from satsim.util import configure_eager
from satsim.math.interpolate import lagrange

configure_eager()


def test_fftconv2_conv2():

    a = np.array([[1, 2, 3],
                  [3, 4, 5]])
    b = np.array([[2, 3, 4],
                  [4, 5, 6]])
    c = np.array([[  7, 16, 17],
                  [30, 62, 58]])

    d = fftconv2(tf.convert_to_tensor(a), tf.convert_to_tensor(b)).numpy()
    np.testing.assert_array_almost_equal(c, d, decimal=4)

    e = fftconv2p(tf.convert_to_tensor(a), tf.convert_to_tensor(b), dtype=tf.complex128).numpy()
    np.testing.assert_array_almost_equal(c, e, decimal=4)

    f = conv2(tf.convert_to_tensor(a), tf.convert_to_tensor(b)).numpy()
    np.testing.assert_array_almost_equal(c, f, decimal=4)


def test_fftconv2_big():

    s = 700
    try:
        a = datasets.face(gray=True)
    except Exception:
        pytest.skip("Dataset not available")

    a = a[0:s,0:s]
    b = np.outer(signal.windows.gaussian(s, 8), signal.windows.gaussian(s, 8))
    c = signal.fftconvolve(a, b, mode='same')

    # warning, takes a long time (just a check to see if equal)
    # c2 = signal.convolve2d(a,b, mode='same')
    # np.testing.assert_array_almost_equal(c, c2, decimal=9)

    d = fftconv2(tf.convert_to_tensor(a), tf.convert_to_tensor(b), dtype=tf.complex128).numpy()
    np.testing.assert_array_almost_equal(c, d, decimal=7)

    e = fftconv2p(tf.convert_to_tensor(a), tf.convert_to_tensor(b), dtype=tf.float32).numpy()
    np.testing.assert_allclose(c, e, rtol=0.2, atol=5e-5)

    curr_tf_ver = tf.__version__
    tf.__version__ = '1.12.0'
    f = fftconv2p(tf.convert_to_tensor(a), tf.convert_to_tensor(b), dtype=tf.float32).numpy()
    np.testing.assert_allclose(c, f, rtol=0.001, atol=6.0)
    tf.__version__ = curr_tf_ver


def test_fft_exception():

    with pytest.raises(Exception) as e:
        fftshift([], 3)
        assert str(e.value) == '1 or 2 dimensional tensors supported.'


def test_fftshift():

    x = tf.convert_to_tensor([0, 1, 2, 3, 4, -4, -3, -2, -1])
    y = tf.convert_to_tensor([-4, -3, -2, -1, 0, 1, 2, 3, 4])

    np.testing.assert_array_equal(fftshift(x, dims=1).numpy(), np.fft.fftshift(x))
    np.testing.assert_array_equal(fftshift(y, dims=1).numpy(), np.fft.fftshift(y))


def test_snr():

    x = tf.convert_to_tensor([0.0, 100.0, 40.0, 0.0, 0.0, 0.0, 0.0])
    y = tf.convert_to_tensor([2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0])

    snr = signal_to_noise_ratio(x,y,2)

    np.testing.assert_array_almost_equal(snr.numpy(), [0, 9.712858, 5.897678, 0, 0, 0, 0])


def test_degree_wrap():

    a = diff_degrees(10, 10)
    assert(a == 0)

    a = diff_degrees(10, 100)
    assert(a == 90)

    a = diff_degrees(100, 10)
    assert(a == -90)

    a = diff_degrees(350, 10)
    assert(a == 20)

    a = diff_degrees(10, 350)
    assert(a == -20)

    a = mean_degrees(10, 10)
    assert(a == 10)

    a = mean_degrees(10, 100)
    assert(a == 55)

    a = mean_degrees(100, 10)
    assert(a == 55)

    a = mean_degrees(350, 10)
    assert(a == 0)

    a = mean_degrees(10, 350)
    assert(a == 0)

    a = mean_degrees(340, 10)
    assert(a == 355)

    a = mean_degrees(10, 340)
    assert(a == 355)

    a = mean_degrees(355, 10)
    assert(a == 2.5)

    a = mean_degrees(10, 355)
    assert(a == 2.5)

    a = interp_degrees(np.linspace(0, 10, 11), 0, 10, 350, 10)
    np.testing.assert_array_equal(a, [350., 352., 354., 356., 358., 0., 2., 4., 6., 8., 10.])

    a = interp_degrees(np.linspace(0, 10, 11), 0, 10, 10, 350)
    np.testing.assert_array_equal(a, [ 10., 8., 6., 4., 2., 0., 358., 356., 354., 352., 350.])

    a = interp_degrees(np.linspace(0, 10, 11), 0, 10, -80, 80, normalize_360=False)
    np.testing.assert_array_equal(a, [-80., -64., -48., -32., -16., 0., 16., 32., 48., 64., 80.])


def test_lagrange():

    y = [5,2,1,4,3]
    p = np.poly1d(y)

    ls = 10
    xs = np.linspace(0,10,ls)
    ys = p(xs)

    lh = 100
    xh = np.linspace(-5,15,lh)
    yh = p(xh)

    # test scalar array
    is0 = lagrange(xs, ys, xh, 5)
    err = np.sqrt(np.sum((is0 - yh) ** 2))
    assert(err < 5e-6)

    # test vector array
    is3 = lagrange(xs, np.reshape(np.repeat(ys, 3, axis=0), (ls,3)), xh, 5)
    for i in range(3):
        err = np.sqrt(np.sum((is3[:,i] - yh) ** 2))
        assert(err < 5e-6)


def test_lagrange_segment_boundaries():

    xx = [[0,1,2],[2,3,4],[4,5,6]]
    yy = [[1,2,3],[3,6,9],[9,14,19]]
    nxx = np.linspace(-2, 7, 19)

    myy = lagrange(xx, yy, nxx, 5)
    np.testing.assert_array_equal(myy, [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.5, 6.0, 7.5, 9.0, 11.5, 14.0, 16.5, 19.0, 21.5, 24.0])

    # test invalid
    invalid = lagrange(np.zeros((3,3,3)), yy, nxx, 5)
    assert(invalid is None)
