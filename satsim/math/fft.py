from __future__ import division, print_function, absolute_import

from functools import lru_cache

import tensorflow as tf

from satsim.util import get_semantic_version


def fftshift(x, dims=2):
    """Shift the zero-frequency component to the center of the spectrum.
    This function swaps half-spaces for all axes listed.

    Examples::

        x = fftshift([ 0.,  1.,  2.,  3.,  4., -5., -4., -3., -2., -1.])
        x.numpy() # array([-5., -4., -3., -2., -1.,  0.,  1.,  2.,  3.,  4.])

    Args:
        x: `Tensor`, input tensor.
        dims: `int`, number of dimensions in x. (1 or 2)
    Returns:
        A `Tensor`, the shifted tensor.
    """
    if dims == 2:
        axes = [0,1]
        shift = [tf.shape(x)[0] // 2, tf.shape(x)[1] // 2]
    elif dims == 1:
        axes = [0]
        shift = [tf.shape(x)[0] // 2]
    else:
        raise Exception('1 or 2 dimensional tensors supported.')

    return tf.roll(x, shift, axes)


def fftconv2(x, y, dtype=tf.complex64):
    """Convolve two 2-dimensional arrays using FFT. Convolve `x` and `y` using
    the fast Fourier transform method. NOTE: `x` and `y` must have the same
    shape.

    Args:
        x: `Tensor`, 2D input tensor.
        y: `Tensor`, 2D input tensor of the same shape as `x`.
        dtype: `tf.dtype`, `tf.complex64` or `tf.complex128`
    Returns:
        A `Tensor`, The 2D tensor containing the discrete linear convolution of
        x with y.
    """
    # TF 1.12 does not have rfft, need to cast to complex
    x_complex = tf.cast(x,dtype=dtype)

    sx = x_complex.get_shape()[0]
    sy = x_complex.get_shape()[1]

    # For the circular convolution of x and y to be equivalent, you must pad the
    # vectors with zeros to length at least N + L - 1 before you take the DFT.
    sxp = sx - 1
    syp = sy - 1

    x_complex = tf.pad(x_complex, [[0,sxp],[0,syp]])
    x_complex = tf.signal.fft2d(x_complex)

    y_complex = tf.cast(y,dtype=dtype)
    y_complex = tf.pad(y_complex, [[0,sxp],[0,syp]])
    y_complex = tf.signal.fft2d(y_complex)

    # the convolution theorem states that under suitable conditions the Fourier
    # transform of a convolution of two signals is the pointwise product of
    # their Fourier transforms
    fftfull = tf.math.real(
        tf.signal.ifft2d(
            tf.multiply(x_complex,y_complex)))

    # crop the array and return 'same' sized array
    start = [sxp // 2, syp // 2]
    return tf.slice(fftfull, start, [sx, sy])


@lru_cache(maxsize=1)
def _cached_fftp(x, pad, dtype=tf.float32):

    (tf_fft_func, tf_ifft_func, tf_fft_dtype) = _get_tf_rfft(dtype)

    x_complex = tf.cast(x.deref(),dtype=tf_fft_dtype)
    x_complex = tf.pad(x_complex, [[pad,pad],[pad,pad]])
    x_complex = tf_fft_func(x_complex)

    return x_complex


def fftconv2p(x, y, pad=32, dtype=tf.float32, cache_last_y=True):
    """Convolve two 2-dimensional arrays using FFT. Convolve `x` and `y` using
    the fast Fourier transform method. The result is an approximate for the
    convolution. If more numeric precision is required, use `fftconv2` which
    pads the array to N + L - 1. NOTE: `x` and `y` must have the same shape.
    NOTE: x and y dimensions should be a multiple of 2.

    Args:
        x: `Tensor`, 2D input tensor.
        y: `Tensor`, 2D input tensor of the same shape as `x`.
        pad: `int`, number of pixels to pad all sides before calculating the FFT
        dtype: `dtype`, `float32`, `complex64` or `complex128`
        cache_last_y: `bool`, if True, cache the last `y` FFT which improves
            performance if `y` is static and called multiple times. Note `y`
            needs to be a tf.Tensor type if set to True.
    Returns:
        A `Tensor`, The 2D tensor containing the approximate discrete linear
        convolution of x with y.
    """
    (tf_fft_func, tf_ifft_func, tf_fft_dtype) = _get_tf_rfft(dtype)

    sx = tf.shape(x)[0]
    sy = tf.shape(x)[1]

    x_complex = tf.cast(x,dtype=tf_fft_dtype)
    x_complex = tf.pad(x_complex, [[pad,pad],[pad,pad]])
    x_complex = tf_fft_func(x_complex)

    if cache_last_y and tf.is_tensor(y):
        y_complex = _cached_fftp(y.ref(), pad, dtype)
    else:
        y_complex = tf.cast(y,dtype=tf_fft_dtype)
        y_complex = tf.pad(y_complex, [[pad,pad],[pad,pad]])
        y_complex = tf_fft_func(y_complex)

    # the convolution theorem states that under suitable conditions the Fourier
    # transform of a convolution of two signals is the pointwise product of
    # their Fourier transforms
    fftfull = tf.math.real(
        tf_ifft_func(
            tf.multiply(x_complex,y_complex)))

    # shift and crop the array and return 'same' sized array
    return tf.slice(fftshift(fftfull), [pad - 1, pad - 1], [sx, sy])


def _get_tf_rfft(dtype):
    """Returns the real fft function if supported else the complex version is
    returned.
    """
    tf_ver = get_semantic_version(tf)
    if dtype == tf.complex64 or dtype == tf.complex128:  # complex fft requested
        tf_fft_func = tf.signal.fft2d
        tf_ifft_func = tf.signal.ifft2d
        tf_fft_dtype = dtype
    elif tf_ver[0] == 1 and tf_ver[1] < 13:  # real fft requested but not supported
        tf_fft_func = tf.signal.fft2d
        tf_ifft_func = tf.signal.ifft2d
        tf_fft_dtype = tf.complex64 if dtype == tf.float32 else tf.complex128
    else:  # real fft
        tf_fft_func = tf.signal.rfft2d
        tf_ifft_func = tf.signal.irfft2d
        tf_fft_dtype = tf.float32  # TF rfft only supports float32

    return (tf_fft_func, tf_ifft_func, tf_fft_dtype)
