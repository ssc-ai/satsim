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
        dtype: `tf.dtype`, `tf.complex64` or `tf.complex128`. Legacy CPU
            FFT arithmetic uses complex128 internally, preserving the
            requested real output dtype.
    Returns:
        A `Tensor`, The 2D tensor containing the discrete linear convolution of
        x with y.
    """
    # Preserve requested input/output precision while doing legacy CPU FFT
    # arithmetic in double precision. Resolve placement after padding so host
    # inputs used under a GPU scope still take the ordinary GPU path.
    x_complex = tf.cast(x,dtype=dtype)
    input_dtype = x_complex.dtype
    output_dtype = input_dtype.real_dtype

    sx = x_complex.get_shape()[0]
    sy = x_complex.get_shape()[1]

    # For the circular convolution of x and y to be equivalent, you must pad the
    # vectors with zeros to length at least N + L - 1 before you take the DFT.
    sxp = sx - 1
    syp = sy - 1

    x_complex = tf.pad(x_complex, [[0,sxp],[0,syp]])

    def convolve(compute_dtype):
        spectrum = tf.signal.fft2d(tf.cast(x_complex, compute_dtype))
        y_complex = tf.cast(tf.cast(y, input_dtype), compute_dtype)
        y_complex = tf.pad(y_complex, [[0,sxp],[0,syp]])
        y_complex = tf.signal.fft2d(y_complex)
        fftfull = tf.math.real(tf.signal.ifft2d(tf.multiply(spectrum,y_complex)))
        return tf.cast(tf.slice(fftfull, [sxp // 2, syp // 2], [sx, sy]), output_dtype)

    return _with_fft_precision(x_complex, input_dtype, convolve)


@lru_cache(maxsize=1)
def _cached_fftp(x, pad, dtype=tf.float32, compute_dtype=None, device=None, graph=None):
    """Cache one kernel spectrum for its input/FFT precision, device and graph.

    ``graph`` is a cache key: an eager kernel captured by two traced functions
    must not share a symbolic spectrum belonging to only the first graph.
    """
    input_dtype = _get_tf_rfft(dtype)[2]
    compute_dtype = input_dtype if compute_dtype is None else compute_dtype
    tf_fft_func = _get_tf_rfft(compute_dtype)[0]
    with tf.device(device):
        x_complex = tf.cast(tf.cast(x.deref(), input_dtype), compute_dtype)
        x_complex = tf.pad(x_complex, [[pad,pad],[pad,pad]])
        return tf_fft_func(x_complex)


def fftconv2p(x, y, pad=32, dtype=tf.float32, cache_last_y=True):
    """Convolve two 2-dimensional arrays using FFT. Convolve `x` and `y` using
    the fast Fourier transform method. The result is an approximate for the
    convolution. If more numeric precision is required, use `fftconv2` which
    pads the array to N + L - 1. NOTE: `x` and `y` must have the same shape.

    Args:
        x: `Tensor`, 2D input tensor.
        y: `Tensor`, 2D input tensor of the same shape as `x`.
        pad: `int`, number of pixels to pad all sides before calculating the FFT
        dtype: `dtype`, `float32`, `float64`, `complex64` or `complex128`.
            Determines input/output precision. Before TensorFlow 2.15, CPU
            FFT arithmetic uses double precision internally to avoid Eigen
            float32 error; the returned real dtype is unchanged.
        cache_last_y: `bool`, if True, cache the last `y` FFT which improves
            performance if `y` is static and called multiple times. Note `y`
            needs to be a tf.Tensor type if set to True.
    Returns:
        A `Tensor`, The 2D tensor containing the approximate discrete linear
        convolution of x with y.
    """
    tf_fft_dtype = _get_tf_rfft(dtype)[2]

    sx = tf.shape(x)[0]
    sy = tf.shape(x)[1]

    x_complex = tf.cast(x,dtype=tf_fft_dtype)
    output_dtype = tf_fft_dtype.real_dtype
    x_complex = tf.pad(x_complex, [[pad,pad],[pad,pad]])
    device = x_complex.device or None

    def convolve(compute_dtype):
        tf_fft_func, tf_ifft_func, _ = _get_tf_rfft(compute_dtype)
        spectrum = tf_fft_func(tf.cast(x_complex, compute_dtype))

        if cache_last_y and tf.is_tensor(y):
            graph = None if tf.executing_eagerly() else tf.compat.v1.get_default_graph()
            y_complex = _cached_fftp(y.ref(), pad, dtype, compute_dtype, device, graph)
        else:
            y_complex = tf.cast(tf.cast(y, tf_fft_dtype), compute_dtype)
            y_complex = tf.pad(y_complex, [[pad,pad],[pad,pad]])
            y_complex = tf_fft_func(y_complex)

        product = tf.multiply(spectrum,y_complex)
        if tf_ifft_func is tf.signal.irfft2d:
            # Preserve the original grid length for both odd and even widths.
            fftfull = tf_ifft_func(product, fft_length=[sx + 2 * pad, sy + 2 * pad])
        else:
            fftfull = tf.math.real(tf_ifft_func(product))

        return tf.cast(tf.slice(fftshift(fftfull), [pad - 1, pad - 1], [sx, sy]), output_dtype)

    return _with_fft_precision(x_complex, tf_fft_dtype, convolve)


def _with_fft_precision(padded, dtype, convolve):
    """Run an FFT calculation at the appropriate precision on its placed device.

    Eager placement is already known after padding. A traced function can be
    reused on CPU and GPU without retracing, so legacy graphs select precision
    at execution time instead of baking in the tracing host's device choice.
    """
    if dtype not in (tf.float32, tf.complex64) or get_semantic_version(tf) >= [2, 15]:
        return convolve(dtype)
    if tf.executing_eagerly():
        with tf.device(padded.device):
            return convolve(_fft_compute_dtype(dtype, padded.device))
    precise = tf.float64 if dtype == tf.float32 else tf.complex128
    with tf.compat.v1.colocate_with(padded):
        on_cpu = tf.equal(tf.raw_ops.DeviceIndex(device_names=['CPU']), 0)
        return tf.cond(on_cpu, lambda: convolve(precise), lambda: convolve(dtype))


def _fft_compute_dtype(dtype, device):
    """Promote single-precision FFT arithmetic only on the legacy CPU backend.

    Only eager callers use this policy; graph placement is handled at runtime
    by ``_with_fft_precision``. No device initialization or visibility changes
    are needed to inspect an already placed eager tensor.
    """
    if dtype not in (tf.float32, tf.complex64) or get_semantic_version(tf) >= [2, 15]:
        return dtype
    device_type = tf.DeviceSpec.from_string(device or '').device_type
    if device_type == 'CPU':
        return tf.float64 if dtype == tf.float32 else tf.complex128
    return dtype


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
        tf_fft_dtype = tf.float64 if dtype == tf.float64 else tf.float32

    return (tf_fft_func, tf_ifft_func, tf_fft_dtype)
