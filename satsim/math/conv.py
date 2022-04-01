from __future__ import division, print_function, absolute_import

import tensorflow as tf

from satsim.image import crop


def conv2(x, y, pad=32, dtype=tf.float32):
    """Convolve two 2-dimensional arrays with traditional convolution.

    Args:
        x: `Tensor`, input image as a 2D tensor.
        y: `Tensor`, input kernel as a 2D tensor.
        dtype: `dtype`, `float32`, `complex64` or `complex128`
    Returns:
        A `Tensor`, The 2D tensor containing the approximate discrete linear
        convolution of x with y.
    """
    x = tf.cast(x,dtype=dtype)
    y = tf.cast(y,dtype=dtype)
    (w, h) = (tf.shape(x)[0], tf.shape(x)[1])

    x = tf.pad(x, [[pad,pad],[pad,pad]])
    x = tf.reshape(x, [1, tf.shape(x)[0], tf.shape(x)[1], 1])

    y = tf.reshape(y, [tf.shape(y)[0], tf.shape(y)[1], 1, 1])

    out = tf.squeeze(tf.nn.conv2d(x, y[::-1, ::-1, :, :], [1,1,1,1], 'SAME'))
    out = crop(out, pad - 1, pad, w, h)

    return out
