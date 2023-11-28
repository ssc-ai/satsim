from __future__ import division, print_function, absolute_import

import math

import numpy as np
import tensorflow as tf

from satsim.geometry.sprite import load_sprite_from_file
from satsim.math import fftconv2p
from satsim.image.fpa import add_counts
import satsim.tfa as tfa


def scatter_shift_random(image, t, loc, scale, length, spacial_osf=1, normalize=True, mode='fft', interpolation='nearest', dtype=tf.float32):
    """ Shifts the images specified in random directions. If length is
    greater than 1, each shift is combined into a single image by summing.

    Args:
        image: `array`, image to shift.
        t: `time`, simulation time. unused but required by interface.
        loc: `float`, mean of the normal distribution.
        scale: `float`, standard deviation of the normal distribution.
        spacial_osf: `int`, oversample factor of input array. x and y are multiplied by it
        normalize: `boolean`, normalize the output image so its sum is equal to
            the original input image.
        mode: `string`, specify algorithm: fft, roll, shift. default: fft
        interpolation: `string`, specify shift interpolation mode: bilinear, nearest. default: nearest

    Returns:
        An `array`, a two dimensional image.
    """
    if length <= 0:
        return tf.cast(image, dtype=dtype)

    mag = np.random.normal(loc, scale, size=length)
    angle = np.random.uniform(0, 360, size=length)

    return scatter_shift_polar(image, t, mag, angle, 1.0, spacial_osf=spacial_osf, normalize=normalize, mode=mode, interpolation=interpolation, dtype=dtype)


def scatter_shift_polar(image, t, mag, angle, weights, spacial_osf=1, normalize=True, mode='fft', interpolation='nearest', dtype=tf.float32):
    """ Shifts the images specified by polar coordinates. If multiple shifts are
    specified, each shift is combined into a single image by summing.

    Args:
        image: `array`, image to shift.
        t: `time`, simulation time. unused but required by interface.
        mag: `float`, the amount to shift the image in pixels.
        angle: `float`, the direction to shift the image in degrees.
        weights: `float`, a scale to apply each shift by.
        spacial_osf: `int`, oversample factor of input array. x and y are multiplied by it
        normalize: `boolean`, normalize the output image so its sum is equal to
            the original input image.
        mode: `string`, specify algorithm: fft, roll, shift. default: fft
        interpolation: `string`, specify shift interpolation mode: bilinear, nearest. default: nearest

    Returns:
        An `array`, a two dimensional image.
    """
    image = tf.cast(image, dtype=dtype)
    mag = tf.convert_to_tensor(mag, dtype=dtype)
    angle = tf.convert_to_tensor(angle, dtype=dtype)

    angle = angle * math.pi / 180.0
    x = mag * tf.math.sin(angle)
    y = mag * tf.math.cos(angle)

    return scatter_shift(image, t, y, x, weights, spacial_osf, normalize, mode=mode, interpolation=interpolation, dtype=dtype)


def scatter_shift(image, t, y, x, weights, spacial_osf=1, normalize=True, mode='fft', interpolation='nearest', dtype=tf.float32):
    """ Shifts the images specified by cartesian coordinates. If multiple shifts
    are specified, each shift is combined into a single image by summing

    Args:
        image: `array`, image to shift.
        t: `time`, simulation time. unused but required by interface.
        y: `float`, the amount to shift the image up/down in pixels.
        x: `float`, the amount to shift the image left/right in pixels.
        weights: `float`, a scale to apply each shift by.
        spacial_osf: `int`, oversample factor of input array. x and y are multiplied by it
        normalize: `boolean`, normalize the output image so its sum is equal to
            the original input image.
        mode: `string`, specify algorithm: fft, roll, shift. default: fft
        interpolation: `string`, specify shift interpolation mode: bilinear, nearest. default: nearest

    Returns:
        An `array`, a two dimensional image.
    """
    orig_image = tf.convert_to_tensor(image, dtype=dtype)
    image = tf.zeros_like(image, dtype=dtype)
    y = tf.convert_to_tensor(y * spacial_osf, dtype=dtype)
    x = tf.convert_to_tensor(x * spacial_osf, dtype=dtype)
    weights = tf.broadcast_to(tf.convert_to_tensor(weights, dtype=dtype), y.shape)

    i = tf.constant(0)

    def c(i, img, oimg, y, x, w):
        return tf.less(i, len(y))

    def b(i, img, oimg, y, x, w):
        return [i + 1, img + tfa.image.translate(oimg, [x[i], y[i]], interpolation) * w[i], oimg, y, x, w]

    def b_roll(i, img, oimg, y, x, w):
        return [i + 1, img + tf.roll(oimg, [y[i], x[i]], axis=[0, 1]) * w[i], oimg, y, x, w]

    if mode == 'roll':
        y = tf.cast(y, tf.int32)
        x = tf.cast(x, tf.int32)
        i, image, _, _, _, _ = tf.while_loop(c, b_roll, (i, image, orig_image, y, x, weights))
    elif mode == 'fft':
        rr = tf.cast(y + image.shape[0] / 2, tf.int32)
        cc = tf.cast(x + image.shape[1] / 2, tf.int32)
        delta = add_counts(tf.zeros_like(image, tf.float32), rr - 1, cc - 1, weights)
        image = fftconv2p(delta, orig_image, pad=1)
    else:
        i, image, _, _, _, _ = tf.while_loop(c, b, (i, image, orig_image, y, x, weights))

    if normalize:
        image = image * tf.math.reduce_sum(orig_image) / tf.math.reduce_sum(image)

    return image


def crop_and_resize(image, t, y_start, x_start, y_box_size, x_box_size):
    """ Crops an image and resizes it to the original image size.

    Args:
        image: `array`, image to shift.
        t: `time`, simulation time. unused but required by interface.
        y_start: `float`, starting y box location in normalized coordinates.
        x_start: `float`, starting x box location in normalized coordinates.
        y_box_size: `float`, y box size in normalized coordinates.
        x_box_size: `float`, x box size in normalized coordinates.

    Returns:
        An `array`, a two dimensional image.
    """
    c = tf.shape(image)

    h = c[0]
    w = c[1]
    y1 = y_start
    x1 = x_start
    y2 = y1 + y_box_size
    x2 = x1 + x_box_size

    if y2 > 1.0:
        y2 = 1.0

    if x2 > 1.0:
        x2 = 1.0

    return tf.squeeze(
        tf.image.crop_and_resize(
            tf.reshape(image, [1, h, w, 1]), [[y1, x1, y2, x2]], [0], [h, w], method='bilinear'))


def flip(image, t, up_down=False, left_right=False):
    """ Flips an image about the y and/or x axis.

    Args:
        image: `array`, image to shift.
        t: `time`, simulation time. unused but required by interface.
        up_down: `boolean`, flip image about the x axis.
        left_right: `boolean`, flip image about the y axis.

    Returns:
        An `array`, a two dimensional image.
    """
    if not up_down and not left_right:
        return image

    c = tf.shape(image)

    h = c[0]
    w = c[1]
    image = tf.reshape(image, [1, h, w, 1])

    if up_down:
        image = tf.image.flip_up_down(image)

    if left_right:
        image = tf.image.flip_left_right(image)

    return tf.squeeze(image)


def null(image, t):
    """ Null function.

    Args:
        image: `array`, image.
        t: `time`, simulation time.

    Returns:
        An `array`, same as input `image`.
    """
    return image


def resize(image, t, height, width, spacial_osf=1, normalize=True, dtype=tf.float32):
    """ Resize the image.

    Args:
        image: `array`, image.
        t: `time`, simulation time in seconds from epoch.
        height: `int`, height of the new image
        width: `int`, width of the new image
        spacial_osf `int`, multiply height and width by this number

    Returns:
        An `array`, same as input `image`.
    """
    orig_image = tf.cast(image, dtype=dtype)
    image = image[tf.newaxis, ..., tf.newaxis]
    image = tf.image.resize(image, [height * spacial_osf, width * spacial_osf])

    image = tf.squeeze(image)
    if normalize:
        image = image * tf.math.reduce_sum(orig_image) / tf.math.reduce_sum(image)

    return image


def rotate(image, t, angle=0, rate=0):
    """ Rotate the image.

    Args:
        image: `array`, image.
        t: `time`, simulation time in seconds from epoch.
        angle: `float`, angle to rotate image in degrees.
        rate: `float`, rate to rotate image in degrees per second.

    Returns:
        An `array`, same as input `image`.
    """
    image = image[tf.newaxis, ..., tf.newaxis]
    image = tfa.image.rotate(image, (angle + rate * t) * math.pi / 180.0)

    return tf.squeeze(image)


def load_from_file(image, t, filename, normalize=True, dtype=tf.float32):
    """ Load an image from file. Wraps `load_sprite_from_file` for
    the SatSim pipeline interface.

    Args:
        image: `array`, base image. (ignored)
        t: `time`, simulation time in seconds from epoch. (unused)
        filename: `float`, filename of file to load.
        normalize: `boolean`, normalize the output image so its sum is equal to 1.

    Returns:
        An `array`, an image.
    """
    return load_sprite_from_file(filename, normalize, dtype)


def pow(image, t, exponent=1, normalize=True, dtype=tf.float32):
    """ Returns the `image` raised to the power `exponent`.

    Args:
        image: `array`, base image.
        t: `time`, simulation time in seconds from epoch.
        exponent: `float`, exponent value.
        normalize: `boolean`, normalize output so sum is equal to input, `image`.

    Returns:
        An `array`, an image.
    """
    orig_image = tf.convert_to_tensor(image, dtype=dtype)
    image = orig_image ** exponent

    if normalize:
        image = image * tf.math.reduce_sum(orig_image) / tf.math.reduce_sum(image)

    return image
