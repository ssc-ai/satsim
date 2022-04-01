from __future__ import division, print_function, absolute_import

import tensorflow as tf


def rotate_and_translate(h_m1, w_m1, r, c, t, rotation, translation):
    """Rotate and translate a list of input pixels locations. Typically used to
    transform stars to simulate image drift and rotation during sidereal and
    rate track.

    Examples::

        (yy, xx) = rotate_and_translate(5120, 5120, r, c, 5.0, 0.01, [10,5])

    Args:
        h_m1: `int`, image height in number of pixels minus 1.
        w_m1: `int`, image width in number of pixels minus 1.
        r: `list`, list of row pixel coordinates
        c: `list`, list of col pixel coordinates
        t: `float`, time elapsed from epoch
        rotation: `float`, rotation rate in radians/sec clockwise
        translation: `[float,float]`, translation rate in pixel/sec in
            `[row,col]` order

    Returns:
        A `tuple`, containing:
            rr: `list`, list of transformed row pixel locations
            cc: `list`, list of transformed col pixel locations
    """
    angles = rotation * t
    sina = tf.math.sin(angles)
    cosa = tf.math.cos(angles)
    # h = h - 1.0
    # w = w - 1.0
    # y_offset = (h - (sina * w + cosa * h)) * 0.5
    # x_offset = (w - (cosa * w - sina * h)) * 0.5

    # a0 =  cosa
    # a1 =  sina
    # a2 =  x_offset
    # b0 =  sina
    # b1 =  cosa
    # b2 =  y_offset

    # c = tf.to_float(c)
    # r = tf.to_float(r)
    rr = sina * c + cosa * r + (h_m1 - (sina * w_m1 + cosa * h_m1)) * 0.5 + translation[0] * t
    cc = cosa * c - sina * r + (w_m1 - (cosa * w_m1 - sina * h_m1)) * 0.5 + translation[1] * t

    return [rr, cc]


def apply_wrap_around(height, width, r, c, t_start, t_end, rotation, translation, wrap_around):
    """Applies discrete rotation and translation transformations to a set of
    input points. This has the effect of "smearing" or "streaking" the point
    between times `t_start` and `t_end`. The number of transforms calculated is
    specified by the temporal oversample factor, `t_osf`. The total energy is
    conserved.

    Args:
        height: `int`, height of array
        width: `int`, width of width
        r: `list`, list of row pixel coordinates.
        c: `list`, list of column pixel coordinates.
        t_start: `float`, start time in seconds from t_epoch where t_epoch=0
        t_end: `float`, end time in seconds from t_epoch where t_epoch=0
        rotation: `float`, clockwise rotation rate in radians/sec
        translation: '[float, float]', translation rate in pixels/sec in
            `[row,col]` order
        wrap_around: `[[float,float],[float,float],[float,float]]`, if not
            None, wrap out of bounds coordinates to other side of the fpa.
            The boundaries are specified as 3 lists: upper and lower row
            bounds, left and right column bounds, and center coordinate.
            Useful to wrap stars.

    Returns:
        A `tuple`:
            r: `Tensor`, new row positions with wrap around applied
            c: `Tensor`, new column positions with wrap around applied
            wrap_around: `Tensor`, new wrap around bounds
    """
    h = tf.cast(height - 1, dtype=tf.float32)
    w = tf.cast(width - 1, dtype=tf.float32)
    r = tf.cast(r, dtype=tf.float32)
    c = tf.cast(c, dtype=tf.float32)

    if wrap_around is not None:
        wrap_around = tf.cast(wrap_around, dtype=tf.float32)
        (r1, c1) = rotate_and_translate(h, w, 0, 0, t_start, rotation, translation)

        r2 = r1 - wrap_around[2][0]
        c2 = c1 - wrap_around[2][1]
        rwrap = wrap_around[0] - r2
        cwrap = wrap_around[1] - c2

        ri0 = tf.where(r < wrap_around[0][0])
        ri1 = tf.where(r > wrap_around[0][1])
        r = tf.tensor_scatter_nd_add(r, ri0, tf.constant(rwrap[1] - rwrap[0], shape=(ri0.shape[0],)))
        r = tf.tensor_scatter_nd_add(r, ri1, tf.constant(rwrap[0] - rwrap[1], shape=(ri1.shape[0],)))

        ci0 = tf.where(c < wrap_around[1][0])
        ci1 = tf.where(c > wrap_around[1][1])
        c = tf.tensor_scatter_nd_add(c, ci0, tf.constant(cwrap[1] - cwrap[0], shape=(ci0.shape[0],)))
        c = tf.tensor_scatter_nd_add(c, ci1, tf.constant(cwrap[0] - cwrap[1], shape=(ci1.shape[0],)))

        wrap_around = [rwrap, cwrap, [tf.squeeze(r1), tf.squeeze(c1)]]

    return r, c, wrap_around
