from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np

from satsim.geometry.transform import rotate_and_translate
from satsim.util import is_tensorflow_running_on_cpu
from satsim.math import fftconv2p


MAX_PIXEL_VALUE = {
    'int8': 127.0,
    'uint8': 255.0,
    'int16': 32767.0,
    'uint16': 65535.0,
    'int32': 2147483647.0,
    'uint32': 4294967295.0,
}


def downsample(fpa, osf, method='conv2d'):
    """Downsample a 2D image tensor. Typically used to calculate the real pixel
    values in an oversampled image.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in oversampled pixel space.
        osf: `int`, oversample factor of the input image.
        method: `string`, conv2d or pool algorithm

    Returns:
        A `Tensor`, the downsampled 2D image tensor.
    """
    c = tf.shape(fpa)
    h = tf.cast(c[0], tf.int32)
    w = tf.cast(c[1], tf.int32)

    if method == 'pool':
        n = tf.cast(osf * osf, tf.float32)
        return n * tf.squeeze(
            tf.nn.avg_pool(
                tf.reshape(fpa, [1, h, w, 1]),
                [1, osf, osf, 1],
                [1, osf, osf, 1],
                padding='SAME'))

    else:

        filt = tf.ones((osf,osf))
        filt = filt[:, :, tf.newaxis, tf.newaxis]

        return tf.squeeze(
            tf.nn.conv2d(
                input=tf.reshape(fpa, [1, h, w, 1]),
                filters=filt,
                strides=[1, osf, osf, 1],
                padding="SAME"))


def crop(fpa, y_pad, x_pad, y_size, x_size):
    """Crops a 2D image tensor. Typically used to remove pixel padding.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in oversampled pixel space.
        y_pad: `int`, starting y pixel location (number of pad pixels at the top).
        x_pad: `int`, starting x pixel location (number of pad pixels on the left).
        y_size: `int`, total y pixels to crop.
        x_size: `int`, total x pixels to crop.

    Returns:
        A `Tensor`, the cropped 2D image tensor.
    """
    y_pad = tf.cast(y_pad, tf.int32)
    x_pad = tf.cast(x_pad, tf.int32)
    y_size = tf.cast(y_size, tf.int32)
    x_size = tf.cast(x_size, tf.int32)

    return tf.slice(fpa,
                    [y_pad, x_pad],
                    [y_size, x_size])


def analog_to_digital(fpa, gain, fwc, bias=0, dtype='uint16', saturated_pixel_model='max'):
    """Converts photoelectron counts to digital counts. The linear model
    `digital * gain = analog` is used to covert from analog to digital counts.
    Photoelectron counts for each pixel are bound between `0` and `fwc`.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in real pixel space.
        gain: `float`, the digital to analog multiplier.
        fwc: `float`, the maximum value or full well capacity in photoelectrons.
        bias: `float`, bias to add in digital counts. (Default: 0)
        dtype: `string`, pixel data type
        saturated_pixel_model: `string`, `max` will set any value greater than the maximum value
            of the `dtype` to the maximum value, otherwise saturated pixels behavior is undefined

    Returns:
        A `Tensor`, the 2D image tensor as whole number digital counts.
    """
    # add bias and bound max
    fpa_with_bias = fpa + (bias * gain)  # note gain is divided out later
    condition = tf.less(fpa_with_bias, fwc)
    fpa_digital = tf.where(condition, fpa_with_bias, tf.ones(tf.shape(fpa)) * fwc)

    # divide by d2a gain
    fpa_digital = tf.math.floor(fpa_digital / gain)

    # set negative values to zero
    condition = tf.greater(fpa_digital, 0)
    fpa_digital = tf.where(condition, fpa_digital, tf.zeros(tf.shape(fpa)))

    # handle saturated pixels
    if saturated_pixel_model == 'max':
        if dtype in MAX_PIXEL_VALUE:
            condition = tf.less(fpa_digital, MAX_PIXEL_VALUE[dtype])
            fpa_digital = tf.where(condition, fpa_digital, tf.fill(tf.shape(fpa), MAX_PIXEL_VALUE[dtype]))

    return fpa_digital


def mv_to_pe(zeropoint, mv):
    """Converts visual magnitude to photoelectrons per second. The zeropoint of
    an instrument, by definition, is the magnitude of an object that produces
    one count (or photoelectron) per second. The visual magnitude (MV) of an
    arbitrary object producing photoelectrons in an observation of one second is
    therefore:

        `mv = -2.5 * log10(pe) + zeropoint`

        and

        `mv_to_pe(zeropoint, zeropoint) == 1`

    Args:
        zeropoint: `float`, the zeropoint of the fpa.
        mv: `float`, the visual magnitude of the object.

    Returns:
        A `float`, the visual magnitude in photoelectron per second
    """
    zp = np.asarray(zeropoint)
    mv = np.asarray(mv)
    return 10 ** ((zp - mv) / 2.5)


def pe_to_mv(zeropoint, pe):
    """Converts photoelectrons per second to visual magnitude. The zeropoint of
    an instrument, by definition, is the magnitude of an object that produces
    one count (or photoelectron) per second. The visual magnitude (MV) of an
    arbitrary object producing photoelectrons in an observation of one second is
    therefore:

        `mv = -2.5 * log10(pe) + zeropoint`

        and

        `mv_to_pe(zeropoint, zeropoint) == 1`

    Args:
        zeropoint: `float`, the zeropoint of the fpa.
        pe: `float`, the brightness of the object in photoelectron per second.

    Returns:
        A `float`, the visual magnitude
    """

    zp = np.asarray(zeropoint)
    pe = np.asarray(pe)
    return -2.5 * np.log10(pe) + zp


def add_patch(fpa, r, c, cnt, patch, r_offset=0, c_offset=0, mode='fft'):
    """Add a patch to the input image, `fpa`, centered about each row, column coordinate.
    Patch is multiplied by the `cnt`. `patch` is typically normalized.

    Args:
        fpa: `Tensor`, input image as a 2D tensor.
        r: `list`, list of row pixel coordinates as int.
        c: `list`, list of column pixel coordinates as int.
        cnt: `list`, list of absolute counts. For example, dn or pe.
        patch: `Tensor`, patch image as a 2D tensor
        r_offset: `int`, offset to add to `r` values. Useful to account for
            image pad.
        c_offset: `int`, offset to add to `c` values. Useful to account for
            image pad.
        mode: `string`, specify render mode. fft or overlay. default: fft

    Returns:
        A `Tensor`, a reference to the modified input `fpa` image
    """

    patch_rows, patch_cols = tf.shape(patch)
    patch_rows_div2, patch_cols_div2 = tf.cast(tf.shape(patch) / 2, tf.int32)

    # shift to center patch
    r_offset = tf.cast(r_offset, tf.float32)
    c_offset = tf.cast(c_offset, tf.float32)

    # expand fpa to fit image and patch
    fpa = tf.pad(fpa, [[patch_rows, patch_rows], [patch_cols, patch_cols]])
    fpa_rows, fpa_cols = tf.shape(fpa)

    rr = tf.cast(r + r_offset, tf.int32)
    cc = tf.cast(c + c_offset, tf.int32)

    if mode == 'fft':
        patch_full = _to_shape(patch, fpa)
        # TODO need to investigate +1 shift
        delta = add_counts(tf.zeros_like(fpa, tf.float32), rr + 1, cc + 1, cnt)
        fpa =  fftconv2p(delta, patch_full, pad=1)
    else:

        def c(i, img, rr, cc, pe):
            return tf.less(i, len(rr))

        def f(i, img, rr, cc, pe):
            r = rr[i]
            c = cc[i]
            r_end = r + patch_rows
            c_end = c + patch_cols

            overlay = pe[i] * patch

            if r >= 0 and c >= 0 and r_end < fpa_rows and c_end < fpa_cols:
                overlay_pad = tf.pad(overlay, [[r, fpa_rows - r_end], [c, fpa_cols - c_end]])
                img = img + overlay_pad

            return i + 1, img, rr, cc, pe

        i = tf.constant(0)
        i, fpa, _, _, _ = tf.while_loop(c, f, (i, fpa, rr, cc, cnt))

    # crop so that patches are centered
    fpa = tf.slice(fpa, [patch_rows_div2, patch_cols_div2], [fpa_rows - patch_rows - patch_rows, fpa_cols - patch_cols - patch_cols])

    return fpa


def add_counts(fpa, r, c, cnt, r_offset=0, c_offset=0):
    """Add counts (e.g. digital number (dn), photoelectrons (pe)) to the input
    image.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in real or oversample pixel
            space. Should be a `tf.Variable`.
        r: `list`, list of row pixel coordinates as int
        c: `list`, list of column pixel coordinates as int
        cnt: `list`, list of absolute counts. For example, dn or pe.
        r_offset: `int`, offset to add to `r` values. Useful to account for
            image pad.
        c_offset: `int`, offset to add to `c` values. Useful to account for
            image pad.

    Returns:
        A `Tensor`, a reference to the modified input `fpa` image
    """
    r = tf.cast(r, tf.int32) + tf.cast(r_offset, tf.int32)
    c = tf.cast(c, tf.int32) + tf.cast(c_offset, tf.int32)

    # fix for no bounds checking if fpa is int32 or if running CPU
    if is_tensorflow_running_on_cpu() or fpa.dtype == 'int32':
        h, w = fpa.get_shape().as_list()
        valid = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        r = tf.boolean_mask(r, valid)
        c = tf.boolean_mask(c, valid)
        cnt = tf.boolean_mask(tf.convert_to_tensor(cnt, dtype=fpa.dtype), valid)

    rc = tf.stack([r, c], axis=1)
    return tf.compat.v1.tensor_scatter_nd_add(fpa, rc, cnt)


def transform_and_fft(fpa, r, c, cnt, t_start, t_end, t_osf, rotation, translation):
    """Applies discrete rotation and translation transformations to the center
    point and applies the smear to all other points with an FFT. This has the
    effect of  "smearing" or "streaking" the point between times `t_start` and
    `t_end`. This function has the advantage of a much faster runtime than
    `transform_and_add_counts` when there are many points to transform and the
    disadvantage that all points have the identical smear.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in real or oversample pixel
            space. Should be from a Variable node.
        r: `list`, list of row pixel coordinates.
        c: `list`, list of column pixel coordinates.
        cnt: `list`, list of absolute dn or pe counts.
        t_start: `float`, start time in seconds from t_epoch where t_epoch=0
        t_end: `float`, end time in seconds from t_epoch where t_epoch=0
        t_osf: `int`, temporal oversample factor, determines the number of
            discrete transforms to apply between `t_start` and `t_end`. Value
            should be greather than 0.
        rotation: `float`, clockwise rotation rate in radians/sec
        translation: '[float, float]', translation rate in pixels/sec in
            `[row,col]` order

    Returns:
        A `tuple`:
            fpa: `Tensor`, a reference to the modified input `fpa` image
    """

    s = tf.shape(fpa)
    h = tf.cast(s[0], dtype=tf.float32)
    w = tf.cast(s[1], dtype=tf.float32)
    h_minus_1 = h - 1.0
    w_minus_1 = w - 1.0
    r = tf.cast(r, dtype=tf.float32)
    c = tf.cast(c, dtype=tf.float32)
    cnt = tf.cast(cnt, dtype=tf.float32)
    t_osf = tf.cast(t_osf, dtype=tf.int32)

    h_mid = h_minus_1 / 2.0
    w_mid = w_minus_1 / 2.0

    # create PSF by creating point source in the middle then transforming it
    blur = transform_and_add_counts(tf.zeros_like(fpa, tf.float32), [h_mid], [w_mid], [1.0], t_start, t_end, t_osf, rotation, translation)

    # create delta functions
    (rr, cc) = rotate_and_translate(h_minus_1, w_minus_1, r, c, 0.0, rotation, translation)
    delta = add_counts(tf.zeros_like(fpa, tf.float32), rr, cc, cnt)

    # FFT
    return fftconv2p(delta, blur, pad=1)


def transform_and_add_counts(fpa, r, c, cnt, t_start, t_end, t_osf, rotation, translation, batch_size=500, filter_out_of_bounds=True):
    """Applies discrete rotation and translation transformations to a set of
    input points. This has the effect of "smearing" or "streaking" the point
    between times `t_start` and `t_end`. The number of transforms calculated is
    specified by the temporal oversample factor, `t_osf`. The total energy is
    conserved.

    Args:
        fpa: `Tensor`, input image as a 2D tensor in real or oversample pixel
            space. Should be from a Variable node.
        r: `list`, list of row pixel coordinates.
        c: `list`, list of column pixel coordinates.
        cnt: `list`, list of absolute dn or pe counts.
        t_start: `float`, start time in seconds from t_epoch where t_epoch=0
        t_end: `float`, end time in seconds from t_epoch where t_epoch=0
        t_osf: `int`, temporal oversample factor, determines the number of
            discrete transforms to apply between `t_start` and `t_end`. Value
            should be greather than 0.
        rotation: `float`, clockwise rotation rate in radians/sec
        translation: '[float, float]', translation rate in pixels/sec in
            `[row,col]` order
        batch_size: `int`: Number of points to process together. Typically a
            larger batch size will process faster due to vectorization at
            the expense of increased memory useage. Default: 500)

    Returns:
        A `tuple`:
            fpa: `Tensor`, a reference to the modified input `fpa` image
    """
    s = tf.shape(fpa)
    h = tf.cast(s[0], dtype=tf.float32)
    w = tf.cast(s[1], dtype=tf.float32)
    h_minus_1 = h - 1.0
    w_minus_1 = w - 1.0
    r = tf.cast(r, dtype=tf.float32)
    c = tf.cast(c, dtype=tf.float32)
    cnt = tf.cast(cnt, dtype=tf.float32)
    t_osf = tf.cast(t_osf, dtype=tf.int32)

    # divide pe by the number of discrete points to sample, e.g. for t_osf==2 then i==[0,1]
    cnt_os = cnt / tf.cast(t_osf, tf.float32)

    if filter_out_of_bounds:
        (rr0_tmp, cc0_tmp) = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
        (rr1_tmp, cc1_tmp) = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_end, rotation, translation)
        rr_lt_0 = tf.math.logical_and(rr0_tmp < 0, rr1_tmp < 0)
        rr_gt_h = tf.math.logical_and(rr0_tmp > h, rr1_tmp > h)
        cc_lt_0 = tf.math.logical_and(cc0_tmp < 0, cc1_tmp < 0)
        cc_gt_w = tf.math.logical_and(cc0_tmp > w, cc1_tmp > w)
        out_of_bounds = tf.math.logical_or(tf.math.logical_or(rr_lt_0, rr_gt_h), tf.math.logical_or(cc_lt_0, cc_gt_w))
        in_bounds = tf.math.logical_not(out_of_bounds)
        r_filter = tf.boolean_mask(r, in_bounds)
        c_filter = tf.boolean_mask(c, in_bounds)
        cnt_os_filter = tf.boolean_mask(cnt_os, in_bounds)
    else:
        r_filter = r
        c_filter = c
        cnt_os_filter = cnt_os

    batch_size = tf.math.minimum(batch_size, tf.size(r_filter))

    if batch_size == 0:
        return fpa

    # batch the points
    r_batch = _to_batch_1d(r_filter, batch_size)
    c_batch = _to_batch_1d(c_filter, batch_size)
    cnt_os_batch = _to_batch_1d(cnt_os_filter, batch_size)

    def func_condition(i, img, r_batch, c_batch, cnt_os_batch):
        return tf.less(i, r_batch.shape[0])

    def func_eval(i, img, r_batch, c_batch, cnt_os_batch):

        # vectorize
        tt = tf.repeat(tf.linspace(t_start, t_end, t_osf), batch_size)
        rr = tf.tile(r_batch[i, :], [t_osf])
        cc = tf.tile(c_batch[i, :], [t_osf])
        cnt_os = tf.tile(cnt_os_batch[i, :], [t_osf])

        # perform transformation
        (rr, cc) = rotate_and_translate(h_minus_1, w_minus_1, rr, cc, tt, rotation, translation)

        return [i + 1, add_counts(img, rr, cc, cnt_os), r_batch, c_batch, cnt_os_batch]

    i = tf.constant(0)
    i, image, _, _, _ = tf.while_loop(func_condition, func_eval, (i, fpa, r_batch, c_batch, cnt_os_batch))

    return image


def _to_batch_1d(x, batch_size):

    remainder = batch_size - tf.math.floormod(x.shape[0], batch_size)

    x = tf.pad(x, [[0, remainder],])

    num_batches = tf.cast(x.shape[0] / batch_size, tf.int32)

    x = tf.reshape(x, [num_batches, batch_size])

    return x


def _to_shape(a, b):
    y_, x_ = b.shape
    y, x = a.shape
    y_pad = (y_ - y)
    x_pad = (x_ - x)
    return tf.pad(a, ((y_pad // 2, y_pad // 2 + y_pad % 2), (x_pad // 2, x_pad // 2 + x_pad % 2)), mode='constant')
