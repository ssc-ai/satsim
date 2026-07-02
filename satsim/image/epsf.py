from __future__ import division, print_function, absolute_import

import tensorflow as tf

from satsim.geometry.transform import rotate_and_translate
from satsim.image.fpa import add_counts, downsample
from satsim.math import fftconv2p


def _validate_kernel_size(kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer in detector pixels')
    return kernel_size


def _validate_point_rendering(point_rendering):
    point_rendering = str(point_rendering).lower()
    if point_rendering not in ('floor', 'bilinear'):
        raise ValueError("point_rendering must be 'floor' or 'bilinear'")
    return point_rendering


def _delta_epsf_lut(s_osf, kernel_size, dtype):
    center = kernel_size // 2
    stamp = tf.one_hot(center, kernel_size, dtype=dtype)[:, None] * tf.one_hot(center, kernel_size, dtype=dtype)[None, :]
    stamp = tf.reshape(stamp, [1, 1, kernel_size, kernel_size])
    return tf.tile(stamp, [int(s_osf), int(s_osf), 1, 1])


def build_epsf_lut(psf_os, s_osf, kernel_size, normalize=False, dtype=tf.float32):
    """Build a phase-indexed detector-space ePSF lookup table.

    The LUT contract matches the existing FFT render path: each phase stamp is
    rendered by placing a unit-flux oversampled delta on an interior base pixel,
    convolving with ``psf_os``, downsampling with detector-pixel block sums, and
    cropping a detector-pixel stamp centered on the base pixel.
    """
    s_osf = int(s_osf)
    kernel_size = _validate_kernel_size(kernel_size)
    dtype = tf.as_dtype(dtype)

    if psf_os is None:
        return _delta_epsf_lut(s_osf, kernel_size, dtype)

    psf_os = tf.cast(psf_os, dtype)
    psf_shape = tf.shape(psf_os)
    h_os = psf_shape[0]
    w_os = psf_shape[1]

    with tf.control_dependencies([
        tf.debugging.assert_equal(
            tf.math.floormod(h_os, s_osf),
            0,
            message='psf_os height must be divisible by s_osf',
        ),
        tf.debugging.assert_equal(
            tf.math.floormod(w_os, s_osf),
            0,
            message='psf_os width must be divisible by s_osf',
        ),
    ]):
        h_det = h_os // s_osf
        w_det = w_os // s_osf

    half = kernel_size // 2
    min_size = tf.constant(kernel_size, dtype=tf.int32)

    with tf.control_dependencies([
        tf.debugging.assert_greater_equal(h_det, min_size, message='psf_os detector height must be at least kernel_size'),
        tf.debugging.assert_greater_equal(w_det, min_size, message='psf_os detector width must be at least kernel_size'),
    ]):
        base_r = h_det // 2
        base_c = w_det // 2

    stamps_r = []
    for phase_r in range(s_osf):
        stamps_c = []
        for phase_c in range(s_osf):
            q_r = base_r * s_osf + phase_r
            q_c = base_c * s_osf + phase_c
            delta = add_counts(
                tf.zeros_like(psf_os, dtype=dtype),
                [q_r],
                [q_c],
                [tf.constant(1.0, dtype=dtype)],
                interpolation='floor',
            )
            conv_os = fftconv2p(delta, psf_os, pad=1)
            conv_det = downsample(conv_os, s_osf, method='block_sum')
            stamp = tf.slice(conv_det, [base_r - half, base_c - half], [kernel_size, kernel_size])

            if normalize:
                total = tf.reduce_sum(stamp)
                stamp = tf.cond(
                    tf.greater(total, tf.cast(0.0, dtype)),
                    lambda: stamp / total,
                    lambda: stamp,
                )

            stamps_c.append(stamp)
        stamps_r.append(tf.stack(stamps_c, axis=0))

    return tf.cast(tf.stack(stamps_r, axis=0), dtype)


def _expand_oversampled_points(r_os, c_os, cnt, r_offset_os, c_offset_os, point_rendering, dtype):
    r = tf.reshape(tf.cast(r_os, tf.float32), [-1]) + tf.cast(r_offset_os, tf.float32)
    c = tf.reshape(tf.cast(c_os, tf.float32), [-1]) + tf.cast(c_offset_os, tf.float32)
    cnt = tf.reshape(tf.cast(cnt, dtype), [-1])

    if point_rendering == 'bilinear':
        r0_float = tf.floor(r)
        c0_float = tf.floor(c)
        dr = tf.cast(r - r0_float, dtype)
        dc = tf.cast(c - c0_float, dtype)

        r0 = tf.cast(r0_float, tf.int32)
        c0 = tf.cast(c0_float, tf.int32)
        r1 = r0 + 1
        c1 = c0 + 1

        rr = tf.concat([r0, r0, r1, r1], axis=0)
        cc = tf.concat([c0, c1, c0, c1], axis=0)
        weights = tf.concat([
            (1 - dr) * (1 - dc),
            (1 - dr) * dc,
            dr * (1 - dc),
            dr * dc,
        ], axis=0)
        values = tf.tile(cnt, [4]) * weights
        return rr, cc, values

    return tf.cast(tf.floor(r), tf.int32), tf.cast(tf.floor(c), tf.int32), cnt


def add_epsf_counts(fpa, r_os, c_os, cnt, epsf_lut, s_osf, r_offset_os=0, c_offset_os=0, batch_size=1024, point_rendering='bilinear'):
    """Deposit oversampled point sources as detector-space ePSF stamps."""
    point_rendering = _validate_point_rendering(point_rendering)
    s_osf = tf.cast(s_osf, tf.int32)
    batch_size = tf.cast(batch_size, tf.int32)

    dtype = fpa.dtype
    epsf_lut = tf.cast(epsf_lut, dtype)
    kernel_size = tf.shape(epsf_lut)[2]
    half = kernel_size // 2

    rr_os, cc_os, values = _expand_oversampled_points(
        r_os,
        c_os,
        cnt,
        r_offset_os,
        c_offset_os,
        point_rendering,
        dtype,
    )

    n = tf.shape(rr_os)[0]
    batch_size = tf.maximum(tf.constant(1, tf.int32), tf.minimum(batch_size, tf.maximum(n, 1)))
    offsets = tf.range(kernel_size, dtype=tf.int32) - half

    def cond(i, image):
        return tf.less(i, n)

    def body(i, image):
        end = tf.minimum(i + batch_size, n)
        rr_b = rr_os[i:end]
        cc_b = cc_os[i:end]
        values_b = values[i:end]

        phase_r = tf.math.floormod(rr_b, s_osf)
        phase_c = tf.math.floormod(cc_b, s_osf)
        base_r = tf.math.floordiv(rr_b, s_osf)
        base_c = tf.math.floordiv(cc_b, s_osf)

        stamps = tf.gather_nd(epsf_lut, tf.stack([phase_r, phase_c], axis=1))
        stamps = stamps * tf.reshape(values_b, [-1, 1, 1])

        rows = tf.reshape(base_r, [-1, 1, 1]) + tf.reshape(offsets, [1, -1, 1])
        rows = tf.tile(rows, [1, 1, kernel_size])
        cols = tf.reshape(base_c, [-1, 1, 1]) + tf.reshape(offsets, [1, 1, -1])
        cols = tf.tile(cols, [1, kernel_size, 1])

        shape = tf.shape(image)
        valid = (rows >= 0) & (rows < shape[0]) & (cols >= 0) & (cols < shape[1])
        rows = tf.boolean_mask(rows, valid)
        cols = tf.boolean_mask(cols, valid)
        scatter_values = tf.boolean_mask(stamps, valid)

        indices = tf.stack([rows, cols], axis=1)
        image = tf.tensor_scatter_nd_add(image, indices, scatter_values)
        return end, image

    _, fpa = tf.while_loop(cond, body, (tf.constant(0, tf.int32), fpa))
    return fpa


def transform_and_add_epsf(fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation, epsf_lut, s_osf, batch_size=500, filter_out_of_bounds=True, point_rendering='bilinear'):
    """Apply star motion and deposit transformed samples as ePSF stamps."""
    point_rendering = _validate_point_rendering(point_rendering)
    s_osf = tf.cast(s_osf, tf.int32)

    shape = tf.shape(fpa)
    h_os = tf.cast(shape[0] * s_osf, tf.float32)
    w_os = tf.cast(shape[1] * s_osf, tf.float32)
    h_minus_1 = h_os - 1.0
    w_minus_1 = w_os - 1.0

    r = tf.cast(r_os, tf.float32)
    c = tf.cast(c_os, tf.float32)
    cnt = tf.cast(cnt, tf.float32)
    t_osf = tf.cast(t_osf, tf.int32)
    cnt_os = cnt / tf.cast(t_osf, tf.float32)

    if filter_out_of_bounds:
        rr0_tmp, cc0_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
        rr1_tmp, cc1_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_end, rotation, translation)
        rr_lt_0 = tf.math.logical_and(rr0_tmp < 0, rr1_tmp < 0)
        rr_gt_h = tf.math.logical_and(rr0_tmp > h_os, rr1_tmp > h_os)
        cc_lt_0 = tf.math.logical_and(cc0_tmp < 0, cc1_tmp < 0)
        cc_gt_w = tf.math.logical_and(cc0_tmp > w_os, cc1_tmp > w_os)
        out_of_bounds = tf.math.logical_or(tf.math.logical_or(rr_lt_0, rr_gt_h), tf.math.logical_or(cc_lt_0, cc_gt_w))
        in_bounds = tf.math.logical_not(out_of_bounds)
        r = tf.boolean_mask(r, in_bounds)
        c = tf.boolean_mask(c, in_bounds)
        cnt_os = tf.boolean_mask(cnt_os, in_bounds)

    n = tf.shape(r)[0]
    batch_size = tf.cast(batch_size, tf.int32)
    batch_size = tf.maximum(tf.constant(1, tf.int32), tf.minimum(batch_size, tf.maximum(n, 1)))

    def cond(i, image):
        return tf.less(i, n)

    def body(i, image):
        end = tf.minimum(i + batch_size, n)
        r_batch = r[i:end]
        c_batch = c[i:end]
        cnt_batch = cnt_os[i:end]
        current_batch = tf.shape(r_batch)[0]

        tt = tf.repeat(tf.linspace(t_start, t_end, t_osf), current_batch)
        rr = tf.tile(r_batch, [t_osf])
        cc = tf.tile(c_batch, [t_osf])
        ccnt = tf.tile(cnt_batch, [t_osf])

        rr, cc = rotate_and_translate(h_minus_1, w_minus_1, rr, cc, tt, rotation, translation)
        image = add_epsf_counts(
            image,
            rr,
            cc,
            ccnt,
            epsf_lut,
            s_osf,
            batch_size=batch_size,
            point_rendering=point_rendering,
        )
        return end, image

    _, fpa = tf.while_loop(cond, body, (tf.constant(0, tf.int32), fpa))
    return fpa
