from __future__ import division, print_function, absolute_import

import math
import warnings
from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Tuple

import numpy as np
import tensorflow as tf

from satsim.geometry.transform import rotate_and_translate
from satsim.image.epsf_cpu import add_epsf_stamps
from satsim.image.fpa import add_counts, downsample
from satsim.math import fftconv2p


EPSF_BATCH_ELEMENT_BUDGET_DEFAULT = 32000000
EPSF_BATCH_MIN_DEFAULT = 1024
EPSF_BATCH_MAX_DEFAULT = 262144
EPSF_PHASE_ERROR_TARGET_PX = 0.02
# Bounds the transformed coordinates handed to a position callback per batch.
_POSITION_TRANSFORM_BATCH_SAMPLES = 65536
_STAR_ACCUMULATORS = ('auto', 'tensorflow', 'cpu')


def _validate_kernel_size(kernel_size):
    kernel_size = int(kernel_size)
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError('kernel_size must be a positive odd integer in detector pixels')
    return kernel_size


def _validate_point_rendering(point_rendering):
    point_rendering = str(point_rendering).lower()
    if point_rendering not in ('floor', 'bilinear', 'phase_nearest'):
        raise ValueError("point_rendering must be 'floor', 'bilinear', or 'phase_nearest'")
    return point_rendering


def _point_rendering_expansion(point_rendering):
    point_rendering = _validate_point_rendering(point_rendering)
    return 4 if point_rendering == 'bilinear' else 1


def resolve_phase_oversample(s_osf, phase_oversample=None, target_px=EPSF_PHASE_ERROR_TARGET_PX):
    """Return the fine phase-grid multiplier used by ``phase_nearest``."""
    s_osf = int(s_osf)
    if s_osf <= 0:
        raise ValueError('s_osf must be positive')
    if phase_oversample is None:
        target_px = float(target_px)
        if target_px <= 0:
            raise ValueError('target_px must be positive')
        return max(1, int(math.ceil(1.0 / (2.0 * float(s_osf) * target_px))))
    phase_oversample = int(phase_oversample)
    if phase_oversample <= 0:
        raise ValueError('sim.epsf.phase_oversample must be a positive integer')
    return phase_oversample


def phase_nearest_error_bound_px(s_osf, phase_oversample):
    return 1.0 / (2.0 * float(s_osf) * float(phase_oversample))


def effective_epsf_batch_size(
        kernel_size,
        point_rendering='bilinear',
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None,
        temporal_osf=1,
        batch_min=EPSF_BATCH_MIN_DEFAULT,
        batch_max=EPSF_BATCH_MAX_DEFAULT,
        source_count=None):
    """Select an ePSF source batch size from a stamp-element budget.

    ``batch_size_cap`` is honored only when explicitly supplied by a caller.
    This keeps legacy tests and user caps deterministic without imposing the
    old 1024-source limit on default production configs.
    """
    expansion = _point_rendering_expansion(point_rendering)
    kernel_size = tf.cast(kernel_size, tf.int64)
    temporal_osf = tf.maximum(tf.cast(temporal_osf, tf.int64), tf.constant(1, tf.int64))
    batch_min = tf.cast(batch_min, tf.int64)
    batch_max = tf.cast(batch_max, tf.int64)

    if batch_element_budget is None:
        if batch_size_cap is None:
            selected = batch_max
        else:
            selected = tf.minimum(tf.cast(batch_size_cap, tf.int64), batch_max)
    else:
        denom = kernel_size * kernel_size * tf.constant(expansion, tf.int64) * temporal_osf
        budget = tf.cast(batch_element_budget, tf.int64)
        selected = budget // tf.maximum(denom, tf.constant(1, tf.int64))
        selected = tf.maximum(batch_min, selected)
        selected = tf.minimum(batch_max, selected)
        if batch_size_cap is not None:
            selected = tf.minimum(selected, tf.cast(batch_size_cap, tf.int64))

    selected = tf.minimum(selected, tf.constant(np.iinfo(np.int32).max, tf.int64))
    selected = tf.cast(tf.maximum(tf.constant(1, tf.int64), selected), tf.int32)
    if source_count is not None:
        selected = tf.minimum(selected, tf.maximum(tf.cast(source_count, tf.int32), tf.constant(1, tf.int32)))
    return selected


def _to_int(value, name):
    if isinstance(value, int):
        return value
    static_value = tf.get_static_value(value)
    if static_value is not None:
        return int(static_value)
    if hasattr(value, 'numpy'):
        return int(value.numpy())
    raise ValueError('{} must be statically known when building an ePSF LUT'.format(name))


def _make_odd(value):
    value = int(value)
    if value % 2 == 0:
        value += 1
    return value


def _to_numpy(value):
    if hasattr(value, 'numpy'):
        return value.numpy()
    return np.asarray(value)


def _kernel_size_from_lut(epsf_lut):
    shape = epsf_lut.shape.as_list() if hasattr(epsf_lut.shape, 'as_list') else list(epsf_lut.shape)
    if shape[2] is not None:
        return int(shape[2])
    return _to_int(tf.shape(epsf_lut)[2], 'epsf_lut kernel_size')


def _validate_crop_size(crop_size, full_size, name='crop_size', allow_full=True):
    crop_size = _validate_kernel_size(crop_size)
    full_size = int(full_size)
    if crop_size > full_size or (not allow_full and crop_size >= full_size):
        relation = 'less than or equal to' if allow_full else 'strictly less than'
        raise ValueError('{} must be {} the full ePSF kernel_size'.format(name, relation))
    if (full_size - crop_size) % 2 != 0:
        raise ValueError('{} must have the same parity as the full ePSF kernel_size'.format(name))
    return crop_size


def crop_epsf_lut(epsf_lut, crop_size):
    """Return a centered crop of a phase-indexed ePSF LUT."""
    full_size = _kernel_size_from_lut(epsf_lut)
    crop_size = _validate_crop_size(crop_size, full_size)
    offset = (full_size - crop_size) // 2
    return epsf_lut[:, :, offset:offset + crop_size, offset:offset + crop_size]


def epsf_wing_profile(epsf_lut, sizes):
    """Measure cropped-wing amplitude and energy loss for candidate sizes."""
    lut = np.asarray(_to_numpy(epsf_lut), dtype=np.float64)
    if lut.ndim != 4 or lut.shape[2] != lut.shape[3]:
        raise ValueError('epsf_lut must have shape [phase_r, phase_c, kernel_size, kernel_size]')

    full_size = int(lut.shape[2])
    full_energy = np.sum(lut, axis=(2, 3))
    profile = {}

    for size in sizes:
        size = _validate_crop_size(size, full_size)
        offset = (full_size - size) // 2
        mask = np.ones([full_size, full_size], dtype=bool)
        mask[offset:offset + size, offset:offset + size] = False

        outside = lut[:, :, mask]
        crop = lut[:, :, offset:offset + size, offset:offset + size]
        crop_energy = np.sum(crop, axis=(2, 3))
        energy_loss = full_energy - crop_energy

        profile[size] = {
            'wing_max': float(np.max(np.abs(outside))) if outside.size > 0 else 0.0,
            'enclosed_energy_min': float(np.min(crop_energy)),
            'enclosed_energy_max': float(np.max(crop_energy)),
            'full_energy_min': float(np.min(full_energy)),
            'full_energy_max': float(np.max(full_energy)),
            'energy_loss_max': float(np.max(np.abs(energy_loss))),
        }

    return profile


def _crop_size(core_size, full_size):
    crop_size = _make_odd(core_size)
    if crop_size > full_size:
        crop_size = full_size
    if (full_size - crop_size) % 2 != 0:
        crop_size = min(full_size, crop_size + 1)
    return _validate_crop_size(crop_size, full_size)


def resolve_epsf_tiers(epsf_lut, crop_config=None, trail_det=None):
    """Resolve an ePSF crop config into sorted flux tiers.

    Each tier has an inclusive ``flux_max`` in total source photoelectrons and
    a centered detector-space ``kernel_size``. ``None`` means tiering is off.
    ``trail_det`` is deprecated and ignored.
    """
    if trail_det is not None:
        _warn_deprecated('resolve_epsf_tiers trail_det is ignored; stars no longer deposit a trailed LUT')
    if not crop_config or crop_config.get('mode', 'off') == 'off':
        return None

    full_size = _kernel_size_from_lut(epsf_lut)
    mode = str(crop_config.get('mode', 'off')).lower()
    if mode not in ('manual', 'auto'):
        raise ValueError("sim.epsf.crop.mode must be 'off', 'manual', or 'auto'")

    tiers = []

    if mode == 'manual':
        threshold_pe = float(crop_config['threshold_pe'])
        core_size = int(crop_config['kernel_size'])
        crop_size = _crop_size(core_size, full_size)
        if crop_size < full_size:
            tiers.append({
                'kernel_size': crop_size,
                'core_size': core_size,
                'flux_max': threshold_pe,
                'mode': 'manual',
            })

    elif mode == 'auto':
        noise_fraction = float(crop_config.get('noise_fraction', 0.1))
        sigma_pix = float(crop_config['sigma_pix'])
        if noise_fraction <= 0:
            raise ValueError('sim.epsf.crop.noise_fraction must be positive')
        if sigma_pix < 0:
            raise ValueError('sim.epsf.crop sigma_pix must be non-negative')

        candidates = []
        for core_size in crop_config.get('sizes', []):
            crop_size = _crop_size(core_size, full_size)
            if crop_size < full_size:
                candidates.append((int(core_size), crop_size))

        seen = set()
        candidates = [(core, size) for core, size in candidates if not (size in seen or seen.add(size))]
        candidates.sort(key=lambda item: item[1])
        profile = epsf_wing_profile(epsf_lut, [size for _, size in candidates])

        last_flux_max = -math.inf
        for core_size, crop_size in candidates:
            wing_max = profile[crop_size]['wing_max']
            energy_loss = profile[crop_size]['energy_loss_max']

            wing_flux_max = math.inf if wing_max <= 0 else noise_fraction * sigma_pix / wing_max
            loss_flux_max = math.inf if energy_loss <= 0 else (noise_fraction / energy_loss) ** 2
            flux_max = min(wing_flux_max, loss_flux_max)
            binding = 'wing' if wing_flux_max <= loss_flux_max else 'loss'

            if flux_max > last_flux_max:
                tiers.append({
                    'kernel_size': crop_size,
                    'core_size': core_size,
                    'flux_max': float(flux_max),
                    'mode': 'auto',
                    'wing_max': float(wing_max),
                    'energy_loss_max': float(energy_loss),
                    'wing_flux_max': float(wing_flux_max),
                    'loss_flux_max': float(loss_flux_max),
                    'binding': binding,
                })
                last_flux_max = flux_max

    if not tiers:
        return None

    if tiers[-1]['kernel_size'] != full_size or not math.isinf(tiers[-1]['flux_max']):
        tiers.append({
            'kernel_size': full_size,
            'core_size': crop_config.get('full_core_size', full_size),
            'flux_max': math.inf,
            'mode': 'full',
        })

    return tiers


def summarize_epsf_tiers(tiers, cnt):
    """Return plain-Python tier counts and stamp-element totals."""
    if not tiers:
        return None

    cnt_np = np.asarray(_to_numpy(tf.reshape(tf.cast(cnt, tf.float32), [-1])), dtype=np.float64)
    prev = -math.inf
    summary = []
    total_count = 0
    total_stamp_elements = 0

    for tier in tiers:
        flux_max = float(tier['flux_max'])
        mask = (cnt_np > prev) & (cnt_np <= flux_max)
        count = int(np.sum(mask))
        kernel_size = int(tier['kernel_size'])
        total_count += count
        total_stamp_elements += count * kernel_size * kernel_size
        summary.append({
            'kernel_size': kernel_size,
            'core_size': int(tier.get('core_size', kernel_size)),
            'flux_max_pe': None if math.isinf(flux_max) else flux_max,
            'wing_flux_max_pe': None if 'wing_flux_max' not in tier or math.isinf(tier['wing_flux_max']) else float(tier['wing_flux_max']),
            'loss_flux_max_pe': None if 'loss_flux_max' not in tier or math.isinf(tier['loss_flux_max']) else float(tier['loss_flux_max']),
            'binding': tier.get('binding'),
            'count': count,
            'stamp_elements': count * kernel_size * kernel_size,
            'mode': tier.get('mode', ''),
        })
        prev = flux_max

    full_size = int(tiers[-1]['kernel_size'])
    return {
        'tiers': summary,
        'source_count': total_count,
        'stamp_elements': total_stamp_elements,
        'full_stamp_elements': int(len(cnt_np) * full_size * full_size),
    }


def _tier_mask(cnt, min_exclusive, max_inclusive):
    cnt = tf.reshape(tf.cast(cnt, tf.float32), [-1])
    mask = cnt <= tf.cast(max_inclusive, tf.float32)
    if min_exclusive != -math.inf:
        mask = tf.logical_and(mask, cnt > tf.cast(min_exclusive, tf.float32))
    return mask


def filter_epsf_sources_in_bounds(
        fpa, r_os, c_os, cnt, t_start, t_end, rotation, translation, s_osf, kernel_size=None,
        position_offset_os=0.0):
    """Drop sources whose ePSF stamps cannot reach ``fpa`` during the exposure.

    A dropped source deposits exactly zero into ``fpa`` for every time in
    ``[t_start, t_end]`` and every ``point_rendering`` when rendered with the
    same ``position_offset_os``. Pass the full LUT ``kernel_size`` so the
    bound holds for every crop tier.

    A stamp of odd ``kernel_size`` reaches the frame only when its detector
    base lies within ``kernel_size // 2`` pixels of it. Bilinear expansion
    moves a base by at most one oversampled pixel, so a halo of
    ``(kernel_size // 2 + 1) * s_osf`` oversampled pixels covers it together
    with any offset in ``[-s_osf, s_osf - 1]``; positions are shifted by the
    part of ``position_offset_os`` outside that range.

    Rotation turns a source on a circle of radius ``rho`` about the frame
    center while translation moves it linearly, so its acceleration is
    ``rho * rotation**2``. At every time in the exposure it therefore lies
    within ``rho * (rotation * (t_end - t_start))**2 / 8`` of the straight
    line between its endpoint positions, and a source is dropped when both
    endpoints lie beyond the halo plus that distance on the same side.

    Omitting ``kernel_size`` is deprecated; it uses no halo, which keeps every
    source the previous center-only filter kept.
    """
    if kernel_size is None:
        _warn_deprecated('filter_epsf_sources_in_bounds without kernel_size drops sources whose ePSF wings '
                         'reach the frame; pass the LUT kernel_size')
    s_osf = tf.cast(s_osf, tf.int32)
    shape = tf.shape(fpa)
    h_os = tf.cast(shape[0] * s_osf, tf.float32)
    w_os = tf.cast(shape[1] * s_osf, tf.float32)
    h_minus_1 = h_os - 1.0
    w_minus_1 = w_os - 1.0
    s_osf_f = tf.cast(s_osf, tf.float32)
    if kernel_size is None:
        halo = 0.0
    else:
        halo = tf.cast((tf.cast(kernel_size, tf.int32) // 2 + 1) * s_osf, tf.float32)

    r = tf.reshape(tf.cast(r_os, tf.float32), [-1])
    c = tf.reshape(tf.cast(c_os, tf.float32), [-1])
    cnt = tf.reshape(tf.cast(cnt, tf.float32), [-1])

    sweep = tf.cast(rotation, tf.float32) * (tf.cast(t_end, tf.float32) - tf.cast(t_start, tf.float32))
    radius = tf.sqrt(tf.square(r - h_minus_1 / 2.0) + tf.square(c - w_minus_1 / 2.0))
    margin = halo + radius * tf.square(sweep) / 8.0
    r_shift, c_shift = [
        tf.cast(offset, tf.float32) - tf.clip_by_value(tf.cast(offset, tf.float32), -s_osf_f, s_osf_f - 1.0)
        for offset in _offset_pair(position_offset_os)
    ]

    rr0_tmp, cc0_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
    rr1_tmp, cc1_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_end, rotation, translation)
    rr0_tmp, rr1_tmp = rr0_tmp + r_shift, rr1_tmp + r_shift
    cc0_tmp, cc1_tmp = cc0_tmp + c_shift, cc1_tmp + c_shift
    rr_lt_0 = tf.math.logical_and(rr0_tmp < -margin, rr1_tmp < -margin)
    rr_gt_h = tf.math.logical_and(rr0_tmp > h_os + margin, rr1_tmp > h_os + margin)
    cc_lt_0 = tf.math.logical_and(cc0_tmp < -margin, cc1_tmp < -margin)
    cc_gt_w = tf.math.logical_and(cc0_tmp > w_os + margin, cc1_tmp > w_os + margin)
    out_of_bounds = tf.math.logical_or(
        tf.math.logical_or(rr_lt_0, rr_gt_h),
        tf.math.logical_or(cc_lt_0, cc_gt_w),
    )
    in_bounds = tf.math.logical_not(out_of_bounds)

    return tf.boolean_mask(r, in_bounds), tf.boolean_mask(c, in_bounds), tf.boolean_mask(cnt, in_bounds)


def _offset_pair(position_offset_os):
    if isinstance(position_offset_os, (tuple, list)):
        return tuple(position_offset_os)
    return (position_offset_os, position_offset_os)


def _delta_epsf_lut(s_osf, kernel_size, dtype):
    center = kernel_size // 2
    stamp = tf.one_hot(center, kernel_size, dtype=dtype)[:, None] * tf.one_hot(center, kernel_size, dtype=dtype)[None, :]
    stamp = tf.reshape(stamp, [1, 1, kernel_size, kernel_size])
    return tf.tile(stamp, [int(s_osf), int(s_osf), 1, 1])


def _build_delta_phase_lut(s_osf, kernel_size, phase_oversample, normalize, dtype):
    h_det = kernel_size + 2
    w_det = kernel_size + 2
    h_os = h_det * s_osf
    w_os = w_det * s_osf
    base_r = h_det // 2
    base_c = w_det // 2
    half = kernel_size // 2
    phase_count = s_osf * phase_oversample

    stamps_r = []
    for phase_r in range(phase_count):
        stamps_c = []
        for phase_c in range(phase_count):
            phase_r_offset = phase_r / float(phase_oversample)
            phase_c_offset = phase_c / float(phase_oversample)
            if phase_oversample > 1:
                phase_r_offset += 0.5 / float(phase_oversample)
                phase_c_offset += 0.5 / float(phase_oversample)
            q_r = tf.cast(base_r * s_osf, tf.float32) + phase_r_offset
            q_c = tf.cast(base_c * s_osf, tf.float32) + phase_c_offset
            delta = add_counts(
                tf.zeros([h_os, w_os], dtype=dtype),
                [q_r],
                [q_c],
                [tf.constant(1.0, dtype=dtype)],
                interpolation='bilinear',
            )
            delta_det = downsample(delta, s_osf, method='block_sum')
            stamp = tf.slice(delta_det, [base_r - half, base_c - half], [kernel_size, kernel_size])
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


def build_epsf_lut(psf_os, s_osf, kernel_size, normalize=False, dtype=tf.float32, phase_oversample=1):
    """Build a phase-indexed detector-space ePSF lookup table.

    The LUT contract matches the existing FFT render path: each phase stamp is
    rendered by placing a unit-flux oversampled delta on an interior base pixel,
    convolving with ``psf_os``, downsampling with detector-pixel block sums, and
    cropping a detector-pixel stamp centered on the base pixel.
    """
    s_osf = int(s_osf)
    kernel_size = _validate_kernel_size(kernel_size)
    phase_oversample = resolve_phase_oversample(s_osf, phase_oversample)
    phase_count = s_osf * phase_oversample
    dtype = tf.as_dtype(dtype)

    if psf_os is None:
        if phase_oversample == 1:
            return _delta_epsf_lut(s_osf, kernel_size, dtype)
        return _build_delta_phase_lut(s_osf, kernel_size, phase_oversample, normalize, dtype)

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
    for phase_r in range(phase_count):
        stamps_c = []
        for phase_c in range(phase_count):
            phase_r_offset = phase_r / float(phase_oversample)
            phase_c_offset = phase_c / float(phase_oversample)
            if phase_oversample > 1:
                phase_r_offset += 0.5 / float(phase_oversample)
                phase_c_offset += 0.5 / float(phase_oversample)
            q_r = tf.cast(base_r * s_osf, tf.float32) + phase_r_offset
            q_c = tf.cast(base_c * s_osf, tf.float32) + phase_c_offset
            delta = add_counts(
                tf.zeros_like(psf_os, dtype=dtype),
                [q_r],
                [q_c],
                [tf.constant(1.0, dtype=dtype)],
                interpolation='floor' if phase_oversample == 1 else 'bilinear',
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


def _epsf_lookup_coords(r_os, c_os, cnt, r_offset_os, c_offset_os, point_rendering, dtype, s_osf, epsf_lut):
    phase_count = tf.shape(epsf_lut)[0]

    if point_rendering == 'phase_nearest':
        with tf.control_dependencies([
            tf.debugging.assert_equal(
                tf.math.floormod(phase_count, s_osf),
                0,
                message='phase_nearest ePSF LUT phase count must be divisible by s_osf',
            ),
        ]):
            phase_oversample = phase_count // s_osf

        r = tf.reshape(tf.cast(r_os, tf.float32), [-1]) + tf.cast(r_offset_os, tf.float32)
        c = tf.reshape(tf.cast(c_os, tf.float32), [-1]) + tf.cast(c_offset_os, tf.float32)
        values = tf.reshape(tf.cast(cnt, dtype), [-1])
        phase_oversample_f = tf.cast(phase_oversample, tf.float32)
        q_r = tf.cast(tf.floor(r * phase_oversample_f), tf.int32)
        q_c = tf.cast(tf.floor(c * phase_oversample_f), tf.int32)
        phase_r = tf.math.floormod(q_r, phase_count)
        phase_c = tf.math.floormod(q_c, phase_count)
        base_r = tf.math.floordiv(q_r, phase_count)
        base_c = tf.math.floordiv(q_c, phase_count)
        return base_r, base_c, phase_r, phase_c, values

    with tf.control_dependencies([
        tf.debugging.assert_equal(
            phase_count,
            s_osf,
            message='floor/bilinear ePSF LUT phase count must equal s_osf',
        ),
    ]):
        rr_os, cc_os, values = _expand_oversampled_points(
            r_os,
            c_os,
            cnt,
            r_offset_os,
            c_offset_os,
            point_rendering,
            dtype,
        )

    phase_r = tf.math.floormod(rr_os, s_osf)
    phase_c = tf.math.floormod(cc_os, s_osf)
    base_r = tf.math.floordiv(rr_os, s_osf)
    base_c = tf.math.floordiv(cc_os, s_osf)
    return base_r, base_c, phase_r, phase_c, values


def add_epsf_counts(
        fpa, r_os, c_os, cnt, epsf_lut, s_osf, r_offset_os=0, c_offset_os=0,
        batch_size=None, point_rendering='bilinear',
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None):
    """Deposit oversampled point sources as detector-space ePSF stamps."""
    return _add_epsf_counts_impl(
        fpa, r_os, c_os, cnt, epsf_lut, s_osf, r_offset_os, c_offset_os,
        batch_size, point_rendering, batch_element_budget, batch_size_cap)


def _add_epsf_counts_impl(
        fpa, r_os, c_os, cnt, epsf_lut, s_osf, r_offset_os=0, c_offset_os=0,
        batch_size=None, point_rendering='bilinear',
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None, parallel_iterations=10):
    """Shared deposition; GPU star graphs bound the number of in-flight batches."""
    point_rendering = _validate_point_rendering(point_rendering)
    s_osf = tf.cast(s_osf, tf.int32)

    dtype = fpa.dtype
    epsf_lut = tf.cast(epsf_lut, dtype)
    kernel_size = tf.shape(epsf_lut)[2]
    half = kernel_size // 2
    # Centers can sit up to half a stamp outside the frame and still deposit
    # visible wing flux, so the internal scatter pad must cover both sides of
    # the stamp around that off-frame center.
    stamp_pad = kernel_size - 1
    fpa_shape = tf.shape(fpa)
    padded_fpa = tf.pad(fpa, [[stamp_pad, stamp_pad], [stamp_pad, stamp_pad]])

    r_os = tf.reshape(tf.cast(r_os, tf.float32), [-1])
    c_os = tf.reshape(tf.cast(c_os, tf.float32), [-1])
    cnt = tf.reshape(tf.cast(cnt, dtype), [-1])
    n = tf.shape(r_os)[0]
    if batch_size_cap is None and batch_size is not None:
        batch_size_cap = batch_size
    batch_size = effective_epsf_batch_size(
        kernel_size,
        point_rendering=point_rendering,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
        temporal_osf=1,
        source_count=n,
    )
    offsets = tf.range(kernel_size, dtype=tf.int32) - half

    def cond(i, image):
        return tf.less(i, n)

    def body(i, image):
        end = tf.minimum(i + batch_size, n)
        base_r, base_c, phase_r, phase_c, values_b = _epsf_lookup_coords(
            r_os[i:end],
            c_os[i:end],
            cnt[i:end],
            r_offset_os,
            c_offset_os,
            point_rendering,
            dtype,
            s_osf,
            epsf_lut,
        )
        intersects = (
            (base_r + half >= 0)
            & (base_r - half < fpa_shape[0])
            & (base_c + half >= 0)
            & (base_c - half < fpa_shape[1])
        )
        base_r = tf.boolean_mask(base_r, intersects)
        base_c = tf.boolean_mask(base_c, intersects)
        phase_r = tf.boolean_mask(phase_r, intersects)
        phase_c = tf.boolean_mask(phase_c, intersects)
        values_b = tf.boolean_mask(values_b, intersects)

        stamps = tf.gather_nd(epsf_lut, tf.stack([phase_r, phase_c], axis=1))
        stamps = stamps * tf.reshape(values_b, [-1, 1, 1])

        rows = tf.reshape(base_r + stamp_pad, [-1, 1, 1]) + tf.reshape(offsets, [1, -1, 1])
        rows = tf.tile(rows, [1, 1, kernel_size])
        cols = tf.reshape(base_c + stamp_pad, [-1, 1, 1]) + tf.reshape(offsets, [1, 1, -1])
        cols = tf.tile(cols, [1, kernel_size, 1])

        shape = tf.shape(image)
        flat_indices = tf.reshape(rows * shape[1] + cols, [-1])
        scatter_values = tf.reshape(stamps, [-1])
        flat_image = tf.reshape(image, [-1])
        # Negative ids would be dropped silently; for odd kernels intersects and stamp_pad keep every id in range.
        flat_image = flat_image + tf.math.unsorted_segment_sum(scatter_values, flat_indices, tf.size(flat_image))
        image = tf.reshape(flat_image, shape)
        return end, image

    _, padded_fpa = tf.while_loop(
        cond, body, (tf.constant(0, tf.int32), padded_fpa), parallel_iterations=parallel_iterations)
    return tf.slice(padded_fpa, [stamp_pad, stamp_pad], fpa_shape)


def _compile_gpu_epsf_deposition():
    """Build a reusable graph without initializing devices or capturing LUT values.

    One loop iteration at a time avoids overlapping large stamp intermediates.
    Shape relaxation reuses graphs as source counts, image sizes and LUT shapes
    change. Motion and user callbacks run outside this function, as before.
    """
    deposit = partial(_add_epsf_counts_impl, parallel_iterations=1)
    try:
        return tf.function(deposit, reduce_retracing=True)
    except TypeError:  # TensorFlow versions before reduce_retracing was available.
        return tf.function(deposit, experimental_relax_shapes=True)


_add_epsf_counts_graph = _compile_gpu_epsf_deposition()


def transform_and_add_epsf(
        fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation,
        epsf_lut, s_osf, batch_size=None, filter_out_of_bounds=True,
        point_rendering='bilinear', epsf_tiers=None,
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None, position_offset_os=0.0, position_transform=None,
        accumulator='auto'):
    """Apply star motion and deposit each star's ePSF at exposure-bin midpoints.

    Every star deposits ``cnt / t_osf`` at each of ``t_osf`` midpoint samples
    through the LUT crop of its total-flux tier. Stars whose rotation and
    translation are statically exactly zero, with no ``position_transform``,
    deposit their full count once. ``filter_out_of_bounds`` drops only sources
    whose stamps cannot reach ``fpa`` (see ``filter_epsf_sources_in_bounds``)
    and is skipped with a ``position_transform``.

    ``accumulator`` selects how the stamps are summed. ``'tensorflow'`` uses
    TensorFlow ops on any device and in graphs. ``'cpu'`` sums in float64 with
    a compiled CPU kernel and returns ``fpa.dtype``; it requires eager
    execution, a float32 or float64 ``fpa`` and every tensor argument already
    on the CPU, and raises ``ValueError`` otherwise rather than copying
    anything. ``'auto'`` uses ``'cpu'`` when those conditions hold and
    ``'tensorflow'`` otherwise. Both give the same samples, phases, weights and
    tiers; only the summation order and precision differ. On the CPU path a
    ``position_transform`` runs under the image's device scope. If a batch of
    its outputs is not on the CPU, ``'cpu'`` raises ``ValueError`` and
    ``'auto'`` deposits that batch and the rest of the render with TensorFlow.
    """
    fpa = tf.convert_to_tensor(fpa)
    if batch_size_cap is None and batch_size is not None:
        batch_size_cap = batch_size
    stars = _plan_star_deposition(
        fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation,
        epsf_lut, s_osf,
        epsf_tiers=epsf_tiers,
        filter_out_of_bounds=filter_out_of_bounds,
        point_rendering=point_rendering,
        position_offset_os=position_offset_os,
        position_transform=position_transform,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
    )
    return _deposit_stars(fpa, stars, accumulator)


@dataclass(frozen=True)
class _StarGroup:
    """Stars that deposit through one optical LUT.

    ``lut`` is the centered crop for one flux tier, or the full LUT when
    tiering is off. Every source's total ``counts`` lies in that tier.
    ``rows`` and ``cols`` are float32 epoch positions in padded-frame
    oversampled pixels. ``source_batch_size`` is the number of sources whose
    samples are transformed and deposited together.
    """
    lut: object
    rows: tf.Tensor
    cols: tf.Tensor
    counts: tf.Tensor
    source_batch_size: tf.Tensor


@dataclass(frozen=True)
class _StarPlan:
    """Every star sample to deposit; built only by ``_plan_star_deposition``.

    ``groups`` partitions the planned (filtered) sources by flux tier.
    ``samples`` is 1 when rotation and translation are statically exactly
    zero and there is no ``position_transform``, and ``t_osf`` otherwise.
    ``time_grid`` holds the ``samples`` exposure-bin midpoints, and each
    sample deposits a source's ``counts / samples``. ``input_devices`` lists
    the devices of the caller's tensor arguments as passed, before any cast or
    copy.
    """
    groups: Tuple[_StarGroup, ...]
    samples: tf.Tensor
    time_grid: tf.Tensor
    rotation: tf.Tensor
    translation: Tuple[tf.Tensor, tf.Tensor]
    frame_extent_os: Tuple[tf.Tensor, tf.Tensor]
    offset_os: Tuple[object, object]
    s_osf: tf.Tensor
    point_rendering: str
    position_transform: Optional[Callable]
    batch_element_budget: Optional[int]
    batch_size_cap: Optional[int]
    input_devices: Tuple[str, ...]


def _plan_star_deposition(
        fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation,
        epsf_lut, s_osf, epsf_tiers=None, filter_out_of_bounds=True,
        point_rendering='bilinear', position_offset_os=0.0, position_transform=None,
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT, batch_size_cap=None):
    """Return the ``_StarPlan`` for ``transform_and_add_epsf`` arguments.

    The only place stars are filtered, collapsed to one sample, split into
    flux tiers and batched. Works in eager and graph mode.
    """
    point_rendering = _validate_point_rendering(point_rendering)
    input_devices = _input_devices(
        fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation, epsf_lut, s_osf, position_offset_os)
    s_osf = tf.cast(s_osf, tf.int32)
    shape = tf.shape(fpa)
    h_minus_1 = tf.cast(shape[0] * s_osf, tf.float32) - 1.0
    w_minus_1 = tf.cast(shape[1] * s_osf, tf.float32) - 1.0
    rotation_f = tf.cast(rotation, tf.float32)
    translation_f = (tf.cast(translation[0], tf.float32), tf.cast(translation[1], tf.float32))
    offset_os = _offset_pair(position_offset_os)

    if filter_out_of_bounds and position_transform is None:
        rows, cols, counts = filter_epsf_sources_in_bounds(
            fpa, r_os, c_os, cnt, t_start, t_end, rotation_f, translation_f, s_osf,
            tf.shape(epsf_lut)[2], position_offset_os=offset_os,
        )
    else:
        rows, cols, counts = [tf.reshape(tf.cast(x, tf.float32), [-1]) for x in (r_os, c_os, cnt)]

    if _is_exact_zero_motion(rotation, translation, position_transform):
        samples = tf.constant(1, tf.int32)
    else:
        samples = tf.cast(t_osf, tf.int32)
    t_start_f = tf.cast(t_start, tf.float32)
    t_duration = tf.cast(t_end, tf.float32) - t_start_f
    time_grid = t_start_f + (tf.cast(tf.range(samples), tf.float32) + 0.5) * (t_duration / tf.cast(samples, tf.float32))

    if epsf_tiers:
        selections = []
        prev = -math.inf
        for tier in epsf_tiers:
            selections.append((crop_epsf_lut(epsf_lut, tier['kernel_size']), _tier_mask(counts, prev, tier['flux_max'])))
            prev = float(tier['flux_max'])
    else:
        selections = [(epsf_lut, None)]

    groups = []
    for lut, mask in selections:
        if mask is None:
            group_rows, group_cols, group_counts = rows, cols, counts
        elif tf.executing_eagerly() and not bool(tf.reduce_any(mask)):
            continue
        else:
            group_rows, group_cols, group_counts = [tf.boolean_mask(x, mask) for x in (rows, cols, counts)]
        source_batch_size = effective_epsf_batch_size(
            tf.shape(lut)[2],
            point_rendering=point_rendering,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
            temporal_osf=samples,
            source_count=tf.shape(group_rows)[0],
        )
        if position_transform is not None:
            source_batch_size = tf.minimum(source_batch_size, tf.maximum(1, _POSITION_TRANSFORM_BATCH_SAMPLES // samples))
        groups.append(_StarGroup(lut, group_rows, group_cols, group_counts, source_batch_size))

    return _StarPlan(
        groups=tuple(groups),
        samples=samples,
        time_grid=time_grid,
        rotation=rotation_f,
        translation=translation_f,
        frame_extent_os=(h_minus_1, w_minus_1),
        offset_os=offset_os,
        s_osf=s_osf,
        point_rendering=point_rendering,
        position_transform=position_transform,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
        input_devices=input_devices,
    )


def _input_devices(*values):
    """Return the device of every tensor in ``values``, walking nested lists and tuples.

    Python scalars, strings, ``None`` and numeric NumPy arrays live on the host
    and add nothing. Any other leaf, such as a sparse tensor or an object
    array, adds an empty device so it never counts as a CPU input.
    """
    devices = []
    for value in tf.nest.flatten(values):
        if isinstance(value, (tf.Tensor, tf.Variable)):
            devices.append(value.device)
        elif not _is_host_value(value):
            devices.append('')
    return tuple(devices)


def _is_host_value(value):
    if value is None or isinstance(value, (bool, int, float, str, np.number, np.bool_)):
        return True
    return isinstance(value, np.ndarray) and value.dtype.kind in 'biuf'


def _is_exact_zero_motion(rotation, translation, position_transform):
    """Return True when no star can move during the exposure.

    Requires rotation and both translation components to be statically
    exactly zero (``-0.0`` included) and no ``position_transform``. A value
    that is not static, such as a dynamic graph tensor, never qualifies.
    """
    if position_transform is not None:
        return False
    statics = [tf.get_static_value(value) for value in (rotation, translation[0], translation[1])]
    return all(static is not None and np.all(np.asarray(static) == 0) for static in statics)


def _deposit_stars(fpa, stars, accumulator):
    """Deposit every sample of the ``_StarPlan`` ``stars`` into ``fpa``.

    The only place the accumulator is chosen. Placement is judged on the
    caller's inputs and on every planned tensor the accumulator reads.
    """
    devices = stars.input_devices + _input_devices(
        stars.time_grid, stars.rotation, stars.translation, stars.frame_extent_os, stars.s_osf,
        [(group.lut, group.rows, group.cols, group.counts) for group in stars.groups])
    if _choose_star_accumulator(accumulator, tf.executing_eagerly(), devices, fpa.dtype) == 'cpu':
        return _accumulate_stars_cpu(fpa, stars, accumulator)
    return _accumulate_stars_tf(fpa, stars)


def _choose_star_accumulator(requested, executing_eagerly, devices, image_dtype):
    """Return ``'cpu'`` or ``'tensorflow'`` for an ``accumulator`` request.

    A render is eligible for ``'cpu'`` only when it executes eagerly, the image
    dtype is float32 or float64, and every device string names a CPU (an empty
    string, as graph tensors report, does not). Whether the host has a GPU is
    never consulted. ``'auto'`` returns ``'cpu'`` exactly when eligible,
    ``'tensorflow'`` always returns ``'tensorflow'``, and ``'cpu'`` raises
    ``ValueError`` when the render is not eligible.
    """
    if requested not in _STAR_ACCUMULATORS:
        raise ValueError("accumulator must be 'auto', 'tensorflow', or 'cpu'")
    if requested == 'tensorflow':
        return 'tensorflow'
    if not executing_eagerly:
        requirement = 'eager execution'
    elif image_dtype not in (tf.float32, tf.float64):
        requirement = 'a float32 or float64 image'
    elif not _on_cpu(devices):
        requirement = 'every render tensor on the CPU'
    else:
        return 'cpu'
    if requested == 'cpu':
        raise ValueError('the cpu ePSF star accumulator requires {}'.format(requirement))
    return 'tensorflow'


def _on_cpu(devices):
    return all(tf.DeviceSpec.from_string(device).device_type == 'CPU' for device in devices)


@dataclass(frozen=True)
class _PendingBatch:
    """A transformed sample batch the CPU accumulator handed to TensorFlow.

    ``rows``, ``cols`` and ``values`` are ``_star_sample_batch`` output for
    sources ``start:end`` of ``groups[group_index]``; ``end`` is where the
    rest of that group resumes.
    """
    group_index: int
    end: tf.Tensor
    rows: tf.Tensor
    cols: tf.Tensor
    values: tf.Tensor


def _accumulate_stars_tf(fpa, stars, pending=None):
    """Deposit ``stars`` with TensorFlow ops; graph-safe on any device.

    With a ``_PendingBatch``, deposit it, then the rest of its group and every
    later group; earlier sources are already in ``fpa``.
    """
    groups = stars.groups
    if pending is not None:
        group = groups[pending.group_index]
        fpa = _add_star_samples(fpa, stars, group, pending.rows, pending.cols, pending.values)
        fpa = _deposit_star_group(fpa, stars, group, start=pending.end)
        groups = groups[pending.group_index + 1:]
    for group in groups:
        fpa = _deposit_star_group(fpa, stars, group)
    return fpa


def _deposit_star_group(fpa, stars, group, start=0):
    n = tf.shape(group.rows)[0]

    def cond(start, image):
        return tf.less(start, n)

    def body(start, image):
        end, rr, cc, vv = _star_sample_batch(stars, group, start)
        return end, _add_star_samples(image, stars, group, rr, cc, vv)

    _, fpa = tf.while_loop(cond, body, (tf.convert_to_tensor(start, tf.int32), fpa))
    return fpa


def _add_star_samples(image, stars, group, rr, cc, vv):
    row_offset_os, col_offset_os = stars.offset_os
    deposit = add_epsf_counts
    # Compile only eager, multi-sample GPU stars. Keep CPU, existing graph and
    # single-sample deposition on their original paths; targets never come here.
    if (tf.executing_eagerly() and stars.time_grid.shape[0] not in (None, 1)
            and tf.DeviceSpec.from_string(image.device).device_type == 'GPU'):
        deposit = _add_epsf_counts_graph
    return deposit(
        image,
        rr,
        cc,
        vv,
        group.lut,
        stars.s_osf,
        r_offset_os=row_offset_os,
        c_offset_os=col_offset_os,
        point_rendering=stars.point_rendering,
        batch_element_budget=stars.batch_element_budget,
        batch_size_cap=stars.batch_size_cap,
    )


def _accumulate_stars_cpu(fpa, stars, requested):
    """Deposit ``stars`` with the compiled float64 kernel; return an image of ``fpa.dtype``.

    Computes the same sample positions, phases, bilinear weights and
    dtype-cast LUT values as ``_accumulate_stars_tf``; only the sum differs.
    Callers must have routed here with ``_choose_star_accumulator``, so every
    input tensor is already on the CPU. ``position_transform`` outputs are
    placed only when it runs, so each batch is checked here: off the CPU,
    ``'auto'`` hands that batch and the partial sum to ``_accumulate_stars_tf``
    without transforming it again, and ``'cpu'`` raises ``ValueError``.
    """
    with tf.device(fpa.device):
        image, pending = _sum_stars_on_cpu(fpa, stars, requested)
        partial = tf.constant(image.astype(fpa.dtype.as_numpy_dtype))
    if pending is None:
        return partial
    return _accumulate_stars_tf(partial, stars, pending)


def _sum_stars_on_cpu(fpa, stars, requested):
    """Return the float64 image sum and the first ``_PendingBatch`` not on the CPU, or ``None``."""
    row_offset_os, col_offset_os = stars.offset_os
    image = np.array(fpa.numpy(), dtype=np.float64)
    for group_index, group in enumerate(stars.groups):
        lut = tf.cast(group.lut, fpa.dtype)
        lut_values = lut.numpy()
        n = tf.shape(group.rows)[0]
        start = 0
        while start < n:
            end, rr, cc, vv = _star_sample_batch(stars, group, start)
            if not _on_cpu(_input_devices(rr, cc)):
                if requested == 'cpu':
                    raise ValueError('the cpu ePSF star accumulator requires position_transform outputs on the CPU')
                return image, _PendingBatch(group_index, end, rr, cc, vv)
            coords = _epsf_lookup_coords(
                rr, cc, vv, row_offset_os, col_offset_os, stars.point_rendering, fpa.dtype, stars.s_osf, lut)
            add_epsf_stamps(image, *(coord.numpy() for coord in coords), lut_values)
            start = end
    return image, None


def _star_sample_batch(stars, group, start):
    """Return ``(end, rows, cols, values)`` for the samples of ``group`` sources ``start:end``.

    Both accumulators walk a group by calling this from ``start = 0`` until
    ``end`` reaches the group's source count, so they deposit the same
    batches. Rows and columns are sample-major positions from
    ``rotate_and_translate`` at each ``time_grid`` midpoint, then
    ``position_transform`` when set; each value is a per-sample count.
    """
    end = tf.minimum(start + group.source_batch_size, tf.shape(group.rows)[0])
    rows = group.rows[start:end]
    cols = group.cols[start:end]
    values = group.counts[start:end] / tf.cast(stars.samples, tf.float32)
    tt = tf.repeat(stars.time_grid, tf.shape(rows)[0])
    rr = tf.tile(rows, [stars.samples])
    cc = tf.tile(cols, [stars.samples])
    vv = tf.tile(values, [stars.samples])
    h_minus_1, w_minus_1 = stars.frame_extent_os
    rr, cc = rotate_and_translate(h_minus_1, w_minus_1, rr, cc, tt, stars.rotation, stars.translation)
    if stars.position_transform is not None:
        rr, cc = stars.position_transform(rr, cc)
    return end, rr, cc, vv


def _warn_deprecated(message):
    warnings.warn(message, DeprecationWarning, stacklevel=3)


def build_trailed_epsf_lut(*args, **kwargs):
    """Removed; ``streak`` stars deposit ``build_epsf_lut`` stamps at exposure-bin midpoints."""
    raise NotImplementedError(
        'build_trailed_epsf_lut was removed; build the optical LUT with build_epsf_lut and deposit '
        'moving stars with transform_and_add_epsf')


def transform_and_add_trailed_epsf(*args, **kwargs):
    """Removed; use ``transform_and_add_epsf``."""
    raise NotImplementedError('transform_and_add_trailed_epsf was removed; use transform_and_add_epsf')


def add_epsf_counts_tiered(*args, **kwargs):
    """Removed; use ``transform_and_add_epsf`` with ``epsf_tiers``."""
    raise NotImplementedError('add_epsf_counts_tiered was removed; use transform_and_add_epsf with epsf_tiers')
