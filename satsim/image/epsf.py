from __future__ import division, print_function, absolute_import

import math

import numpy as np
import tensorflow as tf

from satsim.geometry.transform import rotate_and_translate
from satsim.image.fpa import add_counts, downsample, transform_and_add_counts
from satsim.math import fftconv2p


EPSF_BATCH_ELEMENT_BUDGET_DEFAULT = 32000000
EPSF_BATCH_MIN_DEFAULT = 1024
EPSF_BATCH_MAX_DEFAULT = 262144
EPSF_PHASE_ERROR_TARGET_PX = 0.02


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


def _to_float(value, name):
    static_value = tf.get_static_value(value)
    if static_value is not None:
        return float(static_value)
    if hasattr(value, 'numpy'):
        return float(value.numpy())
    try:
        return float(value)
    except TypeError:
        raise ValueError('{} must be statically known when building an ePSF LUT'.format(name))


def _image_dim_to_int(image, axis, name):
    static_shape = image.shape.as_list()
    if static_shape[axis] is not None:
        return int(static_shape[axis])
    return _to_int(tf.shape(image)[axis], name)


def _match_detector_parity(size, reference_size):
    if size % 2 != reference_size % 2:
        return size + 1
    return size


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


def _pad_to_shape(image, height, width):
    h = _image_dim_to_int(image, 0, 'image height')
    w = _image_dim_to_int(image, 1, 'image width')
    if h > height or w > width:
        raise ValueError('image cannot be padded to a smaller shape')

    pad_h = height - h
    pad_w = width - w
    return tf.pad(
        image,
        [
            [pad_h // 2, pad_h - pad_h // 2],
            [pad_w // 2, pad_w - pad_w // 2],
        ],
    )


def crop_epsf_lut(epsf_lut, crop_size):
    """Return a centered crop of a phase-indexed ePSF LUT."""
    full_size = _kernel_size_from_lut(epsf_lut)
    crop_size = _validate_crop_size(crop_size, full_size)
    offset = (full_size - crop_size) // 2
    return epsf_lut[:, :, offset:offset + crop_size, offset:offset + crop_size]


def epsf_wing_profile(epsf_lut, sizes):
    """Measure cropped-wing amplitude and energy loss for candidate sizes.

    Values are measured on the active LUT, so the profile applies equally to
    static and trailed ePSFs.
    """
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


def _crop_size_for_trail(core_size, trail_det, full_size):
    crop_size = _make_odd(int(core_size) + int(trail_det))
    if crop_size > full_size:
        crop_size = full_size
    if (full_size - crop_size) % 2 != 0:
        crop_size = min(full_size, crop_size + 1)
    return _validate_crop_size(crop_size, full_size)


def resolve_epsf_tiers(epsf_lut, crop_config=None, trail_det=0):
    """Resolve an ePSF crop config into sorted flux tiers.

    Each tier has an inclusive ``flux_max`` in total source photoelectrons and
    a centered detector-space ``kernel_size``. ``None`` means tiering is off.
    """
    if not crop_config or crop_config.get('mode', 'off') == 'off':
        return None

    full_size = _kernel_size_from_lut(epsf_lut)
    mode = str(crop_config.get('mode', 'off')).lower()
    if mode not in ('manual', 'auto'):
        raise ValueError("sim.epsf.crop.mode must be 'off', 'manual', or 'auto'")

    tiers = []
    trail_det = int(trail_det)

    if mode == 'manual':
        threshold_pe = float(crop_config['threshold_pe'])
        core_size = int(crop_config['kernel_size'])
        crop_size = _crop_size_for_trail(core_size, trail_det, full_size)
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
            crop_size = _crop_size_for_trail(core_size, trail_det, full_size)
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


def filter_epsf_sources_in_bounds(fpa, r_os, c_os, cnt, t_start, t_end, rotation, translation, s_osf):
    """Filter sources whose start and end positions are both outside the frame."""
    s_osf = tf.cast(s_osf, tf.int32)
    shape = tf.shape(fpa)
    h_os = tf.cast(shape[0] * s_osf, tf.float32)
    w_os = tf.cast(shape[1] * s_osf, tf.float32)
    h_minus_1 = h_os - 1.0
    w_minus_1 = w_os - 1.0

    r = tf.cast(r_os, tf.float32)
    c = tf.cast(c_os, tf.float32)
    cnt = tf.cast(cnt, tf.float32)

    rr0_tmp, cc0_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
    rr1_tmp, cc1_tmp = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_end, rotation, translation)
    rr_lt_0 = tf.math.logical_and(rr0_tmp < 0, rr1_tmp < 0)
    rr_gt_h = tf.math.logical_and(rr0_tmp > h_os, rr1_tmp > h_os)
    cc_lt_0 = tf.math.logical_and(cc0_tmp < 0, cc1_tmp < 0)
    cc_gt_w = tf.math.logical_and(cc0_tmp > w_os, cc1_tmp > w_os)
    out_of_bounds = tf.math.logical_or(
        tf.math.logical_or(rr_lt_0, rr_gt_h),
        tf.math.logical_or(cc_lt_0, cc_gt_w),
    )
    in_bounds = tf.math.logical_not(out_of_bounds)

    return tf.boolean_mask(r, in_bounds), tf.boolean_mask(c, in_bounds), tf.boolean_mask(cnt, in_bounds)


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


def build_trailed_epsf_lut(psf_os, s_osf, kernel_size, t_start, t_end, t_osf,
                           rotation, translation, normalize=False,
                           point_rendering='bilinear', dtype=tf.float32,
                           max_kernel_size=4095, return_info=False,
                           phase_oversample=1):
    """Build a phase-indexed ePSF LUT with shared star motion baked in.

    The trail component mirrors ``transform_and_fft``: a unit point is placed
    at the field center, transformed over the exposure, convolved with the
    optical PSF, and then passed through the regular ePSF LUT builder.
    """
    s_osf = int(s_osf)
    kernel_size = _validate_kernel_size(kernel_size)
    point_rendering = _validate_point_rendering(point_rendering)
    phase_oversample = resolve_phase_oversample(s_osf, phase_oversample)
    dtype = tf.as_dtype(dtype)

    t_osf = _to_int(t_osf, 't_osf')
    if t_osf <= 0:
        raise ValueError('t_osf must be a positive integer')

    t_start_f = _to_float(t_start, 't_start')
    t_end_f = _to_float(t_end, 't_end')
    t_duration = t_end_f - t_start_f
    translation_r = _to_float(translation[0], 'translation[0]')
    translation_c = _to_float(translation[1], 'translation[1]')

    trail_r_os = abs(translation_r * t_duration)
    trail_c_os = abs(translation_c * t_duration)
    trail_r_det = int(math.ceil(trail_r_os / float(s_osf)))
    trail_c_det = int(math.ceil(trail_c_os / float(s_osf)))
    trail_det = max(trail_r_det, trail_c_det)

    effective_kernel = _make_odd(kernel_size + trail_det)
    if (
        trail_det == 0
        and _to_float(rotation, 'rotation') == 0.0
    ):
        result = (
            build_epsf_lut(
                psf_os,
                s_osf,
                kernel_size,
                normalize=normalize,
                dtype=dtype,
                phase_oversample=phase_oversample,
            ),
            effective_kernel,
        )
        if return_info:
            return result + ({'trail_det': trail_det},)
        return result
    if max_kernel_size is not None:
        max_kernel_size = _to_int(max_kernel_size, 'max_kernel_size')
    if max_kernel_size is not None and effective_kernel > max_kernel_size:
        raise ValueError(
            'trailed ePSF effective kernel_size {} exceeds the maximum {}; '
            'reduce exposure/rate or use sim.mode "fftconv2p"'.format(
                effective_kernel,
                max_kernel_size,
            )
        )

    if psf_os is None:
        psf_det_h = kernel_size
        psf_det_w = kernel_size
        padded_psf_os = None
    else:
        psf_os = tf.cast(psf_os, dtype)
        psf_h_os = _image_dim_to_int(psf_os, 0, 'psf_os height')
        psf_w_os = _image_dim_to_int(psf_os, 1, 'psf_os width')
        if psf_h_os % s_osf != 0 or psf_w_os % s_osf != 0:
            raise ValueError('psf_os dimensions must be divisible by s_osf')
        psf_det_h = psf_h_os // s_osf
        psf_det_w = psf_w_os // s_osf
        padded_psf_os = psf_os

    field_h_det = max(psf_det_h + trail_r_det + 2, effective_kernel)
    field_w_det = max(psf_det_w + trail_c_det + 2, effective_kernel)
    field_h_det = _match_detector_parity(field_h_det, psf_det_h)
    field_w_det = _match_detector_parity(field_w_det, psf_det_w)
    field_h_os = field_h_det * s_osf
    field_w_os = field_w_det * s_osf

    if padded_psf_os is not None:
        padded_psf_os = _pad_to_shape(padded_psf_os, field_h_os, field_w_os)

    h_mid = (field_h_os - 1.0) / 2.0
    w_mid = (field_w_os - 1.0) / 2.0
    build_point_rendering = 'bilinear' if point_rendering == 'phase_nearest' else point_rendering
    if build_point_rendering == 'bilinear':
        h_mid = math.floor(h_mid)
        w_mid = math.floor(w_mid)

    trail_field = transform_and_add_counts(
        tf.zeros([field_h_os, field_w_os], dtype),
        [h_mid],
        [w_mid],
        [1.0],
        0.0,
        t_duration,
        t_osf,
        rotation,
        translation,
        interpolation=build_point_rendering,
    )

    if padded_psf_os is None:
        trailed_psf_os = trail_field
    else:
        trailed_psf_os = fftconv2p(trail_field, padded_psf_os, pad=1)

    result = (
        build_epsf_lut(
            trailed_psf_os,
            s_osf,
            effective_kernel,
            normalize=normalize,
            dtype=dtype,
            phase_oversample=phase_oversample,
        ),
        effective_kernel,
    )
    if return_info:
        return result + ({'trail_det': trail_det},)
    return result


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
        flat_image = tf.tensor_scatter_nd_add(flat_image, tf.reshape(flat_indices, [-1, 1]), scatter_values)
        image = tf.reshape(flat_image, shape)
        return end, image

    _, padded_fpa = tf.while_loop(cond, body, (tf.constant(0, tf.int32), padded_fpa))
    return tf.slice(padded_fpa, [stamp_pad, stamp_pad], fpa_shape)


def add_epsf_counts_tiered(
        fpa, r_os, c_os, cnt, epsf_lut, s_osf, tiers, r_offset_os=0,
        c_offset_os=0, batch_size=None, point_rendering='bilinear',
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None):
    """Deposit sources using center-cropped ePSF LUTs selected by total flux."""
    if not tiers:
        return add_epsf_counts(
            fpa,
            r_os,
            c_os,
            cnt,
            epsf_lut,
            s_osf,
            r_offset_os=r_offset_os,
            c_offset_os=c_offset_os,
            batch_size=batch_size,
            point_rendering=point_rendering,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
        )

    r = tf.reshape(tf.cast(r_os, tf.float32), [-1])
    c = tf.reshape(tf.cast(c_os, tf.float32), [-1])
    cnt = tf.reshape(tf.cast(cnt, tf.float32), [-1])
    prev = -math.inf
    for tier in tiers:
        mask = _tier_mask(cnt, prev, tier['flux_max'])
        if hasattr(mask, 'numpy') and not np.any(mask.numpy()):
            prev = float(tier['flux_max'])
            continue

        tier_lut = crop_epsf_lut(epsf_lut, tier['kernel_size'])
        fpa = add_epsf_counts(
            fpa,
            tf.boolean_mask(r, mask),
            tf.boolean_mask(c, mask),
            tf.boolean_mask(cnt, mask),
            tier_lut,
            s_osf,
            r_offset_os=r_offset_os,
            c_offset_os=c_offset_os,
            batch_size=batch_size,
            point_rendering=point_rendering,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
        )
        prev = float(tier['flux_max'])

    return fpa


def transform_and_add_trailed_epsf(fpa, r_os, c_os, cnt, t_start, t_end,
                                   rotation, translation, epsf_lut, s_osf,
                                   batch_size=None, filter_out_of_bounds=True,
                                   point_rendering='bilinear', epsf_tiers=None,
                                   batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
                                   batch_size_cap=None):
    """Apply FFT-style shared star motion and deposit one trailed ePSF per star."""
    point_rendering = _validate_point_rendering(point_rendering)
    s_osf = tf.cast(s_osf, tf.int32)

    shape = tf.shape(fpa)
    h_os = tf.cast(shape[0] * s_osf, tf.float32)
    w_os = tf.cast(shape[1] * s_osf, tf.float32)
    h_minus_1 = h_os - 1.0
    w_minus_1 = w_os - 1.0

    if filter_out_of_bounds:
        r, c, cnt = filter_epsf_sources_in_bounds(
            fpa,
            r_os,
            c_os,
            cnt,
            t_start,
            t_end,
            rotation,
            translation,
            s_osf,
        )
    else:
        r = tf.cast(r_os, tf.float32)
        c = tf.cast(c_os, tf.float32)
        cnt = tf.cast(cnt, tf.float32)

    rr, cc = rotate_and_translate(h_minus_1, w_minus_1, r, c, t_start, rotation, translation)
    if epsf_tiers:
        return add_epsf_counts_tiered(
            fpa,
            rr,
            cc,
            cnt,
            epsf_lut,
            s_osf,
            epsf_tiers,
            batch_size=batch_size,
            point_rendering=point_rendering,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
        )

    return add_epsf_counts(
        fpa,
        rr,
        cc,
        cnt,
        epsf_lut,
        s_osf,
        batch_size=batch_size,
        point_rendering=point_rendering,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
    )


def transform_and_add_epsf(
        fpa, r_os, c_os, cnt, t_start, t_end, t_osf, rotation, translation,
        epsf_lut, s_osf, batch_size=None, filter_out_of_bounds=True,
        point_rendering='bilinear', epsf_tiers=None,
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None):
    """Apply star motion and deposit transformed samples as ePSF stamps."""
    point_rendering = _validate_point_rendering(point_rendering)
    s_osf = tf.cast(s_osf, tf.int32)

    shape = tf.shape(fpa)
    h_os = tf.cast(shape[0] * s_osf, tf.float32)
    w_os = tf.cast(shape[1] * s_osf, tf.float32)
    h_minus_1 = h_os - 1.0
    w_minus_1 = w_os - 1.0

    t_osf = tf.cast(t_osf, tf.int32)

    if filter_out_of_bounds:
        r, c, cnt_total = filter_epsf_sources_in_bounds(
            fpa,
            r_os,
            c_os,
            cnt,
            t_start,
            t_end,
            rotation,
            translation,
            s_osf,
        )
    else:
        r = tf.cast(r_os, tf.float32)
        c = tf.cast(c_os, tf.float32)
        cnt_total = tf.cast(cnt, tf.float32)

    cnt_os = cnt_total / tf.cast(t_osf, tf.float32)

    if epsf_tiers:
        prev = -math.inf
        for tier in epsf_tiers:
            mask = _tier_mask(cnt_total, prev, tier['flux_max'])
            if hasattr(mask, 'numpy') and not np.any(mask.numpy()):
                prev = float(tier['flux_max'])
                continue
            fpa = transform_and_add_epsf(
                fpa,
                tf.boolean_mask(r, mask),
                tf.boolean_mask(c, mask),
                tf.boolean_mask(cnt_total, mask),
                t_start,
                t_end,
                t_osf,
                rotation,
                translation,
                crop_epsf_lut(epsf_lut, tier['kernel_size']),
                s_osf,
                batch_size=batch_size,
                filter_out_of_bounds=False,
                point_rendering=point_rendering,
                epsf_tiers=None,
                batch_element_budget=batch_element_budget,
                batch_size_cap=batch_size_cap,
            )
            prev = float(tier['flux_max'])
        return fpa

    n = tf.shape(r)[0]
    kernel_size = tf.shape(epsf_lut)[2]
    if batch_size_cap is None and batch_size is not None:
        batch_size_cap = batch_size
    source_batch_size = effective_epsf_batch_size(
        kernel_size,
        point_rendering=point_rendering,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
        temporal_osf=t_osf,
        source_count=n,
    )
    deposit_batch_size = effective_epsf_batch_size(
        kernel_size,
        point_rendering=point_rendering,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
        temporal_osf=1,
    )
    t_start_f = tf.cast(t_start, tf.float32)
    t_duration = tf.cast(t_end, tf.float32) - t_start_f
    t_osf_f = tf.cast(t_osf, tf.float32)
    time_grid = t_start_f + (tf.cast(tf.range(t_osf), tf.float32) + 0.5) * (t_duration / t_osf_f)

    def cond(i, image):
        return tf.less(i, n)

    def body(i, image):
        end = tf.minimum(i + source_batch_size, n)
        r_batch = r[i:end]
        c_batch = c[i:end]
        cnt_batch = cnt_os[i:end]
        current_batch = tf.shape(r_batch)[0]

        tt = tf.repeat(time_grid, current_batch)
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
            batch_size=deposit_batch_size,
            point_rendering=point_rendering,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
        )
        return end, image

    _, fpa = tf.while_loop(cond, body, (tf.constant(0, tf.int32), fpa))
    return fpa
