from __future__ import division, print_function, absolute_import

import math
import logging

import tensorflow as tf
import numpy as np

from satsim.math import fftconv2p
from satsim.image.fpa import downsample, crop, add_counts, transform_and_add_counts, transform_and_fft
from satsim.image.epsf import (
    add_epsf_counts,
    build_trailed_epsf_lut,
    EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
    filter_epsf_sources_in_bounds,
    resolve_epsf_tiers,
    summarize_epsf_tiers,
    transform_and_add_epsf,
    transform_and_add_trailed_epsf,
)

logger = logging.getLogger(__name__)


def _apply_psf_support_center_correction(position, offset=-0.5):
    """Apply the PSF-support correction to a pixel-center position.

    FFT convolution and ePSF lookup tables built from even-sized PSF support
    place their center half an oversampled pixel above the supplied coordinate;
    odd support is already centered. Applying the correction at the deposit
    boundary makes both cases follow the canonical pixel-center convention.
    """
    return tf.cast(position, tf.float32) + tf.cast(offset, tf.float32)


def _psf_deposit_offset(psf_os, axis, default=-0.5):
    """Return the center correction for one PSF support axis."""
    if psf_os is None:
        return default
    size = psf_os.shape[axis]
    if size is not None:
        return -0.5 if int(size) % 2 == 0 else 0.0
    dynamic_size = tf.shape(psf_os)[axis]
    return tf.where(tf.equal(tf.math.floormod(dynamic_size, 2), 0), -0.5, 0.0)


def normalize_star_render_mode(star_render_mode):
    """Normalize supported star rendering modes.

    ``fft`` is kept as a compatibility alias for the shared-streak renderer.
    """
    star_render_mode = str(star_render_mode).lower()
    if star_render_mode == 'fft':
        return 'streak'
    if star_render_mode not in ('transform', 'streak'):
        raise ValueError("star_render_mode must be 'transform', 'streak', or legacy alias 'fft'")
    return star_render_mode


def render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=True, star_render_mode='transform', point_rendering='bilinear'):
    """ Render an image in sub sections. Useful if target supersample image does not fit in GPU memory.

    Args:
        h: `int`, image height in number of real pixels.
        w: `int`, image width in number of real pixels.
        h_sub: `int`, sub section height to render in number of real pixels.
        w_sub: `int`, sub section width to render in number of real pixels.
        h_pad_os: `int`, total vertical pad in oversample space.
        w_pad_os: `int`, total horizontal pad in oversample space.
        s_osf: `int`, spacial oversample factor.
        psf_os: `image`, point spread function with size of `h_sub` by `w_sub` plus padding in oversampled space.
        r_obs_os: `int`, list of target row coordinates in oversampled space.
        c_obs_os: `int`, list of target column coordinates in oversampled space.
        pe_obs_os: `int`, list of target brightnesses in photoelectrons per pixel.
        r_stars_os: `list`, list of star row coordinates in oversampled space at epoch.
        c_stars_os: `list`, list of star column coordinates in oversampled space at epoch.
        pe_stars_os: `list`, list of star brightnesses in photoelectrons per second.
        t_start_star: `float`, start time in seconds from epoch.
        t_end_star: `float`, end time in seconds from epoch.
        t_osf: `int`, temporal oversample factor for star transformation.
        star_rot_rate: `float`, star rotation rate in degrees per second.
        star_tran_os: `[float, float]`, star translation rate in oversampled pixel per second (row, col).
        render_separate: `boolean`, if `True` render targets and stars seperately, required to calculate SNR.
        star_render_mode: `string`, star render mode. `streak` or `transform`. default=transform
        point_rendering: `string`, point source rendering mode. `bilinear`
            preserves sub-pixel centroids; `floor` preserves legacy integer
            deposition.

    Returns:
        A `tuple`, containing:
            fpa_conv_star: `image`, rendered image with stars. Stars and targets if `render_separate` is `False`.
            fpa_conv_targ: `image`, rendered image with targets. Zeros if `render_separate` is `False`.
            _: `None`, to make output compatible with `render_full`
            _: `None`, to make output compatible with `render_full`
            _: `None`, to make output compatible with `render_full`
    """
    star_render_mode = normalize_star_render_mode(star_render_mode)

    # calculate subsection render dimensions
    h_fpa_os = tf.cast(h * s_osf, tf.int32)
    w_fpa_os = tf.cast(w * s_osf, tf.int32)
    h_sub_os = tf.cast(h_sub * s_osf, tf.int32)
    w_sub_os = tf.cast(w_sub * s_osf, tf.int32)
    h_sub_os_f32 = tf.cast(h_sub_os, tf.float32)
    w_sub_os_f32 = tf.cast(w_sub_os, tf.float32)
    h_sub_pad_os = tf.cast(h_sub_os + h_pad_os, tf.int32)
    w_sub_pad_os = tf.cast(w_sub_os + w_pad_os, tf.int32)
    h_pad_os_div2 = tf.cast(h_pad_os / 2, tf.int32)
    w_pad_os_div2 = tf.cast(w_pad_os / 2, tf.int32)

    # calculate the number of row and column divisions
    n_h_div = math.ceil(h_fpa_os / h_sub_os)
    n_w_div = math.ceil(w_fpa_os / w_sub_os)

    # numpy arrays to hold piecewise renders #TODO convert to TensorFlow
    fpa_conv_star = np.zeros((n_h_div, n_w_div, h_sub, w_sub))
    fpa_conv_targ = np.zeros((n_h_div, n_w_div, h_sub, w_sub))

    logger.debug('Rendering {}x{} divisions with {}x{} pixels.'.format(n_h_div, n_w_div, h_sub, w_sub))

    # render #TODO convert to TensorFlow
    for ir in range(n_h_div):
        logger.debug('Rendering row {} of {}.'.format(ir + 1, n_h_div))
        ir_f32 = tf.cast(ir, tf.float32)
        for ic in range(n_w_div):
            ic_f32 = tf.cast(ic, tf.float32)
            r_sub_os_start = ir_f32 * h_sub_os_f32
            c_sub_os_start = ic_f32 * w_sub_os_f32

            # shift stars and obs
            r_stars_sub = r_stars_os - r_sub_os_start
            c_stars_sub = c_stars_os - c_sub_os_start
            pe_stars_sub = pe_stars_os

            r_obs_sub = r_obs_os - r_sub_os_start
            c_obs_sub = c_obs_os - c_sub_os_start
            pe_obs_sub = pe_obs_os

            fpa_conv_star[ir][ic], fpa_conv_targ[ir][ic], _, _, _ = render_full(h_sub_os, w_sub_os, h_sub_pad_os, w_sub_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_sub, c_obs_sub, pe_obs_sub, r_stars_sub, c_stars_sub, pe_stars_sub, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, star_render_mode=star_render_mode, point_rendering=point_rendering)

    fpa_conv_star_stitch = tf.cast(fpa_conv_star, tf.float32)
    fpa_conv_targ_stitch = tf.cast(fpa_conv_targ, tf.float32)
    fpa_conv_star_stitch = tf.transpose(fpa_conv_star_stitch, perm=[0, 2, 1, 3])
    fpa_conv_targ_stitch = tf.transpose(fpa_conv_targ_stitch, perm=[0, 2, 1, 3])
    fpa_conv_star_stitch = tf.reshape(fpa_conv_star_stitch, (h_sub * n_h_div, w_sub * n_w_div))
    fpa_conv_targ_stitch = tf.reshape(fpa_conv_targ_stitch, (h_sub * n_h_div, w_sub * n_w_div))

    # crop #TODO convert to TensorFlow
    fpa_conv_star = tf.cast(fpa_conv_star_stitch[0:h,0:w], tf.float32)
    fpa_conv_targ = tf.cast(fpa_conv_targ_stitch[0:h,0:w], tf.float32)

    return fpa_conv_star, fpa_conv_targ, None, None, None


def render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=True, obs_model=None, star_render_mode='transform', point_rendering='bilinear'):
    """ Render an image.

    Args:
        h_fpa_os: `int`, image height in oversampled space.
        w_fpa_os: `int`, image width in oversampled space.
        h_fpa_pad_os: `int`, image height with pad in oversample space.
        w_fpa_pad_os: `int`, image width with pad in oversample space.
        h_pad_os_div2: `int`, image height with pad divided by 2 in oversample space.
        w_pad_os_div2: `int`, image width with pad divided by 2 in oversample space.
        s_osf: `int`, spacial oversample factor.
        psf_os: `image`, point spread function with size of image with pad in oversampled space.
        r_obs_os: `int`, list of target row coordinates in oversampled space.
        c_obs_os: `int`, list of target column coordinates in oversampled space.
        pe_obs_os: `int`, list of target brightnesses in photoelectrons per pixel.
        r_stars_os: `list`, list of star row coordinates in oversampled space at epoch.
        c_stars_os: `list`, list of star column coordinates in oversampled space at epoch.
        pe_stars_os: `list`, list of star brightnesses in photoelectrons per second.
        t_start_star: `float`, start time in seconds from epoch.
        t_end_star: `float`, end time in seconds from epoch.
        t_osf: `int`, temporal oversample factor for star transformation.
        star_rot_rate: `float`, star rotation rate in degrees per second.
        star_tran_os: `[float, float]`, star translation rate in oversampled pixel per second (row, col).
        render_separate: `boolean`, if `True` render targets and stars seperately, required to calculate SNR.
        obs_model: `list`, list of image arrays in photoelectrons. each array should be the size
            `h_fpa_pad_os` by `w_fpa_pad_os` and is simply added into the image. default=None
        star_render_mode: `string`, star render mode. `streak` or `transform`. default=transform
        point_rendering: `string`, point source rendering mode. `bilinear`
            preserves sub-pixel centroids; `floor` preserves legacy integer
            deposition.

    Returns:
        A `tuple`, containing:
            fpa_conv_star: `image`, rendered image with stars. Stars and targets if `render_separate` is `False`.
            fpa_conv_targ: `image`, rendered image with targets. Zeros if `render_separate` is `False`.
            fpa_os_w_targets: `image`, oversample image with targets before blur.
            fpa_conv_os: `None`, rendered oversample image with pad.
            fpa_conv_crop: `None`, rendered oversample image with no pad.

    """
    # render stars
    star_render_mode = normalize_star_render_mode(star_render_mode)
    fpa_os_w_stars = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
    point_rendering = str(point_rendering).lower()
    if point_rendering not in ('floor', 'bilinear'):
        raise ValueError("point_rendering must be 'floor' or 'bilinear' for FFT rendering")
    downsample_method = 'block_sum' if point_rendering == 'bilinear' else 'pool'

    # The unblurred block-sum path already follows the canonical convention.
    # FFT convolution with even PSF support carries the historical
    # half-subpixel deposit offset; odd support is already centered.
    star_position_offset = (0.0, 0.0)
    if psf_os is not None:
        row_offset = _psf_deposit_offset(psf_os, 0)
        col_offset = _psf_deposit_offset(psf_os, 1)
        r_obs_os = _apply_psf_support_center_correction(r_obs_os, row_offset)
        c_obs_os = _apply_psf_support_center_correction(c_obs_os, col_offset)
        star_position_offset = (row_offset, col_offset)

    if star_render_mode == 'streak':
        fpa_os_w_stars = transform_and_fft(fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, interpolation=point_rendering, position_offset=star_position_offset)
    else:
        fpa_os_w_stars = transform_and_add_counts(fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, interpolation=point_rendering, position_offset=star_position_offset)

    # render modeled targets
    fpa_os_w_targets = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
    if obs_model is not None and len(obs_model) > 0:
        for om in obs_model:
            fpa_os_w_targets = fpa_os_w_targets + tf.cast(om, tf.float32)

        # mask stars (occultation)
        condition = tf.math.greater(fpa_os_w_targets, tf.zeros_like(fpa_os_w_targets, tf.float32))
        fpa_os_w_stars = tf.where(condition, tf.zeros_like(fpa_os_w_stars, tf.float32), fpa_os_w_stars)

    # render point source targets
    fpa_os_w_targets = add_counts(fpa_os_w_targets, r_obs_os, c_obs_os, pe_obs_os, h_pad_os_div2, w_pad_os_div2, interpolation=point_rendering)

    if psf_os is None:
        if render_separate:
            fpa_conv_crop = crop(fpa_os_w_stars, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
            fpa_conv_star = downsample(fpa_conv_crop, s_osf, downsample_method)

            fpa_conv_crop_targ = crop(fpa_os_w_targets, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
            fpa_conv_targ = downsample(fpa_conv_crop_targ, s_osf, downsample_method)

            return fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, None, fpa_conv_crop

        fpa_os_sum = fpa_os_w_stars + fpa_os_w_targets
        fpa_conv_crop = crop(fpa_os_sum, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_ds = downsample(fpa_conv_crop, s_osf, downsample_method)

        return fpa_conv_ds, tf.zeros_like(fpa_conv_ds), fpa_os_w_targets, None, fpa_conv_crop

    if render_separate:

        # blur stars
        fpa_conv_os = fftconv2p(fpa_os_w_stars, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_star = downsample(fpa_conv_crop, s_osf, downsample_method)

        # blur targets
        fpa_conv_os = fftconv2p(fpa_os_w_targets, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_targ = downsample(fpa_conv_crop, s_osf, downsample_method)

        return fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop

    else:

        # blur targets and stars together
        fpa_os_sum = fpa_os_w_stars + fpa_os_w_targets
        if(tf.reduce_all(tf.equal(fpa_os_sum, 0))):
            fpa_conv_os = fpa_os_sum
        else:
            fpa_conv_os = fftconv2p(fpa_os_sum, psf_os, pad=1)

        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_ds = downsample(fpa_conv_crop, s_osf, downsample_method)

        return fpa_conv_ds, tf.zeros_like(fpa_conv_ds), fpa_os_w_targets, fpa_conv_os, fpa_conv_crop


def render_epsf(
        h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os,
        h_pad_os_div2, w_pad_os_div2, s_osf, epsf_lut,
        r_obs_os, c_obs_os, pe_obs_os,
        r_stars_os, c_stars_os, pe_stars_os,
        t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os,
        render_separate=True, obs_model=None, star_render_mode='transform',
        point_rendering='bilinear', batch_size=None,
        batch_element_budget=EPSF_BATCH_ELEMENT_BUDGET_DEFAULT,
        batch_size_cap=None, phase_oversample=1,
        fallback_to_fft_for_models=False, psf_os=None, epsf_normalize=False,
        epsf_crop=None, epsf_metadata=None):
    """Render directly in detector space using an ePSF lookup table.

    Pass the ``psf_os`` used to build ``epsf_lut`` so source registration can
    account for odd versus even PSF support. If it is omitted, even PSF
    support is assumed.
    """
    point_rendering = str(point_rendering).lower()
    if point_rendering not in ('floor', 'bilinear', 'phase_nearest'):
        raise ValueError("point_rendering must be 'floor', 'bilinear', or 'phase_nearest'")
    if batch_size_cap is None and batch_size is not None:
        batch_size_cap = batch_size

    star_render_mode = normalize_star_render_mode(star_render_mode)
    unsupported_model = obs_model is not None and len(obs_model) > 0
    if unsupported_model:
        if fallback_to_fft_for_models:
            if point_rendering == 'phase_nearest':
                raise ValueError('point_rendering="phase_nearest" is only supported by native ePSF paths and cannot use FFT fallback')
            return render_full(
                h_fpa_os,
                w_fpa_os,
                h_fpa_pad_os,
                w_fpa_pad_os,
                h_pad_os_div2,
                w_pad_os_div2,
                s_osf,
                psf_os,
                r_obs_os,
                c_obs_os,
                pe_obs_os,
                r_stars_os,
                c_stars_os,
                pe_stars_os,
                t_start_star,
                t_end_star,
                t_osf,
                star_rot_rate,
                star_tran_os,
                render_separate=render_separate,
                obs_model=obs_model,
                star_render_mode=star_render_mode,
                point_rendering=point_rendering,
            )

        raise NotImplementedError('sim.mode="epsf" does not support obs_model targets without fallback_to_fft_for_models')

    # ePSF LUTs share the even-support center offset of FFT convolution.
    # Apply the correction once at the native renderer boundary.
    row_offset = _psf_deposit_offset(psf_os, 0)
    col_offset = _psf_deposit_offset(psf_os, 1)
    r_obs_os = _apply_psf_support_center_correction(r_obs_os, row_offset)
    c_obs_os = _apply_psf_support_center_correction(c_obs_os, col_offset)

    s_osf_i = tf.cast(s_osf, tf.int32)
    h_fpa_os_i = tf.cast(h_fpa_os, tf.int32)
    w_fpa_os_i = tf.cast(w_fpa_os, tf.int32)
    h_fpa_pad_os_i = tf.cast(h_fpa_pad_os, tf.int32)
    w_fpa_pad_os_i = tf.cast(w_fpa_pad_os, tf.int32)
    h_pad_os_div2_i = tf.cast(h_pad_os_div2, tf.int32)
    w_pad_os_div2_i = tf.cast(w_pad_os_div2, tf.int32)

    with tf.control_dependencies([
        tf.debugging.assert_equal(tf.math.floormod(h_fpa_os_i, s_osf_i), 0, message='h_fpa_os must be divisible by s_osf'),
        tf.debugging.assert_equal(tf.math.floormod(w_fpa_os_i, s_osf_i), 0, message='w_fpa_os must be divisible by s_osf'),
        tf.debugging.assert_equal(tf.math.floormod(h_fpa_pad_os_i, s_osf_i), 0, message='h_fpa_pad_os must be divisible by s_osf'),
        tf.debugging.assert_equal(tf.math.floormod(w_fpa_pad_os_i, s_osf_i), 0, message='w_fpa_pad_os must be divisible by s_osf'),
        tf.debugging.assert_equal(tf.math.floormod(h_pad_os_div2_i, s_osf_i), 0, message='h_pad_os_div2 must be divisible by s_osf'),
        tf.debugging.assert_equal(tf.math.floormod(w_pad_os_div2_i, s_osf_i), 0, message='w_pad_os_div2 must be divisible by s_osf'),
    ]):
        h_det = h_fpa_os_i // s_osf_i
        w_det = w_fpa_os_i // s_osf_i
        h_pad_det = h_fpa_pad_os_i // s_osf_i
        w_pad_det = w_fpa_pad_os_i // s_osf_i
        h_pad_det_div2 = h_pad_os_div2_i // s_osf_i
        w_pad_det_div2 = w_pad_os_div2_i // s_osf_i

    fpa_star = tf.zeros([h_pad_det, w_pad_det], tf.float32)
    if star_render_mode == 'streak':
        base_kernel_size = epsf_lut.shape.as_list()[2]
        if base_kernel_size is None:
            base_kernel_size = int(tf.shape(epsf_lut)[2].numpy())
        trailed_epsf_lut, _, trailed_info = build_trailed_epsf_lut(
            psf_os,
            s_osf,
            base_kernel_size,
            t_start_star,
            t_end_star,
            t_osf,
            star_rot_rate,
            star_tran_os,
            normalize=epsf_normalize,
            point_rendering=point_rendering,
            dtype=tf.float32,
            max_kernel_size=tf.minimum(h_pad_det, w_pad_det),
            return_info=True,
            phase_oversample=phase_oversample,
        )
        epsf_tiers = resolve_epsf_tiers(
            trailed_epsf_lut,
            epsf_crop,
            trail_det=trailed_info.get('trail_det', 0),
        )
        if epsf_metadata is not None and epsf_tiers is not None:
            _, _, visible_pe_stars_os = filter_epsf_sources_in_bounds(
                fpa_star,
                r_stars_os,
                c_stars_os,
                pe_stars_os,
                t_start_star,
                t_end_star,
                star_rot_rate,
                star_tran_os,
                s_osf,
            )
            epsf_metadata.update(summarize_epsf_tiers(epsf_tiers, visible_pe_stars_os))
            epsf_metadata['mode'] = epsf_crop.get('mode')
            epsf_metadata['sigma_pix'] = epsf_crop.get('sigma_pix')
            epsf_metadata['trail_det'] = trailed_info.get('trail_det', 0)
        fpa_star = transform_and_add_trailed_epsf(
            fpa_star,
            r_stars_os,
            c_stars_os,
            pe_stars_os,
            t_start_star,
            t_end_star,
            star_rot_rate,
            star_tran_os,
            trailed_epsf_lut,
            s_osf,
            batch_size=batch_size,
            point_rendering=point_rendering,
            epsf_tiers=epsf_tiers,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
            position_offset_os=(row_offset, col_offset),
        )
    else:
        epsf_tiers = resolve_epsf_tiers(epsf_lut, epsf_crop)
        if epsf_metadata is not None and epsf_tiers is not None:
            _, _, visible_pe_stars_os = filter_epsf_sources_in_bounds(
                fpa_star,
                r_stars_os,
                c_stars_os,
                pe_stars_os,
                t_start_star,
                t_end_star,
                star_rot_rate,
                star_tran_os,
                s_osf,
            )
            epsf_metadata.update(summarize_epsf_tiers(epsf_tiers, visible_pe_stars_os))
            epsf_metadata['mode'] = epsf_crop.get('mode')
            epsf_metadata['sigma_pix'] = epsf_crop.get('sigma_pix')
            epsf_metadata['trail_det'] = 0
        fpa_star = transform_and_add_epsf(
            fpa_star,
            r_stars_os,
            c_stars_os,
            pe_stars_os,
            t_start_star,
            t_end_star,
            t_osf,
            star_rot_rate,
            star_tran_os,
            epsf_lut,
            s_osf,
            batch_size=batch_size,
            point_rendering=point_rendering,
            epsf_tiers=epsf_tiers,
            batch_element_budget=batch_element_budget,
            batch_size_cap=batch_size_cap,
            position_offset_os=(row_offset, col_offset),
        )

    fpa_targ = tf.zeros([h_pad_det, w_pad_det], tf.float32)
    fpa_targ = add_epsf_counts(
        fpa_targ,
        r_obs_os,
        c_obs_os,
        pe_obs_os,
        epsf_lut,
        s_osf,
        r_offset_os=h_pad_os_div2,
        c_offset_os=w_pad_os_div2,
        batch_size=batch_size,
        point_rendering=point_rendering,
        batch_element_budget=batch_element_budget,
        batch_size_cap=batch_size_cap,
    )

    if render_separate:
        fpa_conv_star = crop(fpa_star, h_pad_det_div2, w_pad_det_div2, h_det, w_det)
        fpa_conv_targ = crop(fpa_targ, h_pad_det_div2, w_pad_det_div2, h_det, w_det)
        return fpa_conv_star, fpa_conv_targ, None, None, None

    combined = crop(fpa_star + fpa_targ, h_pad_det_div2, w_pad_det_div2, h_det, w_det)
    return combined, tf.zeros_like(combined), None, None, None
