from __future__ import division, print_function, absolute_import

import math
import logging

import tensorflow as tf
import numpy as np

from satsim.math import fftconv2p
from satsim.image.fpa import downsample, crop, add_counts, transform_and_add_counts, transform_and_fft

logger = logging.getLogger(__name__)


def render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=True, star_render_mode='transform'):
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
        star_render_mode: `string`, star render mode. `fft` or `transform`. default=transform

    Returns:
        A `tuple`, containing:
            fpa_conv_star: `image`, rendered image with stars. Stars and targets if `render_separate` is `False`.
            fpa_conv_targ: `image`, rendered image with targets. Zeros if `render_separate` is `False`.
            _: `None`, to make output compatible with `render_full`
            _: `None`, to make output compatible with `render_full`
            _: `None`, to make output compatible with `render_full`
    """

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

            fpa_conv_star[ir][ic], fpa_conv_targ[ir][ic], _, _, _ = render_full(h_sub_os, w_sub_os, h_sub_pad_os, w_sub_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_sub, c_obs_sub, pe_obs_sub, r_stars_sub, c_stars_sub, pe_stars_sub, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, star_render_mode=star_render_mode)

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


def render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=True, obs_model=None, star_render_mode='transform'):
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
        star_render_mode: `string`, star render mode. `fft` or `transform`. default=transform

    Returns:
        A `tuple`, containing:
            fpa_conv_star: `image`, rendered image with stars. Stars and targets if `render_separate` is `False`.
            fpa_conv_targ: `image`, rendered image with targets. Zeros if `render_separate` is `False`.
            fpa_os_w_targets: `image`, oversample image with targets before blur.
            fpa_conv_os: `None`, rendered oversample image with pad.
            fpa_conv_crop: `None`, rendered oversample image with no pad.

    """
    # render stars
    fpa_os_w_stars = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
    if star_render_mode == 'fft':
        fpa_os_w_stars = transform_and_fft(fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os)
    else:
        fpa_os_w_stars = transform_and_add_counts(fpa_os_w_stars, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os)

    # render modeled targets
    fpa_os_w_targets = tf.zeros([h_fpa_pad_os, w_fpa_pad_os], tf.float32)
    if obs_model is not None and len(obs_model) > 0:
        for om in obs_model:
            fpa_os_w_targets = fpa_os_w_targets + tf.cast(om, tf.float32)

        # mask stars (occultation)
        condition = tf.math.greater(fpa_os_w_targets, tf.zeros_like(fpa_os_w_targets, tf.float32))
        fpa_os_w_stars = tf.where(condition, tf.zeros_like(fpa_os_w_stars, tf.float32), fpa_os_w_stars)

    # render point source targets
    fpa_os_w_targets = add_counts(fpa_os_w_targets, r_obs_os, c_obs_os, pe_obs_os, h_pad_os_div2, w_pad_os_div2)

    if render_separate:

        # blur stars
        fpa_conv_os = fftconv2p(fpa_os_w_stars, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_star = downsample(fpa_conv_crop, s_osf, 'pool')

        # blur targets
        fpa_conv_os = fftconv2p(fpa_os_w_targets, psf_os, pad=1)
        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_targ = downsample(fpa_conv_crop, s_osf, 'pool')

        return fpa_conv_star, fpa_conv_targ, fpa_os_w_targets, fpa_conv_os, fpa_conv_crop

    else:

        # blur targets and stars together
        fpa_os_sum = fpa_os_w_stars + fpa_os_w_targets
        if(tf.reduce_all(tf.equal(fpa_os_sum, 0))):
            fpa_conv_os = fpa_os_sum
        else:
            fpa_conv_os = fftconv2p(fpa_os_sum, psf_os, pad=1)

        fpa_conv_crop = crop(fpa_conv_os, h_pad_os_div2, w_pad_os_div2, h_fpa_os, w_fpa_os)
        fpa_conv_ds = downsample(fpa_conv_crop, s_osf, 'pool')

        return fpa_conv_ds, tf.zeros_like(fpa_conv_ds), fpa_os_w_targets, fpa_conv_os, fpa_conv_crop
