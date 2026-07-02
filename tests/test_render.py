"""Tests for `render` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import tensorflow as tf

from satsim.image.render import render_full, render_piecewise
from satsim.image.psf import gen_gaussian


def _centroid(image):
    image = np.asarray(image, dtype=float)
    yy, xx = np.indices(image.shape)
    total = np.sum(image)
    return np.sum(yy * image) / total, np.sum(xx * image) / total


def _detector_to_oversampled(value, osf):
    # render_full receives coordinates in SatSim's oversampled point convention.
    return value * osf + 0.5 * (osf - 2)


def _render_setup(osf=3):
    h = 48
    w = 48
    pad = 8
    h_fpa_os = h * osf
    w_fpa_os = w * osf
    h_fpa_pad_os = (h + 2 * pad) * osf
    w_fpa_pad_os = (w + 2 * pad) * osf
    h_pad_os_div2 = pad * osf
    w_pad_os_div2 = pad * osf
    psf_os = gen_gaussian(h_fpa_pad_os, w_fpa_pad_os, 1.5 * osf).numpy()
    return (
        h_fpa_os,
        w_fpa_os,
        h_fpa_pad_os,
        w_fpa_pad_os,
        h_pad_os_div2,
        w_pad_os_div2,
        osf,
        psf_os,
    )


def _render_target(row, col, pe=1000.0, point_rendering=None, psf=True):
    args = _render_setup()
    h_pad_os_div2 = args[4]
    w_pad_os_div2 = args[5]
    osf = args[6]
    psf_os = args[7] if psf else None
    kwargs = {}
    if point_rendering is not None:
        kwargs['point_rendering'] = point_rendering

    return render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        h_pad_os_div2,
        w_pad_os_div2,
        osf,
        psf_os,
        [_detector_to_oversampled(row, osf)],
        [_detector_to_oversampled(col, osf)],
        [pe],
        [],
        [],
        [],
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        **kwargs,
    )


def _render_star(row, col, pe=1000.0, star_render_mode='transform'):
    args = _render_setup()
    h_pad_os_div2 = args[4]
    w_pad_os_div2 = args[5]
    osf = args[6]

    return render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        h_pad_os_div2,
        w_pad_os_div2,
        osf,
        args[7],
        [],
        [],
        [],
        [h_pad_os_div2 + _detector_to_oversampled(row, osf)],
        [w_pad_os_div2 + _detector_to_oversampled(col, osf)],
        [pe],
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        star_render_mode=star_render_mode,
    )


def test_render_full():

    h_fpa_os = 10
    w_fpa_os = 10
    h_fpa_pad_os = 14
    w_fpa_pad_os = 14
    h_pad_os_div2 = 2
    w_pad_os_div2 = 2
    s_osf = 1
    psf_os = np.zeros([h_fpa_pad_os, w_fpa_pad_os])
    psf_os[7, 7] = 1.0
    r_obs_os = []
    c_obs_os = []
    pe_obs_os = []
    r_stars_os = [4, 5]
    c_stars_os = [4, 5]
    pe_stars_os = [1000, 500]
    t_start_star = 0.0
    t_end_star = 1.0
    t_osf = 100
    star_rot_rate = 0.0
    star_tran_os = [0.0, 1.0]

    # TODO: these tests should be more thorough

    render_separate = True
    obs_model = None
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500, decimal=3)

    render_separate = True
    obs_model = None
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, None, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500, decimal=3)

    render_separate = True
    obs_model = None
    star_render_mode = 'fft'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500, decimal=3)

    render_separate = False
    obs_model = None
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)

    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500, decimal=3)

    render_separate = False
    obs_model = None
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500, decimal=3)

    # test obs_model with occulation
    render_separate = False
    obs_model = [
        np.ones([h_fpa_pad_os, w_fpa_pad_os]) * 2.0,
        np.ones([h_fpa_pad_os, w_fpa_pad_os])
    ]
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 10.0 * 10.0 * 3.0, decimal=3)

    # test obs_model non-occulation
    render_separate = False
    model = np.zeros([h_fpa_pad_os, w_fpa_pad_os])
    model[8:10, 8:10] = 5.0
    obs_model = [
        model
    ]
    star_render_mode = 'transform'
    img = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)
    np.testing.assert_almost_equal(tf.reduce_sum(img[0]), 1500 + 5.0 * 4.0, decimal=3)


def test_render_full_defaults_to_bilinear_point_rendering():

    default_img = _render_target(20.25, 21.75)[1].numpy()
    bilinear_img = _render_target(20.25, 21.75, point_rendering='bilinear')[1].numpy()
    floor_img = _render_target(20.25, 21.75, point_rendering='floor')[1].numpy()

    np.testing.assert_allclose(default_img, bilinear_img)
    assert(not np.allclose(default_img, floor_img))


def test_render_full_bilinear_no_psf_preserves_oversampled_truth_and_flux():

    row = 20.25
    col = 21.75
    pe = 1234.5
    args = _render_setup()
    osf = args[6]
    h_pad_os_div2 = args[4]
    w_pad_os_div2 = args[5]

    img = _render_target(row, col, pe=pe, psf=False)
    fpa_targ = img[2].numpy()
    detector_targ = img[1].numpy()

    np.testing.assert_allclose(np.sum(fpa_targ), pe, atol=1e-5)
    np.testing.assert_allclose(np.sum(detector_targ), pe, atol=1e-5)
    np.testing.assert_allclose(
        _centroid(fpa_targ),
        [
            h_pad_os_div2 + _detector_to_oversampled(row, osf),
            w_pad_os_div2 + _detector_to_oversampled(col, osf),
        ],
        atol=1e-6,
    )


def test_render_full_bilinear_point_rendering_matches_subpixel_target_centroid():

    r_true = 20.25
    c_true = 21.75

    floor_img = _render_target(r_true, c_true, point_rendering='floor')[1].numpy()
    bilinear_img = _render_target(r_true, c_true, point_rendering='bilinear')[1].numpy()

    floor_centroid = _centroid(floor_img)
    bilinear_centroid = _centroid(bilinear_img)

    assert(abs(floor_centroid[0] - r_true) > 0.02)
    assert(abs(floor_centroid[1] - c_true) > 0.02)
    np.testing.assert_allclose(bilinear_centroid, [r_true, c_true], atol=0.02)


def test_render_full_bilinear_target_centroid_and_photometry_are_phase_stable():

    pe = 1234.5
    for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.50, 0.50), (0.73, 0.37), (0.90, 0.88)]:
        row = 20 + row_phase
        col = 21 + col_phase
        target = _render_target(row, col, pe=pe, point_rendering='bilinear')[1].numpy()

        np.testing.assert_allclose(np.sum(target), pe, atol=1e-3)
        np.testing.assert_allclose(_centroid(target), [row, col], atol=0.02)


def test_render_full_bilinear_point_rendering_matches_subpixel_star_centroid():

    r_true = 20.25
    c_true = 21.75

    bilinear_img = _render_star(r_true, c_true)[0].numpy()

    np.testing.assert_allclose(_centroid(bilinear_img), [r_true, c_true], atol=0.02)


def test_render_full_bilinear_star_centroid_and_photometry_are_phase_stable():

    pe = 987.5
    for star_render_mode in ['transform', 'fft']:
        for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.50, 0.50), (0.73, 0.37)]:
            row = 20 + row_phase
            col = 21 + col_phase
            star = _render_star(row, col, pe=pe, star_render_mode=star_render_mode)[0].numpy()

            np.testing.assert_allclose(np.sum(star), pe, atol=1e-3)
            np.testing.assert_allclose(_centroid(star), [row, col], atol=0.02)


def test_render_piecewise():

    h = 100
    w = 100
    h_sub = 10
    w_sub = 10
    h_pad_os = 4
    w_pad_os = 4
    s_osf = 1
    psf_os = np.zeros([h_sub * s_osf + h_pad_os, w_sub * s_osf + w_pad_os])
    psf_os[6:7, 6:7] = 0.25
    r_obs_os = []
    c_obs_os = []
    pe_obs_os = []
    r_stars_os = [4, 5, 35]
    c_stars_os = [4, 5, 35]
    pe_stars_os = [1000, 500, 2000]
    t_start_star = 0.0
    t_end_star = 1.0
    t_osf = 100
    star_rot_rate = 0.0
    star_tran_os = [0.5, 1.0]

    render_separate = True
    obs_model = None
    star_render_mode = 'transform'
    img = render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, star_render_mode=star_render_mode)

    star_render_mode = 'fft'
    fimg = render_piecewise(h, w, h_sub, w_sub, h_pad_os, w_pad_os, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, star_render_mode=star_render_mode)

    h_fpa_os = 100
    w_fpa_os = 100
    h_fpa_pad_os = 104
    w_fpa_pad_os = 104
    h_pad_os_div2 = 2
    w_pad_os_div2 = 2
    s_osf = 1
    psf_os = np.zeros([h_fpa_pad_os, w_fpa_pad_os])
    psf_os[51:52, 51:52] = 0.25

    star_render_mode = 'transform'
    img2 = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)

    star_render_mode = 'fft'
    fimg2 = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)

    np.testing.assert_allclose(img[0].numpy(), img2[0].numpy(), rtol=30.0, atol=0.001)
    np.testing.assert_allclose(fimg[0].numpy(), fimg2[0].numpy(), rtol=8.0, atol=0.002)
