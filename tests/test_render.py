"""Tests for `render` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import tensorflow as tf

from satsim.image.render import render_full, render_piecewise


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
