"""Tests for `render` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.render import render_full, render_piecewise, render_epsf
from satsim.image.epsf import build_epsf_lut
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


def _render_epsf_target(row, col, pe=1000.0, point_rendering='bilinear', kernel_size=31, phase_oversample=1):
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, kernel_size, phase_oversample=phase_oversample)

    return render_epsf(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
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
        point_rendering=point_rendering,
        phase_oversample=phase_oversample,
    )


def _delta_render_setup(osf=3):
    h = 24
    w = 26
    pad = 4
    h_fpa_os = h * osf
    w_fpa_os = w * osf
    h_fpa_pad_os = (h + 2 * pad) * osf
    w_fpa_pad_os = (w + 2 * pad) * osf
    h_pad_os_div2 = pad * osf
    w_pad_os_div2 = pad * osf
    psf_os = np.zeros([h_fpa_pad_os, w_fpa_pad_os], dtype=np.float32)
    psf_os[h_fpa_pad_os // 2, w_fpa_pad_os // 2] = 1.0
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
    star_render_mode = 'streak'
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
    for star_render_mode in ['transform', 'streak', 'fft']:
        for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.50, 0.50), (0.73, 0.37)]:
            row = 20 + row_phase
            col = 21 + col_phase
            star = _render_star(row, col, pe=pe, star_render_mode=star_render_mode)[0].numpy()

            np.testing.assert_allclose(np.sum(star), pe, atol=1e-3)
            np.testing.assert_allclose(_centroid(star), [row, col], atol=0.02)


def test_render_epsf_matches_fft_delta_psf_for_all_integer_phases():
    args = _delta_render_setup(osf=3)
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 5, normalize=False)

    for phase_r in range(osf):
        for phase_c in range(osf):
            r_obs_os = [10 * osf + phase_r]
            c_obs_os = [11 * osf + phase_c]
            pe_obs_os = [123.0]

            full = render_full(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                osf,
                args[7],
                r_obs_os,
                c_obs_os,
                pe_obs_os,
                [],
                [],
                [],
                0.0,
                1.0,
                1,
                0.0,
                [0.0, 0.0],
                render_separate=True,
                point_rendering='floor',
            )[1].numpy()

            epsf = render_epsf(
                args[0],
                args[1],
                args[2],
                args[3],
                args[4],
                args[5],
                osf,
                epsf_lut,
                r_obs_os,
                c_obs_os,
                pe_obs_os,
                [],
                [],
                [],
                0.0,
                1.0,
                1,
                0.0,
                [0.0, 0.0],
                render_separate=True,
                point_rendering='floor',
            )[1].numpy()

            np.testing.assert_allclose(epsf, full, atol=1e-4)


def test_render_epsf_matches_fft_gaussian_with_even_psf_support():
    pe = 1234.0
    for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.50, 0.50), (0.73, 0.37)]:
        row = 20 + row_phase
        col = 21 + col_phase
        full = _render_target(row, col, pe=pe, point_rendering='bilinear')[1].numpy()
        epsf = _render_epsf_target(row, col, pe=pe, point_rendering='bilinear', kernel_size=31)[1].numpy()

        np.testing.assert_allclose(epsf, full, atol=1e-4)
        np.testing.assert_allclose(_centroid(epsf), _centroid(full), atol=1e-5)


def test_render_epsf_default_does_not_renormalize_cropped_kernel_photometry():
    args = _render_setup()
    osf = args[6]
    pe = 1000.0
    row = 20.0
    col = 21.0
    broad_psf = gen_gaussian(args[2], args[3], 8.0 * osf).numpy()
    default_lut = build_epsf_lut(broad_psf, osf, 5)
    normalized_lut = build_epsf_lut(broad_psf, osf, 5, normalize=True)

    render_args = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
    )
    source_args = (
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
    )

    default = render_epsf(*render_args, default_lut, *source_args, render_separate=True)[1].numpy()
    normalized = render_epsf(*render_args, normalized_lut, *source_args, render_separate=True)[1].numpy()

    assert(np.sum(default) < 0.95 * pe)
    np.testing.assert_allclose(np.sum(normalized), pe, atol=1e-3)


def test_render_epsf_bilinear_target_centroid_and_photometry_are_phase_stable():
    pe = 4321.0
    for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.50, 0.50), (0.73, 0.37)]:
        row = 20 + row_phase
        col = 21 + col_phase
        target = _render_epsf_target(row, col, pe=pe)[1].numpy()

        np.testing.assert_allclose(np.sum(target), pe, atol=1e-3)
        np.testing.assert_allclose(_centroid(target), [row, col], atol=0.03)


def test_render_epsf_phase_nearest_target_centroid_within_quantization_bound():
    pe = 4321.0
    osf = _render_setup()[6]
    phase_oversample = 4
    bound = 1.0 / (2.0 * osf * phase_oversample)

    for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.73, 0.37)]:
        row = 20 + row_phase
        col = 21 + col_phase
        target = _render_epsf_target(
            row,
            col,
            pe=pe,
            point_rendering='phase_nearest',
            phase_oversample=phase_oversample,
        )[1].numpy()

        np.testing.assert_allclose(np.sum(target), pe, atol=1e-3)
        np.testing.assert_allclose(_centroid(target), [row, col], atol=bound + 0.01)


def test_render_epsf_bilinear_star_centroid_and_photometry_are_phase_stable():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    pe = 3210.0

    for row_phase, col_phase in [(0.10, 0.20), (0.25, 0.75), (0.73, 0.37)]:
        row = 20 + row_phase
        col = 21 + col_phase
        star = render_epsf(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            osf,
            epsf_lut,
            [],
            [],
            [],
            [args[4] + _detector_to_oversampled(row, osf)],
            [args[5] + _detector_to_oversampled(col, osf)],
            [pe],
            0.0,
            1.0,
            1,
            0.0,
            [0.0, 0.0],
            render_separate=True,
        )[0].numpy()

        np.testing.assert_allclose(np.sum(star), pe, atol=1e-3)
        np.testing.assert_allclose(_centroid(star), [row, col], atol=0.03)


def test_render_epsf_tiered_bright_star_matches_full_kernel():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [args[4] + _detector_to_oversampled(20.25, osf)]
    c_stars_os = [args[5] + _detector_to_oversampled(21.75, osf)]
    pe_stars_os = [1000.0]

    common = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
    )

    expected = render_epsf(*common, render_separate=True)[0].numpy()
    metadata = {}
    actual = render_epsf(
        *common,
        render_separate=True,
        epsf_crop={
            'mode': 'manual',
            'threshold_pe': 10.0,
            'kernel_size': 5,
        },
        epsf_metadata=metadata,
    )[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert(metadata['source_count'] == 1)
    assert(metadata['tiers'][0]['count'] == 0)
    assert(metadata['tiers'][-1]['kernel_size'] == 31)
    assert(metadata['tiers'][-1]['count'] == 1)


def test_render_epsf_tiered_crop_applies_to_stars_not_targets():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_obs_os = [_detector_to_oversampled(20.25, osf)]
    c_obs_os = [_detector_to_oversampled(21.75, osf)]
    pe_obs_os = [1000.0]

    common = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        r_obs_os,
        c_obs_os,
        pe_obs_os,
        [],
        [],
        [],
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
    )

    expected = render_epsf(*common, render_separate=True)[1].numpy()
    actual = render_epsf(
        *common,
        render_separate=True,
        epsf_crop={
            'mode': 'manual',
            'threshold_pe': 1e9,
            'kernel_size': 5,
        },
    )[1].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)


def test_render_epsf_crop_off_matches_absent_crop():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [args[4] + _detector_to_oversampled(20.25, osf)]
    c_stars_os = [args[5] + _detector_to_oversampled(21.75, osf)]
    pe_stars_os = [1000.0]

    common = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
    )

    expected = render_epsf(*common, render_separate=True)[0].numpy()
    metadata = {}
    actual = render_epsf(
        *common,
        render_separate=True,
        epsf_crop={'mode': 'off'},
        epsf_metadata=metadata,
    )[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-5)
    assert(metadata == {})


def test_render_epsf_tier_metadata_counts_visible_stars_only():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)

    for star_render_mode in ('transform', 'streak'):
        metadata = {}
        render_epsf(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            osf,
            epsf_lut,
            [],
            [],
            [],
            [args[4] + _detector_to_oversampled(20.25, osf), -100.0],
            [args[5] + _detector_to_oversampled(21.75, osf), -100.0],
            [5.0, 5.0],
            0.0,
            1.0,
            1,
            0.0,
            [0.0, 0.0],
            render_separate=True,
            star_render_mode=star_render_mode,
            epsf_crop={
                'mode': 'manual',
                'threshold_pe': 10.0,
                'kernel_size': 5,
            },
            epsf_metadata=metadata,
            psf_os=args[7],
        )

        assert(metadata['source_count'] == 1)
        assert(sum(tier['count'] for tier in metadata['tiers']) == 1)


def test_render_epsf_star_streak_matches_render_full_moving_stars_without_fallback():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [
        args[4] + _detector_to_oversampled(18.25, osf),
        args[4] + _detector_to_oversampled(22.75, osf),
        args[4] + _detector_to_oversampled(27.10, osf),
    ]
    c_stars_os = [
        args[5] + _detector_to_oversampled(20.50, osf),
        args[5] + _detector_to_oversampled(25.10, osf),
        args[5] + _detector_to_oversampled(19.80, osf),
    ]
    pe_stars_os = [500.0, 750.0, 600.0]
    star_tran_os = [3.0, 6.0]

    expected = render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        args[7],
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        24,
        0.0,
        star_tran_os,
        render_separate=True,
        star_render_mode='streak',
    )[0].numpy()
    actual = render_epsf(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        24,
        0.0,
        star_tran_os,
        render_separate=True,
        star_render_mode='streak',
        psf_os=args[7],
    )[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_render_epsf_star_streak_matches_render_full_absolute_start_translation():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [
        args[4] + _detector_to_oversampled(18.25, osf),
        args[4] + _detector_to_oversampled(22.75, osf),
    ]
    c_stars_os = [
        args[5] + _detector_to_oversampled(20.50, osf),
        args[5] + _detector_to_oversampled(25.10, osf),
    ]
    pe_stars_os = [500.0, 750.0]
    star_tran_os = [3.0, 0.0]

    expected = render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        args[7],
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        5.0,
        6.0,
        24,
        0.0,
        star_tran_os,
        render_separate=True,
        star_render_mode='streak',
    )[0].numpy()
    actual = render_epsf(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        5.0,
        6.0,
        24,
        0.0,
        star_tran_os,
        render_separate=True,
        star_render_mode='streak',
        psf_os=args[7],
    )[0].numpy()

    np.testing.assert_allclose(np.sum(actual), np.sum(expected), atol=1e-3)
    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_render_epsf_star_streak_matches_render_full_rotation_without_fallback():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [
        args[4] + _detector_to_oversampled(18.25, osf),
        args[4] + _detector_to_oversampled(22.75, osf),
    ]
    c_stars_os = [
        args[5] + _detector_to_oversampled(20.50, osf),
        args[5] + _detector_to_oversampled(25.10, osf),
    ]
    pe_stars_os = [500.0, 750.0]

    expected = render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        args[7],
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        24,
        0.02,
        [0.0, 0.0],
        render_separate=True,
        star_render_mode='streak',
    )[0].numpy()
    actual = render_epsf(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        24,
        0.02,
        [0.0, 0.0],
        render_separate=True,
        star_render_mode='streak',
        psf_os=args[7],
    )[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-3)


def test_render_epsf_unsupported_obs_model_raises_without_fallback():
    args = _render_setup()
    epsf_lut = build_epsf_lut(args[7], args[6], 7)
    obs_model = [np.zeros([args[2], args[3]], dtype=np.float32)]

    with pytest.raises(NotImplementedError):
        render_epsf(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            args[6],
            epsf_lut,
            [],
            [],
            [],
            [],
            [],
            [],
            0.0,
            1.0,
            1,
            0.0,
            [0.0, 0.0],
            obs_model=obs_model,
        )


def test_render_epsf_fallback_star_streak_matches_render_full():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    r_stars_os = [
        args[4] + _detector_to_oversampled(20.25, osf),
        args[4] + _detector_to_oversampled(22.75, osf),
    ]
    c_stars_os = [
        args[5] + _detector_to_oversampled(21.50, osf),
        args[5] + _detector_to_oversampled(23.10, osf),
    ]
    pe_stars_os = [500.0, 750.0]

    expected = render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        args[7],
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        star_render_mode='streak',
    )[0].numpy()
    actual = render_epsf(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        epsf_lut,
        [],
        [],
        [],
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        star_render_mode='streak',
        fallback_to_fft_for_models=True,
        psf_os=args[7],
    )[0].numpy()

    np.testing.assert_allclose(actual, expected, atol=1e-4)


def test_render_epsf_fallback_obs_model_matches_render_full():
    args = _render_setup()
    osf = args[6]
    epsf_lut = build_epsf_lut(args[7], osf, 31)
    model = np.zeros([args[2], args[3]], dtype=np.float32)
    model[args[4] + 8:args[4] + 12, args[5] + 7:args[5] + 11] = 5.0
    obs_model = [model]

    common = (
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
    )
    source_args = (
        [],
        [],
        [],
        [],
        [],
        [],
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
    )

    expected = render_full(*common, args[7], *source_args, render_separate=False, obs_model=obs_model)[0].numpy()
    actual = render_epsf(
        *common,
        epsf_lut,
        *source_args,
        render_separate=False,
        obs_model=obs_model,
        fallback_to_fft_for_models=True,
        psf_os=args[7],
    )[0].numpy()

    np.testing.assert_allclose(actual, expected)


def test_render_epsf_phase_nearest_rejects_fft_fallback():
    args = _render_setup()
    osf = args[6]
    phase_oversample = 4
    epsf_lut = build_epsf_lut(args[7], osf, 31, phase_oversample=phase_oversample)
    obs_model = [np.zeros([args[2], args[3]], dtype=np.float32)]

    with pytest.raises(ValueError, match='phase_nearest'):
        render_epsf(
            args[0],
            args[1],
            args[2],
            args[3],
            args[4],
            args[5],
            osf,
            epsf_lut,
            [],
            [],
            [],
            [],
            [],
            [],
            0.0,
            1.0,
            1,
            0.0,
            [0.0, 0.0],
            render_separate=False,
            obs_model=obs_model,
            fallback_to_fft_for_models=True,
            psf_os=args[7],
            point_rendering='phase_nearest',
            phase_oversample=phase_oversample,
        )


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

    star_render_mode = 'streak'
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

    star_render_mode = 'streak'
    fimg2 = render_full(h_fpa_os, w_fpa_os, h_fpa_pad_os, w_fpa_pad_os, h_pad_os_div2, w_pad_os_div2, s_osf, psf_os, r_obs_os, c_obs_os, pe_obs_os, r_stars_os, c_stars_os, pe_stars_os, t_start_star, t_end_star, t_osf, star_rot_rate, star_tran_os, render_separate=render_separate, obs_model=obs_model, star_render_mode=star_render_mode)

    np.testing.assert_allclose(img[0].numpy(), img2[0].numpy(), rtol=30.0, atol=0.001)
    np.testing.assert_allclose(fimg[0].numpy(), fimg2[0].numpy(), rtol=8.0, atol=0.002)
