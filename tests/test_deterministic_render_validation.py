"""Deterministic no-noise/no-atmosphere renderer validation tests.

These tests intentionally bypass operational photometry, backgrounds, sensor
noise, and atmosphere. They compare isolated rendered sources directly against
known truth positions and truth photoelectron counts.
"""

from functools import lru_cache

import numpy as np

from satsim.image.epsf import build_epsf_lut, phase_nearest_error_bound_px
from satsim.image.psf import gen_gaussian
from satsim.image.render import render_epsf, render_full


def _centroid(image):
    image = np.asarray(image, dtype=np.float64)
    total = np.sum(image)
    yy, xx = np.indices(image.shape)
    return np.array([
        np.sum(yy * image) / total,
        np.sum(xx * image) / total,
    ])


def _aperture_flux(image, row, col, radius):
    image = np.asarray(image, dtype=np.float64)
    yy, xx = np.indices(image.shape)
    aperture = (yy - row) ** 2 + (xx - col) ** 2 <= radius ** 2
    return float(np.sum(image[aperture]))


def _detector_to_oversampled(value, osf):
    # render_* receives coordinates in SatSim's oversampled point convention.
    return value * osf + 0.5 * (osf - 2)


@lru_cache(maxsize=None)
def _setup(osf=5, psf=True):
    h = 72
    w = 72
    pad = 16
    h_fpa_os = h * osf
    w_fpa_os = w * osf
    h_fpa_pad_os = (h + 2 * pad) * osf
    w_fpa_pad_os = (w + 2 * pad) * osf
    h_pad_os_div2 = pad * osf
    w_pad_os_div2 = pad * osf
    psf_os = gen_gaussian(h_fpa_pad_os, w_fpa_pad_os, 1.2 * osf).numpy() if psf else None
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


@lru_cache(maxsize=None)
def _epsf_lut(psf=True, phase_oversample=1):
    args = _setup(psf=psf)
    osf = args[6]
    kernel_size = 41 if psf else 3
    return build_epsf_lut(
        args[7],
        osf,
        kernel_size,
        phase_oversample=phase_oversample,
    )


def _fft_image(kind, row, col, pe, psf=True):
    args = _setup(psf=psf)
    osf = args[6]
    psf_os = args[7]

    if kind == 'target':
        r_obs_os = [_detector_to_oversampled(row, osf)]
        c_obs_os = [_detector_to_oversampled(col, osf)]
        pe_obs_os = [pe]
        r_stars_os = []
        c_stars_os = []
        pe_stars_os = []
    elif kind == 'star':
        r_obs_os = []
        c_obs_os = []
        pe_obs_os = []
        r_stars_os = [args[4] + _detector_to_oversampled(row, osf)]
        c_stars_os = [args[5] + _detector_to_oversampled(col, osf)]
        pe_stars_os = [pe]
    else:
        raise ValueError('kind must be target or star')

    result = render_full(
        args[0],
        args[1],
        args[2],
        args[3],
        args[4],
        args[5],
        osf,
        psf_os,
        r_obs_os,
        c_obs_os,
        pe_obs_os,
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        point_rendering='bilinear',
    )
    return result[1 if kind == 'target' else 0].numpy()


def _epsf_image(kind, row, col, pe, psf=True, point_rendering='bilinear', phase_oversample=1):
    args = _setup(psf=psf)
    osf = args[6]
    epsf_lut = _epsf_lut(psf=psf, phase_oversample=phase_oversample)

    if kind == 'target':
        r_obs_os = [_detector_to_oversampled(row, osf)]
        c_obs_os = [_detector_to_oversampled(col, osf)]
        pe_obs_os = [pe]
        r_stars_os = []
        c_stars_os = []
        pe_stars_os = []
    elif kind == 'star':
        r_obs_os = []
        c_obs_os = []
        pe_obs_os = []
        r_stars_os = [args[4] + _detector_to_oversampled(row, osf)]
        c_stars_os = [args[5] + _detector_to_oversampled(col, osf)]
        pe_stars_os = [pe]
    else:
        raise ValueError('kind must be target or star')

    result = render_epsf(
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
        r_stars_os,
        c_stars_os,
        pe_stars_os,
        0.0,
        1.0,
        1,
        0.0,
        [0.0, 0.0],
        render_separate=True,
        point_rendering=point_rendering,
        phase_oversample=phase_oversample,
        epsf_crop={'mode': 'off'},
    )
    return result[1 if kind == 'target' else 0].numpy()


def _assert_truth_metrics(image, row, col, pe, centroid_atol, aperture_radius):
    np.testing.assert_allclose(np.sum(image), pe, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(_centroid(image), [row, col], atol=centroid_atol)
    np.testing.assert_allclose(
        _aperture_flux(image, row, col, aperture_radius),
        pe,
        rtol=2e-4,
        atol=1e-2,
    )


def _assert_truth_aperture_flux(image, row, col, pe, aperture_radius):
    np.testing.assert_allclose(np.sum(image), pe, rtol=0.0, atol=1e-3)
    np.testing.assert_allclose(
        _aperture_flux(image, row, col, aperture_radius),
        pe,
        rtol=0.0,
        atol=1e-3,
    )


def test_no_psf_truth_aperture_flux_matches_for_fft_and_epsf():
    row = 30.73
    col = 33.37
    pe = 2500.0
    aperture_radius = 1.5

    for kind in ['target', 'star']:
        fft = _fft_image(kind, row, col, pe, psf=False)
        epsf = _epsf_image(kind, row, col, pe, psf=False)

        _assert_truth_aperture_flux(fft, row, col, pe, aperture_radius=aperture_radius)
        _assert_truth_aperture_flux(epsf, row, col, pe, aperture_radius=aperture_radius)


def test_gaussian_psf_truth_centroid_and_aperture_flux_match_for_fft_and_epsf():
    row = 30.73
    col = 33.37
    pe = 2500.0
    aperture_radius = 14.0

    for kind in ['target', 'star']:
        fft = _fft_image(kind, row, col, pe, psf=True)
        epsf = _epsf_image(kind, row, col, pe, psf=True)

        _assert_truth_metrics(fft, row, col, pe, centroid_atol=0.02, aperture_radius=aperture_radius)
        _assert_truth_metrics(epsf, row, col, pe, centroid_atol=0.03, aperture_radius=aperture_radius)


def test_phase_nearest_truth_centroid_stays_within_quantization_bound():
    row = 30.73
    col = 33.37
    pe = 2500.0
    osf = _setup()[6]
    phase_oversample = 5
    bound = phase_nearest_error_bound_px(osf, phase_oversample)

    for kind in ['target', 'star']:
        image = _epsf_image(
            kind,
            row,
            col,
            pe,
            psf=True,
            point_rendering='phase_nearest',
            phase_oversample=phase_oversample,
        )
        _assert_truth_metrics(
            image,
            row,
            col,
            pe,
            centroid_atol=bound + 0.02,
            aperture_radius=14.0,
        )


def test_epsf_matches_fft_centroid_and_aperture_flux_in_deterministic_case():
    row = 30.73
    col = 33.37
    pe = 2500.0
    aperture_radius = 14.0

    for kind in ['target', 'star']:
        fft = _fft_image(kind, row, col, pe, psf=True)
        epsf = _epsf_image(kind, row, col, pe, psf=True)

        np.testing.assert_allclose(_centroid(epsf), _centroid(fft), atol=0.02)
        np.testing.assert_allclose(
            _aperture_flux(epsf, row, col, aperture_radius),
            _aperture_flux(fft, row, col, aperture_radius),
            rtol=2e-4,
            atol=1e-2,
        )
