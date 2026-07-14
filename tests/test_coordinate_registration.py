"""Regression tests for SatSim's canonical pixel-center convention.

Before the coordinate-registration fix, Gaussian-PSF FFT and ePSF renders
landed ``0.5 / osf`` detector pixels above the proper SatNet inverse. Target
and star cloud annotations sampled raw ``R / osf`` coordinates instead of
canonical detector coordinates.
"""

import numpy as np
import pytest

from satsim.geometry.draw import gen_line
from satsim.image.coordinates import (
    delta_detector_to_oversampled,
    delta_oversampled_to_detector,
    detector_to_normalized,
    detector_to_oversampled,
    normalized_to_detector,
    oversampled_to_detector,
)
from satsim.image.epsf import build_epsf_lut, phase_nearest_error_bound_px
from satsim.image.psf import gen_gaussian
from satsim.image.render import render_epsf, render_full
from satsim.satsim import _add_cloud_transmission_to_objects, _sample_bilinear_clamped


def _centroid(image):
    image = np.asarray(image, dtype=np.float64)
    yy, xx = np.indices(image.shape)
    total = np.sum(image)
    return np.array([np.sum(yy * image) / total, np.sum(xx * image) / total])


def _render_setup(osf, height=48, width=48):
    pad = 8
    h_fpa_os = height * osf
    w_fpa_os = width * osf
    h_fpa_pad_os = (height + 2 * pad) * osf
    w_fpa_pad_os = (width + 2 * pad) * osf
    h_pad_os = pad * osf
    w_pad_os = pad * osf
    psf_os = gen_gaussian(h_fpa_pad_os, w_fpa_pad_os, 1.5 * osf).numpy()
    return (
        h_fpa_os,
        w_fpa_os,
        h_fpa_pad_os,
        w_fpa_pad_os,
        h_pad_os,
        w_pad_os,
        osf,
        psf_os,
    )


def _render_target(mode, osf, row, col, point_rendering='bilinear', phase_oversample=1, size=48, warm_cache=False):
    args = _render_setup(osf, size, size)
    r_os = [detector_to_oversampled(row, osf)]
    c_os = [detector_to_oversampled(col, osf)]
    source_args = (r_os, c_os, [1000.0], [], [], [], 0.0, 1.0, 1, 0.0, [0.0, 0.0])
    if mode == 'fft':
        return render_full(*args, *source_args, render_separate=True, point_rendering=point_rendering)[1].numpy()

    epsf_lut = build_epsf_lut(
        args[7],
        osf,
        31,
        phase_oversample=phase_oversample,
    )
    if warm_cache:
        # A warm LUT cache supplies no PSF array; registration must derive
        # the support-center correction from the known support shape.
        psf_kwargs = {'psf_os': None, 'psf_support_shape': args[7].shape}
    else:
        psf_kwargs = {'psf_os': args[7]}
    return render_epsf(
        *args[:7],
        epsf_lut,
        *source_args,
        render_separate=True,
        point_rendering=point_rendering,
        phase_oversample=phase_oversample,
        **psf_kwargs,
    )[1].numpy()


@pytest.mark.parametrize('size', [47, 48])
@pytest.mark.parametrize('osf', [1, 2])
def test_warm_epsf_cache_matches_cold_registration(osf, size):
    """Warm-cache renders (no PSF array) must match cold-cache registration
    for both odd and even PSF support."""
    truth = np.array([22.30, 22.70])
    cold = _render_target('epsf', osf, truth[0], truth[1], size=size)
    warm = _render_target('epsf', osf, truth[0], truth[1], size=size, warm_cache=True)
    np.testing.assert_allclose(_centroid(cold), truth, atol=0.03)
    np.testing.assert_allclose(_centroid(warm), truth, atol=0.03)
    np.testing.assert_allclose(_centroid(warm), _centroid(cold), atol=1e-6)


@pytest.mark.parametrize('osf', [1, 2, 5])
def test_coordinate_conversions_round_trip(osf):
    detector = np.array([-0.5, 0.0, 12.3, 47.5])
    oversampled = detector_to_oversampled(detector, osf)
    np.testing.assert_allclose(oversampled_to_detector(oversampled, osf), detector)
    normalized = detector_to_normalized(detector, 48)
    np.testing.assert_allclose(normalized_to_detector(normalized, 48), detector)

    delta = np.array([-2.0, 0.0, 3.25])
    np.testing.assert_allclose(
        delta_oversampled_to_detector(delta_detector_to_oversampled(delta, osf), osf),
        delta,
    )


@pytest.mark.parametrize('mode', ['fft', 'epsf'])
@pytest.mark.parametrize('osf', [1, 2, 5])
def test_canonical_position_matches_noiseless_flux_centroid(mode, osf):
    truth = np.array([20.30, 21.70])
    image = _render_target(mode, osf, truth[0], truth[1])
    np.testing.assert_allclose(_centroid(image), truth, atol=0.03)


@pytest.mark.parametrize('mode', ['fft', 'epsf'])
@pytest.mark.parametrize('size', [47, 48])
def test_canonical_centroid_is_independent_of_psf_support_parity(mode, size):
    truth = np.array([22.30, 22.70])
    image = _render_target(mode, 1, truth[0], truth[1], size=size)
    np.testing.assert_allclose(_centroid(image), truth, atol=0.03)


@pytest.mark.parametrize('osf', [1, 2, 5])
def test_phase_nearest_uses_documented_centroid_bound(osf):
    truth = np.array([20.30, 21.70])
    phase_oversample = 5
    image = _render_target(
        'epsf',
        osf,
        truth[0],
        truth[1],
        point_rendering='phase_nearest',
        phase_oversample=phase_oversample,
    )
    bound = phase_nearest_error_bound_px(osf, phase_oversample)
    np.testing.assert_allclose(_centroid(image), truth, atol=bound + 0.02)


@pytest.mark.parametrize('osf', [1, 2, 5])
def test_normalized_line_origin_is_separate_from_rasterization(osf):
    height = 47
    width = 53
    origin = [0.313, 0.687]
    rr, cc, weights, _ = gen_line(
        height * osf,
        width * osf,
        origin,
        [0.0, 0.0],
        100.0,
        0.0,
        1.0,
    )
    discrete = np.array([
        oversampled_to_detector(np.average(rr, weights=weights), osf),
        oversampled_to_detector(np.average(cc, weights=weights), osf),
    ])
    continuous = np.array([
        normalized_to_detector(origin[0], height),
        normalized_to_detector(origin[1], width),
    ])
    np.testing.assert_allclose(discrete, continuous, atol=0.5 / osf + 1e-12)


def test_target_cloud_sample_exports_weighted_discrete_path_centroid():
    transmission = np.arange(20 * 30, dtype=np.float32).reshape(20, 30) / 1000.0
    ob = {
        'rrr': np.array([5.0, 6.0, 8.0]),
        'rcc': np.array([9.0, 10.0, 13.0]),
        'pp': np.array([1.0, 2.0, 1.0]),
    }
    _add_cloud_transmission_to_objects([ob], transmission)

    assert ob['cloud_sample_row'] == 6.25
    assert ob['cloud_sample_col'] == 10.5
    assert ob['cloud_transmission'] == _sample_bilinear_clamped(transmission, 6.25, 10.5)
