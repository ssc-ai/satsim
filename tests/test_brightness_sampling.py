"""Accuracy and fallback coverage for opt-in brightness sampling (#48)."""
import importlib

import numpy as np
import pytest

from satsim import time
from satsim.geometry.astrometric import create_topocentric
from satsim.geometry.draw import gen_curve_from_points
from satsim.geometry.photometric import interpolate_magnitudes, model_to_mv
from satsim.geometry.sgp4 import create_sgp4
from satsim.image.fpa import mv_to_pe
from satsim.pipeline import _avg_t, _delta_t

sim = importlib.import_module('satsim.satsim')
MODEL = {'mode': 'lambertian_sphere', 'albedo': .2, 'size': 1.}
ADAPTIVE = {'mode': 'adaptive', 'max_interval_seconds': 1., 'tolerance_mag': 1e-4}
ECCENTRIC = (
    '1 00005U 58002B   26097.92152118  .00000210  00000-0  25961-3 0  9992',
    '2 00005  34.2431 171.1570 1836179 181.6781 177.6859 10.85968047435646',
)
LEO = (
    '1 41788U 16059F   21260.56440839  .00022290  00000-0  23487-2 0  9990',
    '2 41788  98.2495   9.3816 0022044 246.7613 113.2488 14.89172957267561',
)
GEO = (
    '1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992',
    '2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866',
)


def test_quadratic_exactness_and_reduced_model_samples():
    offsets = np.linspace(1e8, 1e8 + 1, 1001)
    seen = []

    def model(t):
        seen.extend(t)
        return 10 + (t - 1e8)**2

    actual = interpolate_magnitudes(model, offsets)
    np.testing.assert_allclose(actual, 10 + (offsets - 1e8)**2, rtol=0, atol=1e-13)
    assert len(seen) == 5


def test_error_checks_subdivide_a_curved_model():
    offsets = np.linspace(0., 1., 1001)
    seen = []

    def model(t):
        seen.extend(t)
        return 10 + 20 * t**4

    actual = interpolate_magnitudes(model, offsets, tolerance_mag=1e-5)
    assert 5 < len(seen) < len(offsets)
    np.testing.assert_allclose(actual, 10 + 20 * offsets**4, rtol=0, atol=2e-5)


@pytest.mark.parametrize('n', [0, 1, 2, 3, 4, 5, 20])
def test_sparse_or_long_exposure_uses_exact_samples(n):
    offsets = np.linspace(0., 20., n)
    seen = []

    def model(t):
        seen.extend(t)
        return np.sin(t)

    actual = interpolate_magnitudes(model, offsets)
    np.testing.assert_array_equal(actual, np.sin(offsets))
    assert len(seen) == n


@pytest.mark.parametrize('reason', ['nonfinite', 'budget'])
def test_unreliable_interpolation_falls_back_to_full_evaluation(reason):
    offsets = np.linspace(0., 1., 51)
    calls = []

    def model(t):
        calls.append(t.copy())
        if reason == 'nonfinite':
            return np.where(t == .5, np.inf, 1.)
        return np.sin(17 * t)

    actual = interpolate_magnitudes(model, offsets, tolerance_mag=1e-12)
    np.testing.assert_array_equal(calls[-1], offsets)
    expected = np.where(offsets == .5, np.inf, 1.) if reason == 'nonfinite' else np.sin(17 * offsets)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('orbit,epoch,space', [
    pytest.param(ECCENTRIC, [2026, 4, 7, 22, 0, 0.], False, id='eccentric-ground'),
    pytest.param(LEO, [2021, 9, 17, 23, 55, 30.], False, id='close-leo-ground'),
    pytest.param(GEO, [2015, 4, 24, 9, 37, 44.128], False, id='geo-ground'),
    pytest.param(GEO, [2021, 9, 17, 23, 55, 30.], True, id='geo-space-observer'),
])
@pytest.mark.parametrize('duration', [1., 10., 60., 120.])
def test_adaptive_brightness_matches_exact_across_exposures_and_orbits(orbit, epoch, space, duration):
    observer = create_sgp4(*LEO) if space else create_topocentric(20.7, -156.3, 3.)
    target = create_sgp4(*orbit)
    edges = np.linspace(-duration / 2, duration / 2, 1026)
    t = time.utc_from_list(epoch, _avg_t(edges))
    exact_mv = model_to_mv(observer, target, MODEL, t)
    exact_counts = mv_to_pe(20., exact_mv) * _delta_t(edges)
    actual = sim._apply_photometry(np.zeros(1025), edges, observer, target, MODEL, epoch, 20.,
                                   sampling=ADAPTIVE)
    # Validate sample magnitudes and total photoelectrons independently.
    inferred_mv = 20 - 2.5 * np.log10(actual / _delta_t(edges))
    np.testing.assert_allclose(inferred_mv, exact_mv, rtol=0, atol=2e-4)
    np.testing.assert_allclose(actual.sum(), exact_counts.sum(), rtol=2e-4)


def test_adaptive_preserves_annotation_controls_and_shadow_transition(monkeypatch):
    _, _, counts, edges = gen_curve_from_points(0., 0., 100., 0., 200., 0., 1., 0., 1.)
    seen = []

    def model(observer, target, config, t):
        seen.append(len(t.tt))
        return np.full(len(t.tt), 10.)

    # Put the shadow transition inside an interpolation interval.
    midpoint = time.utc_from_list([2026, 4, 7, 22, 0, 0.], .373).tt
    monkeypatch.setattr(sim, 'model_to_mv', model)
    monkeypatch.setattr(sim, 'earth_shadow_umbra_mask', lambda target, t: (t.tt < midpoint).astype(float))
    actual = sim._apply_photometry(counts, edges, None, object(), MODEL, [2026, 4, 7, 22, 0, 0.], 20.,
                                   chunk_size=32, earth_shadow=True, sampling=ADAPTIVE)
    t = time.utc_from_list([2026, 4, 7, 22, 0, 0.], _avg_t(edges))
    expected = 10000 * _delta_t(edges) * (t.tt < midpoint)
    np.testing.assert_allclose(actual, expected)
    assert sum(seen) == 5
    assert np.all(actual[_delta_t(edges) == 0] == 0)


def test_explicit_model_arrays_use_exact_evaluation_even_when_adaptive_is_requested():
    edges = np.linspace(0., 1., 102)
    model = {**MODEL, 'distance': np.linspace(1000., 2000., 101),
             'phase_angle': np.linspace(20., 50., 101)**1.1}
    expected = mv_to_pe(20., model_to_mv(None, None, model, None)) * _delta_t(edges)
    actual = sim._apply_photometry(np.zeros(101), edges, None, None, model, [2026, 4, 7, 0, 0, 0.], 20.,
                                   chunk_size=16, sampling=ADAPTIVE)
    np.testing.assert_array_equal(actual, expected)


def test_default_setting_keeps_per_sample_brightness(monkeypatch):
    seen = []

    def model(observer, target, config, t):
        seen.extend(t.tt)
        return np.full(len(t.tt), 10.)

    monkeypatch.setattr(sim, 'model_to_mv', model)
    sim._apply_photometry(np.zeros(101), np.linspace(0., 1., 102), None, None, MODEL,
                          [2026, 4, 7, 0, 0, 0.], 20., sampling=sim._brightness_sampling_settings({}))
    assert len(seen) == 101


@pytest.mark.parametrize('sampling', [
    None, 'adaptive', {'mode': 'fast'}, {'typo': 1},
    {'tolerance_mag': 0}, {'tolerance_mag': np.nan}, {'tolerance_mag': True},
    {'max_interval_seconds': 0}, {'max_interval_seconds': np.inf},
])
def test_invalid_sampling_settings_raise(sampling):
    with pytest.raises(ValueError, match='brightness_sampling'):
        sim._brightness_sampling_settings({'brightness_sampling': sampling})
