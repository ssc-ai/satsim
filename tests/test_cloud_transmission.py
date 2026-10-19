"""Coverage, interior variation and transmission-bound regressions."""

import json

import numpy as np
import pytest

from satsim.clouds import CloudGeometry, crop_cloud_field, generate_cloud_field, parse_cloud_layers
from satsim.clouds.field import cloud_layer_from_density, combine_cloud_layers, optical_fields
from satsim.clouds.generation import cloud_density_from_noise
pytestmark = pytest.mark.filterwarnings('ignore:jsonschema.RefResolver is deprecated:DeprecationWarning')

from tests.test_clouds import (
    _cloud_schema_validator,
    _cloudy_and_clear_frames,
    _pre_psf_target_ssp,
)


def config(**kwargs):
    return parse_cloud_layers([dict(type='custom', seed=42, **kwargs)])[0]


@pytest.mark.parametrize('settings', [
    None, {}, {'min': 0.2}, {'max': 0.8},
    {'min': 0, 'max': 0.8}, {'min': 1e-50, 'max': 0.8},
    {'min': -0.1, 'max': 0.8}, {'min': 0.9, 'max': 0.8},
    {'min': 0.2, 'max': 1.1}, {'min': True, 'max': 1},
    {'min': float('nan'), 'max': 1}, {'min': 0.2, 'max': float('inf')},
    {'min': 0.2, 'max': 0.8, 'gamma': 0},
    {'min': 0.2, 'max': 0.8, 'gamma': True},
    {'min': 0.2, 'max': 0.8, 'gamme': 2},
])
def test_invalid_transmission_settings(settings):
    with pytest.raises(ValueError, match='transmission'):
        config(transmission=settings)


@pytest.mark.parametrize('key', ['tau_min', 'tau_max', 'tau_gamma'])
@pytest.mark.parametrize('grouped', [False, True])
def test_transmission_rejects_ambiguous_tau_controls(key, grouped):
    overrides = {'optical': {key: 0.5}} if grouped else {key: 0.5}
    raw = dict(type='custom', transmission={'min': 0.2, 'max': 0.8}, **overrides)
    with pytest.raises(ValueError, match='explicit tau'):
        parse_cloud_layers([raw])
    _, validator = _cloud_schema_validator()
    assert list(validator.iter_errors([raw]))


def test_schema_and_metadata_record_active_model():
    raw = {'type': 'patchy', 'transmission': {'min': 0.2, 'max': 0.8, 'gamma': 2}}
    _, validator = _cloud_schema_validator()
    validator.validate([raw])
    metadata = parse_cloud_layers([raw], sim_seed=1)[0].metadata()
    assert metadata['model'] == 'textured-v2'
    assert metadata['transmission'] == raw['transmission']
    assert 'tau_min' not in metadata
    json.dumps(metadata, allow_nan=False)


def test_transmission_mapping_bounds_clear_sky_and_smooth_edge():
    cfg = config(transmission={'min': 0.2, 'max': 0.8})
    texture = np.array([0, 0.5, 1], dtype=np.float32)
    tau, transmission = optical_fields(np.ones(3), cfg, texture)
    np.testing.assert_allclose(transmission, [0.8, 0.5, 0.2])
    np.testing.assert_allclose(np.exp(-tau), transmission, rtol=1e-6)
    edge = np.linspace(0, 1, 1001, dtype=np.float32)
    _, transmission = optical_fields(edge, cfg, np.full_like(edge, 0.5))
    np.testing.assert_allclose(transmission, 1 - 0.5 * edge, atol=1e-7)
    assert transmission[0] == 1
    assert np.max(np.abs(np.diff(transmission))) < 0.001


@pytest.mark.parametrize('low,high', [(1, 1), (0.4, 0.4), (float(np.finfo(np.float32).tiny), 1)])
def test_equal_bounds_and_nearly_opaque_endpoint(low, high):
    cfg = config(transmission={'min': low, 'max': high})
    tau, transmission = optical_fields(np.ones(3), cfg, np.array([0, 0.5, 1]))
    assert np.all(np.isfinite(tau))
    np.testing.assert_allclose(transmission[[0, -1]], [high, low])
    assert transmission[-1] > 0
    if low == high:
        np.testing.assert_array_equal(transmission, np.full(3, low, dtype=np.float32))


@pytest.mark.parametrize('cloud_type', ['patchy', 'cellular', 'veil', 'sheet', 'fog'])
@pytest.mark.parametrize('coverage', [0.5, 1.0])
def test_generated_clouds_retain_variation_under_flat_envelope(cloud_type, coverage):
    cfg = parse_cloud_layers([{'type': cloud_type, 'coverage': coverage}], sim_seed=42)[0]
    texture = np.linspace(0, 1, 10001, dtype=np.float32)
    envelope = cloud_density_from_noise(texture, coverage, cfg.density_edge_width, cfg.density_floor,
                                        cfg.mask_threshold)
    _, transmission = optical_fields(envelope, cfg, texture)
    # Inspect the largest exactly flat envelope region, not whole-image variance.
    values, counts = np.unique(envelope, return_counts=True)
    plateau = envelope == values[np.argmax(counts)]
    if envelope[plateau][0] > 0:
        assert plateau.sum() > 100
        assert np.ptp(transmission[plateau]) > 1e-4
        assert np.unique(transmission[plateau]).size > 100
    else:
        np.testing.assert_array_equal(transmission[plateau], np.ones(plateau.sum()))


@pytest.mark.parametrize('bounded', [False, True])
def test_full_coverage_is_textured_and_zero_contrast_is_uniform(bounded):
    options = {'transmission': {'min': 0.2, 'max': 0.8}} if bounded else {}
    geometry = CloudGeometry(64, 64, 1, 1)
    cfg = config(coverage=1, **options)
    field = generate_cloud_field((cfg,), geometry)
    assert field.mask.all()
    assert np.ptp(field.transmission) > 0.1
    uniform = generate_cloud_field((config(coverage=1, texture_contrast=0, **options),), geometry)
    assert np.ptp(uniform.transmission) == 0
    clear = generate_cloud_field((config(coverage=0, density_floor=0.5, **options),), geometry)
    np.testing.assert_array_equal(clear.transmission, np.ones((64, 64)))


def test_moving_crops_share_texture_and_layer_optics_without_renormalizing():
    configs = (config(coverage=0.6, transmission={'min': 0.2, 'max': 0.8}),
               config(coverage=1, transmission={'min': 0.5, 'max': 0.7}))
    source = generate_cloud_field(configs, CloudGeometry(80, 80, 1, 1))
    replay = generate_cloud_field(configs, CloudGeometry(80, 80, 1, 1))
    np.testing.assert_array_equal(source.transmission, replay.transmission)
    a = crop_cloud_field(source, (0.25, 0.5), 32, 32, pad_px=(10, 10))
    b = crop_cloud_field(source, (-2.75, -3.5), 32, 32, pad_px=(10, 10))
    np.testing.assert_allclose(a.transmission[3:, 4:], b.transmission[:-3, :-4], atol=1e-7)
    np.testing.assert_allclose(a.transmission, a.layers[0].transmission * a.layers[1].transmission)
    np.testing.assert_allclose(np.exp(-a.tau), a.transmission, rtol=1e-6)
    assert source.transmission.min() < 0.2  # Per-layer lower bounds do not bound the stack.
    json.dumps(a.metadata, allow_nan=False)


def test_bounded_glow_and_stack_use_the_same_transmission():
    from satsim.clouds import cloud_brightness_pe_from_field
    from satsim.image.fpa import mv_to_pe
    cfg = config(transmission={'min': 0.2, 'max': 0.8}, brightness=18)
    layer = cloud_layer_from_density(cfg, np.array([[0, 0.5, 1]]), np.array([[0, 0.5, 1]]))
    field = combine_cloud_layers((layer,))
    glow = cloud_brightness_pe_from_field(field, 22, 2, 3)
    np.testing.assert_allclose(glow, mv_to_pe(22, 18) * 6 * (1 - field.transmission))
    assert glow[0, 0] == 0


@pytest.mark.parametrize('cloud_type,coverage', [('fog', 0.5), ('veil', 0.1), ('sheet', 0.9), ('patchy', 0.9)])
def test_baseline_haze_obeys_ceiling_everywhere(cloud_type, coverage):
    cfg = parse_cloud_layers([{'type': cloud_type, 'coverage': coverage, 'seed': 42,
                               'transmission': {'min': 0.2, 'max': 0.8}}])[0]
    field = generate_cloud_field((cfg,), CloudGeometry(64, 64, 1, 1))
    assert field.density.min() > 0
    assert field.transmission.max() <= 0.8 + 1e-7
    assert field.transmission.min() >= 0.2 - 1e-7
    assert np.ptp(field.transmission) > 1e-4
    # The least-dense baseline region must still carry optical texture.
    baseline = field.density == field.density.min()
    if baseline.sum() > 10:
        assert np.ptp(field.transmission[baseline]) > 1e-5
    crop = crop_cloud_field(field, (-0.25, -0.5), 32, 32)
    assert crop.transmission.max() <= 0.8 + 1e-7


@pytest.mark.parametrize('mode', ['fftconv2p', 'epsf'])
@pytest.mark.parametrize('velocity', [[0, 0], [0, 8]])
def test_rendered_flux_and_annotations_agree_for_bounded_clouds(mode, velocity):
    ssp = _pre_psf_target_ssp(velocity)
    ssp['sim']['mode'] = mode
    if mode == 'epsf':
        ssp['sim']['epsf'] = {'kernel_size': 21}
    ssp['clouds'] = [{'type': 'custom', 'seed': 123, 'coverage': 1,
                      'feature_scales_m': [5, 10], 'transmission': {'min': 0.2, 'max': 0.8}}]
    cloudy, clear = _cloudy_and_clear_frames(ssp)
    measured = np.sum(cloudy[0][5].numpy()) / np.sum(clear[0][5].numpy())
    np.testing.assert_allclose(measured, cloudy[0][3][0]['cloud_transmission'], rtol=1e-3)
    truth = np.asarray(cloudy[0][11]['cloud_transmission'])
    assert truth.min() >= 0.2 - 1e-6
    assert truth.max() <= 0.8 + 1e-6
    assert np.ptp(truth) > 0.1
