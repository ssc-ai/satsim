import copy

import numpy as np
import pytest

from satsim.clouds import (
    CloudGeometry,
    cloud_brightness_pe_from_field,
    cloud_field_from_config,
    generate_cloud_field,
    parse_cloud_layers,
)
from satsim.image.fpa import mv_to_pe


def _ssp(clouds=None):
    ssp = {
        'sim': {'seed': 11},
        'fpa': {
            'height': 32,
            'width': 40,
            'y_fov': 1.0,
            'x_fov': 1.2,
        },
    }
    if clouds is not None:
        ssp['clouds'] = clouds
    return ssp


def _geometry():
    return CloudGeometry(height_px=32, width_px=40, y_fov_deg=1.0, x_fov_deg=1.2)


def test_no_clouds_config_returns_none():
    assert cloud_field_from_config(_ssp()) is None
    assert cloud_field_from_config(_ssp([])) is None


def test_preset_layer_uses_defaults():
    layers = parse_cloud_layers([{'type': 'patchy'}], sim_seed=3)

    assert len(layers) == 1
    assert layers[0].cloud_type == 'patchy'
    assert layers[0].coverage == pytest.approx(0.45)
    assert layers[0].density_edge_width == pytest.approx(0.08)
    assert layers[0].brightness is None
    assert layers[0].locality_degree == 2
    assert layers[0].tau_max == pytest.approx(9.0)


def test_preset_layer_applies_public_overrides():
    layers = parse_cloud_layers([
        {
            'type': 'veil',
            'coverage': 0.25,
            'feature_scales_m': [1000, 2000],
            'density_edge_width': 0.2,
            'density_floor': 0.1,
            'brightness': 17.5,
            'texture_contrast': 0.5,
            'locality_degree': 2,
            'tau_min': 0.03,
            'tau_max': 0.7,
            'tau_gamma': 1.4,
        }
    ])

    layer = layers[0]
    assert layer.coverage == pytest.approx(0.25)
    assert layer.feature_scales_m == (1000.0, 2000.0)
    assert layer.density_edge_width == pytest.approx(0.2)
    assert layer.density_floor == pytest.approx(0.1)
    assert layer.brightness == pytest.approx(17.5)
    assert layer.texture_contrast == pytest.approx(0.5)
    assert layer.locality_degree == 2
    assert layer.tau_min == pytest.approx(0.03)
    assert layer.tau_max == pytest.approx(0.7)
    assert layer.tau_gamma == pytest.approx(1.4)


def test_custom_layer_uses_generic_defaults_with_partial_overrides():
    layers = parse_cloud_layers([
        {
            'type': 'custom',
            'coverage': 0.35,
            'tau_max': 0.8,
        }
    ])

    layer = layers[0]
    assert layer.cloud_type == 'custom'
    assert layer.coverage == pytest.approx(0.35)
    assert layer.feature_scales_m == (40.0, 80.0, 160.0, 320.0, 640.0)
    assert layer.density_edge_width == pytest.approx(0.12)
    assert layer.brightness is None
    assert layer.texture_contrast == pytest.approx(1.0)
    assert layer.tau_min == pytest.approx(0.02)
    assert layer.tau_max == pytest.approx(0.8)
    assert layer.tau_gamma == pytest.approx(1.15)


@pytest.mark.parametrize('coverage', [-0.1, 1.1, 20, True])
def test_coverage_accepts_fractions_only(coverage):
    with pytest.raises(ValueError, match='coverage'):
        parse_cloud_layers([{'type': 'patchy', 'coverage': coverage}])


@pytest.mark.parametrize('field', [
    'coverage_mode',
    'mask_threshold',
    'min_feature_scale_px',
    'amplitude_decay',
    'moon_illumination',
    'cloud_albedo',
    'blur_scale_m',
])
def test_hidden_or_unknown_fields_are_rejected(field):
    with pytest.raises(ValueError, match='Unknown cloud layer field'):
        parse_cloud_layers([{'type': 'patchy', field: 1}])


def test_unknown_type_and_invalid_numeric_ranges_fail():
    with pytest.raises(ValueError, match='Unknown cloud type'):
        parse_cloud_layers([{'type': 'not_a_cloud'}])

    with pytest.raises(ValueError, match='tau_max'):
        parse_cloud_layers([{'type': 'custom', 'tau_min': 2.0, 'tau_max': 1.0}])

    with pytest.raises(ValueError, match='feature_scales_m'):
        parse_cloud_layers([{'type': 'custom', 'feature_scales_m': []}])

    with pytest.raises(ValueError, match='feature_scales_m'):
        parse_cloud_layers([{'type': 'custom', 'feature_scales_m': '40,80'}])

    with pytest.raises(ValueError, match='feature_scales_m'):
        parse_cloud_layers([{'type': 'custom', 'feature_scales_m': [40, False]}])

    with pytest.raises(ValueError, match='seed'):
        parse_cloud_layers([{'type': 'custom', 'seed': 1.2}])

    with pytest.raises(ValueError, match='locality_degree'):
        parse_cloud_layers([{'type': 'custom', 'locality_degree': 1.2}])

    with pytest.raises(ValueError, match='locality_degree'):
        parse_cloud_layers([{'type': 'custom', 'locality_degree': True}])

    with pytest.raises(ValueError, match='brightness'):
        parse_cloud_layers([{'type': 'custom', 'brightness': 'bright'}])

    with pytest.raises(ValueError, match='brightness'):
        parse_cloud_layers([{'type': 'custom', 'brightness': float('inf')}])

    with pytest.raises(ValueError, match='brightness'):
        parse_cloud_layers([{'type': 'custom', 'brightness': True}])


def test_brightness_is_in_layer_metadata():
    layers = parse_cloud_layers([
        {
            'type': 'custom',
            'brightness': 17.5,
        }
    ])

    field = generate_cloud_field(layers, _geometry())

    assert field.layers[0].metadata['brightness'] == pytest.approx(17.5)
    assert field.metadata['layers'][0]['brightness'] == pytest.approx(17.5)


def test_multi_layer_clouds_combine_deterministically():
    layers = parse_cloud_layers([
        {'type': 'patchy', 'coverage': 0.2},
        {'type': 'veil', 'coverage': 0.5},
    ], sim_seed=19)

    first = generate_cloud_field(layers, _geometry())
    second = generate_cloud_field(layers, _geometry())

    np.testing.assert_array_equal(first.density, second.density)
    np.testing.assert_array_equal(first.tau, second.tau)
    np.testing.assert_array_equal(first.transmission, second.transmission)

    expected_transmission = first.layers[0].transmission * first.layers[1].transmission
    expected_tau = first.layers[0].tau + first.layers[1].tau
    expected_mask = first.layers[0].mask | first.layers[1].mask
    expected_density = 1.0 - (1.0 - first.layers[0].density) * (1.0 - first.layers[1].density)

    np.testing.assert_allclose(first.transmission, expected_transmission)
    np.testing.assert_allclose(first.tau, expected_tau)
    np.testing.assert_array_equal(first.mask, expected_mask)
    np.testing.assert_allclose(first.density, expected_density)


def test_cloud_brightness_pe_scales_with_inverted_transmission():
    layers = parse_cloud_layers([
        {'type': 'patchy', 'coverage': 0.2, 'brightness': 17.5},
    ], sim_seed=31)
    field = generate_cloud_field(layers, _geometry())

    zeropoint = 23.0
    exposure_s = 2.5
    pixel_area_arcsec2 = 7.25
    brightness_pe = cloud_brightness_pe_from_field(field, zeropoint, exposure_s, pixel_area_arcsec2)

    expected_base = mv_to_pe(zeropoint, 17.5) * pixel_area_arcsec2 * exposure_s
    expected = expected_base * (1.0 - field.layers[0].transmission)

    np.testing.assert_allclose(brightness_pe, expected, rtol=1e-6)


def test_cloud_brightness_pe_adds_per_bright_layer():
    layers = parse_cloud_layers([
        {'type': 'patchy', 'coverage': 0.2, 'brightness': 17.5},
        {'type': 'veil', 'coverage': 0.5},
        {'type': 'fog', 'coverage': 0.6, 'brightness': 19.0},
    ], sim_seed=37)
    field = generate_cloud_field(layers, _geometry())

    zeropoint = 23.0
    exposure_s = 3.0
    pixel_area_arcsec2 = 5.0
    brightness_pe = cloud_brightness_pe_from_field(field, zeropoint, exposure_s, pixel_area_arcsec2)

    expected = (
        mv_to_pe(zeropoint, 17.5) * pixel_area_arcsec2 * exposure_s * (1.0 - field.layers[0].transmission)
        + mv_to_pe(zeropoint, 19.0) * pixel_area_arcsec2 * exposure_s * (1.0 - field.layers[2].transmission)
    )

    np.testing.assert_allclose(brightness_pe, expected, rtol=1e-6)


def test_cloud_brightness_pe_is_zero_without_bright_layers():
    layers = parse_cloud_layers([
        {'type': 'patchy', 'coverage': 0.2},
        {'type': 'veil', 'coverage': 0.5},
    ], sim_seed=41)
    field = generate_cloud_field(layers, _geometry())

    brightness_pe = cloud_brightness_pe_from_field(field, 23.0, 3.0, 5.0)

    np.testing.assert_array_equal(brightness_pe, np.zeros_like(field.transmission))


def test_cloud_field_from_config_does_not_mutate_config():
    ssp = _ssp([{'type': 'custom', 'coverage': 0.4}])
    original = copy.deepcopy(ssp)

    field = cloud_field_from_config(ssp)

    assert field is not None
    assert ssp == original
