import copy
import math

import numpy as np
import pytest

import satsim.clouds.config as clouds_config
from satsim.clouds import (
    CloudField,
    CloudGeometry,
    CloudLayer,
    cloud_brightness_pe_from_field,
    cloud_source_brightness_pe_from_field,
    cloud_field_from_config,
    cloud_geometry_from_config,
    crop_cloud_field,
    generate_cloud_field,
    parse_cloud_layers,
)
from satsim.clouds.constants import LUNAR_DIRECT_BRIGHTENING, SOLAR_DIRECT_BRIGHTENING
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


def _cloud_field_from_densities(densities, brightness=None, layer_overrides=None):
    densities = [np.asarray(density, dtype=np.float32) for density in densities]
    layer_overrides = layer_overrides or {}
    configs = parse_cloud_layers([
        dict({
            'type': 'custom',
            'brightness': brightness,
            'tau_min': 0.0,
            'tau_max': 1.0,
            'tau_gamma': 1.0,
        }, **layer_overrides)
        for _ in densities
    ], sim_seed=1)
    layers = []
    for config, density in zip(configs, densities):
        layers.append(CloudLayer(
            config=config,
            density=density,
            mask=density > config.mask_threshold,
            tau=np.zeros_like(density, dtype=np.float32),
            transmission=np.ones_like(density, dtype=np.float32),
            metadata=config.metadata(),
        ))
    shape = densities[0].shape
    return CloudField(
        layers=tuple(layers),
        density=np.zeros(shape, dtype=np.float32),
        mask=np.zeros(shape, dtype=bool),
        tau=np.zeros(shape, dtype=np.float32),
        transmission=np.ones(shape, dtype=np.float32),
        metadata={},
    )


def test_no_clouds_config_returns_none():
    assert cloud_field_from_config(_ssp()) is None
    assert cloud_field_from_config(_ssp([])) is None


def test_invalid_cloud_collection_and_layer_shape_fail():
    with pytest.raises(ValueError, match='clouds must be a list'):
        parse_cloud_layers({'type': 'patchy'})

    with pytest.raises(ValueError, match=r'clouds\[0\] must be an object'):
        parse_cloud_layers([None])

    with pytest.raises(ValueError, match=r'clouds\[0\].type is required'):
        parse_cloud_layers([{}])

    with pytest.raises(ValueError, match='enabled must be a boolean'):
        parse_cloud_layers([{'type': 'patchy', 'enabled': 'yes'}])


def test_disabled_layers_are_omitted():
    assert parse_cloud_layers([{'type': 'patchy', 'enabled': False}]) == tuple()

    layers = parse_cloud_layers([
        {'type': 'patchy', 'enabled': False},
        {'type': 'veil', 'coverage': 0.3},
    ], sim_seed=5)

    assert len(layers) == 1
    assert layers[0].cloud_type == 'veil'


def test_preset_layer_uses_defaults():
    layers = parse_cloud_layers([{'type': 'patchy'}], sim_seed=3)

    assert len(layers) == 1
    assert layers[0].cloud_type == 'patchy'
    assert layers[0].coverage == pytest.approx(0.45)
    assert layers[0].density_edge_width == pytest.approx(0.08)
    assert layers[0].brightness is None
    assert layers[0].cloud_range == pytest.approx(3.0)
    assert layers[0].altitude is None
    assert layers[0].wind_speed == pytest.approx(0.0)
    assert layers[0].wind_direction == pytest.approx(0.0)
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
            'range': 6.0,
            'altitude': 1.2,
            'wind_speed': 4.0,
            'wind_direction': 450.0,
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
    assert layer.cloud_range == pytest.approx(6.0)
    assert layer.altitude == pytest.approx(1.2)
    assert layer.wind_speed == pytest.approx(4.0)
    assert layer.wind_direction == pytest.approx(90.0)
    assert layer.texture_contrast == pytest.approx(0.5)
    assert layer.locality_degree == 2
    assert layer.tau_min == pytest.approx(0.03)
    assert layer.tau_max == pytest.approx(0.7)
    assert layer.tau_gamma == pytest.approx(1.4)


def test_grouped_layer_config_applies_public_overrides():
    layers = parse_cloud_layers([
        {
            'type': 'veil',
            'coverage': 0.25,
            'texture': {
                'scales_m': [1000, 2000],
                'edge_width': 0.2,
                'floor': 0.1,
                'contrast': 0.5,
                'locality_degree': 2,
            },
            'illumination': {
                'brightness_mag_arcsec2': 17.5,
            },
            'geometry': {
                'range_km': 6.0,
                'altitude_km': 1.2,
            },
            'motion': {
                'speed_m_per_s': 4.0,
                'direction_deg': 450.0,
            },
            'optical': {
                'tau_min': 0.03,
                'tau_max': 0.7,
                'tau_gamma': 1.4,
            },
        }
    ])

    layer = layers[0]
    assert layer.coverage == pytest.approx(0.25)
    assert layer.feature_scales_m == (1000.0, 2000.0)
    assert layer.density_edge_width == pytest.approx(0.2)
    assert layer.density_floor == pytest.approx(0.1)
    assert layer.brightness == pytest.approx(17.5)
    assert layer.cloud_range == pytest.approx(6.0)
    assert layer.altitude == pytest.approx(1.2)
    assert layer.wind_speed == pytest.approx(4.0)
    assert layer.wind_direction == pytest.approx(90.0)
    assert layer.texture_contrast == pytest.approx(0.5)
    assert layer.locality_degree == 2
    assert layer.tau_min == pytest.approx(0.03)
    assert layer.tau_max == pytest.approx(0.7)
    assert layer.tau_gamma == pytest.approx(1.4)


def test_grouped_layer_config_rejects_duplicate_logical_fields():
    with pytest.raises(ValueError, match='defines brightness both directly and inside illumination'):
        parse_cloud_layers([
            {
                'type': 'custom',
                'brightness': 17.5,
                'illumination': {'brightness': 18.0},
            }
        ])

    with pytest.raises(ValueError, match='defines feature_scales_m more than once'):
        parse_cloud_layers([
            {
                'type': 'custom',
                'texture': {
                    'feature_scales_m': [40.0],
                    'scales_m': [80.0],
                },
            }
        ])


def test_grouped_layer_config_rejects_unknown_or_non_object_groups():
    with pytest.raises(ValueError, match='Unknown cloud layer field'):
        parse_cloud_layers([
            {
                'type': 'custom',
                'texture': {'amplitude_decay': 0.8},
            }
        ])

    with pytest.raises(ValueError, match='motion must be an object'):
        parse_cloud_layers([
            {
                'type': 'custom',
                'motion': 'fast',
            }
        ])


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
    assert layer.cloud_range == pytest.approx(3.0)
    assert layer.altitude is None
    assert layer.wind_speed == pytest.approx(0.0)
    assert layer.wind_direction == pytest.approx(0.0)
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
        parse_cloud_layers([{'type': 'custom', 'feature_scales_m': None}])

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

    with pytest.raises(ValueError, match='range'):
        parse_cloud_layers([{'type': 'custom', 'range': 0.0}])

    with pytest.raises(ValueError, match='altitude'):
        parse_cloud_layers([{'type': 'custom', 'altitude': -1.0}])

    with pytest.raises(ValueError, match='wind_speed'):
        parse_cloud_layers([{'type': 'custom', 'wind_speed': -1.0}])

    with pytest.raises(ValueError, match='wind_speed'):
        parse_cloud_layers([{'type': 'custom', 'wind_speed': 'fast'}])

    with pytest.raises(ValueError, match='wind_direction'):
        parse_cloud_layers([{'type': 'custom', 'wind_direction': None}])

    with pytest.raises(ValueError, match='wind_direction'):
        parse_cloud_layers([{'type': 'custom', 'wind_direction': True}])


def test_brightness_is_in_layer_metadata():
    layers = parse_cloud_layers([
        {
            'type': 'custom',
            'brightness': 17.5,
            'range': 5.0,
            'altitude': 1.0,
            'wind_speed': 3.0,
            'wind_direction': -90.0,
        }
    ])

    field = generate_cloud_field(layers, _geometry())

    assert field.layers[0].metadata['brightness'] == pytest.approx(17.5)
    assert field.layers[0].metadata['range'] == pytest.approx(5.0)
    assert field.layers[0].metadata['altitude'] == pytest.approx(1.0)
    assert field.layers[0].metadata['wind_speed'] == pytest.approx(3.0)
    assert field.layers[0].metadata['wind_direction'] == pytest.approx(270.0)
    assert field.metadata['layers'][0]['brightness'] == pytest.approx(17.5)
    assert field.metadata['layers'][0]['range'] == pytest.approx(5.0)
    assert field.metadata['layers'][0]['altitude'] == pytest.approx(1.0)
    assert field.metadata['layers'][0]['wind_speed'] == pytest.approx(3.0)
    assert field.metadata['layers'][0]['wind_direction'] == pytest.approx(270.0)


def test_cloud_generation_handles_empty_field_and_missing_geometry():
    assert generate_cloud_field(tuple(), _geometry()) is None
    assert cloud_brightness_pe_from_field(None, 23.0, 1.0, 1.0) is None

    with pytest.raises(ValueError, match='Cloud generation requires'):
        cloud_field_from_config({'clouds': [{'type': 'patchy'}]})


def test_top_level_seed_is_used_when_sim_seed_is_absent():
    ssp = _ssp([{'type': 'custom', 'coverage': 0.0}])
    del ssp['sim']
    ssp['seed'] = 23

    field = cloud_field_from_config(ssp)

    assert field.layers[0].config.seed == 23


def test_missing_config_seed_uses_random_cloud_seed(monkeypatch):
    monkeypatch.setattr(clouds_config, '_random_seed', lambda: 12345)
    ssp = _ssp([
        {'type': 'custom', 'coverage': 0.0},
        {'type': 'custom', 'coverage': 0.0},
    ])
    del ssp['sim']

    field = cloud_field_from_config(ssp)

    assert field.layers[0].config.seed == 12345
    assert field.layers[1].config.seed == 22345
    assert field.metadata['layers'][0]['seed'] == 12345
    assert field.metadata['layers'][1]['seed'] == 22345


def test_layer_seed_overrides_random_cloud_seed(monkeypatch):
    monkeypatch.setattr(clouds_config, '_random_seed', lambda: 12345)

    layers = parse_cloud_layers([{'type': 'custom', 'seed': 77}])

    assert layers[0].seed == 77


def test_asymmetric_cloud_padding_preserves_fpa_coordinate_origin():
    ssp = _ssp([{'type': 'custom', 'coverage': 0.0}])
    symmetric = cloud_geometry_from_config(ssp, y_pad_px=5, x_pad_px=4)
    asymmetric = cloud_geometry_from_config(
        ssp,
        y_pad_px=5,
        x_pad_px=4,
        y_pad_after_px=0,
        x_pad_after_px=0,
    )

    def row_coordinate(geometry, row):
        return (
            (float(row) + 0.5) * geometry.y_meters_per_pixel -
            geometry.footprint_height_m * 0.5 -
            geometry.y_center_offset_m
        )

    def col_coordinate(geometry, col):
        return (
            (float(col) + 0.5) * geometry.x_meters_per_pixel -
            geometry.footprint_width_m * 0.5 -
            geometry.x_center_offset_m
        )

    assert asymmetric.height_px == symmetric.height_px - 5
    assert asymmetric.width_px == symmetric.width_px - 4
    assert row_coordinate(asymmetric, 5) == pytest.approx(row_coordinate(symmetric, 5))
    assert col_coordinate(asymmetric, 4) == pytest.approx(col_coordinate(symmetric, 4))


def test_zero_and_full_coverage_layers_have_stable_optics():
    geometry = CloudGeometry(height_px=8, width_px=10, y_fov_deg=1.0, x_fov_deg=1.0)

    clear = generate_cloud_field(parse_cloud_layers([
        {'type': 'custom', 'coverage': 0.0},
    ]), geometry)

    np.testing.assert_array_equal(clear.density, np.zeros_like(clear.density))
    np.testing.assert_array_equal(clear.mask, np.zeros_like(clear.mask, dtype=bool))
    np.testing.assert_array_equal(clear.tau, np.zeros_like(clear.tau))
    np.testing.assert_array_equal(clear.transmission, np.ones_like(clear.transmission))

    opaque = generate_cloud_field(parse_cloud_layers([
        {
            'type': 'custom',
            'coverage': 1.0,
            'tau_min': 0.1,
            'tau_max': 0.7,
            'tau_gamma': 1.0,
        },
    ]), geometry)

    np.testing.assert_array_equal(opaque.density, np.ones_like(opaque.density))
    np.testing.assert_array_equal(opaque.mask, np.ones_like(opaque.mask, dtype=bool))
    np.testing.assert_allclose(opaque.tau, np.full_like(opaque.tau, 0.7))
    np.testing.assert_allclose(opaque.transmission, np.exp(np.full_like(opaque.transmission, -0.7)))


def test_cloud_crop_without_offset_returns_center_view():
    density = np.arange(25, dtype=np.float32).reshape(5, 5) / 24.0
    source = _cloud_field_from_densities([density])

    cropped = crop_cloud_field(source, [0.0, 0.0], 3, 3, pad_px=[1, 1])

    np.testing.assert_allclose(cropped.layers[0].density, density[1:4, 1:4])
    assert cropped.metadata['motion']['crop_offset_px'] == [0.0, 0.0]
    assert cropped.metadata['motion']['clamped'] is False


def test_positive_cloud_crop_offset_moves_features_down_and_right():
    density = np.zeros((7, 7), dtype=np.float32)
    density[3, 3] = 1.0
    source = _cloud_field_from_densities([density])

    static = crop_cloud_field(source, [0.0, 0.0], 3, 3, pad_px=[3, 3])
    moved = crop_cloud_field(source, [1.0, 1.0], 3, 3, pad_px=[3, 3])

    assert static.layers[0].density[0, 0] == pytest.approx(1.0)
    assert moved.layers[0].density[1, 1] == pytest.approx(1.0)


def test_fractional_cloud_crop_offset_uses_bilinear_sampling():
    density = np.array([
        [0.0, 1.0],
        [2.0, 3.0],
    ], dtype=np.float32)
    source = _cloud_field_from_densities([density])

    cropped = crop_cloud_field(source, [-0.5, -0.5], 1, 1, pad_px=[0, 0])

    np.testing.assert_allclose(cropped.layers[0].density, [[1.5]])


def test_cropped_cloud_layers_recombine_physical_stack():
    first_density = np.full((4, 4), 0.2, dtype=np.float32)
    second_density = np.full((4, 4), 0.5, dtype=np.float32)
    source = _cloud_field_from_densities([first_density, second_density])

    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[1, 1])

    expected_density = 1.0 - (1.0 - cropped.layers[0].density) * (1.0 - cropped.layers[1].density)
    expected_tau = cropped.layers[0].tau + cropped.layers[1].tau
    expected_transmission = cropped.layers[0].transmission * cropped.layers[1].transmission

    np.testing.assert_allclose(cropped.density, expected_density)
    np.testing.assert_allclose(cropped.tau, expected_tau)
    np.testing.assert_allclose(cropped.transmission, expected_transmission)
    np.testing.assert_array_equal(cropped.mask, cropped.layers[0].mask | cropped.layers[1].mask)


def test_cloud_crop_accepts_per_layer_offsets():
    first_density = np.zeros((4, 4), dtype=np.float32)
    second_density = np.zeros((4, 4), dtype=np.float32)
    first_density[1, 1] = 1.0
    second_density[1, 2] = 1.0
    source = _cloud_field_from_densities([first_density, second_density])

    cropped = crop_cloud_field(
        source,
        [0.0, 0.0],
        2,
        2,
        pad_px=[1, 1],
        layer_offsets_px=[[0.0, 0.0], [0.0, -1.0]],
    )

    assert cropped.layers[0].density[0, 0] == pytest.approx(1.0)
    assert cropped.layers[1].density[0, 0] == pytest.approx(1.0)
    assert cropped.metadata['motion']['layer_crop_offsets_px'] == [[0.0, 0.0], [0.0, -1.0]]


def test_cloud_brightness_uses_cropped_transmission():
    density = np.arange(25, dtype=np.float32).reshape(5, 5) / 24.0
    source = _cloud_field_from_densities([density], brightness=17.5)
    cropped = crop_cloud_field(source, [0.0, 0.0], 3, 3, pad_px=[1, 1])

    brightness_pe = cloud_brightness_pe_from_field(cropped, 23.0, 2.0, 4.0)
    expected_base = mv_to_pe(23.0, 17.5) * 2.0 * 4.0
    expected = expected_base * (1.0 - cropped.layers[0].transmission)

    np.testing.assert_allclose(brightness_pe, expected, rtol=1e-6)


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


def test_cloud_source_brightness_uses_source_components_and_altitude_gain():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {'artificial': 10.0, 'lunar': 4.0, 'solar': 2.0},
    )
    optical_coupling = 1.0 - np.exp(-1.0)
    expected = optical_coupling * (
        10.0 * 1.5 * math.exp(-2000.0 / 4000.0)
        + 4.0 * 0.35 * math.exp(-2000.0 / 8000.0)
        + 2.0 * 0.25 * math.exp(-2000.0 / 12000.0)
    )

    np.testing.assert_allclose(source_pe, np.full((2, 2), expected), rtol=1e-6)


def test_cloud_lunar_source_uses_direct_illumination_metadata():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {
            'lunar': {
                'pe': 4.0,
                'metadata': {
                    'moon_el': 39.0,
                    'phase_angle': 45.0,
                },
            },
        },
    )

    optical_coupling = 1.0 - np.exp(-1.0)
    phase_fraction = 0.5 * (1.0 + math.cos(math.radians(45.0)))
    expected_gain = (
        LUNAR_DIRECT_BRIGHTENING['gain']
        * math.exp(-2000.0 / LUNAR_DIRECT_BRIGHTENING['scale_height_m'])
        * math.sqrt(math.sin(math.radians(39.0)))
        * math.sqrt(phase_fraction)
    )
    expected = 4.0 * optical_coupling * expected_gain

    np.testing.assert_allclose(source_pe, np.full((2, 2), expected), rtol=1e-6)


def test_cloud_lunar_direct_illumination_is_zero_below_horizon():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {
            'lunar': {
                'pe': 4.0,
                'metadata': {
                    'moon_el': -1.0,
                    'phase_angle': 0.0,
                },
            },
        },
    )

    np.testing.assert_array_equal(source_pe, np.zeros((2, 2), dtype=np.float32))


def test_cloud_solar_source_uses_direct_illumination_metadata():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {
            'solar': {
                'pe': 6.0,
                'metadata': {
                    'sun_el': 12.0,
                },
            },
        },
    )

    optical_coupling = 1.0 - np.exp(-1.0)
    expected_gain = (
        SOLAR_DIRECT_BRIGHTENING['gain']
        * math.exp(-2000.0 / SOLAR_DIRECT_BRIGHTENING['scale_height_m'])
        * math.sqrt(math.sin(math.radians(12.0)))
    )
    expected = 6.0 * optical_coupling * expected_gain

    np.testing.assert_allclose(source_pe, np.full((2, 2), expected), rtol=1e-6)


def test_cloud_solar_source_uses_twilight_ramp_below_horizon():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {
            'solar': {
                'pe': 6.0,
                'metadata': {
                    'sun_el': -12.0,
                },
            },
        },
    )

    optical_coupling = 1.0 - np.exp(-1.0)
    twilight_t = (-12.0 - SOLAR_DIRECT_BRIGHTENING['twilight_min_sun_el_deg']) / abs(
        SOLAR_DIRECT_BRIGHTENING['twilight_min_sun_el_deg']
    )
    twilight_gain = twilight_t * twilight_t * (3.0 - 2.0 * twilight_t)
    expected_gain = (
        SOLAR_DIRECT_BRIGHTENING['gain']
        * math.exp(-2000.0 / SOLAR_DIRECT_BRIGHTENING['scale_height_m'])
        * twilight_gain
    )
    expected = 6.0 * optical_coupling * expected_gain

    np.testing.assert_allclose(source_pe, np.full((2, 2), expected), rtol=1e-6)


def test_cloud_solar_direct_illumination_is_zero_after_astronomical_twilight():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], layer_overrides={'altitude': 2.0})
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    source_pe = cloud_source_brightness_pe_from_field(
        cropped,
        {
            'solar': {
                'pe': 6.0,
                'metadata': {
                    'sun_el': -19.0,
                },
            },
        },
    )

    np.testing.assert_array_equal(source_pe, np.zeros((2, 2), dtype=np.float32))


def test_cloud_brightness_returns_config_and_source_diagnostics():
    density = np.ones((2, 2), dtype=np.float32)
    source = _cloud_field_from_densities([density], brightness=17.5)
    cropped = crop_cloud_field(source, [0.0, 0.0], 2, 2, pad_px=[0, 0])

    components = cloud_brightness_pe_from_field(
        cropped,
        23.0,
        2.0,
        4.0,
        source_components={'artificial': 10.0},
        return_components=True,
    )

    np.testing.assert_allclose(
        components['cloud_brightness_pe'],
        components['cloud_config_brightness_pe'] + components['cloud_source_brightness_pe'],
        rtol=1e-6,
    )
    assert np.mean(components['cloud_config_brightness_pe']) > 0.0
    assert np.mean(components['cloud_source_brightness_pe']) > 0.0


def test_cloud_field_from_config_does_not_mutate_config():
    ssp = _ssp([{'type': 'custom', 'coverage': 0.4}])
    original = copy.deepcopy(ssp)

    field = cloud_field_from_config(ssp)

    assert field is not None
    assert ssp == original


def test_cloud_motion_rate_sidereal_last_frame_keeps_accumulated_offset(monkeypatch):
    from satsim import satsim as satsim_module

    calls = []

    def fake_star_motion(*args, **kwargs):
        calls.append({
            'track_mode': args[18],
            'exposure_s': float(args[6]),
        })
        return None, None, [10.0, -2.0], 0.0

    monkeypatch.setattr(satsim_module, '_calculate_star_position_and_motion', fake_star_motion)

    tt = [2025, 1, 1, 0, 0, 0.0]
    ts_collect_start = satsim_module.time.utc_from_list(tt)
    ts_collect_end = satsim_module.time.utc_from_list(tt, 6.0)

    motion = satsim_module._plan_cloud_motion(
        {},
        num_frames=3,
        t_frame=2.0,
        t_exposure=1.0,
        tt=tt,
        ts_collect_start=ts_collect_start,
        ts_collect_end=ts_collect_end,
        s_osf=1,
        star_tran_os=[0.0, 0.0],
        track_mode='rate-sidereal',
        observer=None,
        track=None,
        star_rot=0.0,
        track_az=None,
        track_el=None,
        track_apparent=True,
        track_ra=None,
        track_dec=None,
        h_fpa_pad_os=10,
        w_fpa_pad_os=10,
        y_fov_pad=1.0,
        x_fov_pad=1.0,
        y_fov=1.0,
        x_fov=1.0,
        y_ifov=0.1,
        x_ifov=0.1,
    )

    assert [call['track_mode'] for call in calls] == ['rate', 'rate', 'rate']
    assert [call['exposure_s'] for call in calls] == pytest.approx([0.5, 2.5, 4.0])
    assert motion['frames'][0]['total_offset_px'] == pytest.approx([5.0, -1.0])
    assert motion['frames'][1]['total_offset_px'] == pytest.approx([25.0, -5.0])
    assert motion['frames'][2]['total_offset_px'] == pytest.approx([40.0, -8.0])
    assert motion['pad_px'] == [42, 0]
    assert motion['pad_after_px'] == [0, 10]


def test_cloud_motion_adds_per_layer_wind_offsets():
    from satsim import satsim as satsim_module

    tt = [2025, 1, 1, 0, 0, 0.0]
    ts_collect_start = satsim_module.time.utc_from_list(tt)
    ts_collect_end = satsim_module.time.utc_from_list(tt, 2.0)
    layer_configs = parse_cloud_layers([
        {'type': 'custom', 'wind_speed': 2.0, 'wind_direction': 90.0, 'range': 0.001},
        {'type': 'custom', 'wind_speed': 1.0, 'wind_direction': 0.0, 'range': 0.001},
    ], sim_seed=1)
    geometry = CloudGeometry(
        height_px=1,
        width_px=1,
        y_fov_deg=math.degrees(1.0),
        x_fov_deg=math.degrees(1.0),
        cloud_range_m=1.0,
    )

    motion = satsim_module._plan_cloud_motion(
        {},
        num_frames=2,
        t_frame=1.0,
        t_exposure=1.0,
        tt=tt,
        ts_collect_start=ts_collect_start,
        ts_collect_end=ts_collect_end,
        s_osf=1,
        star_tran_os=[0.0, 0.0],
        track_mode=None,
        observer=None,
        track=None,
        star_rot=0.0,
        track_az=None,
        track_el=None,
        track_apparent=True,
        track_ra=None,
        track_dec=None,
        h_fpa_pad_os=10,
        w_fpa_pad_os=10,
        y_fov_pad=1.0,
        x_fov_pad=1.0,
        y_fov=1.0,
        x_fov=1.0,
        y_ifov=0.1,
        x_ifov=0.1,
        cloud_layer_configs=layer_configs,
        cloud_geometry=geometry,
    )

    np.testing.assert_allclose(motion['frames'][0]['layer_wind_offsets_px'], [[1.0, 0.0], [0.0, 0.5]], atol=1e-7)
    np.testing.assert_allclose(motion['frames'][1]['layer_wind_offsets_px'], [[3.0, 0.0], [0.0, 1.5]], atol=1e-7)
    np.testing.assert_allclose(motion['frames'][1]['layer_total_offsets_px'], [[3.0, 0.0], [0.0, 1.5]], atol=1e-7)
    assert motion['pad_px'] == [5, 4]
    assert motion['pad_after_px'] == [0, 0]
    assert motion['source_shape_px'] == [6, 5]


def test_cloud_motion_ignores_cardinal_direction_roundoff_padding():
    from satsim import satsim as satsim_module

    tt = [2025, 1, 1, 0, 0, 0.0]
    ts_collect_start = satsim_module.time.utc_from_list(tt)
    ts_collect_end = satsim_module.time.utc_from_list(tt, 2.0)
    layer_configs = parse_cloud_layers([
        {'type': 'custom', 'wind_speed': 2.0, 'wind_direction': 90.0, 'range': 0.01},
        {'type': 'custom', 'wind_speed': 1.0, 'wind_direction': 270.0, 'range': 0.01},
    ], sim_seed=1)
    geometry = CloudGeometry(
        height_px=10,
        width_px=10,
        y_fov_deg=math.degrees(1.0),
        x_fov_deg=math.degrees(1.0),
        cloud_range_m=10.0,
    )

    motion = satsim_module._plan_cloud_motion(
        {},
        num_frames=2,
        t_frame=1.0,
        t_exposure=1.0,
        tt=tt,
        ts_collect_start=ts_collect_start,
        ts_collect_end=ts_collect_end,
        s_osf=1,
        star_tran_os=[0.0, 0.0],
        track_mode=None,
        observer=None,
        track=None,
        star_rot=0.0,
        track_az=None,
        track_el=None,
        track_apparent=True,
        track_ra=None,
        track_dec=None,
        h_fpa_pad_os=10,
        w_fpa_pad_os=10,
        y_fov_pad=1.0,
        x_fov_pad=1.0,
        y_fov=1.0,
        x_fov=1.0,
        y_ifov=0.1,
        x_ifov=0.1,
        cloud_layer_configs=layer_configs,
        cloud_geometry=geometry,
    )

    assert motion['pad_px'] == [5, 0]
    assert motion['pad_after_px'] == [4, 0]
    assert motion['source_shape_px'] == [19, 10]


def test_cloud_motion_rejects_excessive_padded_source(monkeypatch):
    from satsim import satsim as satsim_module

    monkeypatch.setattr(satsim_module, '_MAX_CLOUD_SOURCE_PIXELS', 100)
    tt = [2025, 1, 1, 0, 0, 0.0]
    ts_collect_start = satsim_module.time.utc_from_list(tt)
    ts_collect_end = satsim_module.time.utc_from_list(tt, 2.0)
    layer_configs = parse_cloud_layers([
        {'type': 'custom', 'wind_speed': 10.0, 'wind_direction': 90.0, 'range': 0.001},
    ], sim_seed=1)
    geometry = CloudGeometry(
        height_px=10,
        width_px=10,
        y_fov_deg=math.degrees(1.0),
        x_fov_deg=math.degrees(1.0),
        cloud_range_m=1.0,
    )

    with pytest.raises(ValueError, match='padded source field'):
        satsim_module._plan_cloud_motion(
            {},
            num_frames=2,
            t_frame=1.0,
            t_exposure=1.0,
            tt=tt,
            ts_collect_start=ts_collect_start,
            ts_collect_end=ts_collect_end,
            s_osf=1,
            star_tran_os=[0.0, 0.0],
            track_mode=None,
            observer=None,
            track=None,
            star_rot=0.0,
            track_az=None,
            track_el=None,
            track_apparent=True,
            track_ra=None,
            track_dec=None,
            h_fpa_pad_os=10,
            w_fpa_pad_os=10,
            y_fov_pad=1.0,
            x_fov_pad=1.0,
            y_fov=1.0,
            x_fov=1.0,
            y_ifov=0.1,
            x_ifov=0.1,
            cloud_layer_configs=layer_configs,
            cloud_geometry=geometry,
        )


def _runtime_cloud_ssp(star_translation):
    from satsim import config

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['spacial_osf'] = 1
    ssp['sim']['temporal_osf'] = 1
    ssp['sim']['padding'] = 2
    ssp['sim']['enable_shot_noise'] = False
    ssp['sim']['save_ground_truth'] = True
    ssp['sim']['save_segmentation'] = True
    ssp['fpa']['height'] = 24
    ssp['fpa']['width'] = 24
    ssp['fpa']['num_frames'] = 2
    ssp['fpa']['time']['exposure'] = 1.0
    ssp['fpa']['time']['gap'] = 0.0
    ssp['fpa']['dark_current'] = 0
    ssp['fpa']['bias'] = 0
    ssp['fpa']['noise']['read'] = 0
    ssp['fpa']['noise']['electronic'] = 0
    ssp['background']['galactic'] = 22.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['stars']['motion']['mode'] = 'affine'
    ssp['geometry']['stars']['motion']['rotation'] = 0.0
    ssp['geometry']['stars']['motion']['translation'] = list(star_translation)
    ssp['geometry']['obs']['list'] = []
    ssp['clouds'] = [{
        'type': 'custom',
        'seed': 123,
        'coverage': 0.5,
        'feature_scales_m': [20.0, 40.0],
        'density_edge_width': 0.1,
        'tau_min': 0.05,
        'tau_max': 1.0,
        'brightness': 18.0,
    }]
    return ssp


def _run_runtime_cloud_frames(ssp):
    from satsim.satsim import image_generator
    from satsim.util import configure_eager

    configure_eager()
    return list(image_generator(ssp, output_dir='./.images', with_meta=True, num_sets=1))


def test_runtime_static_cloud_crop_is_stable_across_frames():
    frames = _run_runtime_cloud_frames(_runtime_cloud_ssp([0.0, 0.0]))

    first_bg = frames[0][6].numpy()
    second_bg = frames[1][6].numpy()

    np.testing.assert_allclose(first_bg, second_bg)
    assert frames[0][2]['clouds']['motion']['pad_px'] == [0, 0]
    assert frames[0][2]['clouds']['motion']['total_offset_px'] == [0.0, 0.0]
    assert frames[1][2]['clouds']['motion']['total_offset_px'] == [0.0, 0.0]


def test_runtime_moving_cloud_crop_changes_background_and_ground_truth():
    frames = _run_runtime_cloud_frames(_runtime_cloud_ssp([2.0, 0.0]))

    first_bg = frames[0][6].numpy()
    second_bg = frames[1][6].numpy()
    first_motion = frames[0][2]['clouds']['motion']
    second_motion = frames[1][2]['clouds']['motion']

    assert first_motion['pad_px'] == [5, 0]
    assert second_motion['pad_px'] == [5, 0]
    assert first_motion['total_offset_px'] == [1.0, 0.0]
    assert second_motion['total_offset_px'] == [3.0, 0.0]
    assert not np.allclose(first_bg, second_bg)
    np.testing.assert_allclose(frames[0][11]['background_pe'], first_bg)
    np.testing.assert_allclose(frames[1][11]['background_pe'], second_bg)
    assert frames[0][13]['cloud_segmentation'].shape == first_bg.shape
    assert frames[1][13]['cloud_segmentation'].shape == second_bg.shape
