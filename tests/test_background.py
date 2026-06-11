import json
import math
import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings('ignore', category=DeprecationWarning, message='jsonschema.RefResolver.*')
pytestmark = pytest.mark.filterwarnings(
    'ignore:jsonschema.RefResolver is deprecated:DeprecationWarning'
)

from satsim import config
from satsim.background import (
    apply_background_stray_augmentation,
    background_components_from_config,
    background_frame_components_from_config,
    krisciunas_schaefer_moon_brightness_nl,
    hosek_wilkie_daytime_surface_brightness,
    hosek_wilkie_transition_surface_brightness,
    luminance_to_surface_brightness,
    nano_lamberts_to_surface_brightness,
    patat_twilight_surface_brightness,
    perez_daytime_relative_luminance,
    perez_daytime_surface_brightness,
    surface_brightness_to_luminance,
    surface_brightness_to_pe,
)
from satsim import time as satsim_time
from satsim.geometry.astrometric import create_topocentric, get_los_azel
from satsim.satsim import image_generator
from satsim.util import configure_eager


def _background_ssp():
    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['spacial_osf'] = 1
    ssp['sim']['temporal_osf'] = 1
    ssp['sim']['padding'] = 2
    ssp['sim']['enable_shot_noise'] = False
    ssp['sim']['save_ground_truth'] = True
    ssp['sim']['save_segmentation'] = False
    ssp['fpa']['height'] = 24
    ssp['fpa']['width'] = 24
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1.0
    ssp['fpa']['time']['gap'] = 0.0
    ssp['fpa']['dark_current'] = 0
    ssp['fpa']['bias'] = 0
    ssp['fpa']['noise']['read'] = 0
    ssp['fpa']['noise']['electronic'] = 0
    ssp['background'] = {
        'galactic': 22.0,
        'stray': {'mode': 'none'},
    }
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['stars']['motion']['mode'] = 'affine'
    ssp['geometry']['stars']['motion']['rotation'] = 0.0
    ssp['geometry']['stars']['motion']['translation'] = [0.0, 0.0]
    ssp['geometry']['obs']['list'] = []
    ssp.pop('clouds', None)
    return ssp


def _pixel_area(ssp):
    return (
        ssp['fpa']['y_fov'] / ssp['fpa']['height'] * 3600.0
        * ssp['fpa']['x_fov'] / ssp['fpa']['width'] * 3600.0
    )


def _run_one_frame(ssp):
    configure_eager()
    return list(image_generator(ssp, output_dir='./.images', with_meta=True, num_sets=1))[0]


def test_legacy_background_components_match_existing_scalar_math():
    ssp = {'background': {'galactic': 22.0, 'stray': {'mode': 'none'}}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)

    expected = surface_brightness_to_pe(20.0, 22.0, 4.0, 5.0)
    assert components['background_natural_pe'] == pytest.approx(expected)
    assert components['background_skyglow_pe'] == pytest.approx(0.0)
    assert components['background_stray_pe'] == pytest.approx(0.0)
    assert components['background_pre_cloud_pe'] == pytest.approx(expected)
    assert components['active']['background_natural_pe'] is True
    assert components['active']['background_skyglow_pe'] is False
    assert components['active']['background_stray_pe'] is False


def test_stray_background_preserves_existing_exposure_scaling():
    ssp = {'background': {'galactic': 22.0, 'stray': 2.5}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)

    assert components['background_stray_pe'] == pytest.approx(12.5)
    assert components['background_pre_cloud_pe'] == pytest.approx(
        components['background_natural_pe'] + 12.5
    )
    assert components['active']['background_stray_pe'] is True


def test_skyglow_uses_linear_residual_not_magnitude_delta():
    ssp = {'background': {'galactic': 22.0, 'skyglow': 21.5}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)

    natural = surface_brightness_to_pe(20.0, 22.0, 4.0, 5.0)
    total = surface_brightness_to_pe(20.0, 21.5, 4.0, 5.0)
    assert components['background_natural_pe'] == pytest.approx(natural)
    assert components['background_skyglow_pe'] == pytest.approx(total - natural)
    assert components['background_pre_cloud_pe'] == pytest.approx(total)
    assert components['active']['background_skyglow_pe'] is True


def test_skyglow_accepts_directional_magnitude_field():
    skyglow = np.array([
        [21.5, 22.0],
        [21.0, 21.7],
    ])
    ssp = {'background': {'galactic': 22.0, 'skyglow': skyglow}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)

    natural = surface_brightness_to_pe(20.0, 22.0, 4.0, 5.0)
    total = surface_brightness_to_pe(20.0, skyglow, 4.0, 5.0)
    np.testing.assert_allclose(components['background_natural_pe'], natural)
    np.testing.assert_allclose(components['background_skyglow_pe'], total - natural)
    np.testing.assert_allclose(components['background_pre_cloud_pe'], total)


def test_equal_skyglow_and_galactic_has_zero_artificial_residual():
    ssp = {'background': {'galactic': 22.0, 'skyglow': 22.0}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)

    assert components['background_skyglow_pe'] == pytest.approx(0.0)
    assert components['background_pre_cloud_pe'] == pytest.approx(
        components['background_natural_pe']
    )
    assert components['active']['background_skyglow_pe'] is True


def test_skyglow_rejects_negative_artificial_residual():
    ssp = {'background': {'galactic': 22.0, 'skyglow': 22.1}}

    with pytest.raises(ValueError, match='background.skyglow'):
        background_components_from_config(ssp, 20.0, 4.0, 5.0)


def test_krisciunas_schaefer_moon_brightness_reference_case():
    moon_nl = krisciunas_schaefer_moon_brightness_nl(
        phase_angle_deg=0.0,
        moon_sky_separation_deg=90.0,
        moon_zenith_deg=0.0,
        target_zenith_deg=0.0,
    )

    assert moon_nl == pytest.approx(912.731084459842)
    assert nano_lamberts_to_surface_brightness(moon_nl) == pytest.approx(
        18.930306398551053
    )


def test_frame_moon_component_uses_ground_site_geometry():
    ssp = {'background': {'galactic': 22.0, 'moon': {'mode': 'krisciunas-schaefer'}}}
    observer = create_topocentric('20.746111 N', '156.431667 W', 0.3)
    ts_mid = satsim_time.utc(2015, 4, 24, 9, 7, 30.128)
    ra, dec, _, _, el, _ = get_los_azel(observer, 285.0, 30.0, ts_mid)
    astrometrics = {'ra': ra, 'dec': dec, 'el': el}

    frame_components = background_frame_components_from_config(
        ssp,
        20.0,
        4.0,
        5.0,
        observer,
        'ground',
        astrometrics,
        ts_mid,
    )

    assert frame_components['components']['background_moon_pe'] == pytest.approx(
        8.559580369157867
    )
    assert frame_components['active']['background_moon_pe'] is True
    assert frame_components['metadata']['moon']['mode'] == 'krisciunas-schaefer'
    assert frame_components['metadata']['moon']['moon_el'] > 0.0


def test_frame_moon_component_requires_ground_site():
    ssp = {'background': {'galactic': 22.0, 'moon': {'mode': 'default'}}}

    with pytest.raises(ValueError, match='requires a ground geometry.site'):
        background_frame_components_from_config(
            ssp,
            20.0,
            4.0,
            5.0,
            None,
            None,
            {'ra': 0.0, 'dec': 0.0, 'el': 45.0},
            satsim_time.utc(2015, 4, 24, 9, 7, 30.128),
        )


def test_patat_twilight_surface_brightness_reference_values():
    assert np.isinf(patat_twilight_surface_brightness(90.0))
    assert patat_twilight_surface_brightness(92.0) == pytest.approx(11.84)
    assert patat_twilight_surface_brightness(95.0) == pytest.approx(11.84)
    assert patat_twilight_surface_brightness(100.0) == pytest.approx(18.005)
    assert patat_twilight_surface_brightness(104.0) == pytest.approx(20.885)
    assert np.isinf(patat_twilight_surface_brightness(105.0))


def test_frame_twilight_component_uses_ground_site_geometry():
    ssp = {'background': {'galactic': 22.0, 'twilight': {'mode': 'patat'}}}
    observer = create_topocentric('20.746111 N', '156.431667 W', 0.3)
    ts_mid = satsim_time.utc(2026, 6, 3, 6, 0, 0)
    ra, dec, _, _, el, _ = get_los_azel(observer, 180.0, 60.0, ts_mid)
    astrometrics = {'ra': ra, 'dec': dec, 'el': el}

    frame_components = background_frame_components_from_config(
        ssp,
        20.0,
        4.0,
        5.0,
        observer,
        'ground',
        astrometrics,
        ts_mid,
    )

    assert frame_components['components']['background_twilight_pe'] == pytest.approx(
        15.467652534498008
    )
    assert frame_components['active']['background_twilight_pe'] is True
    assert frame_components['metadata']['twilight']['mode'] == 'patat'
    assert frame_components['metadata']['twilight']['sun_el'] == pytest.approx(
        -12.587933862823263
    )


def test_frame_twilight_component_requires_ground_site():
    ssp = {'background': {'galactic': 22.0, 'twilight': {'mode': 'default'}}}

    with pytest.raises(ValueError, match='requires a ground geometry.site'):
        background_frame_components_from_config(
            ssp,
            20.0,
            4.0,
            5.0,
            None,
            None,
            {'ra': 0.0, 'dec': 0.0, 'el': 45.0},
            satsim_time.utc(2026, 6, 3, 6, 0, 0),
        )


def test_perez_daytime_surface_brightness_reference_values():
    assert luminance_to_surface_brightness(4000.0) == pytest.approx(3.59437640705097)
    assert surface_brightness_to_luminance(3.59437640705097) == pytest.approx(4000.0)
    assert perez_daytime_relative_luminance(0.0, 30.0, 30.0) == pytest.approx(1.0)
    assert perez_daytime_surface_brightness(30.0, 30.0, 0.0) == pytest.approx(
        2.1504110693142953
    )
    assert perez_daytime_surface_brightness(30.0, 30.0, 60.0) == pytest.approx(
        4.325357167686294
    )
    assert np.isinf(perez_daytime_surface_brightness(0.0, 100.0, 100.0))


def test_hosek_wilkie_daytime_surface_brightness_reference_values():
    assert np.isinf(hosek_wilkie_daytime_surface_brightness(0.0, 100.0, 100.0))
    assert hosek_wilkie_daytime_surface_brightness(0.0, 84.0, 84.0) == pytest.approx(
        5.204160453217474
    )
    assert hosek_wilkie_daytime_surface_brightness(0.0, 45.0, 45.0) == pytest.approx(
        4.044577897718539
    )
    assert hosek_wilkie_daytime_surface_brightness(0.0, 20.0, 20.0) == pytest.approx(
        3.3077739006407745
    )


def test_hosek_wilkie_transition_removes_perez_horizon_jump():
    patat_horizon = patat_twilight_surface_brightness(92.0)
    at_horizon, horizon_meta = hosek_wilkie_transition_surface_brightness(0.0, 90.0, 90.0)
    near_horizon, near_meta = hosek_wilkie_transition_surface_brightness(0.0, 89.9, 89.9)
    mid_blend, mid_meta = hosek_wilkie_transition_surface_brightness(0.0, 87.0, 87.0)
    blend_end, end_meta = hosek_wilkie_transition_surface_brightness(0.0, 84.0, 84.0)

    assert at_horizon == pytest.approx(patat_horizon)
    assert horizon_meta['blend_weight'] == pytest.approx(0.0)
    assert near_horizon == pytest.approx(patat_horizon, abs=0.2)
    assert near_horizon > perez_daytime_surface_brightness(0.0, 89.9, 89.9) + 5.0
    assert near_meta['branch'] == 'twilight_daytime_blend'
    assert mid_meta['branch'] == 'twilight_daytime_blend'
    assert end_meta['branch'] == 'hosek-wilkie'
    assert near_horizon > mid_blend > blend_end
    assert end_meta['blend_weight'] == pytest.approx(1.0)


def test_sky_background_validation_fixture_values():
    with open('tests/data/sky_background_validation.json', 'r') as f:
        fixture = json.load(f)

    sqm = fixture['sqm_clear_sky_split']
    total_ratio = 10 ** (-0.4 * (sqm['clear_total_mag'] - sqm['natural_mag']))
    assert total_ratio == pytest.approx(sqm['total_to_natural_ratio'])
    assert total_ratio - 1.0 == pytest.approx(sqm['artificial_to_natural_ratio'])

    moon = fixture['krisciunas_schaefer']
    moon_nl = krisciunas_schaefer_moon_brightness_nl(
        moon['phase_angle_deg'],
        moon['moon_sky_separation_deg'],
        moon['moon_zenith_deg'],
        moon['target_zenith_deg'],
    )
    assert moon_nl == pytest.approx(moon['nano_lamberts'])
    assert nano_lamberts_to_surface_brightness(moon_nl) == pytest.approx(
        moon['surface_brightness']
    )

    patat = fixture['patat_v']
    assert patat_twilight_surface_brightness(patat['sun_zenith_deg']) == pytest.approx(
        patat['surface_brightness']
    )

    cloud = fixture['cloud_amplification']
    clear_ratio = 10 ** (-0.4 * (cloud['clear_total_mag'] - cloud['natural_mag']))
    artificial_ratio = clear_ratio - 1.0
    cloudy_ratio = 10 ** (-0.4 * (cloud['cloudy_total_mag'] - cloud['clear_total_mag']))
    gain = (
        cloudy_ratio * clear_ratio - math.exp(-cloud['mean_tau'])
    ) / artificial_ratio
    assert gain == pytest.approx(cloud['effective_artificial_gain'])


def test_frame_daytime_component_uses_ground_site_geometry():
    ssp = {'background': {'galactic': 22.0, 'daytime': {'mode': 'perez'}}}
    observer = create_topocentric('20.746111 N', '156.431667 W', 0.3)
    ts_mid = satsim_time.utc(2021, 9, 21, 17, 40, 0)
    ra, dec, _, _, el, _ = get_los_azel(observer, 100.0, 60.0, ts_mid)
    astrometrics = {'ra': ra, 'dec': dec, 'el': el}

    frame_components = background_frame_components_from_config(
        ssp,
        20.0,
        4.0,
        5.0,
        observer,
        'ground',
        astrometrics,
        ts_mid,
    )

    assert frame_components['components']['background_daytime_pe'] == pytest.approx(
        154740337.26892114
    )
    assert frame_components['active']['background_daytime_pe'] is True
    assert frame_components['metadata']['daytime']['mode'] == 'perez'
    assert frame_components['metadata']['daytime']['sun_el'] == pytest.approx(
        19.125216650567147
    )


def test_frame_daytime_default_uses_hosek_wilkie():
    ssp = {'background': {'galactic': 22.0, 'daytime': {'mode': 'default'}}}
    observer = create_topocentric('20.746111 N', '156.431667 W', 0.3)
    ts_mid = satsim_time.utc(2021, 9, 21, 17, 40, 0)
    ra, dec, _, _, el, _ = get_los_azel(observer, 100.0, 60.0, ts_mid)
    astrometrics = {'ra': ra, 'dec': dec, 'el': el}

    frame_components = background_frame_components_from_config(
        ssp,
        20.0,
        4.0,
        5.0,
        observer,
        'ground',
        astrometrics,
        ts_mid,
    )
    expected_brightness, _ = hosek_wilkie_transition_surface_brightness(
        90.0 - el,
        90.0 - frame_components['metadata']['daytime']['sun_el'],
        frame_components['metadata']['daytime']['sun_sky_separation'],
    )
    expected_pe = max(
        0.0,
        surface_brightness_to_pe(20.0, expected_brightness, 4.0, 5.0)
        - surface_brightness_to_pe(20.0, 22.0, 4.0, 5.0),
    )

    assert frame_components['components']['background_daytime_pe'] == pytest.approx(
        expected_pe
    )
    assert frame_components['metadata']['daytime']['mode'] == 'hosek-wilkie'
    assert frame_components['metadata']['daytime']['transition']['branch'] == 'hosek-wilkie'
    assert frame_components['metadata']['daytime']['turbidity'] == pytest.approx(3.0)


def test_frame_daytime_component_requires_ground_site():
    ssp = {'background': {'galactic': 22.0, 'daytime': {'mode': 'default'}}}

    with pytest.raises(ValueError, match='requires a ground geometry.site'):
        background_frame_components_from_config(
            ssp,
            20.0,
            4.0,
            5.0,
            None,
            None,
            {'ra': 0.0, 'dec': 0.0, 'el': 45.0},
            satsim_time.utc(2021, 9, 21, 17, 40, 0),
        )


def test_unknown_background_mode_fails_clearly():
    ssp = {'background': {'galactic': 22.0, 'moon': {'mode': 'bad'}}}

    with pytest.raises(ValueError, match='Unknown background.moon.mode'):
        background_components_from_config(ssp, 20.0, 4.0, 5.0)


def test_stray_augmentation_is_recorded_as_stray_residual():
    ssp = {'background': {'galactic': 22.0, 'skyglow': 21.5}}
    components = background_components_from_config(ssp, 20.0, 4.0, 5.0)
    augmented = np.full((2, 2), components['background_pre_cloud_pe'] + 3.0)

    updated = apply_background_stray_augmentation(components, augmented)

    np.testing.assert_allclose(updated['background_pre_cloud_pe'], augmented)
    np.testing.assert_allclose(updated['background_stray_pe'], np.full((2, 2), 3.0))
    assert updated['active']['background_stray_pe'] is True
    assert updated['metadata']['stray_augmentation_enabled'] is True


def test_runtime_skyglow_background_ground_truth_components_sum_to_final_background():
    ssp = _background_ssp()
    ssp['background']['skyglow'] = 21.5

    frame = _run_one_frame(ssp)
    bg = frame[6].numpy()
    ground_truth = frame[11]

    natural = surface_brightness_to_pe(
        ssp['fpa']['zeropoint'],
        22.0,
        _pixel_area(ssp),
        ssp['fpa']['time']['exposure'],
    )
    total = surface_brightness_to_pe(
        ssp['fpa']['zeropoint'],
        21.5,
        _pixel_area(ssp),
        ssp['fpa']['time']['exposure'],
    )

    np.testing.assert_allclose(bg, total)
    np.testing.assert_allclose(ground_truth['background_pe'], total)
    np.testing.assert_allclose(ground_truth['background_natural_pe'], natural)
    np.testing.assert_allclose(ground_truth['background_skyglow_pe'], total - natural)
    assert 'background_stray_pe' not in ground_truth
    assert frame[2]['background']['skyglow_enabled'] is True
    assert frame[2]['background']['modes'] == {
        'moon': 'none',
        'twilight': 'none',
        'daytime': 'none',
    }


def test_runtime_skyglow_background_accepts_image_field():
    ssp = _background_ssp()
    skyglow = np.linspace(21.5, 22.0, 24 * 24, dtype=np.float32).reshape(24, 24)
    ssp['background']['skyglow'] = skyglow

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    natural = surface_brightness_to_pe(
        ssp['fpa']['zeropoint'],
        22.0,
        _pixel_area(ssp),
        ssp['fpa']['time']['exposure'],
    )
    total = surface_brightness_to_pe(
        ssp['fpa']['zeropoint'],
        skyglow,
        _pixel_area(ssp),
        ssp['fpa']['time']['exposure'],
    )

    np.testing.assert_allclose(ground_truth['background_pe'], total, rtol=1e-6)
    np.testing.assert_allclose(
        ground_truth['background_skyglow_pe'],
        np.maximum(0.0, total - natural),
        rtol=1e-6,
    )


def test_runtime_legacy_background_without_skyglow_preserves_scalar_output():
    ssp = _background_ssp()

    frame = _run_one_frame(ssp)
    bg = frame[6].numpy()
    ground_truth = frame[11]

    natural = surface_brightness_to_pe(
        ssp['fpa']['zeropoint'],
        22.0,
        _pixel_area(ssp),
        ssp['fpa']['time']['exposure'],
    )

    np.testing.assert_allclose(bg, natural)
    np.testing.assert_allclose(ground_truth['background_pe'], natural)
    np.testing.assert_allclose(ground_truth['background_natural_pe'], natural)
    assert 'background_skyglow_pe' not in ground_truth
    assert 'background_stray_pe' not in ground_truth
    assert frame[2]['background']['skyglow_enabled'] is False


def test_runtime_cloud_attenuates_skyglow_components_and_adds_cloud_brightness():
    ssp = _background_ssp()
    ssp['background']['skyglow'] = 21.5
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

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]
    component_sum = (
        ground_truth['background_natural_pe']
        + ground_truth['background_skyglow_pe']
        + ground_truth['cloud_brightness_pe']
    )

    np.testing.assert_allclose(ground_truth['background_pe'], frame[6].numpy())
    np.testing.assert_allclose(component_sum, ground_truth['background_pe'], rtol=1e-6)
    assert ground_truth['background_natural_pe'].shape == (24, 24)
    assert ground_truth['background_skyglow_pe'].shape == (24, 24)
    assert ground_truth['cloud_brightness_pe'].shape == (24, 24)


def test_runtime_cloud_source_brightening_uses_artificial_skyglow():
    ssp = _background_ssp()
    ssp['background']['skyglow'] = 21.5
    ssp['clouds'] = [{
        'type': 'custom',
        'seed': 321,
        'coverage': 0.5,
        'feature_scales_m': [20.0, 40.0],
        'density_edge_width': 0.1,
        'tau_min': 0.05,
        'tau_max': 1.0,
    }]

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    assert 'cloud_brightness_pe' in ground_truth
    assert 'cloud_source_brightness_pe' in ground_truth
    assert np.mean(ground_truth['cloud_source_brightness_pe']) > 0.0
    np.testing.assert_allclose(
        ground_truth['cloud_brightness_pe'],
        ground_truth['cloud_source_brightness_pe'],
    )
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        (
            ground_truth['background_natural_pe']
            + ground_truth['background_skyglow_pe']
            + ground_truth['cloud_source_brightness_pe']
        ),
        rtol=1e-6,
    )


def test_runtime_moon_background_is_pre_cloud_ground_truth_component():
    ssp = _background_ssp()
    ssp['geometry']['time'] = [2015, 4, 24, 9, 7, 30.128]
    ssp['background']['moon'] = {'mode': 'krisciunas-schaefer'}
    ssp['geometry']['site'] = {
        'mode': 'topo',
        'lat': '20.746111 N',
        'lon': '156.431667 W',
        'alt': 0.3,
        'gimbal': {'mode': 'wcs', 'rotation': 0.0},
        'track': {'mode': 'fixed', 'az': 285.0, 'el': 30.0},
    }

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    assert 'background_moon_pe' in ground_truth
    assert np.mean(ground_truth['background_moon_pe']) > 0.0
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        ground_truth['background_natural_pe'] + ground_truth['background_moon_pe'],
    )
    assert frame[2]['background']['modes']['moon'] == 'krisciunas-schaefer'
    assert frame[2]['background']['moon']['moon_el'] > 0.0


def test_runtime_twilight_background_is_pre_cloud_ground_truth_component():
    ssp = _background_ssp()
    ssp['geometry']['time'] = [2026, 6, 3, 6, 0, 0]
    ssp['background']['twilight'] = {'mode': 'patat'}
    ssp['geometry']['site'] = {
        'mode': 'topo',
        'lat': '20.746111 N',
        'lon': '156.431667 W',
        'alt': 0.3,
        'gimbal': {'mode': 'wcs', 'rotation': 0.0},
        'track': {'mode': 'fixed', 'az': 180.0, 'el': 60.0},
    }

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    assert 'background_twilight_pe' in ground_truth
    assert np.mean(ground_truth['background_twilight_pe']) > 0.0
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        ground_truth['background_natural_pe'] + ground_truth['background_twilight_pe'],
    )
    assert frame[2]['background']['modes']['twilight'] == 'patat'
    assert frame[2]['background']['twilight']['sun_el'] < 0.0


def test_runtime_daytime_background_is_pre_cloud_ground_truth_component():
    ssp = _background_ssp()
    ssp['geometry']['time'] = [2021, 9, 21, 17, 40, 0]
    ssp['background']['daytime'] = {'mode': 'perez'}
    ssp['geometry']['site'] = {
        'mode': 'topo',
        'lat': '20.746111 N',
        'lon': '156.431667 W',
        'alt': 0.3,
        'gimbal': {'mode': 'wcs', 'rotation': 0.0},
        'track': {'mode': 'fixed', 'az': 100.0, 'el': 60.0},
    }

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    assert 'background_daytime_pe' in ground_truth
    assert np.mean(ground_truth['background_daytime_pe']) > 0.0
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        ground_truth['background_natural_pe'] + ground_truth['background_daytime_pe'],
        rtol=1e-6,
    )
    assert frame[2]['background']['modes']['daytime'] == 'perez'
    assert frame[2]['background']['daytime']['sun_el'] > 0.0


def test_runtime_default_daytime_background_uses_hosek_wilkie():
    ssp = _background_ssp()
    ssp['geometry']['time'] = [2021, 9, 21, 17, 40, 0]
    ssp['background']['daytime'] = {'mode': 'default'}
    ssp['geometry']['site'] = {
        'mode': 'topo',
        'lat': '20.746111 N',
        'lon': '156.431667 W',
        'alt': 0.3,
        'gimbal': {'mode': 'wcs', 'rotation': 0.0},
        'track': {'mode': 'fixed', 'az': 100.0, 'el': 60.0},
    }

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    assert 'background_daytime_pe' in ground_truth
    assert np.mean(ground_truth['background_daytime_pe']) > 0.0
    assert frame[2]['background']['modes']['daytime'] == 'hosek-wilkie'
    assert frame[2]['background']['daytime']['mode'] == 'hosek-wilkie'
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        ground_truth['background_natural_pe'] + ground_truth['background_daytime_pe'],
        rtol=1e-6,
    )


def test_runtime_default_daytime_background_is_zero_below_horizon():
    ssp = _background_ssp()
    ssp['geometry']['time'] = [2026, 6, 3, 6, 0, 0]
    ssp['background']['daytime'] = {'mode': 'default'}
    ssp['geometry']['site'] = {
        'mode': 'topo',
        'lat': '20.746111 N',
        'lon': '156.431667 W',
        'alt': 0.3,
        'gimbal': {'mode': 'wcs', 'rotation': 0.0},
        'track': {'mode': 'fixed', 'az': 180.0, 'el': 60.0},
    }

    frame = _run_one_frame(ssp)
    ground_truth = frame[11]

    np.testing.assert_allclose(ground_truth['background_daytime_pe'], 0.0)
    np.testing.assert_allclose(
        ground_truth['background_pe'],
        ground_truth['background_natural_pe'],
    )
    assert frame[2]['background']['daytime']['transition']['branch'] == 'below_horizon'


def _background_schema_validator():
    jsonschema = pytest.importorskip('jsonschema')
    schema_path = os.path.abspath('schema/v1/Background.json')
    with open(schema_path, 'r') as f:
        schema = json.load(f)
    with open(os.path.abspath('schema/v1/types/Float.json'), 'r') as f:
        float_schema = json.load(f)
    base_uri = schema['$id'].rsplit('/', 1)[0] + '/'
    store = {
        schema['$id']: schema,
        base_uri + 'types/Float.json': float_schema,
    }
    generator_dir = os.path.abspath('schema/v1/generators')
    for filename in os.listdir(generator_dir):
        if filename.endswith('.json'):
            with open(os.path.join(generator_dir, filename), 'r') as f:
                store[base_uri + 'generators/' + filename] = json.load(f)
    resolver = jsonschema.RefResolver(
        base_uri=schema['$id'],
        referrer=schema,
        store=store,
    )
    return jsonschema, jsonschema.Draft7Validator(schema, resolver=resolver)


def test_background_schema_accepts_v1_fields_and_daytime_modes():
    _, validator = _background_schema_validator()
    validator.validate({
        'galactic': 22.0,
        'skyglow': 21.5,
        'stray': {'mode': 'none'},
        'moon': {'mode': 'default'},
        'twilight': {'mode': 'patat'},
        'daytime': {'mode': 'hosek-wilkie'},
        'private_legacy_field': True,
    })

    validator.validate({
        'galactic': 22.0,
        'daytime': {'mode': 'perez'},
    })

    validator.validate({
        'galactic': 22.0,
        'skyglow': [[21.5, 21.7], [21.8, 22.0]],
    })


def test_background_schema_rejects_unknown_mode_values_and_fields():
    jsonschema, validator = _background_schema_validator()

    with pytest.raises(jsonschema.ValidationError):
        validator.validate({'galactic': 22.0, 'moon': {'mode': 'bad'}})

    with pytest.raises(jsonschema.ValidationError):
        validator.validate({'galactic': 22.0, 'moon': {'mode': 'none', 'extra': True}})

    with pytest.raises(jsonschema.ValidationError):
        validator.validate({'galactic': 22.0, 'skyglow': '21.5'})
