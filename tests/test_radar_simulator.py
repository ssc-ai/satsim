import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

import satsim.radar.simulator as simulator
import satsim.radar.monostatic as sensor


def _base_ssp():
    return {
        'radar': {
            'tx_power': 1.0e6,
            'tx_frequency': 1.0e9,
            'antenna_diameter': 10.0,
            'efficiency': 0.6,
            'detection': {
                'min_detectable_power': 1.0e-13,
                'snr_threshold': None,
                'angle_error': 0.05,              # deg
                'range_error': 0.0,               # km
                'range_rate_error': 0.0,          # km/s
                'false_alarm_rate': 0.0,
            },
            'field_of_view': {
                'azimuth': [0.0, 180.0],
                'elevation': [0.0, 90.0],
            },
            'range_limits': [0.0, 5000.0],
            'time': {
                'dwell': 1.0,
                'gap': 0.0,
            },
            'num_frames': 1,
        },
        'geometry': {
            'time': [2020, 1, 1, 0, 0, 0.0],
            'site': {
                'lat': '0 N',
                'lon': '0 E',
                'alt': 0.0,
            },
            'obs': {
                'mode': 'list',
                'list': {
                    'mode': 'twobody',
                    'position': [7000.0, 0.0, 0.0],
                    'velocity': [0.0, 7.5, 0.0],
                    'epoch': 0.0,
                    'rcs': 1.0,
                    'name': 'SAT1',
                    'id': 12345,
                },
            },
        },
    }


def test_parse_radar_params_mapping():
    ssp = _base_ssp()
    p = simulator._parse_radar_params(ssp)
    assert isinstance(p, sensor.RadarParams)
    assert p.tx_power == ssp['radar']['tx_power']
    assert p.tx_frequency == ssp['radar']['tx_frequency']
    assert p.antenna_diameter == ssp['radar']['antenna_diameter']
    assert p.efficiency == ssp['radar']['efficiency']
    assert p.min_detectable_power == ssp['radar']['detection']['min_detectable_power']
    assert p.angle_error == 0.05
    assert p.range_error == 0.0
    assert p.range_rate_error == 0.0
    assert p.az_limits == tuple(ssp['radar']['field_of_view']['azimuth'])
    assert p.el_limits == tuple(ssp['radar']['field_of_view']['elevation'])
    assert p.range_limits == tuple(ssp['radar']['range_limits'])
    assert p.dwell == ssp['radar']['time']['dwell']
    assert p.gap == ssp['radar']['time']['gap']
    assert p.num_frames == ssp['radar']['num_frames']


def test_parse_radar_params_rejects_negative_gap():
    ssp = _base_ssp()
    ssp['radar']['time']['gap'] = -1.0
    with pytest.raises(ValueError, match='must be nonnegative'):
        simulator._parse_radar_params(ssp)


@pytest.mark.parametrize('path,value,match', [
    (('tx_power',), 0.0, 'must be positive'),
    (('tx_frequency',), -1.0, 'must be positive'),
    (('antenna_diameter',), -1.0, 'must be nonnegative'),
    (('efficiency',), 1.1, r'range \[0, 1\]'),
    (('detection', 'range_error'), -1.0, 'must be nonnegative'),
    (('detection', 'false_alarm_rate'), 1.1, r'range \[0, 1\]'),
    (('time', 'dwell'), 0.0, 'must be positive'),
    (('num_frames',), 0, 'must be positive'),
])
def test_parse_radar_params_rejects_invalid_physical_values(path, value, match):
    ssp = _base_ssp()
    node = ssp['radar']
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value
    with pytest.raises(ValueError, match=match):
        simulator._parse_radar_params(ssp)


@pytest.mark.parametrize('field', ['field_of_view', 'range_limits'])
def test_parse_radar_params_rejects_reversed_limits(field):
    ssp = _base_ssp()
    if field == 'field_of_view':
        ssp['radar'][field]['elevation'] = [90.0, 0.0]
    else:
        ssp['radar'][field] = [5000.0, 0.0]
    with pytest.raises(ValueError, match='minimum must not exceed maximum'):
        simulator._parse_radar_params(ssp)


def test_parse_radar_params_rejects_multiple_samples():
    ssp = _base_ssp()
    ssp['sim'] = {'samples': 2}
    with pytest.raises(ValueError, match='sim.samples == 1'):
        simulator._parse_radar_params(ssp)


def test_build_observer_ground_and_space(monkeypatch):
    ssp = _base_ssp()
    sentinel = object()
    captured = []
    monkeypatch.setattr(
        simulator,
        'create_observer_from_config',
        lambda site: captured.append(site) or sentinel,
    )
    assert simulator._build_observer(ssp) is sentinel
    assert captured == [ssp['geometry']['site']]


def test_build_target_uses_shared_factory(monkeypatch):
    default_t = [2020, 1, 1, 0, 0, 0.0]
    entry = {'mode': 'observation'}
    sentinel = object()
    captured = []
    monkeypatch.setattr(
        simulator,
        'create_target_from_config',
        lambda value, default_time=None: captured.append((value, default_time)) or sentinel,
    )
    assert simulator._build_target(entry, default_t=default_t) is sentinel
    assert captured == [(entry, default_t)]

    alias = {'mode': 'statevector', 'position': [], 'velocity': [], 'epoch': 0.0}
    assert simulator._build_target(alias, default_t=default_t) is sentinel
    assert captured[-1] == (dict(alias, mode='twobody'), default_t)


def test_simulate_writes_observations(monkeypatch):
    ssp = _base_ssp()
    target_config = ssp['geometry']['obs']['list']
    target_config.pop('id')
    target_config.pop('position')
    target_config.pop('velocity')
    target_config.pop('epoch')
    target_config['mode'] = 'tle'
    target_config['tle'] = [
        '1 25544U 98067A   20029.54791435  .00001264  00000-0  29621-4 0  9993',
        '2 25544  51.6440  30.9682 0005197  77.5934  20.6657 15.49147106211867',
    ]

    # Avoid Skyfield ephemeris loads in unit tests.
    monkeypatch.setattr(simulator, '_build_observer', lambda ssp: object())
    monkeypatch.setattr(simulator, '_build_target', lambda entry, default_t=None: object())

    # Fix LOS for simulator (angles + range)
    def fake_get_los_sim(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        return 0.0, 0.0, 100.0, 45.0, 45.0, None  # rng (km), az, el (ensure detection)

    monkeypatch.setattr(simulator, 'get_los', fake_get_los_sim)

    # Make range_rate deterministic in sensor
    def fake_get_los_sensor(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        radial_velocity = MagicMock()
        radial_velocity.km_per_s = 0.001  # range_rate

        fake_icrf_los = MagicMock()
        fake_icrf_los.frame_latlon_and_rates = MagicMock(
            return_value=(None, None, None, None, None, radial_velocity)
        )

        return 0.0, 0.0, 0.0, 0.0, 0.0, fake_icrf_los  # km

    monkeypatch.setattr(sensor, 'get_los', fake_get_los_sensor)

    # Make measurement noise deterministic (shift == 1-sigma)
    def fake_normal(*args, **kwargs):
        return float(kwargs.get('loc', 0.0)) + float(kwargs.get('scale', 0.0))

    monkeypatch.setattr(simulator.np.random, 'normal', fake_normal)

    out_dir = tempfile.mkdtemp()
    run_dir = simulator.simulate(ssp, out_dir)
    assert (Path(run_dir) / 'config.json').is_file()

    # Locate frame 0 JSON
    obs_path = os.path.join(run_dir, 'AnalyticalObservations')
    files = sorted(os.listdir(obs_path))
    assert any(f.endswith('.0000.json') for f in files)
    f0 = [f for f in files if f.endswith('.0000.json')][0]
    with open(os.path.join(obs_path, f0), 'r') as jf:
        data = json.load(jf)

    assert isinstance(data, list) and len(data) == 1
    m = data[0]
    assert m['type'] == 'RADAR'
    assert m['createdBy'] == 'satsim'
    # Deterministic perturbation due to fake noise (shift == sigma)
    np.testing.assert_allclose(m['azimuth'], 45.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m['elevation'], 45.05, rtol=0, atol=1e-12)
    # Angle uncertainties present (deg)
    np.testing.assert_allclose(m['azimuthUnc'], 0.05, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m['elevationUnc'], 0.05, rtol=0, atol=1e-12)
    # Now outputs are in km and km/s
    np.testing.assert_allclose(m['range'], 100.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(m['rangeRate'], 0.001, rtol=0, atol=1e-9)
    # UDL doppler is line-of-sight velocity in m/s (rangeRate converted to m/s)
    np.testing.assert_allclose(m['doppler'], 1.0, rtol=0, atol=1e-9)
    np.testing.assert_allclose(m['rangeRateUnc'], 0.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(m['dopplerUnc'], 0.0, rtol=0, atol=1e-12)
    assert m['idOnOrbit'] == 'SAT1'
    assert m['satNo'] == 25544

    # Validate SNR proxy (Rmax/R)^4
    rp = simulator._parse_radar_params(ssp)
    rmax = sensor.max_detectable_range(rp, sigma=1.0)
    expected_snr = (rmax / 100.0) ** 4
    np.testing.assert_allclose(m['snr'], expected_snr, rtol=1e-12)


def test_gap_separates_dwell_from_radar_frame_spacing(monkeypatch, tmp_path):
    ssp = _base_ssp()
    ssp['radar']['time']['gap'] = 29.0
    ssp['radar']['num_frames'] = 2

    monkeypatch.setattr(simulator, '_build_observer', lambda ssp: object())
    monkeypatch.setattr(
        simulator,
        '_build_target',
        lambda entry, default_t=None: object(),
    )
    monkeypatch.setattr(
        simulator,
        'get_los',
        lambda *args, **kwargs: (0.0, 0.0, 100.0, 45.0, 45.0, None),
    )
    monkeypatch.setattr(simulator, 'range_rate', lambda *args, **kwargs: 0.0)

    run_dir = simulator.simulate(ssp, str(tmp_path))
    paths = sorted((Path(run_dir) / 'AnalyticalObservations').glob('*.json'))
    times = [json.loads(path.read_text())[0]['obTime'] for path in paths]
    assert times == [
        '2020-01-01T00:00:00.500000Z',
        '2020-01-01T00:00:30.500000Z',
    ]


def test_simulate_from_file(monkeypatch, tmp_path):
    # Avoid Skyfield ephemeris loads in unit tests.
    monkeypatch.setattr(simulator, '_build_observer', lambda ssp: object())
    monkeypatch.setattr(simulator, '_build_target', lambda entry, default_t=None: object())

    # Monkeypatch LOS as above
    def fake_get_los_sim(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        return 0.0, 0.0, 800.0, 20.0, 30.0, None

    monkeypatch.setattr(simulator, 'get_los', fake_get_los_sim)

    def fake_get_los_sensor(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        sec = sensor.time.to_utc_list(t)[5]
        return 0.0, 0.0, sec / 1000.0, 0.0, 0.0, None

    monkeypatch.setattr(sensor, 'get_los', fake_get_los_sensor)

    # Write minimal JSON config
    cfg = _base_ssp()
    cfg['version'] = 'v1'
    cfg['sim'] = {'samples': 1}
    cfg_path = tmp_path / 'radar_cfg.json'
    with open(cfg_path, 'w') as f:
        json.dump(cfg, f)

    out_dir = tempfile.mkdtemp()
    result_dir = simulator.simulate_from_file(str(cfg_path), out_dir)
    assert os.path.isdir(result_dir)
    assert (Path(result_dir) / 'config.json').is_file()


def test_simulate_filters_by_fov_and_range(monkeypatch):
    ssp = _base_ssp()
    # Tight FOV to force rejection
    ssp['radar']['field_of_view'] = {
        'azimuth': [0.0, 1.0],
        'elevation': [0.0, 1.0],
    }

    # LOS reports outside FOV
    def fake_get_los_sim(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        return 0.0, 0.0, 100.0, 90.0, 45.0, None

    monkeypatch.setattr(simulator, 'get_los', fake_get_los_sim)
    monkeypatch.setattr(simulator, '_build_observer', lambda ssp: object())
    monkeypatch.setattr(simulator, '_build_target', lambda entry, default_t=None: object())

    out_dir = tempfile.mkdtemp()
    run_dir = simulator.simulate(ssp, out_dir)
    obs_path = os.path.join(run_dir, 'AnalyticalObservations')
    files = sorted(os.listdir(obs_path))
    f0 = [f for f in files if f.endswith('.0000.json')][0]
    with open(os.path.join(obs_path, f0), 'r') as jf:
        data = json.load(jf)
    assert data == []
