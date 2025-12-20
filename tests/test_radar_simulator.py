import json
import os
import tempfile

import astropy.units as u
import numpy as np

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
    assert p.num_frames == ssp['radar']['num_frames']


def test_build_observer_ground_and_space(monkeypatch):
    # Ground
    ssp = _base_ssp()
    # Avoid Skyfield ephemeris loads in unit tests.
    sentinel_topo = object()
    sentinel_sgp4 = object()
    monkeypatch.setattr(simulator, 'create_topocentric', lambda lat, lon, alt: sentinel_topo)
    monkeypatch.setattr(simulator, 'create_sgp4', lambda tle1, tle2: sentinel_sgp4)

    obs = simulator._build_observer(ssp)
    assert obs is sentinel_topo
    # Space using TLE keys
    ssp2 = _base_ssp()
    ssp2['geometry']['site'] = {
        'tle1': '1 25544U 98067A   20029.54791435  .00001264  00000-0  29621-4 0  9993',
        'tle2': '2 25544  51.6440  30.9682 0005197  77.5934  20.6657 15.49147106211867',
    }
    obs2 = simulator._build_observer(ssp2)
    assert obs2 is sentinel_sgp4


def test_build_target_modes(monkeypatch):
    default_t = [2020, 1, 1, 0, 0, 0.0]
    sentinel_sgp4 = object()
    sentinel_twobody = object()
    sentinel_eph = object()

    captured_twobody = {}
    captured_ephemeris = {}

    def fake_create_sgp4(tle1, tle2):
        return sentinel_sgp4

    def fake_create_twobody(position, velocity, epoch):
        captured_twobody['position'] = position
        captured_twobody['velocity'] = velocity
        captured_twobody['epoch'] = epoch
        return sentinel_twobody

    def fake_create_ephemeris_object(positions, velocities, times, epoch):
        captured_ephemeris['positions'] = positions
        captured_ephemeris['velocities'] = velocities
        captured_ephemeris['times'] = times
        captured_ephemeris['epoch'] = epoch
        return sentinel_eph

    monkeypatch.setattr(simulator, 'create_sgp4', fake_create_sgp4)
    monkeypatch.setattr(simulator, 'create_twobody', fake_create_twobody)
    monkeypatch.setattr(simulator, 'create_ephemeris_object', fake_create_ephemeris_object)

    # TLE
    tle_entry = {
        'mode': 'tle',
        'tle1': '1 25544U 98067A   20029.54791435  .00001264  00000-0  29621-4 0  9993',
        'tle2': '2 25544  51.6440  30.9682 0005197  77.5934  20.6657 15.49147106211867',
    }
    assert simulator._build_target(tle_entry, default_t=default_t) is sentinel_sgp4

    # Statevector
    sv_entry = {
        'mode': 'twobody',
        'position': [7000.0, 0.0, 0.0],
        'velocity': [0.0, 7.5, 0.0],
        'epoch': 0.0,
    }
    assert simulator._build_target(sv_entry, default_t=default_t) is sentinel_twobody
    assert captured_twobody['position'].unit == u.km
    assert captured_twobody['velocity'].unit.is_equivalent(u.km / u.s)
    np.testing.assert_allclose(captured_twobody['position'].value, sv_entry['position'])
    np.testing.assert_allclose(captured_twobody['velocity'].to_value(u.km / u.s), sv_entry['velocity'])

    # Back-compat alias
    sv_entry_alias = dict(sv_entry)
    sv_entry_alias['mode'] = 'statevector'
    assert simulator._build_target(sv_entry_alias, default_t=default_t) is sentinel_twobody

    # Ephemeris
    eph_entry = {
        'mode': 'ephemeris',
        'epoch': [2020, 1, 1, 0, 0, 0.0],
        'seconds_from_epoch': [0.0, 10.0, 20.0],
        'positions': [[7000.0, 0.0, 0.0], [7001.0, 0.0, 0.0], [7002.0, 0.0, 0.0]],
        'velocities': [[0.0, 7.5, 0.0], [0.0, 7.5, 0.0], [0.0, 7.5, 0.0]],
    }
    assert simulator._build_target(eph_entry, default_t=default_t) is sentinel_eph
    assert captured_ephemeris['positions'] == eph_entry['positions']
    assert captured_ephemeris['velocities'] == eph_entry['velocities']
    assert captured_ephemeris['times'] == eph_entry['seconds_from_epoch']

    # Observation -> unsupported
    obs_entry = {'mode': 'observation'}
    assert simulator._build_target(obs_entry, default_t=default_t) is None

    # Default TLE keys without mode
    def_entry = {
        'tle1': '1 25544U 98067A   20029.54791435  .00001264  00000-0  29621-4 0  9993',
        'tle2': '2 25544  51.6440  30.9682 0005197  77.5934  20.6657 15.49147106211867',
    }
    assert simulator._build_target(def_entry, default_t=default_t) is sentinel_sgp4


def test_simulate_writes_observations(monkeypatch):
    ssp = _base_ssp()

    # Avoid Skyfield ephemeris loads in unit tests.
    monkeypatch.setattr(simulator, '_build_observer', lambda ssp: object())
    monkeypatch.setattr(simulator, '_build_target', lambda entry, default_t=None: object())

    # Fix LOS for simulator (angles + range)
    def fake_get_los_sim(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        return 0.0, 0.0, 100.0, 45.0, 45.0, None  # rng (km), az, el (ensure detection)

    monkeypatch.setattr(simulator, 'get_los', fake_get_los_sim)

    # Make range_rate deterministic in sensor by mapping time->range linearly
    def fake_get_los_sensor(observer, target, t, deflection=False, aberration=False, stellar_aberration=False):
        sec = sensor.time.to_utc_list(t)[5]
        return 0.0, 0.0, sec / 1000.0, 0.0, 0.0, None  # km

    monkeypatch.setattr(sensor, 'get_los', fake_get_los_sensor)

    # Make measurement noise deterministic (shift == 1-sigma)
    def fake_normal(*args, **kwargs):
        return float(kwargs.get('loc', 0.0)) + float(kwargs.get('scale', 0.0))

    monkeypatch.setattr(simulator.np.random, 'normal', fake_normal)

    out_dir = tempfile.mkdtemp()
    run_dir = simulator.simulate(ssp, out_dir)

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
    assert m['satNo'] == 12345

    # Validate SNR proxy (Rmax/R)^4
    rp = simulator._parse_radar_params(ssp)
    rmax = sensor.max_detectable_range(rp, sigma=1.0)
    expected_snr = (rmax / 100.0) ** 4
    np.testing.assert_allclose(m['snr'], expected_snr, rtol=1e-12)


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
