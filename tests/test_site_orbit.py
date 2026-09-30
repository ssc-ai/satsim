import json
import logging
import os
import re

import numpy as np
import pytest
from astropy import units as u

from satsim import config, gen_images, time
from satsim.geometry.astrometric import get_los, load_earth
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.twobody import create_twobody
from satsim.util import MultithreadedTaskQueue

SITE_TLE = [
    "1 37168U 10048A   15115.45079343  .00003419  00000-0  18056-3 0  9997",
    "2 37168  97.7781  90.4142 0086116 212.8391 262.6197 15.13064320804554",
]
TRACK_TLE = [
    "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
    "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866",
]
START = [2015, 4, 24, 9, 37, 44.128]
EXPOSURE = 1.0
GAP = 5.0
MIDPOINTS = [0.5, 6.5]
OB_TIMES = ['2015-04-24T09:37:44.628000Z', '2015-04-24T09:37:50.628000Z']
SENSOR_KEYS = ['senx', 'seny', 'senz', 'senvelx', 'senvely', 'senvelz']
SITE_FORMS = ['twobody', 'ephemeris-flat', 'ephemeris-integer', 'ephemeris-segmented']
POSITION_ATOL = 1e-6
VELOCITY_ATOL = 1e-9
ANGLE_ATOL = 1e-9

INTEGER_SECONDS = list(range(12))
INTEGER_POSITIONS = [[7000 - t, 7 * t, 3 * t] for t in INTEGER_SECONDS]
INTEGER_VELOCITIES = [[-t, 7 + t, 3 - t] for t in INTEGER_SECONDS]
INTEGER_MIDPOINT_STATES = [
    [6999.5, 3.5, 1.5, -0.5, 7.5, 2.5],
    [6993.5, 45.5, 19.5, -6.5, 13.5, -3.5],
]


def _state(body, offset):
    sv = (body - load_earth()).at(time.utc_from_list(START, offset))
    return np.concatenate([sv.position.km, sv.velocity.km_per_s])


def _tle_state(offset):
    return _state(create_sgp4(*SITE_TLE), offset)


def _site_case(form):
    if form == 'tle':
        return {'tle': SITE_TLE}, [_tle_state(t) for t in MIDPOINTS]
    if form == 'twobody':
        epoch_state = _tle_state(MIDPOINTS[0])
        site = {'position': epoch_state[:3].tolist(), 'velocity': epoch_state[3:].tolist(), 'epoch': MIDPOINTS[0]}
        twobody = create_twobody(epoch_state[:3] * u.km, epoch_state[3:] * u.km / u.s, time.utc_from_list(START, MIDPOINTS[0]))
        return site, [epoch_state, _state(twobody, MIDPOINTS[1])]
    if form == 'ephemeris-flat':
        seconds = np.arange(-4.0, 13.0, 2.0)
        samples = np.array([_tle_state(t) for t in seconds])
        site = {
            'positions': samples[:, :3].tolist(),
            'velocities': samples[:, 3:].tolist(),
            'seconds_from_epoch': seconds.tolist(),
            'epoch': START,
        }
        return site, [_tle_state(t) for t in MIDPOINTS]
    if form == 'ephemeris-integer':
        site = {
            'positions': INTEGER_POSITIONS,
            'velocities': INTEGER_VELOCITIES,
            'seconds_from_epoch': INTEGER_SECONDS,
            'epoch': 0,
        }
        return site, INTEGER_MIDPOINT_STATES
    site = {
        'positions': [INTEGER_POSITIONS[:6], INTEGER_POSITIONS[5:]],
        'velocities': [INTEGER_VELOCITIES[:6], INTEGER_VELOCITIES[5:]],
        'seconds_from_epoch': [INTEGER_SECONDS[:6], INTEGER_SECONDS[5:]],
        'epoch': START,
    }
    return site, INTEGER_MIDPOINT_STATES


def _run(site, tmp_path, obs=(), sim=None):
    ssp = config.load_json('./tests/config_site_tle_simple.json')
    ssp['sim'].update({'mode': 'none', 'analytical_obs': True}, **(sim or {}))
    ssp['fpa']['num_frames'] = len(MIDPOINTS)
    ssp['fpa']['time'] = {'exposure': EXPOSURE, 'gap': GAP}
    ssp['geometry']['time'] = START
    ssp['geometry']['site'] = {
        'gimbal': {'mode': 'wcs', 'rotation': 0},
        'track': {'mode': 'rate', 'tle': TRACK_TLE},
        **site,
    }
    ssp['geometry']['obs']['list'] = [{'mode': 'tle', 'tle': TRACK_TLE, 'mv': 10, 'name': 'tracked'}, *obs]

    queue = MultithreadedTaskQueue()
    dirname = gen_images(ssp, eager=True, output_dir=str(tmp_path), queue=queue)
    queue.waitUntilEmpty()
    queue.stop()

    obs_dir = os.path.join(dirname, 'AnalyticalObservations')
    frames = []
    for name in sorted(os.listdir(obs_dir)):
        with open(os.path.join(obs_dir, name)) as f:
            frames.append(json.load(f))
    return dirname, frames


def _assert_sensor_state(record, expected):
    state = [record[k] for k in SENSOR_KEYS]
    np.testing.assert_allclose(state[:3], expected[:3], rtol=0, atol=POSITION_ATOL)
    np.testing.assert_allclose(state[3:], expected[3:], rtol=0, atol=VELOCITY_ATOL)


@pytest.mark.parametrize('form', SITE_FORMS, ids=SITE_FORMS)
def test_site_orbit_writes_sensor_state_at_frame_midpoints(form, tmp_path):
    site, expected = _site_case(form)

    _, frames = _run(site, tmp_path)

    assert len(frames) == len(MIDPOINTS)
    for records, ob_time, state in zip(frames, OB_TIMES, expected):
        assert [r['idOnOrbit'] for r in records] == ['tracked']
        assert records[0]['obTime'] == ob_time
        _assert_sensor_state(records[0], state)


def test_site_twobody_matches_site_tle_at_epoch(tmp_path):
    _, tle_frames = _run(_site_case('tle')[0], tmp_path / 'tle')
    _, twobody_frames = _run(_site_case('twobody')[0], tmp_path / 'twobody')

    tle, twobody = tle_frames[0][0], twobody_frames[0][0]
    assert twobody['obTime'] == tle['obTime'] == OB_TIMES[0]
    _assert_sensor_state(twobody, [tle[k] for k in SENSOR_KEYS])
    np.testing.assert_allclose(
        [twobody['ra'], twobody['declination']],
        [tle['ra'], tle['declination']],
        rtol=0,
        atol=ANGLE_ATOL,
    )


@pytest.mark.parametrize('form', ['twobody', 'ephemeris-flat'], ids=['twobody', 'ephemeris-flat'])
@pytest.mark.parametrize('sim, tracks, visible', [
    pytest.param({'enable_fov_filter': False}, 2, ['tracked', 'offset'], id='filter-off'),
    pytest.param({'enable_fov_filter': True}, 2, ['tracked', 'offset'], id='filter-default-radius'),
    pytest.param({'enable_fov_filter': True, 'fov_filter_radius': 0.05}, 1, ['tracked'], id='filter-narrow-radius'),
])
def test_site_orbit_fov_filter(form, sim, tracks, visible, tmp_path, caplog):
    ts_mid = time.utc_from_list(START, MIDPOINTS[0])
    ra, dec, *_ = get_los(create_sgp4(*SITE_TLE), create_sgp4(*TRACK_TLE), ts_mid)
    offset = {
        'mode': 'observation',
        'ra': float(ra),
        'dec': float(dec) + 0.2,
        'time': time.to_utc_list(ts_mid),
        'range': 36000.0,
        'mv': 10,
        'name': 'offset',
    }

    with caplog.at_level(logging.DEBUG, logger='satsim.satsim'):
        _, frames = _run(_site_case(form)[0], tmp_path, obs=[offset], sim={'enable_profiler': True, **sim})

    assert [[r['idOnOrbit'] for r in records] for records in frames] == [visible] * len(MIDPOINTS)
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
    propagated = [int(n) for n in re.findall(r'Object profile: .*\btracks=(\d+)', caplog.text)]
    assert propagated == [tracks] * len(MIDPOINTS)


@pytest.mark.parametrize('form', SITE_FORMS, ids=SITE_FORMS)
def test_site_orbit_czml_has_moving_observer_and_fov_cone(form, tmp_path):
    site, expected = _site_case(form)

    dirname, _ = _run(site, tmp_path)

    with open(os.path.join(dirname, 'satsim.czml')) as f:
        packets = {p['id']: p for p in json.load(f)}
    position = packets['GS0']['position']
    samples = np.reshape(position['cartesian'], (-1, 4))
    assert position['referenceFrame'] == 'INERTIAL'
    np.testing.assert_allclose(samples[:, 0], MIDPOINTS, rtol=0, atol=1e-6)
    np.testing.assert_allclose(samples[:, 1:], np.array(expected)[:, :3] * 1000, rtol=0, atol=1.0)
    assert np.linalg.norm(samples[1, 1:] - samples[0, 1:]) > 1000

    cone = packets['GS0_FOV']
    assert cone['position'] == {'reference': 'GS0#position'}
    assert cone['agi_rectangularSensor']['show'] is True
    assert len(cone['orientation']['unitQuaternion']) == 5 * len(MIDPOINTS)
