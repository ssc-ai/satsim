import astropy.units as u
import numpy as np
import pytest

import satsim.geometry.factory as factory


TLE = (
    '1 25544U 98067A   20029.54791435  .00001264  00000-0  29621-4 0  9993',
    '2 25544  51.6440  30.9682 0005197  77.5934  20.6657 15.49147106211867',
)


def test_create_observer_supports_ground_and_tle_sites(monkeypatch):
    ground = object()
    space = object()
    monkeypatch.setattr(
        factory,
        'create_topocentric',
        lambda lat, lon, alt: ground if (lat, lon, alt) == ('1 N', '2 E', 0.3) else None,
    )
    monkeypatch.setattr(
        factory,
        'create_sgp4',
        lambda tle1, tle2: space if (tle1, tle2) == TLE else None,
    )
    assert factory.create_observer_from_config({
        'lat': '1 N', 'lon': '2 E', 'alt': 0.3,
    }) is ground
    assert factory.create_observer_from_config({'tle': list(TLE)}) is space


def test_create_target_supports_standard_ranging_modes(monkeypatch):
    default_time = [2020, 1, 1, 0, 0, 0.0]
    sentinels = {name: object() for name in ('tle', 'twobody', 'ephemeris')}
    captured = {}

    monkeypatch.setattr(factory, 'create_sgp4', lambda *args: sentinels['tle'])

    def create_twobody(position, velocity, epoch):
        captured['position'] = position
        captured['velocity'] = velocity
        captured['twobody_epoch'] = epoch
        return sentinels['twobody']

    def create_ephemeris(positions, velocities, seconds, epoch):
        captured['positions'] = positions
        captured['velocities'] = velocities
        captured['seconds'] = seconds
        captured['ephemeris_epoch'] = epoch
        return sentinels['ephemeris']

    monkeypatch.setattr(factory, 'create_twobody', create_twobody)
    monkeypatch.setattr(factory, 'create_ephemeris_object', create_ephemeris)

    assert factory.create_target_from_config({
        'mode': 'tle', 'tle': list(TLE),
    }, default_time) is sentinels['tle']

    state = {
        'mode': 'twobody',
        'position': [7000.0, 0.0, 0.0],
        'velocity': [0.0, 7.5, 0.0],
        'epoch': 0.0,
    }
    assert factory.create_target_from_config(state, default_time) is sentinels['twobody']
    assert captured['position'].unit == u.km
    assert captured['velocity'].unit.is_equivalent(u.km / u.s)
    np.testing.assert_allclose(captured['position'].value, state['position'])

    ephemeris = {
        'mode': 'ephemeris',
        'positions': [[7000.0, 0.0, 0.0]],
        'velocities': [[0.0, 7.5, 0.0]],
        'seconds_from_epoch': [0.0],
        'epoch': default_time,
    }
    assert factory.create_target_from_config(
        ephemeris, default_time
    ) is sentinels['ephemeris']
    assert captured['positions'] == ephemeris['positions']
    assert captured['velocities'] == ephemeris['velocities']
    assert captured['seconds'] == ephemeris['seconds_from_epoch']

    assert factory.create_target_from_config(
        {'mode': 'observation'}, default_time
    ) is None


def test_target_factory_rejects_invalid_modes_and_tles():
    with pytest.raises(ValueError, match='Unsupported ranging target mode'):
        factory.create_target_from_config({'mode': 'unknown'})
    with pytest.raises(ValueError, match='Unsupported ranging target mode'):
        factory.create_target_from_config({'mode': 'statevector'})
    with pytest.raises(ValueError, match='two non-empty strings'):
        factory.create_target_from_config({'mode': 'tle', 'tle': ['one']})
    with pytest.raises(ValueError, match='explicit id'):
        factory.target_id_from_config({'mode': 'twobody'})


def test_target_id_uses_explicit_id_or_tle_norad_number():
    assert factory.target_id_from_config({'id': 7}) == 7
    assert factory.target_id_from_config({'tle': list(TLE)}) == 25544
