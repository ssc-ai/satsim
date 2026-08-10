import copy
import json
from pathlib import Path

from jsonschema import Draft7Validator, ValidationError
import pytest
from referencing import Registry, Resource

from satsim.config import load_json, transform
from satsim.passive_rf.config import (
    load_receivers,
    load_run_config,
)
from satsim.passive_rf.model import simulate_run_config


ROOT = Path(__file__).parents[1]
SCHEMA_DIR = ROOT / 'schema' / 'v1'
CONFIG_PATH = Path(__file__).parent / 'config_passive_rf.json'


def _config():
    return transform(
        load_json(str(CONFIG_PATH)),
        dirname=str(CONFIG_PATH.parent),
    )


def _shared_config(document):
    shared = copy.deepcopy(document['passive_rf'][0])
    document['passive_rf'] = shared
    return shared


def test_load_run_config_and_units():
    config = load_run_config(_config())
    assert list(config.receivers) == ['RX_A', 'RX_B', 'RX_C']
    assert set(config.observers) == {'RX_A', 'RX_B', 'RX_C'}
    assert config.receivers['RX_B'].antenna_gain_dbi == 29.5
    assert config.receivers['RX_A'].az_limits == (0.0, 100.0)
    assert config.receivers['RX_B'].az_limits == (0.0, 90.0)
    assert config.receivers['RX_C'].el_limits == (-90.0, 0.0)
    assert list(config.targets) == [25544]
    assert list(config.target_objects) == [25544]
    target = config.targets[25544]
    assert target.name == 'ISS (ZARYA)'
    assert target.frequency_hz == 2.2e9
    assert target.bandwidth_hz == 500000.0
    assert target.eirp_dbw == 45.0
    assert target.band == 'S-band'
    assert config.estimator.integration_time_s == 1.0
    assert config.estimator.coherent_time_s == 0.005
    assert config.estimator.caf_loss_db == 3.0
    assert config.estimator.tdoa_floor_by_sensor['RX_A'] == 1.2e-8
    assert len(config.frame_times) == 2
    assert len(config.observations) == 6


def test_frame_flow_matches_radar_midpoint_and_all_unique_pairs():
    config = load_run_config(_config())
    assert config.frame_times[0].utc_iso() == '2020-01-29T13:10:00Z'
    assert config.frame_times[1].utc_iso() == '2020-01-29T13:10:01Z'
    assert [(item.sensor1, item.sensor2) for item in config.observations[:3]] == [
        ('RX_A', 'RX_B'),
        ('RX_A', 'RX_C'),
        ('RX_B', 'RX_C'),
    ]


def test_gap_separates_dwell_from_frame_spacing():
    document = _config()
    for receiver in document['passive_rf']:
        receiver['time']['gap'] = 29.0
    config = load_run_config(document)
    assert config.frame_times[0].utc_iso() == '2020-01-29T13:10:00Z'
    assert config.frame_times[1].utc_iso() == '2020-01-29T13:10:30Z'
    assert config.estimator.integration_time_s == 1.0


def test_shared_passive_rf_dict_applies_to_every_site():
    document = _config()
    shared = _shared_config(document)
    shared['antenna_gain'] = 27.5
    config = load_run_config(document)
    assert {receiver.antenna_gain_dbi for receiver in config.receivers.values()} == {27.5}
    assert set(config.estimator.tdoa_floor_by_sensor) == {'RX_A', 'RX_B', 'RX_C'}


def test_passive_rf_array_must_align_with_geometry_sites():
    document = _config()
    document['passive_rf'].pop()
    with pytest.raises(ValueError, match='length must match'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][1]['time']['dwell'] = 2.0
    with pytest.raises(ValueError, match='same timing and run controls'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][1]['time']['gap'] = 1.0
    with pytest.raises(ValueError, match='same timing and run controls'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][1]['estimator']['caf_loss'] = 4.0
    with pytest.raises(ValueError, match='same estimator controls'):
        load_run_config(document)


def test_receiver_conversion_supports_cardinals_and_validates_sites():
    document = _config()
    document['geometry']['site'][0]['lat'] = '18.8 N'
    document['geometry']['site'][0]['lon'] = '99.1 E'
    receivers, observers = load_receivers(document)
    assert 'RX_A' in receivers
    assert 'RX_A' in observers

    document['geometry']['site'][0]['lat'] = 91
    with pytest.raises(ValueError, match=r'lat must be in \[-90'):
        load_receivers(document)

    document = _config()
    document['geometry']['site'] = document['geometry']['site'][:1]
    document['passive_rf'] = document['passive_rf'][:1]
    with pytest.raises(ValueError, match='at least two'):
        load_receivers(document)

    document = _config()
    document['geometry']['site'][0]['tle'] = ['line1', 'line2']
    with pytest.raises(ValueError, match='must be a ground site'):
        load_receivers(document)


def test_site_names_are_required_and_unique():
    document = _config()
    document['geometry']['site'][0].pop('name')
    with pytest.raises(ValueError, match='name must be a non-empty string'):
        load_run_config(document)

    document = _config()
    document['geometry']['site'][1]['name'] = 'RX_A'
    with pytest.raises(ValueError, match='Duplicate geometry.site name'):
        load_run_config(document)


def test_targets_are_loaded_from_geometry_object_properties():
    document = _config()
    target = document['geometry']['obs']['list'][0]
    target['id'] = 77777
    run = load_run_config(document)
    assert list(run.targets) == [77777]
    assert list(run.target_objects) == [77777]


def test_optional_band_is_not_invented():
    document = _config()
    document['geometry']['obs']['list'][0].pop('band')
    run = load_run_config(document)
    assert run.targets[25544].band is None
    assert all('band' not in record for record in simulate_run_config(run))


def test_id_on_orbit_is_only_emitted_for_named_targets():
    document = _config()
    document['geometry']['obs']['list'][0].pop('name')
    records = simulate_run_config(load_run_config(document))
    assert all(record['satNo'] == 25544 for record in records)
    assert all('idOnOrbit' not in record for record in records)


@pytest.mark.parametrize('mode', ['twobody', 'ephemeris'])
def test_targets_use_shared_satsim_ranging_modes(mode):
    document = _config()
    target = document['geometry']['obs']['list'][0]
    target.pop('tle')
    target['mode'] = mode
    if mode == 'twobody':
        target.update({
            'position': [7000.0, 0.0, 0.0],
            'velocity': [0.0, 7.5, 0.0],
            'epoch': [2020, 1, 29, 13, 9, 59.5],
        })
    else:
        target.update({
            'positions': [
                [7000.0, 0.0, 0.0],
                [7000.0, 75.0, 0.0],
                [6999.0, 150.0, 0.0],
            ],
            'velocities': [[0.0, 7.5, 0.0]] * 3,
            'seconds_from_epoch': [0.0, 10.0, 20.0],
            'epoch': [2020, 1, 29, 13, 9, 59.5],
        })
    run = load_run_config(document)
    assert list(run.targets) == [25544]
    assert list(run.target_objects) == [25544]


@pytest.mark.parametrize('change,match', [
    ('missing_tle', 'TLE target requires'),
    ('bad_mode', 'ranging-capable'),
    ('missing_signature', 'requires frequency, bandwidth, and eirp'),
    ('bad_frequency', 'must be a finite number'),
    ('partial_signature', 'RF target is missing'),
])
def test_invalid_native_targets(change, match):
    document = _config()
    target = document['geometry']['obs']['list'][0]
    if change == 'missing_tle':
        target.pop('tle')
    elif change == 'bad_mode':
        target.pop('tle')
        target['mode'] = 'observation'
    elif change == 'missing_signature':
        for key in ('frequency', 'bandwidth', 'eirp', 'band'):
            target.pop(key)
        target['rcs'] = 1.0
    elif change == 'bad_frequency':
        target['frequency'] = None
    else:
        target.pop('bandwidth')
    with pytest.raises(ValueError, match=match):
        load_run_config(document)


def test_duplicate_target_ids_are_rejected():
    document = _config()
    duplicate = copy.deepcopy(document['geometry']['obs']['list'][0])
    document['geometry']['obs']['list'].append(duplicate)
    with pytest.raises(ValueError, match='Duplicate passive RF target id'):
        load_run_config(document)


def test_estimator_rejects_invalid_shapes_and_models():
    document = _config()
    document['passive_rf'][0]['estimator']['rms_bandwidth'] = 'triangle'
    with pytest.raises(ValueError, match='rms_bandwidth'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][0]['estimator']['uncertainty'] = []
    with pytest.raises(ValueError, match='uncertainty must be an object'):
        load_run_config(document)


def test_field_of_view_is_per_receiver_and_validated():
    document = _config()
    document['passive_rf'][0]['field_of_view'] = []
    with pytest.raises(ValueError, match='field_of_view must be an object'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][0]['field_of_view']['azimuth'] = [0.0]
    with pytest.raises(ValueError, match=r'azimuth must contain'):
        load_run_config(document)

    document = _config()
    document['passive_rf'][0]['field_of_view']['elevation'] = [10.0, 0.0]
    with pytest.raises(ValueError, match='minimum must not exceed maximum'):
        load_run_config(document)


def test_run_config_rejects_samples_and_timing():
    document = _config()
    document['sim']['samples'] = 2
    with pytest.raises(ValueError, match='sim.samples == 1'):
        load_run_config(document)

    document = _config()
    for receiver in document['passive_rf']:
        receiver['time']['dwell'] = 0
    with pytest.raises(ValueError, match='must be positive'):
        load_run_config(document)

    document = _config()
    for receiver in document['passive_rf']:
        receiver['time']['gap'] = -1
    with pytest.raises(ValueError, match='must be nonnegative'):
        load_run_config(document)

    document = _config()
    document['geometry']['time'] = [2020, 1]
    with pytest.raises(ValueError, match='geometry.time must be'):
        load_run_config(document)


def test_run_config_requires_native_blocks():
    document = _config()
    document.pop('passive_rf')
    with pytest.raises(ValueError, match='requires a passive_rf object or array'):
        load_run_config(document)

    document = _config()
    document.pop('geometry')
    with pytest.raises(ValueError, match='requires a geometry object'):
        load_run_config(document)


def _data(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _registry():
    resources = []
    for path in SCHEMA_DIR.rglob('*.json'):
        schema = _data(path)
        if '$id' in schema:
            resources.append((schema['$id'], Resource.from_contents(schema)))
    return Registry().with_resources(resources)


def _validate(schema_name, data):
    schema = _data(SCHEMA_DIR / schema_name)
    Draft7Validator(schema, registry=_registry()).validate(data)


def _radar_config():
    return {
        'tx_power': 1.0e6,
        'tx_frequency': 1.0e9,
        'antenna_diameter': 10.0,
        'efficiency': 0.6,
        'field_of_view': {
            'azimuth': [0.0, 360.0],
            'elevation': [0.0, 90.0],
        },
        'range_limits': [0.0, 50000.0],
        'detection': {
            'min_detectable_power': 1.0e-16,
            'snr_threshold': 1.0,
            'angle_error': 0.01,
            'range_error': 0.1,
            'range_rate_error': 0.001,
            'false_alarm_rate': 0.0,
        },
        'time': {'dwell': 1.0, 'gap': 0.0},
        'num_frames': 1,
    }


def _analytical_document(mode):
    document = {
        'version': 1,
        'sim': {'samples': 1},
        'geometry': {
            'site': {'name': 'SENSOR', 'lat': 0.0, 'lon': 0.0, 'alt': 0.0},
            'obs': {'mode': 'list', 'list': []},
        },
    }
    if mode == 'radar':
        document['radar'] = _radar_config()
    else:
        document.update({'fpa': {}, 'background': {}})
        document['geometry']['stars'] = {'mode': 'none'}
    return document


def test_document_schema_accepts_passive_rf_fixture():
    _validate('Document.json', _data(CONFIG_PATH))


def test_passive_rf_schema_accepts_shared_dict_and_aligned_array():
    config = _data(CONFIG_PATH)
    _validate('PassiveRF.json', config['passive_rf'])
    _validate('PassiveRF.json', config['passive_rf'][0])


def test_passive_rf_schema_rejects_unsupported_fields():
    config = _data(CONFIG_PATH)['passive_rf'][0]
    for invalid_key in (
        'receiver_pairs', 'transmitters', 'observation_plan', 'tle_bank',
        'min_elevation', 'fdoa_sign', 'verbose',
    ):
        invalid = copy.deepcopy(config)
        invalid[invalid_key] = []
        with pytest.raises(ValidationError):
            _validate('PassiveRF.json', invalid)

    for invalid_key, value in (('fdoa_sign', -1.0), ('verbose', True)):
        invalid = copy.deepcopy(config)
        invalid['estimator'][invalid_key] = value
        with pytest.raises(ValidationError):
            _validate('PassiveRF.json', invalid)


def test_passive_rf_and_radar_share_field_of_view_schema():
    config = _data(CONFIG_PATH)
    field_of_view = config['passive_rf'][0]['field_of_view']
    _validate('types/FieldOfView.json', field_of_view)
    radar = _radar_config()
    radar['field_of_view'] = field_of_view
    _validate('Radar.json', radar)

    invalid = copy.deepcopy(config['passive_rf'][0])
    invalid['field_of_view']['azimuth'] = [0.0]
    with pytest.raises(ValidationError):
        _validate('PassiveRF.json', invalid)


def test_passive_rf_and_radar_share_dwell_gap_timing_schema():
    config = _data(CONFIG_PATH)['passive_rf'][0]
    config['time'] = {'dwell': 1.0, 'gap': 29.0}
    _validate('PassiveRF.json', config)
    radar = _radar_config()
    radar['time'] = config['time']
    _validate('Radar.json', radar)

    config['time']['gap'] = -1.0
    with pytest.raises(ValidationError):
        _validate('PassiveRF.json', config)
    with pytest.raises(ValidationError):
        _validate('Radar.json', radar)


def test_document_schema_requires_geometry_for_passive_rf():
    config = _data(CONFIG_PATH)
    config.pop('geometry')
    with pytest.raises(ValidationError):
        _validate('Document.json', config)


def test_document_schema_rejects_mixed_mode():
    config = _data(CONFIG_PATH)
    config['fpa'] = {}
    config['background'] = {}
    with pytest.raises(ValidationError):
        _validate('Document.json', config)


def test_geometry_site_accepts_object_or_passive_rf_array():
    config = _data(CONFIG_PATH)
    _validate('Geometry.json', config['geometry'])
    config['geometry']['site'] = config['geometry']['site'][0]
    _validate('Geometry.json', config['geometry'])


def test_geometry_site_array_uses_full_site_validation():
    config = _data(CONFIG_PATH)
    config['geometry']['site'][0]['track'] = {'mode': 'invalid'}
    with pytest.raises(ValidationError):
        _validate('Geometry.json', config['geometry'])


def test_document_schema_enforces_mode_specific_site_shapes():
    passive = _data(CONFIG_PATH)
    _validate('Document.json', passive)
    passive['geometry']['site'] = passive['geometry']['site'][0]
    with pytest.raises(ValidationError):
        _validate('Document.json', passive)

    radar = _analytical_document('radar')
    _validate('Document.json', radar)
    radar['geometry']['site'] = [radar['geometry']['site']] * 2
    with pytest.raises(ValidationError):
        _validate('Document.json', radar)

    eo = _analytical_document('eo')
    _validate('Document.json', eo)
    eo['geometry']['site'] = [eo['geometry']['site']] * 2
    with pytest.raises(ValidationError):
        _validate('Document.json', eo)


@pytest.mark.parametrize('track', [
    {'mode': 'fixed', 'az': 0.0, 'el': 45.0},
    {'mode': 'radec', 'ra': 180.0, 'dec': 10.0},
    {'mode': 'sidereal'},
    {'mode': 'rate-sidereal', 'tle': ['line 1', 'line 2']},
    {
        'mode': 'rate',
        'position': [7000.0, 0.0, 0.0],
        'velocity': [0.0, 7.5, 0.0],
        'epoch': 0.0,
    },
])
def test_site_schema_accepts_unambiguous_tracking_modes(track):
    _validate('types/Site.json', {'track': track})


def test_radar_schema_requires_core_fields_and_physical_bounds():
    with pytest.raises(ValidationError):
        _validate('Radar.json', {})

    invalid_cases = (
        ('tx_power', 0.0),
        ('tx_frequency', -1.0),
        ('antenna_diameter', -1.0),
        ('efficiency', 1.1),
        ('num_frames', 0),
    )
    for key, value in invalid_cases:
        radar = _radar_config()
        radar[key] = value
        with pytest.raises(ValidationError):
            _validate('Radar.json', radar)


def test_geometry_target_uses_direct_rf_properties_like_radar_rcs():
    config = _data(CONFIG_PATH)
    target = config['geometry']['obs']['list'][0]
    assert target['mode'] == 'tle'
    assert len(target['tle']) == 2
    assert {
        key for key in target
        if key in {'frequency', 'bandwidth', 'eirp', 'band'}
    } == {'frequency', 'bandwidth', 'eirp', 'band'}
    _validate('Document.json', config)

    target['center_frequency'] = target.pop('frequency')
    with pytest.raises(ValidationError):
        _validate('Document.json', config)


def test_geometry_rf_target_accepts_twobody_mode():
    config = _data(CONFIG_PATH)
    target = config['geometry']['obs']['list'][0]
    target.pop('tle')
    target.update({
        'mode': 'twobody',
        'position': [7000.0, 0.0, 0.0],
        'velocity': [0.0, 7.5, 0.0],
        'epoch': [2020, 1, 29, 13, 9, 59.5],
    })
    _validate('Document.json', config)


def test_passive_rf_array_requires_at_least_two_entries():
    config = _data(CONFIG_PATH)
    with pytest.raises(ValidationError):
        _validate('PassiveRF.json', config['passive_rf'][:1])
