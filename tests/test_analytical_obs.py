import os
import json
import copy
from satsim import config, gen_images
from satsim.util import MultithreadedTaskQueue
from tests.test_satsim import _gen_name


def test_analytical_observations():
    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['analytical_obs'] = True
    ssp['fpa']['detection'] = {
        'snr_threshold': 0.0,
        'pixel_error': 0.5,
        'false_alarm_rate': 1.0,
        'max_false': 2
    }
    ssp['fpa']['num_frames'] = 1
    ssp['geometry']['site'] = {
        "mode": "topo",
        "lat": "20.746111 N",
        "lon": "156.431667 W",
        "alt": 0.0,
        "gimbal": {"mode": "wcs", "rotation": 0},
        "track": {"mode": "fixed", "az": 0, "el": 90}
    }
    from satsim.geometry.astrometric import create_topocentric, get_los_azel
    from satsim import time
    topo = create_topocentric("20.746111 N", "156.431667 W", 0.0)
    ts = time.utc(2020, 1, 1, 0, 0, 0)
    ra_c, dec_c, _, _, _, _ = get_los_azel(topo, 0, 90, ts,
                                           deflection=False, aberration=False)

    ssp['geometry']['obs']['list'] = [
        {
            "mode": "observation",
            "ra": ra_c,
            "dec": dec_c,
            "time": [2020, 1, 1, 0, 0, 0],
            "range": 1000.0,
            "mv": 10
        }
    ]

    queue = MultithreadedTaskQueue()
    set_name = _gen_name('analytical')
    dirname = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    assert os.path.isdir(dirname)
    obs_dir = os.path.join(dirname, 'AnalyticalObservations')
    files = [f for f in os.listdir(obs_dir) if f.endswith('.json')]
    assert len(files) == 1
    with open(os.path.join(obs_dir, files[0])) as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) >= ssp['fpa']['detection']['max_false']
    assert len(data) <= ssp['fpa']['detection']['max_false'] + 1
    assert 'snrEst' in data[0]
    assert '+00:00' not in data[0]['obTime']
    assert isinstance(data[0]['senlat'], float)
    assert isinstance(data[0]['senlon'], float)


def test_analytical_observations_site_tle():
    ssp = config.load_json('./tests/config_site_tle_simple.json')
    ssp['sim']['analytical_obs'] = True
    ssp['fpa']['detection'] = {
        'snr_threshold': 0.0,
        'pixel_error': 0.5,
        'false_alarm_rate': 1.0,
        'max_false': 1
    }
    queue = MultithreadedTaskQueue()
    set_name = _gen_name('analytical_tle')
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    obs_dir = os.path.join(dirname, 'AnalyticalObservations')
    files = [f for f in os.listdir(obs_dir) if f.endswith('.json')]
    with open(os.path.join(obs_dir, files[0])) as f:
        data = json.load(f)
    assert 'senx' in data[0] and 'senvelx' in data[0]


def test_analytical_observations_threshold():
    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['analytical_obs'] = True
    ssp['fpa']['detection'] = {
        'snr_threshold': 1e6,
        'pixel_error': 0.0,
        'false_alarm_rate': 1.0,
        'max_false': 3
    }
    ssp['fpa']['num_frames'] = 1
    ssp['geometry']['site'] = {
        "mode": "topo",
        "lat": "20.746111 N",
        "lon": "156.431667 W",
        "alt": 0.0,
        "gimbal": {"mode": "wcs", "rotation": 0},
        "track": {"mode": "fixed", "az": 0, "el": 90}
    }
    from satsim.geometry.astrometric import create_topocentric, get_los_azel
    from satsim import time
    topo = create_topocentric("20.746111 N", "156.431667 W", 0.0)
    ts = time.utc(2020, 1, 1, 0, 0, 0)
    ra_c, dec_c, _, _, _, _ = get_los_azel(topo, 0, 90, ts,
                                           deflection=False, aberration=False)

    ssp['geometry']['obs']['list'] = [
        {
            "mode": "observation",
            "ra": ra_c,
            "dec": dec_c,
            "time": [2020, 1, 1, 0, 0, 0],
            "range": 1000.0,
            "mv": 10
        }
    ]

    queue = MultithreadedTaskQueue()
    set_name = _gen_name('analytical2')
    dirname = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    obs_dir = os.path.join(dirname, 'AnalyticalObservations')
    files = [f for f in os.listdir(obs_dir) if f.endswith('.json')]
    with open(os.path.join(obs_dir, files[0])) as f:
        data = json.load(f)

    assert len(data) == ssp['fpa']['detection']['max_false']


def test_truth_annotation_ra_dec():
    ssp = config.load_json('./tests/config_static.json')
    ssp['fpa']['num_frames'] = 1
    ssp['geometry']['site'] = {
        "mode": "topo",
        "lat": "20.746111 N",
        "lon": "156.431667 W",
        "alt": 0.0,
        "gimbal": {"mode": "wcs", "rotation": 0},
        "track": {"mode": "fixed", "az": 0, "el": 90}
    }
    from satsim.geometry.astrometric import create_topocentric, get_los_azel
    from satsim import time

    ts = time.utc(2020, 1, 1, 0, 0, 0)
    topo = create_topocentric("20.746111 N", "156.431667 W", 0.0)
    ra_c, dec_c, _, _, _, _ = get_los_azel(topo, 0, 90, ts,
                                           deflection=False, aberration=False)

    ssp['geometry']['obs']['list'] = [
        {
            "mode": "observation",
            "ra": ra_c,
            "dec": dec_c,
            "time": [2020, 1, 1, 0, 0, 0],
            "range": 1000.0,
            "mv": 10
        }
    ]

    queue = MultithreadedTaskQueue()
    set_name = _gen_name('truthra')
    dirname = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images',
                         output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    anno_dir = os.path.join(dirname, 'Annotations')
    files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    with open(os.path.join(anno_dir, files[0])) as f:
        data = json.load(f)

    found = any('ra' in ob and 'ra_obs' in ob
                for ob in data['data']['objects']
                if ob['class_name'] == 'Satellite')
    assert found


def test_analytical_obs_mode_none():
    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['analytical_obs'] = True
    ssp['sim']['mode'] = 'none'
    ssp['fpa']['detection'] = {
        'snr_threshold': 0.0,
        'pixel_error': 0.0,
        'false_alarm_rate': 0.0,
        'max_false': 0
    }
    ssp['fpa']['num_frames'] = 1
    ssp['geometry']['site'] = {
        "mode": "topo",
        "lat": "20.746111 N",
        "lon": "156.431667 W",
        "alt": 0.0,
        "gimbal": {"mode": "wcs", "rotation": 0},
        "track": {"mode": "fixed", "az": 0, "el": 90}
    }
    from satsim.geometry.astrometric import create_topocentric, get_los_azel
    from satsim import time
    topo = create_topocentric("20.746111 N", "156.431667 W", 0.0)
    ts = time.utc(2020, 1, 1, 0, 0, 0)
    ra_c, dec_c, _, _, _, _ = get_los_azel(topo, 0, 90, ts,
                                           deflection=False, aberration=False)

    ssp['geometry']['obs']['list'] = [
        {
            "mode": "observation",
            "ra": ra_c,
            "dec": dec_c,
            "time": [2020, 1, 1, 0, 0, 0],
            "range": 1000.0,
            "mv": 10
        }
    ]

    queue = MultithreadedTaskQueue()
    set_name = _gen_name('analytical_none')
    dirname = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images',
                         output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    obs_dir = os.path.join(dirname, 'AnalyticalObservations')
    files = [f for f in os.listdir(obs_dir) if f.endswith('.json')]
    assert len(files) == 1
    with open(os.path.join(obs_dir, files[0])) as f:
        data = json.load(f)
    assert len(data) == 1
    anno_dir = os.path.join(dirname, 'Annotations')
    anno_files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
    assert len(anno_files) == 1
