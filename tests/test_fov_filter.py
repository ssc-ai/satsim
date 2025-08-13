import os
import json

from satsim.util.thread import MultithreadedTaskQueue
from satsim import config, gen_images
from satsim.util import configure_eager


def _make_base_ssp():
    configure_eager()
    ssp = config.load_json('./tests/config_site_tle_simple.json')
    # Minimal, noise-free, small FPA config; keep stars disabled
    ssp['fpa']['num_frames'] = 5
    ssp['fpa']['time']['exposure'] = 1.0
    return ssp


def _read_objects_from_debug(dirname):
    # Debug metadata contains per-frame objects list mirroring rendered targets
    with open(os.path.join(dirname, 'Debug', 'metadata_0.json'), 'r') as f:
        md = json.load(f)
    return md['data']['objects']


def test_fov_filter_skips_far_targets():
    ssp = _make_base_ssp()

    # Enable FOV filter; default radius is the half-diagonal of the padded FOV
    ssp['sim']['enable_fov_filter'] = True

    # Use a space-based site with TLE observer and track a TLE.
    track_tle = [
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    other_tle = [
        "1 36412U 10008A   15115.05079343  .00000069  00000-0  00000+0 0  9992",
        "2 36412 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    ssp['geometry']['site']['track'] = {'mode': 'rate', 'tle': track_tle}
    ssp['geometry']['obs']['list'] = [
        {'mode': 'tle', 'tle': track_tle, 'mv': 10},  # near boresight
        {'mode': 'tle', 'tle': other_tle, 'mv': 10},  # farther away
    ]

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)
    objs = _read_objects_from_debug(dirname)

    # Expect only the centered target to remain
    assert len(objs) == 1


def test_fov_filter_disabled_keeps_all():
    ssp = _make_base_ssp()
    track_tle = [
        "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
        "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    other_tle = [
        "1 36412U 10008A   15115.45279343  .00000069  00000-0  00000+0 0  9992",
        "2 36412 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
    ]
    ssp['geometry']['site']['track'] = {'mode': 'rate', 'tle': track_tle}
    ssp['geometry']['obs']['list'] = [
        {'mode': 'tle', 'tle': track_tle, 'mv': 10},
        {'mode': 'tle', 'tle': other_tle, 'mv': 10},
    ]

    # Compare disabled vs enabled counts; disabled should be >= enabled
    ssp['sim']['enable_fov_filter'] = False
    queue = MultithreadedTaskQueue()
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue)
    objs_disabled = _read_objects_from_debug(dirname)

    ssp['sim']['enable_fov_filter'] = True
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue)
    objs_enabled = _read_objects_from_debug(dirname)
    queue.waitUntilEmpty()

    assert len(objs_disabled) == len(objs_enabled)
