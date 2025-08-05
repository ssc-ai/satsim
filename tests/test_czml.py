"""Tests for `czml` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import json

from satsim import config, image_generator
from satsim.io.czml import save_czml
from satsim.util import configure_eager


def test_czml():

    configure_eager()

    ssp = config.load_json('./tests/config_poppy.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    num_czml_samples = 10

    ssp['sim']['czml_samples'] = num_czml_samples
    ssp['sim']['mode'] = 'none'
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 60

    fpa_digital, frame_num, astrometrics, obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, num_shot_noise_samples, obs_cache, ground_truth, star_os_pix, segmentation = next(image_generator(ssp, None, None, None, with_meta=True, num_sets=1))
    j = save_czml(ssp, obs_cache, [astrometrics], './.images/satsim.czml')

    d = json.loads(j)

    # document
    assert(d[0]['id'] == 'document')
    assert(d[0]['clock']['currentTime'] == '2015-04-24T09:07:30.128000Z')
    assert(d[0]['clock']['interval'] == "2015-04-24T09:07:30.128Z/2015-04-24T09:08:32.628Z")

    # site
    assert(d[1]['id'] == 'GS0')
    assert(len(d[1]['position']['cartesian']) == 3)

    # sensor view cone
    assert(d[2]['id'] == 'GS0_FOV')
    assert(len(d[1]['position']['cartesian']) == 3)

    # satellites
    for i in range(3, len(d)):
        assert(d[i]['id'] == i - 3)
        assert(len(d[i]['position']['cartesian']) == num_czml_samples * 4)


def test_czml_space_observer():

    configure_eager()

    ssp = config.load_json('./tests/config_site_tle_simple.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    ssp['sim']['czml_samples'] = 5
    ssp['sim']['mode'] = 'none'
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 60

    fpa_digital, frame_num, astrometrics, obs_os_pix, fpa_conv_star, fpa_conv_targ, bg_tf, dc_tf, rn_tf, num_shot_noise_samples, obs_cache, ground_truth, star_os_pix, segmentation = next(image_generator(ssp, None, None, None, with_meta=True, num_sets=1))
    j = save_czml(ssp, obs_cache, [astrometrics], None)

    d = json.loads(j)

    assert d[1]['id'] == 'GS0'
    # space observer should have a trajectory with multiple points
    assert len(d[1]['position']['cartesian']) > 3
    assert d[2]['id'] == 'GS0_FOV'
