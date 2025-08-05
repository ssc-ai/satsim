import os
import json
from unittest.mock import patch

from satsim import config, gen_images
from satsim.util import configure_eager, MultithreadedTaskQueue
from satsim.geometry import sgp4
from astropy.io import fits as afits


def test_site_tle():
    configure_eager()
    ssp = config.load_json('./tests/config_site_tle_simple.json')
    ssp, _ = config.transform(ssp, max_stages=10, with_debug=True)
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1

    queue = MultithreadedTaskQueue()
    with patch('satsim.satsim.create_topocentric') as ctopo, \
         patch('satsim.satsim.create_sgp4', wraps=sgp4.create_sgp4) as csgp4:
        dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name='site_tle')
        queue.waitUntilEmpty()
        assert os.path.isdir(dirname)
        # observer should be created from TLE so create_topocentric is never called
        assert ctopo.call_count == 0
        # create_sgp4 called for site and track
        assert csgp4.call_count >= 2

        anno_dir = os.path.join(dirname, 'Annotations')
        files = [f for f in os.listdir(anno_dir) if f.endswith('.json')]
        with open(os.path.join(anno_dir, files[0])) as f:
            data = json.load(f)
        md = data['data']['metadata']
        assert 'x' in md and 'vx' in md
        assert 'lat' not in md

        fits_file = os.path.join(dirname, 'ImageFiles', os.listdir(os.path.join(dirname, 'ImageFiles'))[0])
        hdul = afits.open(fits_file)
        hdr = hdul[0].header
        assert 'SITEX' in hdr and 'SITEVX' in hdr
