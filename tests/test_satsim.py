"""Tests for `satsim` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os
import pickle
import json
import copy
from datetime import datetime

import numpy as np
import scipy
from tifffile import imread
from astropy.io import fits as afits


from satsim import config, gen_images
from satsim.util import configure_eager
from satsim.util.system import is_tensorflow_running_on_cpu
from satsim.util import MultithreadedTaskQueue


def test_star_brightness():
    _test_star_brightness(render_size=None)
    _test_star_brightness(render_size=[100, 100])
    _test_star_brightness(render_size=[400, 600])


def test_target_brightness():
    _test_target_brightness(render_size=None)
    _test_target_brightness(render_size=[100, 100])
    _test_target_brightness(render_size=[400, 600])


def test_target_centroid(render_size=None):
    _test_target_centroid(render_size=None)
    _test_target_centroid(render_size=[101, 101])
    _test_target_centroid(render_size=[401, 601])


def _test_star_brightness(render_size=None):

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')

    if render_size is not None:
        ssp['sim']['render_size'] = render_size

    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['mv']['bins'] = [10,11]
    ssp['geometry']['stars']['mv']['density'] = [10000.0]
    ssp['geometry']['obs']['list'] = []

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_0 = pickle.load(f)

    ssp['fpa']['time']['exposure'] = 2

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_1 = pickle.load(f)

    # stars are being resampled so 2x brightness will not be exact
    assert((np.sum(fpa_conv_star_0.flatten()) * 2 - np.sum(fpa_conv_star_1.flatten())) / np.sum(fpa_conv_star_1.flatten()) < 0.1)


def test_star_brightness_polar():

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['mv']['bins'] = [10,11]
    ssp['geometry']['stars']['mv']['density'] = [10000.0]
    ssp['geometry']['stars']['motion']['mode'] = 'affine-polar'
    ssp['geometry']['stars']['motion']['translation'] = [45.0, 3.0]
    ssp['geometry']['obs']['list'] = []

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_0 = pickle.load(f)

    ssp['fpa']['time']['exposure'] = 2

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_1 = pickle.load(f)

    # stars are being resampled so 2x brightness will not be exact
    assert((np.sum(fpa_conv_star_0.flatten()) * 2 - np.sum(fpa_conv_star_1.flatten())) / np.sum(fpa_conv_star_1.flatten()) < 0.1)


def _test_target_brightness(render_size=None):

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')

    if render_size is not None:
        ssp['sim']['render_size'] = render_size

    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    if render_size is None:
        with open(os.path.join(dirname, 'Debug', 'fpa_os_0.pickle'), 'rb') as f:
            fpa_os_0 = pickle.load(f)

        ssp['fpa']['time']['exposure'] = 2

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    if render_size is None:
        with open(os.path.join(dirname, 'Debug', 'fpa_os_0.pickle'), 'rb') as f:
            fpa_os_1 = pickle.load(f)

        np.testing.assert_approx_equal(np.sum(fpa_os_0.flatten()) * 2, np.sum(fpa_os_1.flatten()), significant=6)

    with open(os.path.join(dirname, 'Debug', 'metadata_0.json'), 'r') as f:
        metadata_1 = json.load(f)

    filtered_obs = [item for item in ssp['geometry']['obs']['list'] if item.get('mode') != 'none']
    for a, b in zip(filtered_obs, metadata_1['data']['objects']):
        if 'mv_truth' in a:
            assert(a['mv_truth'] == b['magnitude'])
        if 'pe_truth' in a:
            assert(a['pe_truth'] == b['pe_per_sec'])


def _test_target_centroid(render_size=None):
    """ Note `scipy.ndimage.center_of_mass` returns center of pixel
    as whole numbers and edges as X.5.
    """

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')

    if render_size is not None:
        ssp['sim']['render_size'] = render_size

    ssp['sim']['spacial_osf'] = 3
    ssp['fpa']['height'] = 401
    ssp['fpa']['width'] = 601
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['obs']['list'] = [
        {
            "mode": "line",
            "origin": [0.5, 0.5],
            "velocity": [0, 0],
            "mv": 10
        },
        # {
        #     "mode": "line",
        #     "origin": [0.0, 0.0],
        #     "velocity": [401,601],
        #     "mv": 10
        # },
        # {
        #     "mode": "line",
        #     "origin": [1.0, 0.0],
        #     "velocity": [-401, 601],
        #     "mv": 10
        # }
    ]

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    if render_size is None:

        with open(os.path.join(dirname, 'Debug', 'fpa_os_0.pickle'), 'rb') as f:
            fpa = pickle.load(f)

        # print(scipy.ndimage.center_of_mass(fpa))

        np.testing.assert_equal(scipy.ndimage.center_of_mass(fpa), (np.asarray(fpa.shape) - 1) / 2)

        with open(os.path.join(dirname, 'Debug', 'fpa_conv_os_0.pickle'), 'rb') as f:
            fpa2 = pickle.load(f)

        # print(scipy.ndimage.center_of_mass(fpa2))

        if is_tensorflow_running_on_cpu():
            np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa2), (np.asarray(fpa2.shape) - 1) / 2, rtol=0.03, atol=2e-5)
        else:
            np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa2), (np.asarray(fpa2.shape) - 1) / 2, 1e-5)

        with open(os.path.join(dirname, 'Debug', 'fpa_conv_crop_0.pickle'), 'rb') as f:
            fpa3 = pickle.load(f)

        # print(scipy.ndimage.center_of_mass(fpa3))

        np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa3), (np.asarray(fpa3.shape) - 1) / 2, 1e-5)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_0.pickle'), 'rb') as f:
        fpa4 = pickle.load(f)

    # print(scipy.ndimage.center_of_mass(fpa4))

    np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa4), (np.asarray(fpa4.shape) - 1) / 2, 1e-5)


def test_target_centroid_polar():
    """ Note `scipy.ndimage.center_of_mass` returns center of pixel
    as whole numbers and edges as X.5.
    """

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['spacial_osf'] = 3
    ssp['fpa']['height'] = 401
    ssp['fpa']['width'] = 601
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['obs']['list'] = [
        {
            "mode": "line-polar",
            "origin": [0.5, 0.5],
            "velocity": [90, 30],
            "mv": 1
        },
    ]

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_os_0.pickle'), 'rb') as f:
        fpa = pickle.load(f)

    np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa), [946.0, 1201.0], rtol=1e-6, atol=0)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_0.pickle'), 'rb') as f:
        fpa4 = pickle.load(f)

    np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa4), [215, 300], rtol=1e-3, atol=0)


def test_arcsec():
    """ Note `scipy.ndimage.center_of_mass` returns center of pixel
    as whole numbers and edges as X.5.
    """

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['spacial_osf'] = 3
    ssp['sim']['velocity_units'] = 'arcsec'
    ssp['fpa']['height'] = 101
    ssp['fpa']['width'] = 101
    ssp['fpa']['y_fov'] = 0.2525
    ssp['fpa']['x_fov'] = 0.2525  # 9 arcsec pixels
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['dark_current'] = 0
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 1000.0
    ssp['geometry']['stars']['pad'] = 2.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['obs']['list'] = [
        {
            "mode": "line",
            "origin": [0.5, 0.5],
            "velocity": [9 * 3, 0],  # move 3 pixels/sec
            "mv": 1
        },
    ]

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_0.pickle'), 'rb') as f:
        fpa4 = pickle.load(f)

    np.testing.assert_allclose(scipy.ndimage.center_of_mass(fpa4), np.array([51.5, 50]), 1e-5)


def test_poppy():

    configure_eager()

    queue = MultithreadedTaskQueue()

    ssp = config.load_json('./tests/config_poppy.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1

    dir_name = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue)
    queue.waitUntilEmpty()
    hdul = afits.open(os.path.join(dir_name, 'ImageFiles', 'sat_00000.0000.fits'))
    hdulhdr = hdul[0].header
    assert(hdulhdr['EXPTIME'] == ssp['fpa']['time']['exposure'])
    assert(hdulhdr['TRKMODE'] == ssp['geometry']['site']['track']['mode'])

    ssp['sim']['enable_shot_noise'] = False
    ssp['sim']['save_pickle'] = False
    ssp['geometry']['site']['track']['mode'] = 'sidereal'
    ssp['geometry']['site']['track']['tle'] = [ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2']]
    del(ssp['geometry']['site']['track']['tle1'])
    del(ssp['geometry']['site']['track']['tle2'])
    dir_name = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue)
    queue.waitUntilEmpty()
    hdul = afits.open(os.path.join(dir_name, 'ImageFiles', 'sat_00000.0000.fits'))
    hdulhdr = hdul[0].header
    assert(hdulhdr['EXPTIME'] == ssp['fpa']['time']['exposure'])
    assert(hdulhdr['TRKMODE'] == ssp['geometry']['site']['track']['mode'])

    ssp['fpa']['num_frames'] = 2
    ssp['geometry']['site']['track']['mode'] = 'rate-sidereal'
    dir_name = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue)
    queue.waitUntilEmpty()
    hdul = afits.open(os.path.join(dir_name, 'ImageFiles', 'sat_00000.0000.fits'))
    hdulhdr = hdul[0].header
    assert(hdulhdr['EXPTIME'] == ssp['fpa']['time']['exposure'])
    assert(hdulhdr['TRKMODE'] == 'rate')
    hdul = afits.open(os.path.join(dir_name, 'ImageFiles', 'sat_00000.0001.fits'))
    hdulhdr = hdul[0].header
    assert(hdulhdr['EXPTIME'] == ssp['fpa']['time']['exposure'])
    assert(hdulhdr['TRKMODE'] == 'sidereal')

    ssp['geometry']['site']['track']['mode'] = 'rate'
    ssp['geometry']['site']['track']['position'] = [-35180.62550265, -23252.99066344, 92.95410805]
    ssp['geometry']['site']['track']['velocity'] = [1.69553697, -2.56443628, 1.12318636e-03]
    ssp['geometry']['site']['track']['epoch'] = [2015, 4, 24, 9, 7, 24.128]
    del(ssp['geometry']['site']['track']['tle'])
    gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)


def test_render_modes():
    """ Note `scipy.ndimage.center_of_mass` returns center of pixel
    as whole numbers and edges as X.5.
    """

    configure_eager()

    ssp = config.load_json('./tests/config_static.json')

    ssp['sim']['enable_shot_noise'] = False
    ssp['sim']['spacial_osf'] = 3
    ssp['fpa']['height'] = 400
    ssp['fpa']['width'] = 600
    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1
    ssp['background']['galactic'] = 15.0
    ssp['geometry']['stars']['mv']['bins'] = []
    ssp['geometry']['stars']['mv']['density'] = []
    ssp['geometry']['obs']['list'] = [
        {
            "mode": "line",
            "origin": [0.5, 0.5],
            "velocity": [0, 0],
            "mv": 10
        },
    ]

    ssp['sim']['calculate_snr'] = True

    queue = MultithreadedTaskQueue()

    dirname0 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_render_modes'))
    queue.waitUntilEmpty()

    with open(os.path.join(dirname0, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_0 = pickle.load(f)
    with open(os.path.join(dirname0, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        fpa_conv_targ_0 = pickle.load(f)

    ssp['sim']['calculate_snr'] = False

    dirname1 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_render_modes2'))
    queue.waitUntilEmpty()

    with open(os.path.join(dirname1, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_1 = pickle.load(f)

    # ssp['sim']['render_size'] = [101, 101]  # TODO investigate why odd render_size doesn't work properly
    ssp['sim']['render_size'] = [100, 100]
    ssp['sim']['calculate_snr'] = True

    dirname2 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_render_modes3'))
    queue.waitUntilEmpty()

    with open(os.path.join(dirname2, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_2 = pickle.load(f)
    with open(os.path.join(dirname2, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        fpa_conv_targ_2 = pickle.load(f)

    ssp['sim']['render_size'] = [100, 100]
    ssp['sim']['calculate_snr'] = False

    dirname3 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_render_modes4'))
    queue.waitUntilEmpty()

    with open(os.path.join(dirname3, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_3 = pickle.load(f)

    # print(np.sum(fpa_conv_star_0.flatten()))
    # print(np.sum(fpa_conv_targ_0.flatten()))
    # print(np.sum(fpa_conv_star_1.flatten()))

    # print(np.sum(fpa_conv_star_2.flatten()))
    # print(np.sum(fpa_conv_targ_2.flatten()))
    # print(np.sum(fpa_conv_star_3.flatten()))

    np.testing.assert_array_equal(fpa_conv_star_0 + fpa_conv_targ_0, fpa_conv_star_1)
    np.testing.assert_array_equal(fpa_conv_star_2 + fpa_conv_targ_2, fpa_conv_star_3)

    # print(scipy.ndimage.center_of_mass(fpa_conv_star_0 + fpa_conv_targ_0))
    # print(scipy.ndimage.center_of_mass(fpa_conv_star_2 + fpa_conv_targ_2))

    if is_tensorflow_running_on_cpu():
        np.testing.assert_allclose(fpa_conv_star_0 + fpa_conv_targ_0, fpa_conv_star_2 + fpa_conv_targ_2, rtol=222932.27, atol=8.6796875)
        np.testing.assert_allclose(fpa_conv_star_1, fpa_conv_star_3, rtol=222932.27, atol=8.6796875)
    else:
        np.testing.assert_array_almost_equal(fpa_conv_star_0 + fpa_conv_targ_0, fpa_conv_star_2 + fpa_conv_targ_2, 1)
        np.testing.assert_array_almost_equal(fpa_conv_star_1, fpa_conv_star_3, 1)


def test_piecewise():

    configure_eager()

    # TODO make this test more thorough
    ssp = config.load_json('./tests/config_piecewise.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    assert(dirname is not None)


def test_model():

    configure_eager()

    # TODO make this test more thorough
    ssp = config.load_json('./tests/config_pipeline.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    assert(dirname is not None)


def test_none():

    configure_eager()

    # TODO make this test more thorough
    ssp = config.load_json('./tests/config_none.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    assert(dirname is not None)


def test_crop():

    configure_eager()

    # TODO make this test more thorough
    ssp = config.load_json('./tests/config_piecewise.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    ssp['fpa']['crop'] = {
        "height_offset": 256,
        "width_offset": 0,
        "height": 512,
        "width": 1024
    }

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    assert(dirname is not None)

    with open(os.path.join(dirname, 'Debug', 'fpa_digital_0.pickle'), 'rb') as f:
        f = pickle.load(f)
        assert(f.shape == (512, 1024))

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        f = pickle.load(f)
        assert(f.shape == (512, 1024))

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        f = pickle.load(f)
        assert(f.shape == (512, 1024))


def test_segmentation_annotation():

    queue = MultithreadedTaskQueue()

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['save_segmentation'] = True
    ssp['sim']['star_annotation_threshold'] = 10
    ssp['fpa']['num_frames'] = 1
    ssp['geometry']['stars']['mv']['bins'] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    ssp['geometry']['stars']['mv']['density'] = [0,0,0,0,0,0.73333,0,1.6667,3.7333,11.494,18.172,33.236,36.531,57.311,93.314,149.13,250.63,380.99]

    np.random.seed(42)
    set_name = _gen_name('test_segmentation_annotation')
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        pe_data = pickle.load(f)
        seg_data = imread(dirname + '/Annotations/' + set_name + '.0000_star_segmentation.tiff')
        mask_data = pe_data[seg_data > 0]

    assert(not np.any(mask_data < ssp['fpa']['a2d']['gain']))

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        pe_data = pickle.load(f)
        seg_data = imread(dirname + '/Annotations/' + set_name + '.0000_object_segmentation.tiff')
        mask_data = pe_data[seg_data > 0]

    assert(not np.any(mask_data < ssp['fpa']['a2d']['gain']))

    ssp = config.load_json('./tests/config_piecewise.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    ssp['fpa']['crop'] = {
        "height_offset": 0,
        "width_offset": 0,
        "height": 512,
        "width": 1024
    }
    ssp['sim']['save_segmentation'] = True
    ssp['sim']['star_annotation_threshold'] = 9

    np.random.seed(42)
    set_name = _gen_name('test_segmentation_annotation2')
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=set_name)
    queue.waitUntilEmpty()

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        pe_data = pickle.load(f)
        seg_data = imread(dirname + '/Annotations/' + set_name + '.0000_star_segmentation.tiff')
        mask_data = pe_data[seg_data > 0]

    assert(not np.any(mask_data < ssp['fpa']['a2d']['gain']))

    with open(os.path.join(dirname, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        pe_data = pickle.load(f)
        seg_data = imread(dirname + '/Annotations/' + set_name + '.0000_object_segmentation.tiff')
        mask_data = pe_data[seg_data > 0]

    assert(not np.any(mask_data < ssp['fpa']['a2d']['gain']))


def test_mode_none():

    queue = MultithreadedTaskQueue()

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['mode'] = 'none'

    np.random.seed(42)
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_none'))
    queue.waitUntilEmpty()

    assert(dirname is not None)


def test_mode_misc():

    queue = MultithreadedTaskQueue()

    ssp = config.load_json('./tests/config_static.json')
    ssp['sim']['psf_sample_frequency'] = 'frame'
    ssp['fpa']['num_frames'] = 3
    ssp['augment'] = {
        'image': {
            'post': None
        }
    }

    np.random.seed(42)
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_misc'))
    queue.waitUntilEmpty()

    pad = 2 * ssp['sim']['padding']
    ssp['fpa']['psf'] = np.random.rand((pad + ssp['fpa']['height']) * ssp['sim']['spacial_osf'], (pad + ssp['fpa']['width']) * ssp['sim']['spacial_osf'])
    ssp['augment']['image']['post'] = 1
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_misc2'))
    queue.waitUntilEmpty()

    assert(dirname is not None)

    ssp['augment']['image']['post'] = lambda x: x + 1.0
    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_misc3'))
    queue.waitUntilEmpty()

    assert(dirname is not None)


def test_csv_catalog():

    queue = MultithreadedTaskQueue()

    ssp = config.load_json('./tests/config_static_sttr7_sgp4.json')
    ssp['fpa']['x_fov'] = 20.0
    ssp['fpa']['y_fov'] = 20.0
    ssp['fpa']['zeropoint'] = 17.0
    ssp['geometry']['stars'] = {
        'mode': 'csv',
        'path': './tests/hip_main.txt',
        'motion': { 'mode': 'none'}
    }

    dirname = gen_images(ssp, eager=True, output_dir='./.images', output_debug=True, queue=queue, set_name=_gen_name('test_csv_catalog'))
    queue.waitUntilEmpty()

    assert(dirname is not None)


def _gen_name(name):
    return '{}_{}'.format(datetime.now().isoformat().replace(':', '-'), name)
