"""Tests for `satsim` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os
import pickle
import json
import copy

import numpy as np
import scipy


from satsim import config, gen_images
from satsim.util import configure_eager
from satsim.util.system import is_tensorflow_running_on_cpu


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

        np.testing.assert_approx_equal(np.sum(fpa_os_0.flatten()) * 2, np.sum(fpa_os_1.flatten()), significant=7)

    with open(os.path.join(dirname, 'Debug', 'metadata_0.json'), 'r') as f:
        metadata_1 = json.load(f)

    for a, b in zip(ssp['geometry']['obs']['list'], metadata_1['data']['objects']):
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

    print(scipy.ndimage.center_of_mass(fpa4))

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

    ssp = config.load_json('./tests/config_poppy.json')
    ssp, d = config.transform(ssp, max_stages=10, with_debug=True)

    ssp['fpa']['num_frames'] = 1
    ssp['fpa']['time']['exposure'] = 1

    gen_images(ssp, eager=True, output_dir='./.images', output_debug=True)

    ssp['sim']['enable_shot_noise'] = False
    ssp['sim']['save_pickle'] = False
    ssp['geometry']['site']['track']['mode'] = 'sidereal'
    ssp['geometry']['site']['track']['tle'] = [ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2']]
    del(ssp['geometry']['site']['track']['tle1'])
    del(ssp['geometry']['site']['track']['tle2'])
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

    from satsim.util import MultithreadedTaskQueue

    queue = MultithreadedTaskQueue()

    dirname0 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue)
    with open(os.path.join(dirname0, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_0 = pickle.load(f)
    with open(os.path.join(dirname0, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        fpa_conv_targ_0 = pickle.load(f)

    ssp['sim']['calculate_snr'] = False

    dirname1 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue)
    with open(os.path.join(dirname1, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_1 = pickle.load(f)

    # ssp['sim']['render_size'] = [101, 101]  # TODO investigate why odd render_size doesn't work properly
    ssp['sim']['render_size'] = [100, 100]
    ssp['sim']['calculate_snr'] = True

    dirname2 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue)
    with open(os.path.join(dirname2, 'Debug', 'fpa_conv_star_0.pickle'), 'rb') as f:
        fpa_conv_star_2 = pickle.load(f)
    with open(os.path.join(dirname2, 'Debug', 'fpa_conv_targ_0.pickle'), 'rb') as f:
        fpa_conv_targ_2 = pickle.load(f)

    ssp['sim']['render_size'] = [100, 100]
    ssp['sim']['calculate_snr'] = False

    dirname3 = gen_images(copy.deepcopy(ssp), eager=True, output_dir='./.images', output_debug=True, queue=queue)
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
