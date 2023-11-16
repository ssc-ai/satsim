"""Tests for `satsim.config`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os
import numpy as np

from satsim import config


def test_json():

    config.load_json('./tests/config_static.json')


def test_yaml():

    config.load_yaml('./tests/config.yml')


def test_transform(config_file='./tests/config_dynamic.json'):

    c = config.load_json(config_file)
    t = config.transform(c, dirname='./tests')

    dc = t['fpa']['dark_current']
    g = t['fpa']['gain']
    b = t['fpa']['bias']

    assert(dc.shape == (t['fpa']['height'], t['fpa']['width']))

    config.save_json('./tests/transform.json', t, save_pickle=True)

    c2 = config.load_json('./tests/transform.json')
    t2 = config.transform(c2, dirname='./tests')

    assert(os.path.isfile('./tests/transform.json'))
    assert(os.path.isfile('./tests/00000.pickle'))
    assert(os.path.isfile('./tests/00001.pickle'))
    assert(os.path.isfile('./tests/00002.pickle'))

    # check maps are equal in value
    np.testing.assert_array_equal(dc, t2['fpa']['dark_current'])
    np.testing.assert_array_equal(g, t2['fpa']['gain'])
    np.testing.assert_array_equal(b, t2['fpa']['bias'])

    # delete maps and check equality of all others
    del(t['fpa']['dark_current'])
    del(t2['fpa']['dark_current'])
    del(t['fpa']['gain'])
    del(t2['fpa']['gain'])
    del(t['fpa']['bias'])
    del(t2['fpa']['bias'])

    assert(t == t2)

    os.remove('./tests/transform.json')
    os.remove('./tests/00000.pickle')
    os.remove('./tests/00001.pickle')
    os.remove('./tests/00002.pickle')

    c = config.load_json(config_file)
    t = config.transform(c, dirname='./tests')

    config.save_json('./tests/transform.json', t, save_pickle=False)

    c2 = config.load_json('./tests/transform.json')

    c2['fpa']['dark_current'] == 'n/a'
    c2['fpa']['gain'] == 'n/a'
    c2['fpa']['bias'] == 'n/a'

    assert(os.path.isfile('./tests/00000.pickle') is False)
    assert(os.path.isfile('./tests/00001.pickle') is False)
    assert(os.path.isfile('./tests/00002.pickle') is False)

    os.remove('./tests/transform.json')


def test_transform_legacy():

    test_transform('./tests/config_dynamic_legacy.json')


def test_generator():

    tconfig = config.parse_generator({
        "module": "satsim.generator.obs.geometry",
        "function": "cone",
        "kwargs": {
            "n": 5,
            "t": 0,
            "direction": [45, 15],
            "velocity": [1.0, 3.0],
            "origin": [0.5, 0.5],
            "mv": [15.0, 17.0]
        }
    })

    assert('list' in tconfig)

    output = tconfig['list']

    assert(len(output) == 7)

    for i in range(5):
        assert(output[i]['mode'] == 'line')
        assert('origin' in output[i])
        assert('velocity' in output[i])
        assert('mv' in output[i])


def test_dynamic_json():

    c = config.load_json('./tests/config_dynamic.json')
    assert(config._has_rkey_deep(c, '$sample') is True)

    t = config.transform(c, max_stages=2, with_debug=False)
    assert(config._has_rkey_deep(t, '$sample') is False)


def test_generator_json(config_file='./tests/config_generator.json'):

    c = config.load_json(config_file)
    assert(config._has_rkey_deep(c, '$generator') is True)

    t, td = config.transform(c, max_stages=10, with_debug=True)

    assert(len(td) == 7)
    assert(config._has_rkey_deep(t, '$generator') is False)
    assert(config._has_rkey_deep(t, '$sample') is False)

    config.save_debug(td, './')

    assert(os.path.isfile('./config_pass_1.json'))
    assert(os.path.isfile('./config_pass_2.json'))
    assert(os.path.isfile('./config_pass_3.json'))
    assert(os.path.isfile('./config_pass_4.json'))
    assert(os.path.isfile('./config_pass_5.json'))
    assert(os.path.isfile('./config_pass_6.json'))
    assert(os.path.isfile('./config_pass_7.json'))

    os.remove('./config_pass_1.json')
    os.remove('./config_pass_2.json')
    os.remove('./config_pass_3.json')
    os.remove('./config_pass_4.json')
    os.remove('./config_pass_5.json')
    os.remove('./config_pass_6.json')
    os.remove('./config_pass_7.json')


def test_generator_json_legacy():

    test_generator_json('./tests/config_generator_legacy.json')


def test_dict_merge():

    a = {
        'a': 1,
        'b': {
            'b1': 2,
            'b2': 3,
        },
    }
    b = {
        'a': 1,
        'b': {
            'b1': 4,
        },
        'c': 'new'
    }

    config.dict_merge(a, b)

    m = a

    assert(m['a'] == 1)
    assert(m['b']['b2'] == 3)
    assert(m['b']['b1'] == 4)
    assert(m['c'] == 'new')


def test_import():

    c = config.load_json('./tests/config_import.json')
    assert(config._has_rkey_deep(c, '$import') is True)

    t = config.transform(c, max_stages=2, with_debug=False)
    assert(config._has_rkey_deep(t, '$import') is False)

    assert(t['fpa']['num_frames'] == 1)

    p = config.load_json('./tests/config_partial.json')

    assert(p['num_frames'] == 16)

    del t['fpa']['num_frames']
    del p['num_frames']

    assert(t['fpa'] == p)


def test_function_pipeline():

    c = config.load_json('./tests/config_pipeline.json')
    assert(config._has_rkey_deep(c, '$pipeline') is True)

    c = config.transform(c, max_stages=2, with_debug=False)
    assert(config._has_rkey_deep(c, '$pipeline') is True)

    c = config._transform(c, eval_python=True)
    assert(config._has_rkey_deep(c, '$pipeline') is False)


def test_function():

    c = config.load_json('./tests/config_function.json')
    assert(config._has_rkey_deep(c, '$function') is True)

    t, td = config.transform(c, max_stages=10, with_debug=True)

    assert(len(td) == 8)
    assert(config.has_key_deep(t, '$function') is False)
    assert(config._has_rkey_deep(t, '$sample') is False)


def test_has_key_deep():

    assert(config.has_key_deep(0, 'nothing') is False)

    assert(config.has_key_deep({ 'test': { 'test2': 0}}, 'test2') is True)

    assert(config.has_key_deep({ 'test': { 'test2': 0}}, 'test3') is False)

    assert(config.has_key_deep({ 'test': [{ 'test2': 0}]}, 'test2') is True)

    assert(config.has_key_deep({ 'test': [{ 'test2': 0}]}, 'test3') is False)

    assert(config.has_key_deep({ 'test': [{ 'test2': [{ 'test3': 0}]}]}, 'test3') is True)


def test_cache():

    param = {
        '$cache': './tests/',
        'param': {
            'test': 'test'
        }
    }

    cparam, from_cache = config.parse_cache(param)

    assert(from_cache is False)

    config.save_cache(param, 'cached_value')

    cparam, from_cache = config.parse_cache(param)

    assert(from_cache is True)
    assert(cparam == 'cached_value')

    full_param = {
        'version': 1,
        'sim': {
        },
        'test': param
    }

    assert(full_param['test']['param']['test'] == 'test')

    tform = config.transform(full_param)

    assert(tform['test'] == 'cached_value')

    import shutil
    shutil.rmtree('./tests/.satsim_cache/')
