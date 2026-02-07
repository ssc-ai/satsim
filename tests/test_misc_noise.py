import numpy as np

from satsim.satsim import _eval_misc_noise


def test_eval_misc_noise_seed_offset():
    ssp = {
        'fpa': {
            'height': 3,
            'width': 4,
        }
    }
    misc_param = {
        '$sample': 'random.uniform',
        'low': 0.0,
        'high': 1.0,
        'size': [
            {'$ref': 'fpa.height'},
            {'$ref': 'fpa.width'},
        ],
        'seed': 10,
    }

    val0 = _eval_misc_noise(ssp, misc_param, frame_num=0, input_dir='.')
    val1 = _eval_misc_noise(ssp, misc_param, frame_num=1, input_dir='.')
    val0_repeat = _eval_misc_noise(ssp, misc_param, frame_num=0, input_dir='.')

    assert val0.shape == (3, 4)
    assert np.allclose(val0, val0_repeat)
    assert not np.allclose(val0, val1)
