import datetime
import numpy as np
from satsim.geometry import analytic_obs


def test_pixel_error_distribution():
    ssp = {
        'fpa': {
            'width': 100,
            'height': 100,
            'x_fov': 1.0,
            'y_fov': 1.0,
            'time': {'exposure': 1.0},
            'psf': {'eod': 1.0},
            'detection': {
                'pixel_error': 1.0,
                'snr_threshold': 0.0,
                'false_alarm_rate': 0.0,
                'max_false': 0,
            },
        },
        'sim': {'spacial_osf': 1},
    }

    astrometrics = {
        'time': datetime.datetime.utcnow(),
        'x_ifov': ssp['fpa']['x_fov'] / ssp['fpa']['width'],
        'y_ifov': ssp['fpa']['y_fov'] / ssp['fpa']['height'],
    }

    obs_os_pix = [{
        'rr': [0],
        'cc': [0],
        'pp': [100],
        'ra_obs': 0.0,
        'dec_obs': 0.0,
    }]

    deltas = []
    for _ in range(2000):
        obs = analytic_obs.generate(ssp, obs_os_pix, astrometrics, 0.0, 0.0, 0.0, 0.0)
        delta_ra_pix = (obs[0]['ra']) * np.cos(0.0) / astrometrics['x_ifov']
        delta_dec_pix = obs[0]['declination'] / astrometrics['y_ifov']
        deltas.append([delta_ra_pix, delta_dec_pix])

    arr = np.asarray(deltas)
    axis_std = np.std(arr, axis=0)
    assert np.allclose(axis_std, np.ones(2) / np.sqrt(2), rtol=0.1, atol=0.1)


def test_object_out_of_fov():
    ssp = {
        'fpa': {
            'width': 10,
            'height': 10,
            'x_fov': 1.0,
            'y_fov': 1.0,
            'time': {'exposure': 1.0},
            'psf': {'eod': 1.0},
            'detection': {
                'pixel_error': 0.0,
                'snr_threshold': 0.0,
                'false_alarm_rate': 0.0,
                'max_false': 0,
            },
        },
        'sim': {'spacial_osf': 1},
    }

    astrometrics = {
        'time': datetime.datetime.utcnow(),
        'x_ifov': ssp['fpa']['x_fov'] / ssp['fpa']['width'],
        'y_ifov': ssp['fpa']['y_fov'] / ssp['fpa']['height'],
    }

    obs_os_pix = [{
        'rr': [-1],
        'cc': [-1],
        'pp': [100],
        'ra_obs': 0.0,
        'dec_obs': 0.0,
    }]

    obs = analytic_obs.generate(ssp, obs_os_pix, astrometrics, 0.0, 0.0, 0.0, 0.0)
    assert obs == []


def test_object_in_fov_with_padding():
    ssp = {
        'fpa': {
            'width': 10,
            'height': 10,
            'x_fov': 1.0,
            'y_fov': 1.0,
            'time': {'exposure': 1.0},
            'psf': {'eod': 1.0},
            'detection': {
                'pixel_error': 0.0,
                'snr_threshold': 0.0,
                'false_alarm_rate': 0.0,
                'max_false': 0,
            },
        },
        'sim': {'spacial_osf': 1, 'padding': 2},
    }

    astrometrics = {
        'time': datetime.datetime.utcnow(),
        'x_ifov': ssp['fpa']['x_fov'] / ssp['fpa']['width'],
        'y_ifov': ssp['fpa']['y_fov'] / ssp['fpa']['height'],
    }

    obs_os_pix = [{
        'rr': [1],
        'cc': [1],
        'pp': [100],
        'ra_obs': 0.0,
        'dec_obs': 0.0,
    }]

    obs = analytic_obs.generate(ssp, obs_os_pix, astrometrics, 0.0, 0.0, 0.0, 0.0)
    assert len(obs) == 1
