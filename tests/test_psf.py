"""Tests for `satsim.image.psf` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.image.psf import eod_to_sigma, gen_gaussian, gen_from_poppy_configuration
from satsim.image.fpa import downsample
from satsim.util import configure_eager

configure_eager()


def test_eod_to_sigma():

    np.testing.assert_almost_equal(eod_to_sigma(0.1, 1), 1.2275330469951529)
    np.testing.assert_almost_equal(eod_to_sigma(0.3, 11), 7.317492355442362)
    np.testing.assert_almost_equal(eod_to_sigma(0.5, 21), 9.982925772640492)


def test_gen_gaussian_eod():

    h = 101
    w = 101
    eod = 0.5
    osf = 5

    sigma = eod_to_sigma(eod, osf)
    psf = gen_gaussian(h * osf, w * osf, sigma)

    psf = downsample(psf, osf)

    np.testing.assert_almost_equal(np.max(psf.numpy()), eod, decimal=2)
    np.testing.assert_almost_equal(np.sum(psf.numpy()), 1.0, decimal=5)

    # test non-square
    h = 51
    w = 101
    eod = 0.5
    osf = 51

    sigma = eod_to_sigma(eod, osf)
    psf = gen_gaussian(h * osf, w * osf, sigma)

    psf = downsample(psf, osf)

    np.testing.assert_array_equal(psf.get_shape(), [h,w])
    np.testing.assert_almost_equal(np.max(psf.numpy()), eod, decimal=4)
    np.testing.assert_almost_equal(np.sum(psf.numpy()), 1.0, decimal=5)


def test_gen_poppy():

    config = {
        "mode": "poppy",
        "optical_system": [
            {
                "type": "CompoundAnalyticOptic",
                "opticslist": [
                    {
                        "type": "CircularAperture",
                        "kwargs": {
                            "radius": 0.200
                        }
                    },
                    {
                        "type": "SecondaryObscuration",
                        "kwargs": {
                            "secondary_radius": 0.110,
                            "n_supports": 4,
                            "support_width": 0.010
                        }
                    }
                ]
            },
            {
                "type": "ZernikeWFE",
                "kwargs": {
                    "radius": 0.200,
                    "coefficients": [0, 0, 0, 100e-9]
                }
            }
        ],
        "wavelengths": [500e-9, 600e-9, 700e-9],
        "weights": [0.3, 0.4, 0.3]
    }

    h = 200
    w = 100
    osf = 3
    psf = gen_from_poppy_configuration(h, w, 0.3 / h, 0.3 / w, osf, config)

    assert(psf.shape[0] == h * osf)
    assert(psf.shape[1] == w * osf)

    config['size'] = [50, 50]
    psf2 = gen_from_poppy_configuration(h, w, 0.3 / h, 0.3 / w, osf, config)

    np.testing.assert_array_almost_equal(psf, psf2, decimal=2)


def test_gen_poppy_with_atmosphere():

    config = {
        "mode": "poppy",
        "optical_system": [
            {
                "type": "CompoundAnalyticOptic",
                "opticslist": [
                    {
                        "type": "CircularAperture",
                        "kwargs": {
                            "radius": 0.200
                        }
                    },
                    {
                        "type": "SecondaryObscuration",
                        "kwargs": {
                            "secondary_radius": 0.110,
                            "n_supports": 4,
                            "support_width": 0.010
                        }
                    }
                ]
            }
        ],
        "turbulent_atmosphere": {
            "Cn2": 1.7e-15,
            "propagation_distance": 3000,
            "zones": 5
        },
        "wavelengths": [500e-9, 600e-9, 700e-9],
        "weights": [0.3, 0.4, 0.3]
    }

    h = 200
    w = 100
    osf = 3
    psf = gen_from_poppy_configuration(h, w, 0.3 / h, 0.3 / w, osf, config)
    assert(psf.shape[0] == h * osf)
    assert(psf.shape[1] == w * osf)
