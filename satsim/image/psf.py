from __future__ import division, print_function, absolute_import

from astropy import units as u
import math
import tensorflow as tf
from scipy.special import erfinv
import numpy as np


def eod_to_sigma(eod, osf):
    """Calculates the sigma of a 2D Gaussian required to get a maximum energy
    on a detector (EOD) or single pixel. EOD is specified as a fraction
    (0 < EOD < 1). The 2D Gaussian is assumed to be centered on a real pixel.

    Args:
        eod: `float`, desired maximum energy on detector in real pixel space.
        osf: `int`, desired PSF oversample factor.

    Returns:
        A `float`, the estimated Gaussian `sigma` for `eod` in real pixel space.
    """
    return osf / (2.0 * math.sqrt(2.0) * erfinv(math.sqrt(eod)))


def gen_gaussian(height, width, sigma, dtype=tf.float32):
    """Generate a 2D Gaussian point spread function (PSF).

    Examples::

        osf = 11
        sigma = eod_to_sigma(0.8, osf)
        psf = gen_gaussian(512*osf, 512*osf, sigma)

    Args:
        height: `int`, height of the output PSF.
        width: `int`, width of the output PSF.
        sigma: `float`, the standard deviation of the Gaussian.
        dtype: `tf.dtype`, Defaults to `tf.float32`.

    Returns:
        A `Tensor`, the 2D Gaussian PSF with the shape `[height,width]`
    """
    escale = 1.0 / ( 2.0 * sigma * sigma)
    gscale = escale / math.pi

    r = tf.cast(height, tf.float32) / 2.0 - 0.5
    c = tf.cast(width, tf.float32) / 2.0 - 0.5

    rr = tf.range(-r, r + 1, delta=1, dtype=dtype)
    cc = tf.range(-c, c + 1, delta=1, dtype=dtype)

    ccc = tf.reshape(tf.tile(cc, [height]), [height, width])
    rrr = tf.transpose(tf.reshape(tf.tile(rr, [width]), [width, height]))

    return tf.exp(-(ccc * ccc + rrr * rrr) * escale) * gscale


def gen_from_poppy(optical_system, wavelengths=[600e-9], weights=[1]):
    """Generate a point spread function (PSF) from a POPPY optical system.

    Args:
        optical_system: `object`, POPPY OpticalSystem.
        wavelengths: `array`, an array of wavelengths to simulate.
        weights: `array`, an array of same length of `wavelength`. Each value is
        used to scale the PSF for corresponding wavelength.
    Returns:
        An `array`, the PSF as a two dimensional array.
    """

    with np.errstate(divide='ignore'):
        psf = optical_system.calc_psf(normalize='last', source={
            'wavelengths': wavelengths,
            'weights': weights,
            'oversample': 2,
        })

        return psf[0].data


def gen_from_poppy_configuration(height, width, y_ifov, x_ifov, s_osf, config):
    """Generate a point spread function (PSF) from a POPPY configuration.
    Configuration example::

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
            "wavelengths": [300e-9, 600e-9, 900e-9],
            "weights": [0.3, 0.4, 0.3]
        }

    Args:
        height: `int`, height of the output PSF.
        width: `int`, width of the output PSF.
        y_ifov: `float`, field of view of a pixel in y.
        x_ifov: `float`, field of view of a pixel in x.
        s_osf: `int`, oversample factor.
        param: `dict`, a dictionary that describes the optical system.
            See example.

    Returns:
        An `array`, the PSF as a two dimensional array.
    """
    import poppy
    import importlib
    module = importlib.import_module('poppy')

    ifov = (x_ifov * u.deg / u.pixel).to(u.arcsec / u.pixel)
    optical_system_config = config['optical_system']

    osys = poppy.OpticalSystem(oversample=s_osf, npix=None)

    for c in optical_system_config:
        if c['type'] == 'CompoundAnalyticOptic':
            element_list = []
            for e in c['opticslist']:
                element_list.append(getattr(module, e['type'])(**e['kwargs']))
            osys.add_pupil(poppy.CompoundAnalyticOptic(opticslist=element_list))
        else:
            osys.add_pupil(getattr(module, c['type'])(**c['kwargs']))

    # fix for misspelled key
    if 'turbulant_atmosphere' in config and 'turbulent_atmosphere' not in config:
        config['turbulent_atmosphere'] = config['turbulant_atmosphere']

    if 'turbulent_atmosphere' in config:
        turbulent_atmosphere = config['turbulent_atmosphere']
        Cn2 = turbulent_atmosphere['Cn2'] * u.m**(-2 / 3)
        L = turbulent_atmosphere['propagation_distance'] * u.m
        nz = turbulent_atmosphere['zones']
        dz = L / nz
        for i in range(nz + 1):
            if i == 0 or i == nz:
                phase_screen = poppy.KolmogorovWFE(Cn2=Cn2, dz=dz / 2)
            else:
                phase_screen = poppy.KolmogorovWFE(Cn2=Cn2, dz=dz)
            osys.add_pupil(phase_screen)

    if 'size' in config:
        # generate a cropped PSF
        osys.add_detector(pixelscale=ifov.value, fov_pixels=config['size'])

        m = max(config['size'])
        npix = _calc_npix(osys, ifov * m * u.pixel, config['wavelengths'])
        osys.npix = npix

        h_pad = int((height - config['size'][0]) / 2 * s_osf)
        w_pad = int((width - config['size'][1]) / 2 * s_osf)

        psf = gen_from_poppy(osys, config['wavelengths'], config['weights'])
        psf_pad = np.pad(psf, ((h_pad, h_pad),(w_pad, w_pad)))

        return psf_pad
    else:
        # generate a full frame PSF
        osys.add_detector(pixelscale=ifov.value, fov_pixels=[height, width])

        m = max([height, width])
        npix = _calc_npix(osys, ifov * m * u.pixel, config['wavelengths'])
        osys.npix = npix

        return gen_from_poppy(osys, config['wavelengths'], config['weights'])


def _calc_npix(optical_system, fov, wavelengths):
    det_fov = fov.to(u.radian).value
    det_fov = det_fov * 1.1

    optimal_npix = []
    diam = optical_system.planes[0].pupil_diam
    for wl in wavelengths:
        optimal_npix.append(int( ((det_fov / 2.0) * (diam / wl)).value + 1))

    npix = max(optimal_npix)

    # atmosphere requires even number of pixels for symmetry
    if npix % 2 != 0:
        npix += 1

    return npix
