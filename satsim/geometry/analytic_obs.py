from __future__ import division, print_function, absolute_import

import math
import numpy as np


def generate(ssp, obs_os_pix, astrometrics, bg_level, dc_level, rn, en):
    """Generate analytical observations for a frame using precomputed RA/Dec.

    Args:
        ssp: `dict`, SatSim configuration parameters.
        obs_os_pix: `list`, list of detected observations in pixel space. Each
            entry must contain the keys ``ra`` and ``dec`` giving the
            mid-exposure position of the object.
        astrometrics: `dict`, frame astrometric parameters.
        bg_level: `float`, background noise level per pixel.
        dc_level: `float`, dark current level per pixel.
        rn: `float`, read noise standard deviation in photo-electrons.
        en: `float`, electronic noise standard deviation in photo-electrons.

    Notes:
        ``pixel_error`` is interpreted as the standard deviation of the
        positional error in pixel units. The error is applied isotropically in
        the focal plane so that the per-axis standard deviation is
        ``pixel_error / sqrt(2)``.

    Returns:
        `list` of observations in the JSON output format.
    """
    obs_cfg = ssp['fpa'].get('detection', {})
    pixel_error = obs_cfg.get('pixel_error', 0.0)
    snr_threshold = obs_cfg.get('snr_threshold', 0.0)
    false_alarm_rate = obs_cfg.get('false_alarm_rate', 0.0)
    max_false = int(obs_cfg.get('max_false', 10))

    height = ssp['fpa'].get('crop', {}).get('height', ssp['fpa']['height'])
    width = ssp['fpa'].get('crop', {}).get('width', ssp['fpa']['width'])

    y_ifov = astrometrics.get('y_ifov',
                              ssp['fpa']['y_fov'] / ssp['fpa']['height'])
    x_ifov = astrometrics.get('x_ifov',
                              ssp['fpa']['x_fov'] / ssp['fpa']['width'])

    s_osf = ssp['sim'].get('spacial_osf', 1)
    eod = 1.0
    if isinstance(ssp['fpa'].get('psf'), dict) and 'eod' in ssp['fpa']['psf']:
        eod = ssp['fpa']['psf']['eod']

    obs_list = []

    # convert radial pixel error to per-axis standard deviation
    axis_error = pixel_error / math.sqrt(2.0) if pixel_error else 0.0

    for ob in obs_os_pix:
        if 'ra_obs' not in ob or 'dec_obs' not in ob:
            continue

        rr = np.asarray(ob['rr'], dtype=np.int32)
        cc = np.asarray(ob['cc'], dtype=np.int32)
        pp = np.asarray(ob['pp'], dtype=float)

        # bin oversampled pixels into real pixel space and ignore any pixels
        # that fall outside the cropped image
        r_real = rr // s_osf
        c_real = cc // s_osf
        bins = {}
        for r_pix, c_pix, val in zip(r_real, c_real, pp):
            if (0 <= r_pix < height and
                    0 <= c_pix < width):
                bins[(r_pix, c_pix)] = bins.get((r_pix, c_pix), 0.0) + val

        if not bins:
            continue

        peak_signal = max(bins.values()) * eod
        snr = float(peak_signal / math.sqrt(peak_signal + bg_level + dc_level + rn * rn + en * en))
        if snr < snr_threshold:
            continue

        ra_obs = ob['ra_obs']
        dec_obs = ob['dec_obs']
        ra_m = ra_obs + np.random.normal(scale=axis_error * x_ifov) / math.cos(math.radians(dec_obs))
        dec_m = dec_obs + np.random.normal(scale=axis_error * y_ifov)

        entry = {
            'obTime': astrometrics['time'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'ra': float(ra_m),
            'declination': float(dec_m),
            'snrEst': float(snr),
            'expDuration': float(ssp['fpa']['time']['exposure']),
            'uct': False,
            'createdBy': 'satsim',
            'type': 'OPTICAL'
        }
        if 'x' in astrometrics:
            entry.update({
                'senx': float(astrometrics['x']),
                'seny': float(astrometrics['y']),
                'senz': float(astrometrics['z']),
                'senvelx': float(astrometrics['vx']),
                'senvely': float(astrometrics['vy']),
                'senvelz': float(astrometrics['vz']),
            })
        else:
            entry.update({
                'senlat': float(astrometrics.get('lat', 0)),
                'senlon': float(astrometrics.get('lon', 0)),
                'senalt': float(astrometrics.get('alt', 0)),
            })
        obs_list.append(entry)

        if 'object_name' in ob and ob['object_name'] is not None and ob['object_name'] != '':
            obs_list[-1]['idOnOrbit'] = ob['object_name']

        if 'object_id' in ob and ob['object_id'] is not None and ob['object_id'] != '':
            obs_list[-1]['satNo'] = ob['object_id']

    center_ra = astrometrics.get('ra', 0.0)
    center_dec = astrometrics.get('dec', 0.0)
    count = 0
    while count < max_false and np.random.random() < false_alarm_rate:
        c_pix = np.random.uniform(-width / 2.0, width / 2.0)
        r_pix = np.random.uniform(-height / 2.0, height / 2.0)
        ra_m = center_ra + c_pix * x_ifov / math.cos(math.radians(center_dec))
        dec_m = center_dec + r_pix * y_ifov
        ra_m += np.random.normal(scale=axis_error * x_ifov) / math.cos(math.radians(dec_m))
        dec_m += np.random.normal(scale=axis_error * y_ifov)
        entry = {
            'obTime': astrometrics['time'].strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            'ra': float(ra_m),
            'declination': float(dec_m),
            'snrEst': 0.0,
            'expDuration': float(ssp['fpa']['time']['exposure']),
            'uct': True,
            'createdBy': 'satsim',
            'type': 'OPTICAL'
        }
        if 'x' in astrometrics:
            entry.update({
                'senx': float(astrometrics['x']),
                'seny': float(astrometrics['y']),
                'senz': float(astrometrics['z']),
                'senvelx': float(astrometrics['vx']),
                'senvely': float(astrometrics['vy']),
                'senvelz': float(astrometrics['vz']),
            })
        else:
            entry.update({
                'senlat': float(astrometrics.get('lat', 0)),
                'senlon': float(astrometrics.get('lon', 0)),
                'senalt': float(astrometrics.get('alt', 0)),
            })
        obs_list.append(entry)
        count += 1

    return obs_list
