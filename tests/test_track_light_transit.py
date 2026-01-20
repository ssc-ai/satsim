"""Tests for tracker light transit toggle."""

from satsim import config, time
from satsim.geometry.sgp4 import create_sgp4
from satsim.satsim import _calculate_star_position_and_motion


def _compute_boresight(enable_light_transit):
    """Return boresight RA/Dec for a simple site/track config."""
    ssp = config.load_json('./tests/config_site_tle_simple.json')
    ssp['geometry']['site']['track']['light_transit_correction'] = enable_light_transit

    # Ensure defaults used by the tracker are present
    ssp['sim'].setdefault('enable_deflection', False)
    ssp['sim'].setdefault('enable_light_transit', True)
    ssp['sim'].setdefault('enable_stellar_aberration', True)
    ssp['fpa'].setdefault('flip_up_down', False)
    ssp['fpa'].setdefault('flip_left_right', False)

    s_osf = ssp['sim']['spacial_osf']
    padding = ssp['sim']['padding']
    h_pad_os = 2 * padding * s_osf
    w_pad_os = 2 * padding * s_osf
    h_fpa_os = ssp['fpa']['height'] * s_osf
    w_fpa_os = ssp['fpa']['width'] * s_osf
    h_fpa_pad_os = h_fpa_os + h_pad_os
    w_fpa_pad_os = w_fpa_os + w_pad_os

    y_ifov = ssp['fpa']['y_fov'] / ssp['fpa']['height']
    x_ifov = ssp['fpa']['x_fov'] / ssp['fpa']['width']
    y_ifov_os = y_ifov / s_osf
    x_ifov_os = x_ifov / s_osf
    y_fov = h_fpa_os * y_ifov_os
    x_fov = w_fpa_os * x_ifov_os
    y_fov_pad = h_fpa_pad_os * y_ifov_os
    x_fov_pad = w_fpa_pad_os * x_ifov_os

    ts_collect_start = time.utc_from_list(ssp['geometry']['time'])
    t_exposure = ssp['fpa']['time']['exposure']
    t_frame = ssp['fpa']['time']['gap'] + t_exposure
    ts_collect_end = time.utc_from_list(ssp['geometry']['time'], t_frame * ssp['fpa']['num_frames'])
    ts_start = ts_collect_start
    ts_end = time.utc_from_list(ssp['geometry']['time'], t_exposure)

    observer = create_sgp4(ssp['geometry']['site']['tle1'], ssp['geometry']['site']['tle2'])
    track = create_sgp4(ssp['geometry']['site']['track']['tle1'], ssp['geometry']['site']['track']['tle2'])

    astrometrics = {}
    _calculate_star_position_and_motion(
        ssp,
        astrometrics,
        ts_collect_start,
        ts_collect_end,
        ts_start,
        ts_end,
        t_exposure,
        h_fpa_pad_os,
        w_fpa_pad_os,
        y_fov_pad,
        x_fov_pad,
        y_fov,
        x_fov,
        y_ifov,
        x_ifov,
        observer,
        track,
        ssp['geometry']['site']['gimbal']['rotation'],
        ssp['geometry']['site']['track']['mode'],
        [ssp['geometry']['site']['track'].get('az', 0)] * 2,
        [ssp['geometry']['site']['track'].get('el', 0)] * 2,
        enable_light_transit,
    )

    return astrometrics['ra'], astrometrics['dec']


def test_track_pointing_changes_with_light_transit_toggle():
    ra_on, dec_on = _compute_boresight(True)
    ra_off, dec_off = _compute_boresight(False)

    ra_diff = abs(ra_on - ra_off)
    dec_diff = abs(dec_on - dec_off)

    # Light transit correction should shift pointing slightly
    assert ra_diff > 1e-7 or dec_diff > 1e-7
