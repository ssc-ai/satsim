"""Twilight background models."""

import numpy as np

from satsim.background.units import surface_brightness_residual_pe
from satsim.geometry.astrometric import get_los, load_sun


PATAT_V_COEFFICIENTS = (11.84, 1.518, -0.057)
PATAT_BRIGHT_END_SURFACE_BRIGHTNESS = PATAT_V_COEFFICIENTS[0]


def patat_twilight_surface_brightness(sun_zenith_deg):
    """Return Patat V-band zenith twilight brightness in mag/arcsec^2."""
    sun_zenith = float(sun_zenith_deg)
    if sun_zenith <= 90.0 or sun_zenith >= 105.0:
        return np.inf

    zeta = max(95.0, sun_zenith)
    delta = zeta - 95.0
    a0, a1, a2 = PATAT_V_COEFFICIENTS
    return a0 + a1 * delta + a2 * delta * delta


def patat_twilight_component(
    ssp,
    observer,
    ts_mid,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    """Return the Patat twilight PE residual and diagnostic metadata."""
    sun = load_sun()
    _, _, _, sun_az, sun_el, _ = get_los(observer, sun, ts_mid)
    sun_zenith = 90.0 - sun_el
    twilight_brightness = patat_twilight_surface_brightness(sun_zenith)
    if not np.isfinite(twilight_brightness):
        twilight_pe = 0.0
    else:
        twilight_pe = surface_brightness_residual_pe(
            zeropoint,
            twilight_brightness,
            ssp['background']['galactic'],
            pixel_area_arcsec2,
            exposure_s,
        )

    return twilight_pe, {
        'mode': 'patat',
        'brightness': None if not np.isfinite(twilight_brightness) else twilight_brightness,
        'sun_az': sun_az,
        'sun_el': sun_el,
        'sun_zenith': sun_zenith,
        'residual_over_galactic': True,
    }
