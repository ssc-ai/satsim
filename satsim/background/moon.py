"""Lunar background models."""

import math
import numpy as np

from satsim.background.units import (
    nano_lamberts_to_surface_brightness,
    surface_brightness_to_pe,
)
from satsim.geometry.astrometric import (
    angle_between,
    angle_from_los,
    get_los,
    load_moon,
    load_sun,
)


DEFAULT_V_EXTINCTION = 0.172
MIN_MOON_SEPARATION_DEG = 0.25


def krisciunas_schaefer_moon_brightness_nl(
    phase_angle_deg,
    moon_sky_separation_deg,
    moon_zenith_deg,
    target_zenith_deg,
    extinction=DEFAULT_V_EXTINCTION,
):
    """Return the K&S scattered moonlight contribution in nanoLamberts."""
    if moon_zenith_deg >= 90.0 or target_zenith_deg >= 90.0:
        return 0.0

    separation = max(float(moon_sky_separation_deg), MIN_MOON_SEPARATION_DEG)
    separation_rad = math.radians(separation)
    rayleigh = 10.0 ** 5.36 * (1.06 + math.cos(separation_rad) ** 2)
    if separation < 10.0:
        mie = 6.2e7 / (separation * separation)
    else:
        mie = 10.0 ** (6.15 - separation / 40.0)
    scattering = rayleigh + mie

    phase = abs(float(phase_angle_deg))
    lunar_intensity = 10.0 ** (-0.4 * (3.84 + 0.026 * phase + 4.0e-9 * phase ** 4))
    moon_airmass = _krisciunas_schaefer_airmass(moon_zenith_deg)
    target_airmass = _krisciunas_schaefer_airmass(target_zenith_deg)

    moon_extinction = 10.0 ** (-0.4 * extinction * moon_airmass)
    target_scattering = 1.0 - 10.0 ** (-0.4 * extinction * target_airmass)
    return scattering * lunar_intensity * moon_extinction * target_scattering


def krisciunas_schaefer_moon_component(
    observer,
    ts_mid,
    ra_deg,
    dec_deg,
    target_el_deg,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    """Return the K&S moonlight PE contribution and diagnostic metadata."""
    moon = load_moon()
    sun = load_sun()
    _, _, _, moon_az, moon_el, _ = get_los(observer, moon, ts_mid)
    phase_angle_deg = angle_between(moon, observer, sun, ts_mid)
    moon_separation_deg = angle_from_los(observer, moon, ra_deg, dec_deg, ts_mid)
    moon_nl = krisciunas_schaefer_moon_brightness_nl(
        phase_angle_deg,
        moon_separation_deg,
        90.0 - moon_el,
        90.0 - target_el_deg,
    )
    moon_brightness = nano_lamberts_to_surface_brightness(moon_nl)
    moon_pe = (
        0.0
        if not np.isfinite(moon_brightness)
        else surface_brightness_to_pe(
            zeropoint,
            moon_brightness,
            pixel_area_arcsec2,
            exposure_s,
        )
    )
    return moon_pe, {
        'mode': 'krisciunas-schaefer',
        'brightness': None if not np.isfinite(moon_brightness) else moon_brightness,
        'nano_lamberts': moon_nl,
        'phase_angle': phase_angle_deg,
        'moon_sky_separation': moon_separation_deg,
        'moon_az': moon_az,
        'moon_el': moon_el,
        'target_el': target_el_deg,
        'extinction': DEFAULT_V_EXTINCTION,
    }


def _krisciunas_schaefer_airmass(zenith_deg):
    zenith = min(max(float(zenith_deg), 0.0), 89.9)
    return (1.0 - 0.96 * math.sin(math.radians(zenith)) ** 2) ** -0.5
