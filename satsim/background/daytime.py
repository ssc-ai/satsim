"""Daytime background models."""

import math
import numpy as np

from satsim.background.hosek_wilkie import (
    default_ground_albedo,
    default_turbidity,
    hosek_wilkie_luminance,
)
from satsim.background.twilight import PATAT_BRIGHT_END_SURFACE_BRIGHTNESS
from satsim.background.units import (
    luminance_to_surface_brightness,
    surface_brightness_to_luminance,
    surface_brightness_residual_pe,
)
from satsim.geometry.astrometric import angle_from_los, get_los, load_sun


CIE_CLEAR_SKY_LOW_TURBIDITY = (-1.0, -0.32, 10.0, -3.0, 0.45)
DEFAULT_DAYTIME_ZENITH_LUMINANCE_CD_M2 = 4000.0
SOLAR_BLEND_START_EL_DEG = 0.0
SOLAR_BLEND_END_EL_DEG = 6.0


def perez_daytime_relative_luminance(
    target_zenith_deg,
    sun_zenith_deg,
    sun_sky_separation_deg,
):
    """Return Perez/CIE relative luminance for a target sky direction."""
    if sun_zenith_deg >= 90.0 or target_zenith_deg >= 90.0:
        return 0.0

    target_zenith = min(max(float(target_zenith_deg), 0.0), 89.9)
    sun_zenith = min(max(float(sun_zenith_deg), 0.0), 89.9)
    separation = min(max(float(sun_sky_separation_deg), 0.0), 180.0)
    target = _perez_luminance_distribution(target_zenith, separation)
    zenith = _perez_luminance_distribution(0.0, sun_zenith)
    if zenith <= 0.0:
        return 0.0
    return max(0.0, target / zenith)


def perez_daytime_surface_brightness(
    target_zenith_deg,
    sun_zenith_deg,
    sun_sky_separation_deg,
):
    """Return fixed-normalization Perez daytime brightness in mag/arcsec^2."""
    relative_luminance = perez_daytime_relative_luminance(
        target_zenith_deg,
        sun_zenith_deg,
        sun_sky_separation_deg,
    )
    if relative_luminance <= 0.0:
        return np.inf

    zenith_brightness = luminance_to_surface_brightness(
        DEFAULT_DAYTIME_ZENITH_LUMINANCE_CD_M2
    )
    return zenith_brightness - 2.5 * math.log10(relative_luminance)


def perez_daytime_component(
    ssp,
    observer,
    ts_mid,
    ra_deg,
    dec_deg,
    target_el_deg,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    """Return the Perez daytime PE residual and diagnostic metadata."""
    sun = load_sun()
    _, _, _, sun_az, sun_el, _ = get_los(observer, sun, ts_mid)
    sun_zenith = 90.0 - sun_el
    target_zenith = 90.0 - target_el_deg
    sun_separation = angle_from_los(observer, sun, ra_deg, dec_deg, ts_mid)
    daytime_brightness = perez_daytime_surface_brightness(
        target_zenith,
        sun_zenith,
        sun_separation,
    )
    if not np.isfinite(daytime_brightness):
        daytime_pe = 0.0
    else:
        daytime_pe = surface_brightness_residual_pe(
            zeropoint,
            daytime_brightness,
            ssp['background']['galactic'],
            pixel_area_arcsec2,
            exposure_s,
        )

    return daytime_pe, {
        'mode': 'perez',
        'brightness': None if not np.isfinite(daytime_brightness) else daytime_brightness,
        'sun_az': sun_az,
        'sun_el': sun_el,
        'sun_zenith': sun_zenith,
        'sun_sky_separation': sun_separation,
        'target_el': target_el_deg,
        'zenith_luminance_cd_m2': DEFAULT_DAYTIME_ZENITH_LUMINANCE_CD_M2,
        'residual_over_galactic': True,
    }


def hosek_wilkie_daytime_surface_brightness(
    target_zenith_deg,
    sun_zenith_deg,
    sun_sky_separation_deg,
):
    """Return Hosek-Wilkie daytime brightness in mag/arcsec^2."""
    luminance = hosek_wilkie_luminance(
        target_zenith_deg,
        sun_zenith_deg,
        sun_sky_separation_deg,
    )
    return luminance_to_surface_brightness(luminance)


def hosek_wilkie_transition_surface_brightness(
    target_zenith_deg,
    sun_zenith_deg,
    sun_sky_separation_deg,
):
    """Return Hosek-Wilkie brightness with a twilight-to-daytime horizon blend."""
    sun_el = 90.0 - float(sun_zenith_deg)
    if sun_el < SOLAR_BLEND_START_EL_DEG or target_zenith_deg >= 90.0:
        return np.inf, {
            'branch': 'below_horizon',
            'blend_weight': 0.0,
            'hosek_wilkie_brightness': None,
            'hosek_wilkie_luminance_cd_m2': 0.0,
            'blended_luminance_cd_m2': 0.0,
        }

    hw_luminance = hosek_wilkie_luminance(
        target_zenith_deg,
        sun_zenith_deg,
        sun_sky_separation_deg,
    )
    if sun_el >= SOLAR_BLEND_END_EL_DEG:
        hw_brightness = luminance_to_surface_brightness(hw_luminance)
        return hw_brightness, {
            'branch': 'hosek-wilkie',
            'blend_weight': 1.0,
            'hosek_wilkie_brightness': hw_brightness,
            'hosek_wilkie_luminance_cd_m2': hw_luminance,
            'blended_luminance_cd_m2': hw_luminance,
        }

    blend_t = (
        (sun_el - SOLAR_BLEND_START_EL_DEG)
        / (SOLAR_BLEND_END_EL_DEG - SOLAR_BLEND_START_EL_DEG)
    )
    weight = _smoothstep(blend_t)
    twilight_luminance = surface_brightness_to_luminance(
        PATAT_BRIGHT_END_SURFACE_BRIGHTNESS
    )
    blended_luminance = (1.0 - weight) * twilight_luminance + weight * hw_luminance
    hw_brightness = luminance_to_surface_brightness(hw_luminance)
    return luminance_to_surface_brightness(blended_luminance), {
        'branch': 'twilight_daytime_blend',
        'blend_weight': weight,
        'hosek_wilkie_brightness': hw_brightness,
        'hosek_wilkie_luminance_cd_m2': hw_luminance,
        'blended_luminance_cd_m2': blended_luminance,
    }


def hosek_wilkie_daytime_component(
    ssp,
    observer,
    ts_mid,
    ra_deg,
    dec_deg,
    target_el_deg,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    """Return the Hosek-Wilkie daytime PE residual and diagnostic metadata."""
    sun = load_sun()
    _, _, _, sun_az, sun_el, _ = get_los(observer, sun, ts_mid)
    sun_zenith = 90.0 - sun_el
    target_zenith = 90.0 - target_el_deg
    sun_separation = angle_from_los(observer, sun, ra_deg, dec_deg, ts_mid)
    daytime_brightness, transition_metadata = hosek_wilkie_transition_surface_brightness(
        target_zenith,
        sun_zenith,
        sun_separation,
    )
    if not np.isfinite(daytime_brightness):
        daytime_pe = 0.0
    else:
        daytime_pe = surface_brightness_residual_pe(
            zeropoint,
            daytime_brightness,
            ssp['background']['galactic'],
            pixel_area_arcsec2,
            exposure_s,
        )

    metadata = {
        'mode': 'hosek-wilkie',
        'brightness': None if not np.isfinite(daytime_brightness) else daytime_brightness,
        'sun_az': sun_az,
        'sun_el': sun_el,
        'sun_zenith': sun_zenith,
        'sun_sky_separation': sun_separation,
        'target_el': target_el_deg,
        'target_zenith': target_zenith,
        'turbidity': default_turbidity(),
        'ground_albedo': default_ground_albedo(),
        'transition': transition_metadata,
        'residual_over_galactic': True,
    }
    return daytime_pe, metadata


def _perez_luminance_distribution(zenith_deg, separation_deg):
    a, b, c, d, e = CIE_CLEAR_SKY_LOW_TURBIDITY
    theta = math.radians(zenith_deg)
    gamma = math.radians(separation_deg)
    gradation = 1.0 + a * math.exp(b / max(math.cos(theta), 1e-6))
    indicatrix = 1.0 + c * math.exp(d * gamma) + e * math.cos(gamma) ** 2
    return gradation * indicatrix


def _smoothstep(value):
    value = min(max(float(value), 0.0), 1.0)
    return value * value * (3.0 - 2.0 * value)
