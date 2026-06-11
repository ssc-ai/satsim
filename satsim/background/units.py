"""Background unit conversions."""

import math
import numpy as np

from satsim.image.fpa import mv_to_pe


V_BAND_ZERO_LUMINANCE_CD_M2 = 1.096e5


def surface_brightness_to_pe(zeropoint, brightness, pixel_area_arcsec2, exposure_s):
    """Convert visual magnitude per arcsec^2 to PE per pixel per exposure."""
    return mv_to_pe(zeropoint, brightness) * pixel_area_arcsec2 * exposure_s


def surface_brightness_residual_pe(zeropoint, brightness, baseline_brightness,
                                   pixel_area_arcsec2, exposure_s):
    """Return the positive PE residual between two surface-brightness levels."""
    total_pe = surface_brightness_to_pe(
        zeropoint,
        brightness,
        pixel_area_arcsec2,
        exposure_s,
    )
    baseline_pe = surface_brightness_to_pe(
        zeropoint,
        baseline_brightness,
        pixel_area_arcsec2,
        exposure_s,
    )
    return np.maximum(0.0, total_pe - baseline_pe)


def nano_lamberts_to_surface_brightness(nano_lamberts):
    """Convert V-band sky brightness from nanoLamberts to mag/arcsec^2."""
    if nano_lamberts <= 0:
        return np.inf
    return (20.7233 - math.log(nano_lamberts / 34.08)) / 0.92104


def luminance_to_surface_brightness(luminance_cd_m2):
    """Convert V-like photopic luminance to mag/arcsec^2."""
    if luminance_cd_m2 <= 0:
        return np.inf
    return -2.5 * math.log10(luminance_cd_m2 / V_BAND_ZERO_LUMINANCE_CD_M2)


def surface_brightness_to_luminance(brightness):
    """Convert V-like mag/arcsec^2 to luminance in cd/m^2."""
    return V_BAND_ZERO_LUMINANCE_CD_M2 * 10.0 ** (-0.4 * float(brightness))
