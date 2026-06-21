"""Background component assembly and config mode dispatch."""

import numpy as np

from satsim.background.daytime import (
    hosek_wilkie_daytime_component,
    perez_daytime_component,
)
from satsim.background.moon import krisciunas_schaefer_moon_component
from satsim.background.twilight import patat_twilight_component
from satsim.background.units import (
    surface_brightness_residual_pe,
    surface_brightness_to_pe,
)


MODE_OPTIONS = {
    'moon': ('none', 'default', 'krisciunas-schaefer'),
    'twilight': ('none', 'default', 'patat'),
    'daytime': ('none', 'default', 'perez', 'hosek-wilkie'),
}
DEFAULT_MODES = {
    'moon': 'krisciunas-schaefer',
    'twilight': 'patat',
    'daytime': 'hosek-wilkie',
}
IMPLEMENTED_MODES = {
    'moon': ('krisciunas-schaefer',),
    'twilight': ('patat',),
    'daytime': ('perez', 'hosek-wilkie'),
}
DAYTIME_COMPONENTS = {
    'perez': perez_daytime_component,
    'hosek-wilkie': hosek_wilkie_daytime_component,
}


def background_components_from_config(ssp, zeropoint, pixel_area_arcsec2, exposure_s):
    """Build static pre-cloud background PE components from ``ssp['background']``.

    Returned component values are in photoelectrons per pixel per exposure.
    Static components include galactic sky, artificial skyglow residual, and
    legacy stray light. Frame-dependent Sun and Moon terms are added by
    :func:`background_frame_components_from_config`.
    """
    background = ssp['background']
    modes = background_modes(background)

    natural_pe = surface_brightness_to_pe(
        zeropoint,
        background['galactic'],
        pixel_area_arcsec2,
        exposure_s,
    )

    skyglow_enabled = 'skyglow' in background
    if skyglow_enabled:
        if np.any(np.asarray(background['skyglow']) > np.asarray(background['galactic'])):
            raise ValueError(
                'background.skyglow must be less than or equal to '
                'background.galactic; larger magnitude values are darker and '
                'would imply negative artificial skyglow.'
            )
        skyglow_pe = surface_brightness_residual_pe(
            zeropoint,
            background['skyglow'],
            background['galactic'],
            pixel_area_arcsec2,
            exposure_s,
        )
    else:
        skyglow_pe = 0.0

    stray_enabled = _has_stray(background)
    stray_pe = _stray_pe(background.get('stray'), exposure_s)
    pre_cloud_pe = natural_pe + skyglow_pe + stray_pe

    return {
        'background_natural_pe': natural_pe,
        'background_skyglow_pe': skyglow_pe,
        'background_stray_pe': stray_pe,
        'background_pre_cloud_pe': pre_cloud_pe,
        'background_pe': pre_cloud_pe,
        'active': {
            'background_natural_pe': True,
            'background_skyglow_pe': skyglow_enabled,
            'background_stray_pe': stray_enabled,
            'background_pre_cloud_pe': False,
        },
        'metadata': {
            'galactic': background['galactic'],
            'skyglow': background.get('skyglow'),
            'skyglow_enabled': skyglow_enabled,
            'stray_enabled': stray_enabled,
            'stray_augmentation_enabled': False,
            'modes': modes,
        },
    }


def background_frame_components_from_config(
    ssp,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
    observer,
    site_mode,
    astrometrics,
    ts_mid,
):
    """Build frame-dependent background PE components from Sun/Moon geometry.

    The returned dictionary has ``components``, ``active``, and ``metadata``
    entries matching the static component structure. ``astrometrics`` must
    contain the current frame boresight fields used for angular separation.
    """
    background = ssp['background']
    modes = background_modes(background)
    components = {}
    active = {}
    metadata = {}

    for key, mode in modes.items():
        if mode == 'none':
            continue
        _require_ground_background_component(key, site_mode, observer)
        component_key, build_component = FRAME_COMPONENTS[key]
        component_pe, component_metadata = build_component(
            mode,
            ssp,
            observer,
            ts_mid,
            astrometrics,
            zeropoint,
            pixel_area_arcsec2,
            exposure_s,
        )
        components[component_key] = component_pe
        active[component_key] = True
        metadata[key] = component_metadata

    return {
        'components': components,
        'active': active,
        'metadata': metadata,
    }


def apply_background_stray_augmentation(components, pre_cloud_pe):
    """Return ``components`` updated after legacy stray-background augmentation."""
    updated = dict(components)
    active = dict(components['active'])
    metadata = dict(components['metadata'])

    updated['background_pre_cloud_pe'] = pre_cloud_pe
    updated['background_pe'] = pre_cloud_pe
    updated['background_stray_pe'] = (
        pre_cloud_pe
        - components['background_natural_pe']
        - components['background_skyglow_pe']
    )
    active['background_stray_pe'] = True
    metadata['stray_augmentation_enabled'] = True
    updated['active'] = active
    updated['metadata'] = metadata
    return updated


def background_modes(background):
    """Resolve configured moon, twilight, and daytime modes to concrete models."""
    modes = {}
    for key, valid_modes in MODE_OPTIONS.items():
        raw = background.get(key)
        if raw is None:
            mode = 'none'
        else:
            if not isinstance(raw, dict):
                raise ValueError('background.{} must be an object with a mode field.'.format(key))
            mode = raw.get('mode', 'none')
        if mode not in valid_modes:
            raise ValueError(
                'Unknown background.{}.mode {!r}; expected one of {}.'.format(
                    key,
                    mode,
                    ', '.join(valid_modes),
                )
            )
        if mode == 'default':
            mode = DEFAULT_MODES[key]
        if mode != 'none' and mode not in IMPLEMENTED_MODES[key]:
            raise ValueError(
                'background.{}.mode {!r} is not implemented yet.'.format(key, mode)
            )
        modes[key] = mode
    return modes


def _has_stray(background):
    if 'stray' not in background:
        return False
    stray = background['stray']
    return not (isinstance(stray, dict) and stray.get('mode') == 'none')


def _stray_pe(stray, exposure_s):
    if stray is None:
        return 0.0
    if isinstance(stray, dict) and stray.get('mode') == 'none':
        return 0.0
    return stray * exposure_s


def _require_ground_background_component(key, site_mode, observer):
    if site_mode != 'ground' or observer is None:
        raise ValueError(
            'background.{}.mode requires a ground geometry.site with lat/lon.'.format(key)
        )


def _moon_frame_component(
    mode,
    ssp,
    observer,
    ts_mid,
    astrometrics,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    return krisciunas_schaefer_moon_component(
        observer,
        ts_mid,
        astrometrics['ra'],
        astrometrics['dec'],
        astrometrics['el'],
        zeropoint,
        pixel_area_arcsec2,
        exposure_s,
    )


def _twilight_frame_component(
    mode,
    ssp,
    observer,
    ts_mid,
    astrometrics,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    return patat_twilight_component(
        ssp,
        observer,
        ts_mid,
        zeropoint,
        pixel_area_arcsec2,
        exposure_s,
    )


def _daytime_frame_component(
    mode,
    ssp,
    observer,
    ts_mid,
    astrometrics,
    zeropoint,
    pixel_area_arcsec2,
    exposure_s,
):
    return DAYTIME_COMPONENTS[mode](
        ssp,
        observer,
        ts_mid,
        astrometrics['ra'],
        astrometrics['dec'],
        astrometrics['el'],
        zeropoint,
        pixel_area_arcsec2,
        exposure_s,
    )


FRAME_COMPONENTS = {
    'moon': ('background_moon_pe', _moon_frame_component),
    'twilight': ('background_twilight_pe', _twilight_frame_component),
    'daytime': ('background_daytime_pe', _daytime_frame_component),
}
