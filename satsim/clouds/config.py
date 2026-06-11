"""Cloud configuration parsing."""

import math

import numpy as np

from satsim.clouds.constants import (
    CLOUD_TYPE_NAMES,
    CUSTOM_CLOUD_TYPE,
    CUSTOM_DEFAULTS,
    DEFAULT_CLOUD_RANGE_M,
    GROUP_FIELDS,
    PRESETS,
    PUBLIC_LAYER_FIELDS,
    SEED_MAX,
)
from satsim.clouds.models import CloudGeometry, CloudLayerConfig
from satsim.util.validation import (
    finite_number,
    integer,
    nonnegative_number,
    optional_finite_number,
    optional_unit_interval,
    positive_integer,
    positive_number,
    unit_interval,
)


def cloud_layers_from_config(ssp):
    """Return normalized enabled cloud layer configs from a SatSim config."""
    return parse_cloud_layers(ssp.get('clouds'), sim_seed=_config_seed(ssp))


def cloud_geometry_from_config(ssp, y_pad_px=0, x_pad_px=0, y_pad_after_px=None, x_pad_after_px=None):
    """Return cloud-plane geometry derived from FPA size, FOV, and padding."""
    try:
        fpa = ssp['fpa']
        y_pad_after_px = y_pad_px if y_pad_after_px is None else y_pad_after_px
        x_pad_after_px = x_pad_px if x_pad_after_px is None else x_pad_after_px
        fpa_height_px = int(fpa['height'])
        fpa_width_px = int(fpa['width'])
        y_pad_px = int(y_pad_px)
        x_pad_px = int(x_pad_px)
        y_pad_after_px = int(y_pad_after_px)
        x_pad_after_px = int(x_pad_after_px)
        height_px = fpa_height_px + y_pad_px + y_pad_after_px
        width_px = fpa_width_px + x_pad_px + x_pad_after_px
        y_meters_per_pixel = DEFAULT_CLOUD_RANGE_M * math.radians(float(fpa['y_fov'])) / float(fpa_height_px)
        x_meters_per_pixel = DEFAULT_CLOUD_RANGE_M * math.radians(float(fpa['x_fov'])) / float(fpa_width_px)
        return CloudGeometry(
            height_px=height_px,
            width_px=width_px,
            y_fov_deg=float(fpa['y_fov']) * height_px / float(fpa_height_px),
            x_fov_deg=float(fpa['x_fov']) * width_px / float(fpa_width_px),
            y_center_offset_m=0.5 * (y_pad_px - y_pad_after_px) * y_meters_per_pixel,
            x_center_offset_m=0.5 * (x_pad_px - x_pad_after_px) * x_meters_per_pixel,
        )
    except KeyError as exc:
        raise ValueError('Cloud generation requires fpa.height, fpa.width, fpa.y_fov, and fpa.x_fov.') from exc


def parse_cloud_layers(clouds, sim_seed=None):
    """Validate raw cloud layer objects and return ``CloudLayerConfig`` values."""
    if clouds is None:
        return tuple()
    if not isinstance(clouds, list):
        raise ValueError('clouds must be a list of cloud layer objects.')

    if sim_seed is None:
        sim_seed = _random_seed()

    layers = []
    for index, raw_layer in enumerate(clouds):
        layers.extend(_parse_cloud_layer(raw_layer, index, sim_seed))
    return tuple(layers)


def _parse_cloud_layer(raw_layer, index, sim_seed):
    if not isinstance(raw_layer, dict):
        raise ValueError('clouds[{}] must be an object.'.format(index))

    unknown = sorted(set(raw_layer) - PUBLIC_LAYER_FIELDS)
    if unknown:
        raise ValueError('Unknown cloud layer field(s) in clouds[{}]: {}.'.format(index, ', '.join(unknown)))

    if 'type' not in raw_layer:
        raise ValueError('clouds[{}].type is required.'.format(index))

    cloud_type = str(raw_layer['type'])
    if cloud_type == CUSTOM_CLOUD_TYPE:
        base = dict(CUSTOM_DEFAULTS)
    elif cloud_type in PRESETS:
        base = dict(PRESETS[cloud_type])
    else:
        valid = ', '.join(CLOUD_TYPE_NAMES + (CUSTOM_CLOUD_TYPE,))
        raise ValueError("Unknown cloud type '{}'. Valid cloud types: {}.".format(cloud_type, valid))

    enabled = raw_layer.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('clouds[{}].enabled must be a boolean.'.format(index))
    if not enabled:
        return tuple()

    layer = _normalize_cloud_layer(raw_layer, index)

    seed = raw_layer.get('seed')
    if seed is None:
        seed = integer('seed', sim_seed) + index * 10000
    seed = integer('seed', seed)

    for key in (
        'coverage',
        'feature_scales_m',
        'density_edge_width',
        'density_floor',
        'brightness',
        'range',
        'altitude',
        'wind_speed',
        'wind_direction',
        'texture_contrast',
        'locality_degree',
        'tau_min',
        'tau_max',
        'tau_gamma',
    ):
        if key in layer:
            base[key] = layer[key]

    coverage = unit_interval('coverage', base['coverage'])
    feature_scales_m = _feature_scales(base['feature_scales_m'])
    density_edge_width = nonnegative_number('density_edge_width', base['density_edge_width'])
    density_floor = optional_unit_interval('density_floor', base['density_floor'])
    brightness = optional_finite_number('brightness', base.get('brightness'))
    cloud_range = positive_number('range', base.get('range', DEFAULT_CLOUD_RANGE_M / 1000.0))
    altitude = None
    if base.get('altitude') is not None:
        altitude = positive_number('altitude', base.get('altitude'))
    wind_speed = nonnegative_number('wind_speed', base.get('wind_speed', 0.0))
    wind_direction = finite_number('wind_direction', base.get('wind_direction', 0.0))
    texture_contrast = unit_interval('texture_contrast', base['texture_contrast'])
    locality_degree = positive_integer('locality_degree', base['locality_degree'])

    tau_min = nonnegative_number('tau_min', base['tau_min'])
    tau_max = nonnegative_number('tau_max', base['tau_max'])
    if tau_max < tau_min:
        raise ValueError('tau_max must be greater than or equal to tau_min.')
    tau_gamma = positive_number('tau_gamma', base['tau_gamma'])

    return (CloudLayerConfig(
        name=str(raw_layer.get('name', 'cloud_{}'.format(index))),
        cloud_type=cloud_type,
        seed=seed,
        texture_seed=seed + int(base['seed_offset']),
        coverage=coverage,
        feature_scales_m=feature_scales_m,
        density_edge_width=density_edge_width,
        density_floor=density_floor,
        brightness=brightness,
        cloud_range=cloud_range,
        altitude=altitude,
        wind_speed=wind_speed,
        wind_direction=float(wind_direction) % 360.0,
        texture_contrast=texture_contrast,
        locality_degree=locality_degree,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_gamma=tau_gamma,
        mask_threshold=float(base['mask_threshold']),
        min_feature_scale_px=float(base['min_feature_scale_px']),
        amplitude_decay=float(base['amplitude_decay']),
    ),)


def _normalize_cloud_layer(raw_layer, index):
    normalized = {
        key: value
        for key, value in raw_layer.items()
        if key not in GROUP_FIELDS
    }
    for group_name, aliases in GROUP_FIELDS.items():
        if group_name in raw_layer:
            _merge_cloud_layer_group(normalized, group_name, raw_layer[group_name], aliases, index)
    return normalized


def _merge_cloud_layer_group(normalized, group_name, group, aliases, index):
    if not isinstance(group, dict):
        raise ValueError('clouds[{}].{} must be an object.'.format(index, group_name))

    seen = {}
    for raw_key, value in group.items():
        if raw_key not in aliases:
            raise ValueError(
                'Unknown cloud layer field(s) in clouds[{}].{}: {}.'.format(index, group_name, raw_key)
            )
        canonical_key = aliases[raw_key]
        if canonical_key in seen:
            raise ValueError(
                'clouds[{}].{} defines {} more than once.'.format(index, group_name, canonical_key)
            )
        if canonical_key in normalized:
            raise ValueError(
                'clouds[{}] defines {} both directly and inside {}.'.format(index, canonical_key, group_name)
            )
        normalized[canonical_key] = value
        seen[canonical_key] = raw_key


def _config_seed(ssp):
    if isinstance(ssp.get('sim'), dict) and 'seed' in ssp['sim']:
        return integer('sim.seed', ssp['sim']['seed'])
    if 'seed' in ssp:
        return integer('seed', ssp['seed'])
    return None


def _random_seed():
    return int(np.random.RandomState().randint(0, SEED_MAX))


def _feature_scales(value):
    if isinstance(value, str):
        raise ValueError('feature_scales_m must be a list of positive numbers.')
    try:
        scales = tuple(positive_number('feature_scales_m', scale) for scale in value)
    except TypeError:
        raise ValueError('feature_scales_m must be a list of positive numbers.')
    if not scales:
        raise ValueError('feature_scales_m must contain at least one scale.')
    return scales
