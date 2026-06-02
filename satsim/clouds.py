from __future__ import division, print_function, absolute_import

import math
from dataclasses import dataclass

import numpy as np
from opensimplex import OpenSimplex

from satsim.image.fpa import mv_to_pe
from satsim.util.validation import (
    integer,
    nonnegative_number,
    optional_finite_number,
    optional_unit_interval,
    positive_integer,
    positive_number,
    unit_interval,
)


CLOUD_TYPE_NAMES = ('patchy', 'cellular', 'veil', 'sheet', 'fog')
CUSTOM_CLOUD_TYPE = 'custom'
DEFAULT_CLOUD_RANGE_M = 3000.0

_PUBLIC_LAYER_FIELDS = {
    'type',
    'name',
    'enabled',
    'seed',
    'coverage',
    'feature_scales_m',
    'density_edge_width',
    'density_floor',
    'brightness',
    'texture_contrast',
    'locality_degree',
    'tau_min',
    'tau_max',
    'tau_gamma',
}

_INTERNAL_DEFAULTS = {
    'coverage_mode': 'density',
    'mask_threshold': 0.02,
    'min_feature_scale_px': 4.0,
    'amplitude_decay': 0.7,
}

_CUSTOM_DEFAULTS = dict(_INTERNAL_DEFAULTS, **{
    'seed_offset': 0,
    'coverage': 0.5,
    'feature_scales_m': (40.0, 80.0, 160.0, 320.0, 640.0),
    'density_edge_width': 0.12,
    'density_floor': None,
    'brightness': None,
    'texture_contrast': 1.0,
    'locality_degree': 1,
    'tau_min': 0.02,
    'tau_max': 3.5,
    'tau_gamma': 1.15,
})

_PRESETS = {
    'patchy': dict(_INTERNAL_DEFAULTS, **{
        'seed_offset': 101,
        'coverage': 0.45,
        'feature_scales_m': (
            2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0, 256.0,
            512.0, 1024.0, 2048.0, 4096.0, 8192.0, 16384.0,
        ),
        'min_feature_scale_px': 12.0,
        'density_edge_width': 0.08,
        'density_floor': None,
        'mask_threshold': 0.03,
        'amplitude_decay': 0.65,
        'texture_contrast': 1.0,
        'locality_degree': 2,
        'tau_min': 0.2,
        'tau_max': 9.0,
        'tau_gamma': 0.55,
    }),
    'cellular': dict(_INTERNAL_DEFAULTS, **{
        'seed_offset': 202,
        'coverage': 0.55,
        'feature_scales_m': (960.0, 1920.0, 3840.0, 7680.0, 15360.0),
        'density_edge_width': 0.16,
        'density_floor': None,
        'mask_threshold': 0.02,
        'amplitude_decay': 0.85,
        'texture_contrast': 0.9,
        'locality_degree': 1,
        'tau_min': 0.02,
        'tau_max': 3.5,
        'tau_gamma': 1.15,
    }),
    'veil': dict(_INTERNAL_DEFAULTS, **{
        'seed_offset': 303,
        'coverage': 0.40,
        'feature_scales_m': (640.0, 1280.0, 2560.0, 5120.0, 10240.0, 20480.0),
        'density_edge_width': 0.42,
        'density_floor': 0.06,
        'mask_threshold': 0.02,
        'amplitude_decay': 1.1,
        'texture_contrast': 0.6,
        'locality_degree': 1,
        'tau_min': 0.01,
        'tau_max': 0.9,
        'tau_gamma': 1.15,
    }),
    'sheet': dict(_INTERNAL_DEFAULTS, **{
        'seed_offset': 404,
        'coverage': 0.90,
        'feature_scales_m': (2048.0, 4096.0, 8192.0, 16384.0, 32768.0),
        'min_feature_scale_px': 8.0,
        'density_edge_width': 0.34,
        'density_floor': 0.4,
        'mask_threshold': 0.02,
        'amplitude_decay': 1.3,
        'texture_contrast': 0.52,
        'locality_degree': 1,
        'tau_min': 0.5,
        'tau_max': 10.0,
        'tau_gamma': 1.15,
    }),
    'fog': dict(_INTERNAL_DEFAULTS, **{
        'seed_offset': 505,
        'coverage': 0.95,
        'feature_scales_m': (8192.0, 16384.0, 32768.0, 65536.0),
        'density_edge_width': 0.45,
        'density_floor': 0.45,
        'mask_threshold': 0.01,
        'amplitude_decay': 1.6,
        'texture_contrast': 0.28,
        'locality_degree': 1,
        'tau_min': 0.05,
        'tau_max': 1.5,
        'tau_gamma': 1.15,
    }),
}


@dataclass(frozen=True)
class CloudGeometry:
    height_px: int
    width_px: int
    y_fov_deg: float
    x_fov_deg: float
    cloud_range_m: float = DEFAULT_CLOUD_RANGE_M

    @property
    def x_meters_per_pixel(self):
        return self.cloud_range_m * math.radians(self.x_fov_deg) / float(self.width_px)

    @property
    def y_meters_per_pixel(self):
        return self.cloud_range_m * math.radians(self.y_fov_deg) / float(self.height_px)

    @property
    def meters_per_pixel(self):
        return 0.5 * (self.x_meters_per_pixel + self.y_meters_per_pixel)

    @property
    def footprint_width_m(self):
        return self.width_px * self.x_meters_per_pixel

    @property
    def footprint_height_m(self):
        return self.height_px * self.y_meters_per_pixel


@dataclass(frozen=True)
class CloudLayerConfig:
    name: str
    cloud_type: str
    seed: int
    texture_seed: int
    coverage: float
    feature_scales_m: tuple
    density_edge_width: float
    density_floor: float
    brightness: float
    texture_contrast: float
    locality_degree: int
    tau_min: float
    tau_max: float
    tau_gamma: float
    mask_threshold: float
    min_feature_scale_px: float
    amplitude_decay: float

    def metadata(self):
        return {
            'name': self.name,
            'type': self.cloud_type,
            'seed': int(self.seed),
            'texture_seed': int(self.texture_seed),
            'coverage': float(self.coverage),
            'feature_scales_m': list(self.feature_scales_m),
            'density_edge_width': float(self.density_edge_width),
            'density_floor': self.density_floor,
            'brightness': None if self.brightness is None else float(self.brightness),
            'texture_contrast': float(self.texture_contrast),
            'locality_degree': int(self.locality_degree),
            'tau_min': float(self.tau_min),
            'tau_max': float(self.tau_max),
            'tau_gamma': float(self.tau_gamma),
        }


@dataclass(frozen=True)
class CloudLayer:
    config: CloudLayerConfig
    density: np.ndarray
    mask: np.ndarray
    tau: np.ndarray
    transmission: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class CloudField:
    layers: tuple
    density: np.ndarray
    mask: np.ndarray
    tau: np.ndarray
    transmission: np.ndarray
    metadata: dict


def cloud_field_from_config(ssp):
    """Generate a combined cloud field from a SatSim configuration."""
    layer_configs = parse_cloud_layers(ssp.get('clouds'), sim_seed=_sim_seed(ssp))
    if not layer_configs:
        return None

    geometry = cloud_geometry_from_config(ssp)
    return generate_cloud_field(layer_configs, geometry)


def cloud_geometry_from_config(ssp):
    try:
        fpa = ssp['fpa']
        return CloudGeometry(
            height_px=int(fpa['height']),
            width_px=int(fpa['width']),
            y_fov_deg=float(fpa['y_fov']),
            x_fov_deg=float(fpa['x_fov']),
        )
    except KeyError as exc:
        raise ValueError('Cloud generation requires fpa.height, fpa.width, fpa.y_fov, and fpa.x_fov.') from exc


def parse_cloud_layers(clouds, sim_seed=7):
    if clouds is None:
        return tuple()
    if not isinstance(clouds, list):
        raise ValueError('clouds must be a list of cloud layer objects.')

    layers = []
    for index, raw_layer in enumerate(clouds):
        layers.extend(_parse_cloud_layer(raw_layer, index, sim_seed))
    return tuple(layers)


def cloud_brightness_pe_from_field(cloud_field, zeropoint, exposure_s, pixel_area_arcsec2):
    """Return per-pixel cloud glow in photoelectrons for the generated field."""
    if cloud_field is None:
        return None

    brightness_pe = np.zeros_like(cloud_field.transmission, dtype=np.float32)
    for layer in cloud_field.layers:
        brightness = layer.config.brightness
        if brightness is None:
            continue
        base_pe = float(mv_to_pe(zeropoint, brightness)) * float(pixel_area_arcsec2) * float(exposure_s)
        brightness_pe += (base_pe * (1.0 - layer.transmission)).astype(np.float32)

    return brightness_pe.astype(np.float32)


def generate_cloud_field(layer_configs, geometry):
    layers = tuple(generate_cloud_layer(config, geometry) for config in layer_configs)
    if not layers:
        return None

    transmission = np.ones_like(layers[0].transmission, dtype=np.float32)
    clear_density = np.ones_like(layers[0].density, dtype=np.float32)
    tau = np.zeros_like(layers[0].tau, dtype=np.float32)
    mask = np.zeros_like(layers[0].mask, dtype=bool)

    for layer in layers:
        transmission *= layer.transmission
        clear_density *= (1.0 - layer.density)
        tau += layer.tau
        mask |= layer.mask

    density = (1.0 - clear_density).clip(0.0, 1.0).astype(np.float32)
    metadata = {
        'layers': [layer.metadata for layer in layers],
        'stats': {
            'density': _array_stats(density),
            'mask': {'coverage_fraction': float(mask.mean())},
            'tau': _array_stats(tau),
            'transmission': _array_stats(transmission),
        },
    }
    return CloudField(
        layers=layers,
        density=density,
        mask=mask,
        tau=tau.astype(np.float32),
        transmission=transmission.astype(np.float32),
        metadata=metadata,
    )


def generate_cloud_layer(config, geometry):
    texture = _cloud_texture(config, geometry)
    density = _cloud_density_from_noise(
        texture,
        coverage=config.coverage,
        edge_width=config.density_edge_width,
        density_floor=config.density_floor,
    )
    mask = density > config.mask_threshold
    tau, transmission = _optical_fields(density, config)
    metadata = config.metadata()
    metadata['stats'] = {
        'density': _array_stats(density),
        'mask': {'coverage_fraction': float(mask.mean())},
        'tau': _array_stats(tau),
        'transmission': _array_stats(transmission),
    }
    return CloudLayer(
        config=config,
        density=density,
        mask=mask,
        tau=tau,
        transmission=transmission,
        metadata=metadata,
    )


def _parse_cloud_layer(raw_layer, index, sim_seed):
    if not isinstance(raw_layer, dict):
        raise ValueError('clouds[{}] must be an object.'.format(index))

    unknown = sorted(set(raw_layer) - _PUBLIC_LAYER_FIELDS)
    if unknown:
        raise ValueError('Unknown cloud layer field(s) in clouds[{}]: {}.'.format(index, ', '.join(unknown)))

    if 'type' not in raw_layer:
        raise ValueError('clouds[{}].type is required.'.format(index))

    cloud_type = str(raw_layer['type'])
    if cloud_type == CUSTOM_CLOUD_TYPE:
        base = dict(_CUSTOM_DEFAULTS)
    elif cloud_type in _PRESETS:
        base = dict(_PRESETS[cloud_type])
    else:
        valid = ', '.join(CLOUD_TYPE_NAMES + (CUSTOM_CLOUD_TYPE,))
        raise ValueError("Unknown cloud type '{}'. Valid cloud types: {}.".format(cloud_type, valid))

    enabled = raw_layer.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('clouds[{}].enabled must be a boolean.'.format(index))
    if not enabled:
        return tuple()

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
        'texture_contrast',
        'locality_degree',
        'tau_min',
        'tau_max',
        'tau_gamma',
    ):
        if key in raw_layer:
            base[key] = raw_layer[key]

    coverage = unit_interval('coverage', base['coverage'])
    feature_scales_m = _feature_scales(base['feature_scales_m'])
    density_edge_width = nonnegative_number('density_edge_width', base['density_edge_width'])
    density_floor = optional_unit_interval('density_floor', base['density_floor'])
    brightness = optional_finite_number('brightness', base.get('brightness'))
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
        texture_contrast=texture_contrast,
        locality_degree=locality_degree,
        tau_min=tau_min,
        tau_max=tau_max,
        tau_gamma=tau_gamma,
        mask_threshold=float(base['mask_threshold']),
        min_feature_scale_px=float(base['min_feature_scale_px']),
        amplitude_decay=float(base['amplitude_decay']),
    ),)


def _cloud_texture(config, geometry):
    feature_scales_m = _active_feature_scales_m(config, geometry)
    texture = _open_simplex_fractal(
        geometry=geometry,
        feature_scales_m=feature_scales_m,
        seed=config.texture_seed,
        amplitude_decay=config.amplitude_decay,
    )
    for index in range(1, config.locality_degree):
        texture *= _open_simplex_fractal(
            geometry=geometry,
            feature_scales_m=feature_scales_m,
            seed=config.texture_seed + index * 7919,
            amplitude_decay=config.amplitude_decay,
        )
    texture = _normalize_unit(texture)
    texture = 0.5 + (texture - 0.5) * config.texture_contrast
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def _active_feature_scales_m(config, geometry):
    active = tuple(
        float(scale_m)
        for scale_m in config.feature_scales_m
        if scale_m / geometry.meters_per_pixel >= config.min_feature_scale_px
    )
    return active if active else (max(config.feature_scales_m),)


def _cloud_density_from_noise(noise, coverage, edge_width, density_floor):
    values = np.clip(np.asarray(noise, dtype=np.float32), 0.0, 1.0)
    cloud_fraction = unit_interval('coverage', coverage)
    nonnegative_number('density_edge_width', edge_width)
    if density_floor is not None:
        unit_interval('density_floor', density_floor)

    if cloud_fraction == 0.0:
        return np.zeros_like(values, dtype=np.float32)
    if cloud_fraction == 1.0:
        return np.ones_like(values, dtype=np.float32)

    threshold = 0.82 - 0.66 * cloud_fraction
    width = 0.06 + float(edge_width) + 0.10 * cloud_fraction
    gamma = max(0.45, 1.65 - 1.15 * cloud_fraction)
    gain = 0.50 + 0.72 * cloud_fraction
    floor = max(0.0, cloud_fraction - 0.85) * 0.25 if density_floor is None else float(density_floor)
    support = _smoothstep(values, threshold - width, threshold + width)
    density = floor + (1.0 - floor) * np.power(support, gamma) * gain
    return np.clip(density, 0.0, 1.0).astype(np.float32)


def _optical_fields(density, config):
    cloud = np.clip(np.asarray(density, dtype=np.float32), 0.0, 1.0)
    thickness = np.power(cloud, config.tau_gamma, dtype=np.float32)
    tau = config.tau_min + thickness * (config.tau_max - config.tau_min)
    tau = np.where(cloud > 0.0, tau * cloud, 0.0).astype(np.float32)
    transmission = np.exp(-tau).astype(np.float32)
    return tau, transmission


def _open_simplex_fractal(geometry, feature_scales_m, seed, amplitude_decay):
    xs = (np.arange(geometry.width_px, dtype=np.float64) + 0.5) * geometry.x_meters_per_pixel
    ys = (np.arange(geometry.height_px, dtype=np.float64) + 0.5) * geometry.y_meters_per_pixel
    xs -= geometry.footprint_width_m * 0.5
    ys -= geometry.footprint_height_m * 0.5

    largest = max(feature_scales_m)
    field = np.zeros((geometry.height_px, geometry.width_px), dtype=np.float64)
    total_weight = 0.0
    for octave_index, scale_m in enumerate(feature_scales_m):
        weight = (float(scale_m) / largest) ** amplitude_decay
        field += _noise_grid(seed + octave_index * 1009, xs / scale_m, ys / scale_m) * weight
        total_weight += weight

    field /= total_weight
    return _normalize_unit(field).astype(np.float32)


def _noise_grid(seed, xs, ys):
    generator = OpenSimplex(seed=int(seed))
    array_noise = getattr(generator, 'noise2array', None)
    if callable(array_noise):
        values = np.asarray(array_noise(xs, ys), dtype=np.float64)
        if values.shape == (xs.size, ys.size):
            values = values.T
        if values.shape == (ys.size, xs.size):
            return values

    scalar_noise = getattr(generator, 'noise2', None) or getattr(generator, 'noise2d')
    values = np.empty((ys.size, xs.size), dtype=np.float64)
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            values[row, col] = scalar_noise(float(x), float(y))
    return values


def _sim_seed(ssp):
    if isinstance(ssp.get('sim'), dict) and 'seed' in ssp['sim']:
        return integer('sim.seed', ssp['sim']['seed'])
    if 'seed' in ssp:
        return integer('seed', ssp['seed'])
    return 7


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


def _smoothstep(values, edge0, edge1):
    if edge0 == edge1:
        return (values >= edge1).astype(np.float32)
    x = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _normalize_unit(values):
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / span


def _array_stats(values):
    array = np.asarray(values, dtype=np.float64)
    return {
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
    }
