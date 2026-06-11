"""Cloud texture generation."""

import numpy as np
from opensimplex import OpenSimplex

from satsim.clouds.config import cloud_geometry_from_config, cloud_layers_from_config
from satsim.clouds.constants import DEFAULT_CLOUD_RANGE_M
from satsim.clouds.field import (
    cloud_layer_from_density,
    combine_cloud_layers,
    normalize_unit,
    smoothstep,
)
from satsim.clouds.models import CloudGeometry
from satsim.util.validation import nonnegative_number, unit_interval


def cloud_field_from_config(ssp, y_pad_px=0, x_pad_px=0, layer_configs=None,
                            y_pad_after_px=None, x_pad_after_px=None):
    """Generate a combined cloud field from a SatSim configuration."""
    if layer_configs is None:
        layer_configs = cloud_layers_from_config(ssp)
    if not layer_configs:
        return None

    geometry = cloud_geometry_from_config(
        ssp,
        y_pad_px=y_pad_px,
        x_pad_px=x_pad_px,
        y_pad_after_px=y_pad_after_px,
        x_pad_after_px=x_pad_after_px,
    )
    field = generate_cloud_field(layer_configs, geometry)
    y_pad_after_px = y_pad_px if y_pad_after_px is None else y_pad_after_px
    x_pad_after_px = x_pad_px if x_pad_after_px is None else x_pad_after_px
    field.metadata['motion'] = {
        'source_shape_px': [int(geometry.height_px), int(geometry.width_px)],
        'pad_px': [int(y_pad_px), int(x_pad_px)],
        'pad_after_px': [int(y_pad_after_px), int(x_pad_after_px)],
    }
    return field


def generate_cloud_field(layer_configs, geometry):
    """Generate and combine all enabled cloud layers for one geometry."""
    layers = tuple(generate_cloud_layer(config, geometry) for config in layer_configs)
    return combine_cloud_layers(layers)


def generate_cloud_layer(config, geometry):
    """Generate one cloud layer density, mask, optical depth, and transmission."""
    texture = cloud_texture(config, layer_geometry(config, geometry))
    density = cloud_density_from_noise(
        texture,
        coverage=config.coverage,
        edge_width=config.density_edge_width,
        density_floor=config.density_floor,
    )
    return cloud_layer_from_density(config, density)


def cloud_texture(config, geometry):
    """Return a normalized procedural texture for one cloud layer."""
    feature_scales_m = active_feature_scales_m(config, geometry)
    texture = open_simplex_fractal(
        geometry=geometry,
        feature_scales_m=feature_scales_m,
        seed=config.texture_seed,
        amplitude_decay=config.amplitude_decay,
    )
    for index in range(1, config.locality_degree):
        texture *= open_simplex_fractal(
            geometry=geometry,
            feature_scales_m=feature_scales_m,
            seed=config.texture_seed + index * 7919,
            amplitude_decay=config.amplitude_decay,
        )
    texture = normalize_unit(texture)
    texture = 0.5 + (texture - 0.5) * config.texture_contrast
    return np.clip(texture, 0.0, 1.0).astype(np.float32)


def active_feature_scales_m(config, geometry):
    """Return texture scales that are resolvable for the current pixel size."""
    active = tuple(
        float(scale_m)
        for scale_m in config.feature_scales_m
        if scale_m / geometry.meters_per_pixel >= config.min_feature_scale_px
    )
    return active if active else (max(config.feature_scales_m),)


def layer_geometry(config, geometry):
    """Return geometry rescaled for a layer-specific cloud range."""
    layer_range_m = float(config.cloud_range) * 1000.0
    if abs(layer_range_m - float(geometry.cloud_range_m)) < 1e-9:
        return geometry
    scale = layer_range_m / float(geometry.cloud_range_m)
    return CloudGeometry(
        height_px=geometry.height_px,
        width_px=geometry.width_px,
        y_fov_deg=geometry.y_fov_deg,
        x_fov_deg=geometry.x_fov_deg,
        cloud_range_m=layer_range_m,
        y_center_offset_m=geometry.y_center_offset_m * scale,
        x_center_offset_m=geometry.x_center_offset_m * scale,
    )


def cloud_density_from_noise(noise, coverage, edge_width, density_floor):
    """Map normalized procedural noise into normalized cloud density."""
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
    support = smoothstep(values, threshold - width, threshold + width)
    density = floor + (1.0 - floor) * np.power(support, gamma) * gain
    return np.clip(density, 0.0, 1.0).astype(np.float32)


def open_simplex_fractal(geometry, feature_scales_m, seed, amplitude_decay):
    """Generate a multi-scale OpenSimplex texture over the cloud plane."""
    xs = (np.arange(geometry.width_px, dtype=np.float64) + 0.5) * geometry.x_meters_per_pixel
    ys = (np.arange(geometry.height_px, dtype=np.float64) + 0.5) * geometry.y_meters_per_pixel
    xs -= geometry.footprint_width_m * 0.5
    ys -= geometry.footprint_height_m * 0.5
    xs -= geometry.x_center_offset_m
    ys -= geometry.y_center_offset_m

    largest = max(feature_scales_m)
    field = np.zeros((geometry.height_px, geometry.width_px), dtype=np.float64)
    total_weight = 0.0
    for octave_index, scale_m in enumerate(feature_scales_m):
        weight = (float(scale_m) / largest) ** amplitude_decay
        field += noise_grid(seed + octave_index * 1009, xs / scale_m, ys / scale_m) * weight
        total_weight += weight

    field /= total_weight
    return normalize_unit(field).astype(np.float32)


def noise_grid(seed, xs, ys):
    """Return an OpenSimplex noise grid with SatSim row/column orientation."""
    values = np.asarray(OpenSimplex(seed=int(seed)).noise2array(xs, ys), dtype=np.float64)
    return values.T if values.shape == (xs.size, ys.size) else values
