"""Cloud field composition, cropping, and brightness helpers."""

import math

import numpy as np

from satsim.clouds.constants import (
    LUNAR_DIRECT_BRIGHTENING,
    SOLAR_DIRECT_BRIGHTENING,
    SOURCE_BRIGHTENING,
)
from satsim.clouds.models import CloudField, CloudLayer
from satsim.image.fpa import mv_to_pe


def cloud_brightness_pe_from_field(cloud_field, zeropoint, exposure_s, pixel_area_arcsec2,
                                   source_components=None, return_components=False):
    """Return per-pixel cloud glow in photoelectrons for the generated field."""
    if cloud_field is None:
        return None

    configured_pe = np.zeros_like(cloud_field.transmission, dtype=np.float32)
    for layer in cloud_field.layers:
        brightness = layer.config.brightness
        if brightness is None:
            continue
        base_pe = float(mv_to_pe(zeropoint, brightness)) * float(pixel_area_arcsec2) * float(exposure_s)
        configured_pe += (base_pe * (1.0 - layer.transmission)).astype(np.float32)

    source_pe = cloud_source_brightness_pe_from_field(cloud_field, source_components)
    total = (configured_pe + source_pe).astype(np.float32)
    if return_components:
        return {
            'cloud_brightness_pe': total,
            'cloud_config_brightness_pe': configured_pe.astype(np.float32),
            'cloud_source_brightness_pe': source_pe.astype(np.float32),
        }
    return total


def cloud_source_brightness_pe_from_field(cloud_field, source_components=None):
    """Return source-driven cloud glow from artificial, lunar, and solar PE fields."""
    if cloud_field is None:
        return None
    source_components = source_components or {}
    if not source_components:
        return np.zeros_like(cloud_field.transmission, dtype=np.float32)

    source_pe = np.zeros_like(cloud_field.transmission, dtype=np.float32)
    for source_name, source_value in source_components.items():
        if source_value is None:
            continue
        source_value, source_metadata = _source_value_and_metadata(source_value)
        if source_value is None:
            continue
        source_array = _as_cloud_array(source_value, cloud_field.transmission.shape)
        if not np.any(source_array):
            continue
        for layer in cloud_field.layers:
            height_m = _cloud_source_height_m(layer.config)
            optical_coupling = (1.0 - layer.transmission).astype(np.float32)
            source_pe += source_array * optical_coupling * _cloud_source_gain(
                source_name,
                source_metadata,
                height_m,
            )

    return source_pe.astype(np.float32)


def crop_cloud_field(cloud_field, offset_px, height_px, width_px, pad_px=(0, 0), layer_offsets_px=None):
    """Return an FPA-sized cloud field sampled from a padded source field."""
    if cloud_field is None:
        return None

    pad_y, pad_x = float(pad_px[0]), float(pad_px[1])
    offset_y, offset_x = float(offset_px[0]), float(offset_px[1])
    if layer_offsets_px is None:
        layer_offsets = [[offset_y, offset_x] for _ in cloud_field.layers]
    else:
        layer_offsets = [[float(offset[0]), float(offset[1])] for offset in layer_offsets_px]
        if len(layer_offsets) != len(cloud_field.layers):
            raise ValueError('layer_offsets_px length must match the number of cloud layers.')

    layers = []
    clamped = False
    crop_starts = []
    for layer, layer_offset in zip(cloud_field.layers, layer_offsets):
        start_y = pad_y - layer_offset[0]
        start_x = pad_x - layer_offset[1]
        crop_starts.append([start_y, start_x])
        density, layer_clamped = sample_bilinear(layer.density, start_y, start_x, height_px, width_px)
        clamped = clamped or layer_clamped
        layers.append(cloud_layer_from_density(layer.config, density))

    field = combine_cloud_layers(tuple(layers))
    field.metadata['motion'] = {
        'source_shape_px': [int(cloud_field.density.shape[0]), int(cloud_field.density.shape[1])],
        'pad_px': [int(pad_px[0]), int(pad_px[1])],
        'pad_after_px': cloud_field.metadata.get('motion', {}).get('pad_after_px', [int(pad_px[0]), int(pad_px[1])]),
        'crop_offset_px': [offset_y, offset_x],
        'crop_start_px': [pad_y - offset_y, pad_x - offset_x],
        'layer_crop_offsets_px': layer_offsets,
        'layer_crop_starts_px': crop_starts,
        'clamped': bool(clamped),
    }
    return field


def cloud_layer_from_density(config, density):
    """Build a cloud layer from normalized density and optical config."""
    density = np.asarray(density, dtype=np.float32)
    mask = density > config.mask_threshold
    tau, transmission = optical_fields(density, config)
    metadata = config.metadata()
    metadata['stats'] = {
        'density': array_stats(density),
        'mask': {'coverage_fraction': float(mask.mean())},
        'tau': array_stats(tau),
        'transmission': array_stats(transmission),
    }
    return CloudLayer(
        config=config,
        density=density,
        mask=mask,
        tau=tau,
        transmission=transmission,
        metadata=metadata,
    )


def combine_cloud_layers(layers):
    """Combine cloud layers into one stacked transmission and density field."""
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
            'density': array_stats(density),
            'mask': {'coverage_fraction': float(mask.mean())},
            'tau': array_stats(tau),
            'transmission': array_stats(transmission),
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


def optical_fields(density, config):
    """Return optical depth and transmission arrays for a cloud density field."""
    cloud = np.clip(np.asarray(density, dtype=np.float32), 0.0, 1.0)
    thickness = np.power(cloud, config.tau_gamma, dtype=np.float32)
    tau = config.tau_min + thickness * (config.tau_max - config.tau_min)
    tau = np.where(cloud > 0.0, tau * cloud, 0.0).astype(np.float32)
    transmission = np.exp(-tau).astype(np.float32)
    return tau, transmission


def sample_bilinear(values, start_y, start_x, height, width):
    """Sample a rectangular view from ``values`` using bilinear interpolation."""
    source = np.asarray(values, dtype=np.float32)
    rows = float(start_y) + np.arange(int(height), dtype=np.float32)
    cols = float(start_x) + np.arange(int(width), dtype=np.float32)
    clamped = (
        np.any(rows < 0.0) or np.any(cols < 0.0) or
        np.any(rows > source.shape[0] - 1) or np.any(cols > source.shape[1] - 1)
    )
    rows = np.clip(rows, 0.0, source.shape[0] - 1)
    cols = np.clip(cols, 0.0, source.shape[1] - 1)

    r0 = np.floor(rows).astype(np.int32)
    c0 = np.floor(cols).astype(np.int32)
    r1 = np.clip(r0 + 1, 0, source.shape[0] - 1)
    c1 = np.clip(c0 + 1, 0, source.shape[1] - 1)
    ry = (rows - r0).astype(np.float32)[:, None]
    cx = (cols - c0).astype(np.float32)[None, :]

    top = source[r0[:, None], c0[None, :]] * (1.0 - cx) + source[r0[:, None], c1[None, :]] * cx
    bottom = source[r1[:, None], c0[None, :]] * (1.0 - cx) + source[r1[:, None], c1[None, :]] * cx
    return (top * (1.0 - ry) + bottom * ry).astype(np.float32), bool(clamped)


def smoothstep(values, edge0, edge1):
    """Apply cubic Hermite interpolation between two scalar edges."""
    if edge0 == edge1:
        return (values >= edge1).astype(np.float32)
    x = np.clip((values - edge0) / (edge1 - edge0), 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def normalize_unit(values):
    """Normalize an array to the closed interval ``[0, 1]``."""
    low = float(np.min(values))
    high = float(np.max(values))
    span = high - low
    if span <= 1e-12:
        return np.zeros_like(values)
    return (values - low) / span


def array_stats(values):
    """Return JSON-serializable min, max, and mean statistics for an array."""
    array = np.asarray(values, dtype=np.float64)
    return {
        'min': float(np.min(array)),
        'max': float(np.max(array)),
        'mean': float(np.mean(array)),
    }


def _as_cloud_array(value, shape):
    if hasattr(value, 'numpy'):
        value = value.numpy()
    array = np.asarray(value, dtype=np.float32)
    if array.shape == ():
        return np.full(shape, float(array), dtype=np.float32)
    return np.broadcast_to(array, shape).astype(np.float32)


def _source_value_and_metadata(value):
    if isinstance(value, dict):
        return value.get('pe'), value.get('metadata') or {}
    return value, {}


def _cloud_source_gain(source_name, metadata, height_m):
    if source_name == 'lunar' and metadata:
        return _lunar_direct_source_gain(metadata, height_m)
    if source_name == 'solar' and metadata:
        return _solar_direct_source_gain(metadata, height_m)
    gain, scale_height_m = SOURCE_BRIGHTENING[source_name]
    return gain * math.exp(-height_m / scale_height_m)


def _lunar_direct_source_gain(metadata, height_m):
    moon_el = float(metadata.get('moon_el', 0.0))
    if moon_el <= 0.0:
        return 0.0

    phase_fraction = _moon_illumination_fraction(metadata.get('phase_angle'))
    if phase_fraction <= 0.0:
        return 0.0

    elevation_gain = math.sqrt(math.sin(math.radians(min(moon_el, 90.0))))
    phase_gain = math.sqrt(phase_fraction)
    height_gain = math.exp(-height_m / LUNAR_DIRECT_BRIGHTENING['scale_height_m'])
    return LUNAR_DIRECT_BRIGHTENING['gain'] * height_gain * elevation_gain * phase_gain


def _moon_illumination_fraction(phase_angle_deg):
    if phase_angle_deg is None:
        return 1.0
    phase_rad = math.radians(float(phase_angle_deg))
    return max(0.0, min(1.0, 0.5 * (1.0 + math.cos(phase_rad))))


def _solar_direct_source_gain(metadata, height_m):
    sun_el = float(metadata.get('sun_el', -90.0))
    min_twilight_el = SOLAR_DIRECT_BRIGHTENING['twilight_min_sun_el_deg']
    if sun_el <= min_twilight_el:
        return 0.0

    if sun_el >= 0.0:
        elevation_gain = max(
            0.35,
            math.sqrt(math.sin(math.radians(min(sun_el, 90.0)))),
        )
    else:
        twilight_t = (sun_el - min_twilight_el) / abs(min_twilight_el)
        elevation_gain = smoothstep(twilight_t, 0.0, 1.0)

    height_gain = math.exp(-height_m / SOLAR_DIRECT_BRIGHTENING['scale_height_m'])
    return SOLAR_DIRECT_BRIGHTENING['gain'] * height_gain * elevation_gain


def _cloud_source_height_m(config):
    if config.altitude is not None:
        return float(config.altitude) * 1000.0
    return float(config.cloud_range) * 1000.0
