"""Cloud data models."""

import math
from dataclasses import dataclass

import numpy as np

from satsim.clouds.constants import DEFAULT_CLOUD_RANGE_M


@dataclass(frozen=True)
class CloudGeometry:
    """Cloud-plane sampling geometry derived from the sensor FOV."""

    height_px: int
    width_px: int
    y_fov_deg: float
    x_fov_deg: float
    cloud_range_m: float = DEFAULT_CLOUD_RANGE_M
    y_center_offset_m: float = 0.0
    x_center_offset_m: float = 0.0

    @property
    def x_meters_per_pixel(self):
        """Return horizontal cloud-plane meters per image pixel."""
        return self.cloud_range_m * math.radians(self.x_fov_deg) / float(self.width_px)

    @property
    def y_meters_per_pixel(self):
        """Return vertical cloud-plane meters per image pixel."""
        return self.cloud_range_m * math.radians(self.y_fov_deg) / float(self.height_px)

    @property
    def meters_per_pixel(self):
        """Return the mean cloud-plane meters per image pixel."""
        return 0.5 * (self.x_meters_per_pixel + self.y_meters_per_pixel)

    @property
    def footprint_width_m(self):
        """Return the cloud-plane footprint width in meters."""
        return self.width_px * self.x_meters_per_pixel

    @property
    def footprint_height_m(self):
        """Return the cloud-plane footprint height in meters."""
        return self.height_px * self.y_meters_per_pixel


@dataclass(frozen=True)
class CloudLayerConfig:
    """Validated generation and optical settings for one cloud layer."""

    name: str
    cloud_type: str
    seed: int
    texture_seed: int
    coverage: float
    feature_scales_m: tuple
    density_edge_width: float
    density_floor: float
    brightness: float
    cloud_range: float
    altitude: float
    wind_speed: float
    wind_direction: float
    texture_contrast: float
    locality_degree: int
    tau_min: float
    tau_max: float
    tau_gamma: float
    mask_threshold: float
    min_feature_scale_px: float
    amplitude_decay: float

    def metadata(self):
        """Return JSON-serializable layer settings for run metadata."""
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
            'range': float(self.cloud_range),
            'altitude': None if self.altitude is None else float(self.altitude),
            'wind_speed': float(self.wind_speed),
            'wind_direction': float(self.wind_direction),
            'texture_contrast': float(self.texture_contrast),
            'locality_degree': int(self.locality_degree),
            'tau_min': float(self.tau_min),
            'tau_max': float(self.tau_max),
            'tau_gamma': float(self.tau_gamma),
        }


@dataclass(frozen=True)
class CloudLayer:
    """Generated density, mask, optical, and metadata arrays for one layer."""

    config: CloudLayerConfig
    density: np.ndarray
    mask: np.ndarray
    tau: np.ndarray
    transmission: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class CloudField:
    """Combined cloud stack with per-layer arrays and aggregate metadata."""

    layers: tuple
    density: np.ndarray
    mask: np.ndarray
    tau: np.ndarray
    transmission: np.ndarray
    metadata: dict
