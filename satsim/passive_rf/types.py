"""Normalized passive RF configuration types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class ReceiverParams:
    """Receiver parameters from an aligned ``passive_rf`` entry."""

    name: str
    antenna_gain_dbi: float
    noise_figure_db: float
    system_losses_db: float = 0.0
    antenna_noise_temperature_k: float = 290.0
    az_limits: Optional[Tuple[float, float]] = None
    el_limits: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class TargetRFParams:
    """RF emission properties associated with a SatSim geometry target."""

    target_id: int
    frequency_hz: float
    bandwidth_hz: float
    eirp_dbw: float
    name: Optional[str] = None
    band: Optional[str] = None


@dataclass(frozen=True)
class EstimatorParams:
    integration_time_s: float = 1.0
    coherent_time_s: float = 0.005
    rms_bw_from_rect: bool = True
    caf_loss_db: float = 3.0
    tdoa_floor_s: float = 0.0
    fdoa_floor_hz: float = 0.0
    tdoa_scale: float = 1.0
    fdoa_scale: float = 1.0
    tdoa_floor_by_sensor: Mapping[str, float] = field(default_factory=dict)
    fdoa_floor_by_sensor: Mapping[str, float] = field(default_factory=dict)
    tdoa_scale_by_sensor: Mapping[str, float] = field(default_factory=dict)
    fdoa_scale_by_sensor: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class ObservationRequest:
    target_id: int
    sensor1: str
    sensor2: str
    time: Any
    occurrence: int = 0
    source_index: int = 0


@dataclass(frozen=True)
class ReceiverGeometry:
    """Resolved geometric receiver-to-target state at one epoch."""

    range_m: float
    range_rate_m_s: float
    azimuth_deg: float
    elevation_deg: float


@dataclass(frozen=True)
class PassiveRFRunConfig:
    receivers: Mapping[str, ReceiverParams]
    observers: Mapping[str, Any]
    targets: Mapping[int, TargetRFParams]
    target_objects: Mapping[int, Any]
    estimator: EstimatorParams
    observations: Tuple[ObservationRequest, ...]
    frame_times: Tuple[Any, ...] = ()
    seed: int = 42


__all__ = [
    'EstimatorParams',
    'ObservationRequest',
    'PassiveRFRunConfig',
    'ReceiverGeometry',
    'ReceiverParams',
    'TargetRFParams',
]
