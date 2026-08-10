"""Pure passive RF link-budget and measurement-uncertainty functions."""

from __future__ import annotations

import math
from typing import Mapping, Tuple

from .types import EstimatorParams, ReceiverParams, TargetRFParams


C_M_S = 299_792_458.0
K_B = 1.380649e-23
T0_K = 290.0


def db_to_lin(x_db: float) -> float:
    return 10.0 ** (float(x_db) / 10.0)


def lin_to_db(x: float) -> float:
    if not math.isfinite(x) or x <= 0.0:
        raise ValueError('Linear power ratio must be positive and finite.')
    return 10.0 * math.log10(x)


def fspl_db(frequency_hz: float, range_m: float) -> float:
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError('Frequency must be positive and finite.')
    if not math.isfinite(range_m) or range_m <= 0.0:
        raise ValueError('Range must be positive and finite.')
    return 20.0 * math.log10(4.0 * math.pi * range_m * frequency_hz / C_M_S)


def noise_power_dbw(
    bandwidth_hz: float,
    noise_figure_db: float,
    antenna_noise_temperature_k: float = T0_K,
) -> float:
    if not math.isfinite(bandwidth_hz) or bandwidth_hz <= 0.0:
        raise ValueError('Bandwidth must be positive and finite.')
    if (not math.isfinite(antenna_noise_temperature_k) or
            antenna_noise_temperature_k <= 0.0):
        raise ValueError('Antenna noise temperature must be positive and finite.')
    noise_factor = db_to_lin(noise_figure_db)
    system_temperature = (
        antenna_noise_temperature_k + (noise_factor - 1.0) * T0_K
    )
    if system_temperature <= 0.0:
        raise ValueError('System noise temperature must be positive.')
    return (
        lin_to_db(K_B) + lin_to_db(system_temperature) +
        lin_to_db(bandwidth_hz)
    )


def processing_gain_db(bandwidth_hz: float, coherent_time_s: float) -> float:
    if bandwidth_hz <= 0.0 or coherent_time_s <= 0.0:
        raise ValueError('Bandwidth and coherent time must be positive.')
    return 10.0 * math.log10(bandwidth_hz * coherent_time_s)


def effective_snr_db(snr1_db: float, snr2_db: float) -> float:
    gain1 = db_to_lin(snr1_db)
    gain2 = db_to_lin(snr2_db)
    effective = (gain1 * gain2) / (gain1 + gain2)
    return lin_to_db(effective)


def receiver_snr_db(
    target: TargetRFParams,
    receiver: ReceiverParams,
    range_m: float,
) -> float:
    received_power_dbw = (
        target.eirp_dbw + receiver.antenna_gain_dbi -
        fspl_db(target.frequency_hz, range_m) - receiver.system_losses_db
    )
    noise_dbw = noise_power_dbw(
        target.bandwidth_hz,
        receiver.noise_figure_db,
        receiver.antenna_noise_temperature_k,
    )
    return received_power_dbw - noise_dbw


def tdoa_fdoa_uncertainties(
    bandwidth_hz: float,
    processed_snr_db: float,
    integration_time_s: float,
    use_rectangular_rms: bool = True,
) -> Tuple[float, float]:
    if bandwidth_hz <= 0.0 or integration_time_s <= 0.0:
        raise ValueError('Bandwidth and integration time must be positive.')
    processed_snr = max(db_to_lin(processed_snr_db), 1e-12)
    beta = (
        bandwidth_hz / (2.0 * math.sqrt(3.0))
        if use_rectangular_rms else 0.35 * bandwidth_hz
    )
    sigma_tdoa_s = 1.0 / (
        2.0 * math.pi * beta * math.sqrt(2.0 * processed_snr)
    )
    sigma_fdoa_hz = 1.0 / (
        2.0 * math.pi * math.sqrt(2.0 * processed_snr) * integration_time_s
    )
    return sigma_tdoa_s, sigma_fdoa_hz


def combine_sensor_value(
    global_value: float,
    values_by_sensor: Mapping[str, float],
    sensor1: str,
    sensor2: str,
) -> float:
    values = [
        values_by_sensor[name]
        for name in (sensor1, sensor2)
        if name in values_by_sensor
    ]
    return max(values) if values else global_value


def measurement_uncertainties(
    target: TargetRFParams,
    estimator: EstimatorParams,
    sensor1: str,
    sensor2: str,
    processed_snr_db: float,
) -> Tuple[float, float]:
    sigma_tdoa_s, sigma_fdoa_hz = tdoa_fdoa_uncertainties(
        target.bandwidth_hz,
        processed_snr_db,
        estimator.integration_time_s,
        estimator.rms_bw_from_rect,
    )
    tdoa_scale = combine_sensor_value(
        estimator.tdoa_scale,
        estimator.tdoa_scale_by_sensor,
        sensor1,
        sensor2,
    )
    tdoa_floor = combine_sensor_value(
        estimator.tdoa_floor_s,
        estimator.tdoa_floor_by_sensor,
        sensor1,
        sensor2,
    )
    fdoa_scale = combine_sensor_value(
        estimator.fdoa_scale,
        estimator.fdoa_scale_by_sensor,
        sensor1,
        sensor2,
    )
    fdoa_floor = combine_sensor_value(
        estimator.fdoa_floor_hz,
        estimator.fdoa_floor_by_sensor,
        sensor1,
        sensor2,
    )
    return (
        math.sqrt((tdoa_scale * sigma_tdoa_s) ** 2 + tdoa_floor ** 2),
        math.sqrt((fdoa_scale * sigma_fdoa_hz) ** 2 + fdoa_floor ** 2),
    )


__all__ = [
    'C_M_S',
    'K_B',
    'T0_K',
    'combine_sensor_value',
    'db_to_lin',
    'effective_snr_db',
    'fspl_db',
    'lin_to_db',
    'measurement_uncertainties',
    'noise_power_dbw',
    'processing_gain_db',
    'receiver_snr_db',
    'tdoa_fdoa_uncertainties',
]
