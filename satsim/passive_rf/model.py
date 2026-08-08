"""Passive RF TDOA/FDOA measurement model."""

from __future__ import annotations

import hashlib
import random
from typing import Any, Dict, List, Mapping, Optional

from satsim.geometry.astrometric import get_los, range_rate_from_los
from satsim.geometry.fov import in_fov
from satsim.io.analytical import format_ob_time

from .core import (
    C_M_S,
    db_to_lin,
    effective_snr_db,
    measurement_uncertainties,
    processing_gain_db,
    receiver_snr_db,
)
from .types import (
    EstimatorParams,
    ObservationRequest,
    PassiveRFRunConfig,
    ReceiverGeometry,
    ReceiverParams,
    TargetRFParams,
)


def receiver_geometry(observer, target, epoch) -> ReceiverGeometry:
    """Resolve geometric range, range rate, azimuth, and elevation via SatSim."""
    _, _, range_km, azimuth, elevation, icrf_los = get_los(
        observer,
        target,
        epoch,
        deflection=False,
        aberration=False,
        stellar_aberration=False,
    )
    range_m = float(range_km) * 1000.0
    if range_m <= 0.0:
        raise ValueError('Computed receiver range must be positive.')
    return ReceiverGeometry(
        range_m=range_m,
        range_rate_m_s=range_rate_from_los(icrf_los) * 1000.0,
        azimuth_deg=float(azimuth),
        elevation_deg=float(elevation),
    )


def stable_measurement_seed(run_seed: int, request: ObservationRequest) -> int:
    """Derive a stable seed, including duplicate occurrence identity."""
    identity = '|'.join((
        str(run_seed),
        str(request.target_id),
        request.sensor1,
        request.sensor2,
        format_ob_time(request.time),
        str(request.occurrence),
    ))
    return int.from_bytes(
        hashlib.sha256(identity.encode('utf-8')).digest()[:8],
        byteorder='big',
        signed=False,
    )


def simulate_measurement(
    target: TargetRFParams,
    receiver1: ReceiverParams,
    receiver2: ReceiverParams,
    estimator: EstimatorParams,
    epoch,
    rng: random.Random,
    geometry1: ReceiverGeometry,
    geometry2: ReceiverGeometry,
) -> Dict[str, Any]:
    """Evaluate one resolved receiver-pair measurement.

    Target and receiver states are evaluated geometrically at one common UTC
    reception time. File, network, wall-clock, and global RNG access remain
    outside this kernel.
    """
    range1_m = geometry1.range_m
    range_rate1_m_s = geometry1.range_rate_m_s
    range2_m = geometry2.range_m
    range_rate2_m_s = geometry2.range_rate_m_s

    tdoa_true_s = (range2_m - range1_m) / C_M_S
    # SatSim fixes the pair convention to frequency * d(TDOA)/dt.
    fdoa_true_hz = (
        target.frequency_hz * (range_rate2_m_s - range_rate1_m_s) / C_M_S
    )

    snr1_db = receiver_snr_db(target, receiver1, range1_m)
    snr2_db = receiver_snr_db(target, receiver2, range2_m)
    raw_snr_db = effective_snr_db(snr1_db, snr2_db)
    coherent_interval = min(
        estimator.integration_time_s,
        estimator.coherent_time_s,
    )
    processed_snr_db = (
        raw_snr_db +
        processing_gain_db(target.bandwidth_hz, coherent_interval) -
        estimator.caf_loss_db
    )

    sigma_tdoa_s, sigma_fdoa_hz = measurement_uncertainties(
        target,
        estimator,
        receiver1.name,
        receiver2.name,
        processed_snr_db,
    )
    measured_tdoa_s = tdoa_true_s + rng.gauss(0.0, sigma_tdoa_s)
    measured_fdoa_hz = fdoa_true_hz + rng.gauss(0.0, sigma_fdoa_hz)

    record = {
        'obTime': format_ob_time(epoch),
        'idSensor1': receiver1.name,
        'idSensor2': receiver2.name,
        'satNo': target.target_id,
        'frequency': target.frequency_hz,
        'bandwidth': target.bandwidth_hz,
        'snr': float(db_to_lin(processed_snr_db)),
        'snrRawDb': float(raw_snr_db),
        'snrProcessedDb': float(processed_snr_db),
        'tdoaTrue': float(tdoa_true_s),
        'fdoaTrue': float(fdoa_true_hz),
        'tdoa': float(measured_tdoa_s),
        'tdoaUnc': float(sigma_tdoa_s),
        'fdoa': float(measured_fdoa_hz),
        'fdoaUnc': float(sigma_fdoa_hz),
        'type': 'PASSIVE_RF',
        'uct': False,
        'createdBy': 'satsim',
    }
    if target.name:
        record['idOnOrbit'] = target.name
    if target.band:
        record['band'] = target.band
    return record


def simulate_run_config(
    config: PassiveRFRunConfig,
    target_objects: Optional[Mapping[int, Any]] = None,
    observer_objects: Optional[Mapping[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Run a validated native SatSim passive RF configuration."""
    if target_objects is None:
        target_objects = config.target_objects
    if observer_objects is None:
        observer_objects = config.observers

    requests = sorted(
        config.observations,
        key=lambda request: (float(request.time.tt), request.source_index),
    )
    geometry_cache = {}

    def geometry_for(request, receiver_id):
        key = (request.target_id, receiver_id, request.time)
        if key not in geometry_cache:
            geometry_cache[key] = receiver_geometry(
                observer_objects[receiver_id],
                target_objects[request.target_id],
                request.time,
            )
        return geometry_cache[key]

    results = []
    for request in requests:
        target = config.targets[request.target_id]
        receivers = (
            config.receivers[request.sensor1],
            config.receivers[request.sensor2],
        )
        geometries = (
            geometry_for(request, request.sensor1),
            geometry_for(request, request.sensor2),
        )
        visible = True
        for receiver, geometry in zip(receivers, geometries):
            if receiver.az_limits is None and receiver.el_limits is None:
                continue
            if not in_fov(
                geometry.azimuth_deg,
                geometry.elevation_deg,
                receiver.az_limits,
                receiver.el_limits,
            ):
                visible = False
                break
        if not visible:
            continue
        rng = random.Random(stable_measurement_seed(config.seed, request))
        results.append(simulate_measurement(
            target,
            receivers[0],
            receivers[1],
            config.estimator,
            request.time,
            rng,
            geometries[0],
            geometries[1],
        ))
    return results


__all__ = [
    'receiver_geometry',
    'simulate_measurement',
    'simulate_run_config',
    'stable_measurement_seed',
]
