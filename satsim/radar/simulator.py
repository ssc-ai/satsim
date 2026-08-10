from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from satsim import time
from satsim.config import load_json, load_yaml, save_debug, save_json, transform
from satsim.geometry.astrometric import get_los
from satsim.geometry.factory import (
    create_observer_from_config,
    create_target_from_config,
    target_id_from_config,
)
from satsim.io.analytical import format_ob_time, save as save_observations
from satsim.util.validation import (
    finite_number,
    nonnegative_number,
    positive_integer,
    positive_number,
    unit_interval,
)
from .monostatic import (
    RadarParams,
    in_fov,
    in_range_limits,
    detect,
    range_rate,
    range_unc,
)

logger = logging.getLogger(__name__)


def _limits(name, value, nonnegative=False):
    """Validate an optional ordered pair of numerical limits."""
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError('{} must contain [minimum, maximum].'.format(name))
    validator = nonnegative_number if nonnegative else finite_number
    minimum = validator('{}[0]'.format(name), value[0])
    maximum = validator('{}[1]'.format(name), value[1])
    if minimum > maximum:
        raise ValueError('{} minimum must not exceed maximum.'.format(name))
    return minimum, maximum


def _parse_radar_params(ssp: Dict[str, Any]) -> RadarParams:
    """Map a SatSim document into :class:`~satsim.radar.monostatic.RadarParams`."""
    if positive_integer('sim.samples', ssp.get('sim', {}).get('samples', 1)) != 1:
        raise ValueError('RADAR requires sim.samples == 1 in v0.26.0.')
    rc = ssp.get('radar')
    if not isinstance(rc, dict):
        raise ValueError('Configuration requires a radar object.')
    det = rc.get('detection', {})
    fov = rc.get('field_of_view', {})
    timing = rc.get('time', {})
    if not isinstance(det, dict):
        raise ValueError('radar.detection must be an object.')
    if not isinstance(fov, dict):
        raise ValueError('radar.field_of_view must be an object.')
    if not isinstance(timing, dict):
        raise ValueError('radar.time must be an object.')
    rlim = rc.get('range_limits')
    # Derive a sensor identifier from config if not provided
    site = ssp.get('geometry', {}).get('site', {})
    if not isinstance(site, dict):
        raise ValueError('RADAR requires geometry.site to be an object.')
    site_name = site.get('name') or site.get('track', {}).get('name')
    min_detectable_power = det.get('min_detectable_power')
    if min_detectable_power is not None:
        min_detectable_power = positive_number(
            'radar.detection.min_detectable_power',
            min_detectable_power,
        )
    snr_threshold = det.get('snr_threshold')
    if snr_threshold is not None:
        snr_threshold = nonnegative_number(
            'radar.detection.snr_threshold',
            snr_threshold,
        )
    p = RadarParams(
        tx_power=positive_number('radar.tx_power', rc.get('tx_power')),
        tx_frequency=positive_number('radar.tx_frequency', rc.get('tx_frequency')),
        antenna_diameter=nonnegative_number(
            'radar.antenna_diameter', rc.get('antenna_diameter', 0.0)
        ),
        efficiency=unit_interval('radar.efficiency', rc.get('efficiency', 1.0)),
        min_detectable_power=min_detectable_power,
        snr_threshold=snr_threshold,
        angle_error=nonnegative_number(
            'radar.detection.angle_error', det.get('angle_error', 0.0)
        ),
        range_error=nonnegative_number(
            'radar.detection.range_error', det.get('range_error', 0.0)
        ),
        range_rate_error=nonnegative_number(
            'radar.detection.range_rate_error',
            det.get('range_rate_error', 0.0),
        ),
        false_alarm_rate=unit_interval(
            'radar.detection.false_alarm_rate',
            det.get('false_alarm_rate', 0.0),
        ),
        az_limits=_limits('radar.field_of_view.azimuth', fov.get('azimuth')),
        el_limits=_limits('radar.field_of_view.elevation', fov.get('elevation')),
        range_limits=_limits('radar.range_limits', rlim, nonnegative=True),
        dwell=positive_number('radar.time.dwell', timing.get('dwell', 1.0)),
        gap=nonnegative_number(
            'radar.time.gap',
            timing.get('gap', 0.0),
        ),
        num_frames=positive_integer('radar.num_frames', rc.get('num_frames', 1)),
        sensor_id=rc.get('idSensor') or rc.get('sensor_id') or rc.get('id') or rc.get('name') or site_name,
    )
    return p


def _build_observer(ssp: Dict[str, Any]):
    """Create the observing platform (ground site or space-borne observer)."""
    return create_observer_from_config(ssp.get('geometry', {}).get('site', {}))


def _build_target(entry: Dict[str, Any], default_t: Optional[List[Any]] = None):
    """Create a ranging-capable target using the shared SatSim factory."""
    target_config = entry
    if entry.get('mode') == 'statevector':
        target_config = dict(entry)
        target_config['mode'] = 'twobody'
    return create_target_from_config(target_config, default_time=default_t)


def simulate(ssp: Dict[str, Any], output_dir: str = './') -> str:
    """Simulate analytical radar measurements and save per-frame JSON outputs.

    Observation units follow SatSim/UDL conventions:
    - azimuth/elevation: degrees
    - range: kilometers
    - rangeRate: kilometers per second
    - doppler: meters per second (line-of-sight velocity; ``doppler == rangeRate * 1000``)

    Gaussian measurement noise is applied using the 1-sigma values in
    ``radar.detection``. Only detections are emitted (no false alarms yet).

    Returns:
        The output directory used for this run.
    """
    # Parse radar params
    rp = _parse_radar_params(ssp)

    # Time setup
    tt = ssp.get('geometry', {}).get('time', [2020, 1, 1, 0, 0, 0.0])

    # Observer
    observer = _build_observer(ssp)

    # Targets
    obs_cfg = ssp.get('geometry', {}).get('obs', {})
    obs_list = obs_cfg.get('list', [])
    if isinstance(obs_list, dict):
        obs_list = [obs_list]

    targets: List[Tuple[Any, Dict[str, Any]]] = []
    for o in obs_list:
        target = _build_target(o, default_t=tt)
        if target is None:
            continue
        targets.append((target, o))

    # Match EO timestamped folder naming, after configuration validation.
    from datetime import datetime
    set_dir = os.path.join(output_dir, datetime.now().isoformat().replace(':', '-'))
    os.makedirs(set_dir, exist_ok=False)

    # Per-frame loop
    for frame_idx in range(rp.num_frames):
        t_mid = time.utc_from_list(
            tt,
            delta_sec=frame_idx * (rp.dwell + rp.gap) + 0.5 * rp.dwell,
        )

        frame_measurements: List[Dict[str, Any]] = []
        for target, o in targets:
            try:
                _, _, rng, az, el, _ = get_los(
                    observer,
                    target,
                    t_mid,
                    deflection=False,
                    aberration=False,
                    stellar_aberration=False,
                )
            except Exception:
                logger.exception("Radar LOS computation failed for target.")
                continue

            # LOS/FOV and range bounds
            if not in_fov(az, el, rp):
                continue
            if not in_range_limits(rng, rp):
                continue

            # Detection check
            rcs = positive_number('radar target rcs', o.get('rcs', 1.0))
            detected, snr = detect(rp, rcs, rng)
            if not detected:
                continue

            # Range-rate (km/s)
            rr_val = range_rate(observer, target, t_mid)

            # Apply measurement noise
            az_m = az + np.random.normal(scale=rp.angle_error)
            el_m = el + np.random.normal(scale=rp.angle_error)
            # rng is km; keep units consistent in km
            r_m = rng + np.random.normal(scale=rp.range_error)
            # range_rate is km/s
            rr_m = rr_val + np.random.normal(scale=rp.range_rate_error)
            # doppler is line-of-sight velocity in m/s (not Hz)
            dop_mps = rr_m * 1000.0

            entry = {
                'obTime': format_ob_time(t_mid),
                'type': 'RADAR',
                'azimuth': float(az_m),
                'elevation': float(el_m),
                'azimuthUnc': float(rp.angle_error),
                'elevationUnc': float(rp.angle_error),
                'range': float(r_m),       # km
                'rangeRate': float(rr_m),  # km/s
                'rangeRateUnc': float(abs(rp.range_rate_error)),  # km/s
                'doppler': float(dop_mps),  # m/s
                'rangeUnc': float(range_unc(rp)),  # km
                'dopplerUnc': float(abs(rp.range_rate_error) * 1000.0),  # m/s
                'uct': False,
                'snr': float(snr) if snr is not None else None,
                'rcs': float(rcs),
                'createdBy': 'satsim',
            }
            if 'name' in o and o['name']:
                entry['idOnOrbit'] = o['name']
            try:
                target_id = target_id_from_config(o)
            except ValueError:
                target_id = None
            if target_id is not None and target_id != '':
                entry['satNo'] = target_id
            if rp.sensor_id:
                entry['idSensor'] = rp.sensor_id

            # Append
            frame_measurements.append(entry)

        # Save observations in the standard analytical location.
        save_observations(set_dir, frame_idx, frame_measurements)

    save_json(os.path.join(set_dir, 'config.json'), ssp)
    return set_dir


def simulate_from_file(config_file: str, output_dir: str = './') -> str:
    """Load, transform ($sample/$import/$ref), and run the radar simulator."""
    config_file_lower = config_file.lower()
    if config_file_lower.endswith('.json'):
        ssp = load_json(config_file)
    elif config_file_lower.endswith(('.yml', '.yaml')):
        ssp = load_yaml(config_file)
    else:
        raise ValueError('Config file must be JSON or YAML.')
    # Transform (evaluate $sample/$import/$ref) with input dir context and keep debug stages
    input_dir = os.path.dirname(os.path.abspath(config_file))
    ssp_t, stages = transform(ssp, input_dir, with_debug=True)
    run_dir = simulate(ssp_t, output_dir)
    # Save config passes to match EO output structure
    save_debug(stages, run_dir)
    return run_dir
