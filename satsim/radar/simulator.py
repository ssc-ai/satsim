from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import astropy.units as u
import numpy as np

from satsim import time
from satsim.config import transform, load_json, save_debug
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.twobody import create_twobody
from satsim.geometry.ephemeris import create_ephemeris_object
from satsim.geometry.astrometric import create_topocentric, get_los
from satsim.io.analytical import save as save_observations
from .monostatic import (
    RadarParams,
    in_fov,
    in_range_limits,
    detect,
    range_rate,
    range_unc,
)

logger = logging.getLogger(__name__)


def _format_ob_time(t) -> str:
    """Format a Skyfield time as an ISO-8601 UTC timestamp with microseconds."""
    return t.utc_datetime().strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def _parse_radar_params(ssp: Dict[str, Any]) -> RadarParams:
    """Map a SatSim document into :class:`~satsim.radar.monostatic.RadarParams`."""
    rc = ssp['radar']
    det = rc.get('detection', {})
    fov = rc.get('field_of_view', {})
    rlim = rc.get('range_limits')
    # Derive a sensor identifier from config if not provided
    site = ssp.get('geometry', {}).get('site', {})
    site_name = site.get('name') or site.get('track', {}).get('name')
    p = RadarParams(
        tx_power=float(rc['tx_power']),
        tx_frequency=float(rc['tx_frequency']),
        antenna_diameter=float(rc.get('antenna_diameter', 0.0)),
        efficiency=float(rc.get('efficiency', 1.0)),
        min_detectable_power=det.get('min_detectable_power', None),
        snr_threshold=det.get('snr_threshold', None),
        angle_error=float(det.get('angle_error', 0.0)),
        range_error=float(det.get('range_error', 0.0)),  # km
        range_rate_error=float(det.get('range_rate_error', 0.0)),  # km/s
        false_alarm_rate=float(det.get('false_alarm_rate', 0.0)),
        az_limits=tuple(fov['azimuth']) if 'azimuth' in fov else None,
        el_limits=tuple(fov['elevation']) if 'elevation' in fov else None,
        range_limits=tuple(rlim) if rlim is not None else None,
        dwell=float(rc.get('time', {}).get('dwell', 1.0)),
        num_frames=int(rc.get('num_frames', 1)),
        sensor_id=rc.get('idSensor') or rc.get('sensor_id') or rc.get('id') or rc.get('name') or site_name,
    )
    return p


def _build_observer(ssp: Dict[str, Any]):
    """Create the observing platform (ground site or space-borne observer)."""
    site = ssp.get('geometry', {}).get('site', {})
    if 'tle' in site:
        return create_sgp4(site['tle'][0], site['tle'][1])
    if 'tle1' in site:
        return create_sgp4(site['tle1'], site['tle2'])
    lat = site.get('lat', 0.0)
    lon = site.get('lon', 0.0)
    alt = float(site.get('alt', 0.0))
    return create_topocentric(lat, lon, alt)


def _build_target(entry: Dict[str, Any], default_t: Optional[List[Any]] = None):
    """Create a target object from a geometry obs entry.

    Supported modes:
    - ``tle``: SGP4 propagation
    - ``twobody``: two-body state vector propagation (position [km], velocity [km/s])
    - ``ephemeris``: interpolated positions/velocities (positions [km], velocities [km/s])

    ``observation`` is angles-only and is not supported for radar ranging.
    """
    if default_t is None:
        default_t = [2020, 1, 1, 0, 0, 0.0]

    mode = entry.get('mode')

    if mode == 'tle' or (mode is None and ('tle' in entry or 'tle1' in entry)):
        if 'tle' in entry:
            return create_sgp4(entry['tle'][0], entry['tle'][1])
        return create_sgp4(entry['tle1'], entry['tle2'])

    # Alias "statevector" for backwards compatibility.
    if mode in {'twobody', 'statevector'}:
        epoch = time.utc_from_list_or_scalar(entry.get('epoch'), default_t=default_t)
        position = np.array(entry['position']) * u.km
        velocity = np.array(entry['velocity']) * u.km / u.s
        return create_twobody(position, velocity, epoch)

    if mode == 'ephemeris':
        epoch = time.utc_from_list_or_scalar(entry.get('epoch'), default_t=default_t)
        return create_ephemeris_object(
            entry['positions'],
            entry['velocities'],
            entry['seconds_from_epoch'],
            epoch,
        )

    # observation angles-only not supported here for radar ranging
    if mode == 'observation':
        return None

    return None


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
    # Prepare output directory (match EO timestamped folder naming)
    from datetime import datetime
    set_dir = os.path.join(output_dir, datetime.now().isoformat().replace(':', '-'))
    os.makedirs(set_dir, exist_ok=True)

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

    # Per-frame loop
    for frame_idx in range(rp.num_frames):
        t_mid = time.utc_from_list(tt, delta_sec=frame_idx * rp.dwell + 0.5 * rp.dwell)

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
            rcs = float(o.get('rcs', 1.0))
            detected, snr = detect(rp, rcs, rng)
            if not detected:
                continue

            # Range-rate (km/s)
            rr_val = range_rate(observer, target, t_mid, dt=min(1.0, max(rp.dwell, 1e-3)))

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
                'obTime': _format_ob_time(t_mid),
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
            }
            if 'name' in o and o['name']:
                entry['idOnOrbit'] = o['name']
            if 'id' in o and o['id']:
                entry['satNo'] = o['id']
            if rp.sensor_id:
                entry['idSensor'] = rp.sensor_id

            # Append
            frame_measurements.append(entry)

        # Save observations for frame (legacy location)
        save_observations(set_dir, frame_idx, frame_measurements)

    return set_dir


def simulate_from_file(config_file: str, output_dir: str = './') -> str:
    """Load, transform ($sample/$import/$ref), and run the radar simulator."""
    if config_file.endswith('.json'):
        ssp = load_json(config_file)
    else:
        from satsim.config import load_yaml
        ssp = load_yaml(config_file)
    # Transform (evaluate $sample/$import/$ref) with input dir context and keep debug stages
    ssp_t, stages = transform(ssp, os.path.dirname(config_file), with_debug=True)
    run_dir = simulate(ssp_t, output_dir)
    # Save config passes to match EO output structure
    save_debug(stages, run_dir)
    return run_dir
