"""Construct SatSim observers and targets from geometry configuration nodes."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Tuple

import astropy.units as u
import numpy as np

from satsim import time
from satsim.geometry.astrometric import create_topocentric
from satsim.geometry.ephemeris import create_ephemeris_object
from satsim.geometry.sgp4 import create_sgp4
from satsim.geometry.twobody import create_twobody


def tle_lines_from_config(config: Mapping[str, Any]) -> Optional[Tuple[str, str]]:
    """Return a validated TLE pair from a SatSim geometry node, if present."""
    if 'tle' in config:
        tle = config['tle']
        if (not isinstance(tle, (list, tuple)) or len(tle) != 2 or
                not all(isinstance(line, str) and line for line in tle)):
            raise ValueError('tle must contain two non-empty strings.')
        return tle[0], tle[1]
    if 'tle1' in config or 'tle2' in config:
        tle1 = config.get('tle1')
        tle2 = config.get('tle2')
        if not (isinstance(tle1, str) and tle1 and isinstance(tle2, str) and tle2):
            raise ValueError('tle1 and tle2 must both be non-empty strings.')
        return tle1, tle2
    return None


def create_observer_from_config(site: Mapping[str, Any]):
    """Create a ground or space-based observer from ``geometry.site``."""
    tle_lines = tle_lines_from_config(site)
    if tle_lines is not None:
        return create_sgp4(*tle_lines)
    return create_topocentric(
        site.get('lat', 0.0),
        site.get('lon', 0.0),
        float(site.get('alt', 0.0)),
    )


def create_target_from_config(
    entry: Mapping[str, Any],
    default_time: Optional[Sequence[Any]] = None,
):
    """Create a ranging-capable target from a ``geometry.obs.list`` entry.

    TLE, two-body, and ephemeris targets are supported. Angles-only
    observations return ``None`` because they do not define range or range rate.
    """
    if default_time is None:
        default_time = [2020, 1, 1, 0, 0, 0.0]

    mode = entry.get('mode')
    tle_lines = tle_lines_from_config(entry)
    if mode == 'tle' or (mode is None and tle_lines is not None):
        if tle_lines is None:
            raise ValueError('TLE target requires tle or tle1/tle2.')
        return create_sgp4(*tle_lines)

    if mode == 'twobody':
        epoch = time.utc_from_list_or_scalar(
            entry.get('epoch'),
            default_t=list(default_time),
        )
        position = np.asarray(entry['position']) * u.km
        velocity = np.asarray(entry['velocity']) * u.km / u.s
        return create_twobody(position, velocity, epoch)

    if mode == 'ephemeris':
        epoch = time.utc_from_list_or_scalar(
            entry.get('epoch'),
            default_t=list(default_time),
        )
        return create_ephemeris_object(
            entry['positions'],
            entry['velocities'],
            entry['seconds_from_epoch'],
            epoch,
        )

    if mode == 'observation':
        return None
    raise ValueError('Unsupported ranging target mode: {!r}.'.format(mode))


def target_id_from_config(entry: Mapping[str, Any]) -> Any:
    """Return an explicit target ID or derive the NORAD number from a TLE."""
    if 'id' in entry:
        return entry['id']
    tle_lines = tle_lines_from_config(entry)
    if tle_lines is None:
        raise ValueError('Non-TLE targets require an explicit id.')
    try:
        return int(tle_lines[0][2:7])
    except (TypeError, ValueError) as exc:
        raise ValueError('TLE has an invalid NORAD ID.') from exc


__all__ = [
    'create_observer_from_config',
    'create_target_from_config',
    'target_id_from_config',
    'tle_lines_from_config',
]
