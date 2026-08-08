from __future__ import division, print_function, absolute_import

import json
import os
from datetime import timezone


def format_ob_time(value):
    """Format a datetime or Skyfield time as UTC with fixed microseconds."""
    if hasattr(value, 'utc_datetime'):
        value = value.utc_datetime()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.strftime('%Y-%m-%dT%H:%M:%S.%fZ')


def save(dir_name, frame_num, obs_list):
    """Save analytical observations to disk.

    Args:
        dir_name: `str`, directory where the observation set is stored.
        frame_num: `int`, frame number being processed.
        obs_list: `list`, list of observations as returned by
            `satsim.geometry.analytic_obs.generate`.
    """
    obs_dir = os.path.join(dir_name, 'AnalyticalObservations')
    if not os.path.exists(obs_dir):
        os.makedirs(obs_dir, exist_ok=True)
    set_name = os.path.basename(dir_name)
    file_name = '{}.{:04d}.json'.format(set_name, frame_num)
    with open(os.path.join(obs_dir, file_name), 'w') as jf:
        json.dump(obs_list, jf)
