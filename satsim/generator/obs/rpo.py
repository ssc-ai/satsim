from __future__ import division, print_function, absolute_import

import numpy as np

from skyfield.api import EarthSatellite
from skyfield.constants import AU_KM, DAY_S

from satsim import time


def rpo_from_tle(tle, epoch, delta_distance=5, delta_position_direction='random',
                 delta_velocity=0, delta_velocity_direction='random',
                 target_mv=12.0, rpo_mv=12.0, offset=[0.0, 0.0]):
    """Generates an rpo from the input `tle`.

    Args:
        tle: `array`, an array containing the SGP4 two line element set.
        epoch: `array`, UCT time as year, month, day, hour, minute, seconds.
        delta_distance: `float`, distance to position rpo away from target in km. default=5
        delta_position_direction: `string` or `array`, "random" or [x,y,z] direction vector. default="random"
        delta_velocity: `float`, delta velocity between target and rpo in km/s. default=0
        delta_velocity_direction: `string` or `array`, "random" or [x,y,z] direction vector. default="random"
        target_mv: `float`, brightness of target before collision in visual magnitude. default=12
        rpo_mv: `float`, brightness of colliding object before collision in visual magnitude. default=12
        offset: `array`, row column offset of target position on fpa in normalized coordinates. default=[0,0]

    Example usage in SatSim configuration::

        "obs": {
            "$generator": {
                "module": "satsim.generator.obs.rpo",
                "function": "rpo_from_tle",
                "kwargs": {
                    "tle": {
                        "ref": "geometry.site.track.tle"
                    },
                    "epoch": {
                        "ref": "geometry.time"
                    },
                    "delta_distance": 1.5,
                    "delta_position_direction": [0, 1, 0],
                    "delta_velocity": 0.001,
                    "delta_velocity_direction": "random",
                    "target_mv": 12.0,
                    "rpo_mv": 15.0,
                    "offset": [
                        0.1736509022094605,
                        0.22670133968196546
                    ]
                }
            }
        }
    """

    if isinstance(delta_position_direction, str) and delta_position_direction == 'random':
        delta_position_direction = np.random.normal(size=3)
        delta_position_direction /= np.linalg.norm(delta_position_direction)

    if isinstance(delta_velocity_direction, str) and delta_velocity_direction == 'random':
        delta_velocity_direction = np.random.normal(size=3)
        delta_velocity_direction /= np.linalg.norm(delta_velocity_direction)

    sat = EarthSatellite(tle[0], tle[1])
    t = time.utc_from_list(epoch)
    epoch = time.to_utc_list(t)

    # get state vector at time t
    position0, velocity0, _, _ = sat._at(t)
    position0 *= AU_KM
    velocity0 *= AU_KM
    velocity0 /= DAY_S

    position1 = position0 + delta_position_direction * delta_distance
    velocity1 = velocity0 + delta_velocity_direction * delta_velocity

    obs = {
        "mode": "list",
        "list": []
    }

    if target_mv is not None:
        obs['list'].append({
            "mode": "twobody",
            "position": list(position0),
            "velocity": list(velocity0),
            "epoch": epoch,
            "mv": target_mv,
            "offset": offset,
        })

    if rpo_mv is not None:
        obs['list'].append({
            "mode": "twobody",
            "position": list(position1),
            "velocity": list(velocity1),
            "epoch": epoch,
            "mv": rpo_mv,
            "offset": offset,
        })

    return obs
