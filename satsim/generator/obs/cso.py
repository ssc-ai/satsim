from __future__ import division, print_function, absolute_import

import numpy as np


def one(origin=[0.5, 0.5], velocity=[0.0, 0.0], mv=[15.0, 16.0], separation=0.01):
    """Generates two objects that are spaced by `separation`.

    Args:
        origin: `array`, row and column coordinate in normalized image space
        velocity: `array`, row and column velocity in pixel per second
        mv: `array`, brightness of object 1 and object 2
        separation: `float`, distance between object in normalized image space

    Example usage in SatSim configuration::

        "obs": {
            "mode": "list",
            "sample": "random.list",
            "length":
                { "sample": "random.randint", "low": 0, "high": 15 },
            "list": {
                "$generator": {
                    "module": "satsim.generator.obs.cso",
                    "function": "one",
                    "kwargs": {
                        "origin": [
                            { "sample": "random.uniform", "low": 0.05, "high": 0.95 },
                            { "sample": "random.uniform", "low": 0.05, "high": 0.95 }
                        ],
                        "velocity": [
                            { "sample": "random.uniform", "low": -1.0, "high": 1.0 },
                            { "sample": "random.uniform", "low": -1.0, "high": 1.0 }
                        ],
                        "mv": [
                            { "sample": "random.uniform", "low": 12.5, "high": 15.5 },
                            { "sample": "random.uniform", "low": 12.5, "high": 15.5 }
                        ],
                        "separation":
                            { "sample": "random.uniform", "low": -0.02, "high": 0.02 }
                    }
                }
            }
        }
    """
    line1 = {
        "mode": "line",
        "origin": origin,
        "velocity": velocity,
        "mv": mv[0]
    }

    norm = np.linalg.norm(velocity)

    if norm != 0:
        ortho = (np.array([-velocity[1], velocity[0]]) / np.linalg.norm(velocity)) * separation
    else:
        ortho = [0, separation]

    line2 = {
        "mode": "line",
        "origin": [origin[0] + ortho[0], origin[1] + ortho[1]],
        "velocity": velocity,
        "mv": mv[1]
    }

    return [line1, line2]
