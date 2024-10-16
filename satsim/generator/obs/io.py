from __future__ import division, print_function, absolute_import

import numpy as np


def tle_file(filename, lines=3, mv=[10.0, 16.0]):
    """Generates SGP4 objects from a tle file.

    Args:
        filename: `str`, path to file
        lines: `int`, number of lines per object. 2 or 3.
        mv: `array`, min and max brightness from uniform distribution

    Example usage in SatSim configuration::

        "obs": {
            "$generator": {
                "module": "satsim.generator.obs.io",
                "function": "tle_file",
                "kwargs": {
                    "filename": "./tle.txt",
                    "lines": 2,
                    "mv": [10.0, 15.0]
                }
            }
        }
    """
    tle = []
    with open(filename) as f:
        while True:

            if lines == 3:
                f.readline()

            tle1 = f.readline()
            tle2 = f.readline()

            if not tle1 or not tle2:
                break

            tle.append({
                "mode": "tle",
                "tle1": tle1,
                "tle2": tle2,
                "mv": np.random.uniform(mv[0], mv[1])
            })

    obs = {
        "mode": "list",
        "list": tle
    }

    return obs
