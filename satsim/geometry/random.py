from __future__ import division, print_function, absolute_import

import numpy as np

from satsim.math.random import gen_samples_from_bins


def gen_random_points(height, width, y_fov, x_fov, pe_bins, density, pad_mult=1):
    """Generates random points in pixel coordinates of random brightnesses
    based on density bins. Typically used to inject a stationary target such as
    stars onto an oversampled image.

    Examples::

        (rr, cc, pe) = gen_random_points(5120, 5120, 2, 2, [200,100,10], [30,50])

    Args:
        height: `int`, image height in number of pixels.
        width: `int`, image width in number of pixels.
        y_fov: `float`, vertical field of view in degrees
        x_fov: `float`, horizontal field of view in degrees
        pe_bins: `list`, brightness bins
        density: `list`, number of occurrences per 1 square degree for each bin
        pad_mult: `float`, pad multiplier to add to each side of image, for
            example, 2x pad will add increase the size of array to 2 * y_fov
            and 2 * x_fov to all four sides; for a total area equal to
            (5 * y_fov) * (5 * x_fov)

    Returns:
        A `tuple`, containing:
            rr: `list`, numpy list of row pixel locations
            cc: `list`, numpy list of col pixel locations
            pe: `list`, numpy list of pe counts
    """
    sfov = ((2 * pad_mult + 1) * x_fov) * ((2 * pad_mult + 1) * y_fov)

    pe = gen_samples_from_bins(pe_bins, density, sfov)
    n = len(pe)

    rr = np.random.randint(-height * pad_mult, height * (pad_mult + 1), n)
    cc = np.random.randint(-width * pad_mult, width * (pad_mult + 1), n)

    return (rr, cc, pe)
