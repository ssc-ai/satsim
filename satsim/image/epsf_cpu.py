from __future__ import division, print_function, absolute_import

import numpy as np
from numba import njit


@njit(cache=True)
def add_epsf_stamps(image, base_r, base_c, phase_r, phase_c, values, lut):
    """Add ``values[i] * lut[phase_r[i], phase_c[i]]`` centered on ``(base_r[i], base_c[i])``.

    ``image`` is a float64 ``[H, W]`` array updated in place. ``lut`` is a
    ``[P, P, k, k]`` phase table with odd ``k``; stamp offsets run from
    ``-(k // 2)`` to ``k // 2``, as in ``add_epsf_counts``. Stamp pixels outside
    ``image`` are dropped. Each product is formed in float64 and added in index
    order, so the result is deterministic for given inputs.
    """
    kernel_size = lut.shape[2]
    half = kernel_size // 2
    height, width = image.shape
    for i in range(values.shape[0]):
        value = np.float64(values[i])
        top = base_r[i] - half
        left = base_c[i] - half
        x0 = max(0, -left)
        x1 = min(kernel_size, width - left)
        stamp = lut[phase_r[i], phase_c[i]]
        for y in range(max(0, -top), min(kernel_size, height - top)):
            # Contiguous row views let the compiler vectorize the inner loop.
            image_row = image[top + y, left + x0:left + x1]
            stamp_row = stamp[y, x0:x1]
            for x in range(x1 - x0):
                image_row[x] += value * np.float64(stamp_row[x])
