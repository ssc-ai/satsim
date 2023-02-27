import math
import numpy as np
from scipy.interpolate import lagrange as scipy_lagrange


def _lagrange(x, y, new_x, order=5, normalize_x=True):

    if len(x) < order + 2:
        x_crop = x
        n_min = 0
        n_max = len(x)
    else:
        x_end = len(x)
        o_div_2 = (order + 2) / 2
        n = np.searchsorted(x, new_x, side='right')

        n_min = n - math.ceil(o_div_2)
        n_max = n + math.floor(o_div_2)

        if n_min < 0:
            n_max = n_max - n_min
            n_min = 0

        if n_max > x_end:
            n_min = n_min - (n_max - x_end)
            n_max = x_end

        x_crop = x[n_min:n_max]

    if normalize_x:
        x_offset = x[n_min]
        x_crop = np.array(x_crop) - x_offset
        new_x = new_x - x_offset

    if np.ndim(y) == 1:
        y_crop = y[n_min:n_max]
        f = scipy_lagrange(x_crop, y_crop)
        res = f(new_x)
    else:
        y_crop = y[n_min:n_max, :]
        res = np.zeros_like(y[0])
        for i in range(len(res)):
            f = scipy_lagrange(x_crop, y_crop[:,i])
            res[i] = f(new_x)

    return res


def _segment_boundary_lagrange(x, y, new_x, order=5):

    # extrapolate before first x
    if new_x <= x[0][0]:
        return lagrange(x[0], y[0], new_x, order)

    # extrapolate after last x
    elif new_x >= x[-1][-1]:
        return lagrange(x[-1], y[-1], new_x, order)

    # find segment and interpolate
    else:
        for i in range(len(x)):
            if new_x >= x[i][0] and new_x <= x[i][-1]:
                return lagrange(x[i], y[i], new_x, order)


def lagrange(x, y, new_x, order=5):
    """Lagrange interpolator. `x` can be one dimensional or two dimensional (if segment
    boundaries are specified). `y` should have the same length as `x` and the coordinates can
    be multi-dimensional, for example, cartesian coordinates [[x0, y0, z0], [x1, y1, z1], ...].

    Args:
        x: `array_like`, x represents the x-coordinates of a set of data points.
        y: `array_like`, y represents the y-coordinates of a set of data points, i.e., f(x).
        new_x: `float` or `array_like`, x-coordinates to interpolate y-coordinates.
        order: `int`, Lagrange polynomial order. Nearest `order + 2` points to be used for fit.

    Returns:
        A `float` or `array_like`, `new_x` Lagrange interpolated y-coordinates.
    """
    # no segment boundaries
    if np.ndim(x[0]) == 0:
        func = _lagrange
    # segment boundaries
    elif np.ndim(x[0]) == 1:
        func = _segment_boundary_lagrange
    # invalid
    else:
        return None

    if np.ndim(new_x) == 0:
        return func(x, y, new_x, order)
    else:
        return np.array([func(x, y, next_x, order) for next_x in new_x])
