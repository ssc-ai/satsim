import numpy as np


def diff_degrees(a0, a1, max_diff=180):
    """ Calculates the difference between two angles and accounts for crossover at 0 degrees based on `max_diff`.

    Args:
        a0: `float`, an angle in degrees.
        a1: `float`, an angle in degrees.
        max_diff: `float`, maximum angle difference before concidering the opposite angle. default=180.

    Returns:
        A `float`, a1 - a0
    """
    ad = a1 - a0

    # increasing
    if ad > 0 and ad < max_diff:
        return ad
    # decreasing
    elif ad < 0 and ad > -max_diff:
        return ad
    # increasing
    elif ad < 0 and ad < -max_diff:
        return ad + 360
    # decreasing
    elif ad > 0 and ad > max_diff:
        return ad - 360
    # equal
    else:
        return 0


def mean_degrees(a0, a1, max_diff=180, normalize_360=True):
    """ Calculates the mean of two angles and accounts for crossover at 0 degrees based on `max_diff`.

    Args:
        a0: `float`, an angle in degrees.
        a1: `float`, an angle in degrees.
        max_diff: `float`, maximum angle difference before concidering the opposite angle. default=180.
        normalize_360: `boolean`, normalize angle between 0 and 360. default=True

    Returns:
        A `float`, mean of a0 and a1
    """
    # am = (a0 + a1) * 0.5 if math.abs(a0 - a1) < max_diff else (a0 + a1) * 0.5 - 180
    # am = am if am >= 0 else am + 360
    # return am
    d = diff_degrees(a0, a1, max_diff)
    m = (a0 + a0 + d) * 0.5

    if normalize_360:
        m = m % 360

    return m


def interp_degrees(tt, t0, t1, a0, a1, max_diff=180, normalize_360=True):
    """ Interpolate between two angles and accounts for crossover at 0 degrees based on `max_diff`.

    Args:
        tt: `list`, list of times to interpolate between `t0` and `t1`.
        t0: `float`, time for `a0`.
        t1: `float`, time for `a1`.
        a0: `float`, angle in degrees.
        a1: `float`, angle in degrees.
        max_diff: `float`, maximum angle difference before concidering the opposite angle. default=180.
        normalize_360: `boolean`, normalize angle between 0 and 360. default=True

    Returns:
        A `list`, interpolated angles at `tt`
    """
    d = diff_degrees(a0, a1, max_diff)
    aa = np.interp(tt, [t0, t1], [0, d]) + a0

    if normalize_360:
        aa = aa % 360

    return aa
