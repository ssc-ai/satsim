from datetime import timezone

import numpy as np
from skyfield.api import load
from skyfield.timelib import Time

_ts = load.timescale(builtin=True)


class TimeWrapper(Time):
    """ This class implements the `__hash__` function for the Skyfield `Time`. """

    def __init__(self, other):
        """ Constructor.
        
        Args:
            other: `Time`, the Skyfield `Time` object to wrap
        """
        self.ts = other.ts
        self.whole = other.whole
        self.tt_fraction = other.tt_fraction
        self.shape = getattr(self.tt, 'shape', ())

    def __hash__(self):
        """ Custom hash based on `whole` and `tt_fraction`. """
        return hash((self.whole, self.tt_fraction))


def utc(year, month, day, hour, minute, seconds):
    """ Generate a Skyfield `Time` object based on UTC time.
    
    Args:
        year: `int`, UTC year
        month: `int`, UTC month of year
        day: `int`, UTC day of month
        hour: `int`, UTC hour
        minute: `int`, UTC minute
        seconds: `float`, UTC seconds
    
    Returns:
        A Skyfield `Time` object
    """
    return TimeWrapper(_ts.utc(year, month, day, hour, minute, seconds))


def utc_from_list(t_list, delta_sec=0):
    """ Generate a Skyfield `Time` object based on UTC time in array form.
    
    Args:
        t_list: `list`, UTC as [year, month, day, hour, minute, seconds]
        delta_sec: `float`, seconds to add
    
    Returns:
        A Skyfield `Time` object
    """
    return utc(t_list[0], t_list[1], t_list[2], t_list[3], t_list[4], t_list[5] + delta_sec)


def utc_from_list_or_scalar(t, delta_sec=0, default_t=None):
    """ Generate a Skyfield `Time` object based on a UTC time in array or scalar form. Scalar form requires
    `default_t` to be defined.
    
    Args:
        t: `list or float`, UTC as [year, month, day, hour, minute, seconds] or delta seconds from `default_t`
        delta_sec: `float`, seconds to add
        default_t: `list`, UTC as [year, month, day, hour, minute, seconds], required if `t` is None or a scalar
    
    Returns:
        A Skyfield `Time` object
    """
    if isinstance(t, list):
        return utc_from_list(t, delta_sec)
    elif t is None:
        return utc_from_list(default_t, delta_sec)
    else:
        return utc_from_list(default_t, t + delta_sec)


def to_utc_list(t):
    """ Convert a Skyfield `Time` object to UTC in array form.

    Args:
        t: `Time`, time to convert to UTC

    Returns:
        A `list` with UTC values for year, month, day, hour, minutes, seconds
    """
    tu = t.utc
    return [int(tu[0]), int(tu[1]), int(tu[2]), int(tu[3]), int(tu[4]), float(tu[5])]


def from_datetime(dt, utc=False):
    if utc:
        dt = dt.replace(tzinfo=timezone.utc)

    return _ts.from_datetime(dt)


def to_astropy(t):
    """ Convert a Skyfield `Time` object to an AstroPy `Time`.

    Args:
        t: `Time`, a Skyfield `Time`

    Returns:
        An AstroPy `Time` object
    """
    from astropy import time
    return time.Time(t.whole, t.tt_fraction, format='jd', scale='tt').utc


def from_astropy(t):
    """ Convert an AstroPy `Time` to Skyfield `Time`.

    Args:
        t: `Time`, an AstroPy `Time`

    Returns:
        A Skyfield `Time` object
    """
    return _ts.from_astropy(t)


def linspace(t0, t1, num=50):
    """ Return evenly spaced Skyfield times over a specified interval.

    Args:
        t0: `Time`, start Skyfield `Time`
        t1: `Time`, end Skyfield `Time`
        num: `int`, number of samples to generate. default=50

    Returns:
        A `list` of Skyfield times equally spaced between `t0` and `t1`.
    """
    whole0 = t0.whole
    frac0 = t0.tt_fraction
    whole1 = t1.whole
    frac1 = t1.tt_fraction
    return Time(
        _ts,
        np.linspace(whole0, whole1, num),
        np.linspace(frac0, frac1, num),
    )


def delta_sec(t0, t1):
    """ Subtract t0 and t1. (t0 - t1)

    Returns:
        A `float` in seconds.
    """
    return (t0 - t1) * 86400


def mid(t0, t1):
    """ Return the mid of two times.

    Args:
        t0: `Time`, start Skyfield `Time`
        t1: `Time`, end Skyfield `Time`

    Returns:
        A `Time`, mid Skyfield `Time`
    """
    d = t1 - t0

    return t0 + d * 0.5