from __future__ import division, print_function, absolute_import

import math
import numbers


def finite_number(name, value):
    if isinstance(value, bool):
        raise ValueError('{} must be a finite number.'.format(name))
    if not isinstance(value, numbers.Real):
        raise ValueError('{} must be a finite number.'.format(name))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('{} must be a finite number.'.format(name))
    return value


def optional_finite_number(name, value):
    if value is None:
        return None
    return finite_number(name, value)


def positive_number(name, value):
    value = finite_number(name, value)
    if value <= 0.0:
        raise ValueError('{} must be positive.'.format(name))
    return value


def nonnegative_number(name, value):
    value = finite_number(name, value)
    if value < 0.0:
        raise ValueError('{} must be nonnegative.'.format(name))
    return value


def integer(name, value):
    value = finite_number(name, value)
    if not value.is_integer():
        raise ValueError('{} must be an integer.'.format(name))
    return int(value)


def positive_integer(name, value):
    value = integer(name, value)
    if value <= 0:
        raise ValueError('{} must be positive.'.format(name))
    return value


def unit_interval(name, value):
    value = finite_number(name, value)
    if value < 0.0 or value > 1.0:
        raise ValueError('{} must be in the range [0, 1].'.format(name))
    return value


def optional_unit_interval(name, value):
    if value is None:
        return None
    return unit_interval(name, value)
