from __future__ import division, print_function, absolute_import

import numpy as np


def constant(x, t, ssp=None, value=0.0):

    return np.full_like(x, value)


def sin_add(x, t, ssp=None, freq=1, mag_scale=1):

    t = _avg_t(t)
    return (np.sin(t * freq * 2 * np.pi) + 1) * 0.5 * mag_scale + x


def sin(x, t, ssp=None, freq=1, mag_scale=1):

    t = _avg_t(t)
    return (np.sin(t * freq * 2 * np.pi) + 1) * 0.5 * mag_scale * x


def cos(x, t, ssp=None, freq=1, mag_scale=1):

    t = _avg_t(t)
    return (np.cos(t * freq * 2 * np.pi) + 1) * 0.5 * mag_scale * x


def poly(x, t=None, ssp=None, coef=[]):

    return np.poly1d(coef)(x)


def polyt(x, t, ssp=None, coef=[]):

    return np.poly1d(coef)(_avg_t(t)) + x


def glint(x, t, ssp=None, period=0.0, magnitude=0):

    dt = np.floor(np.array(t) / period)

    for i, (t0, t1) in enumerate(zip(dt[0:-1], dt[1:])):
        if t1 - t0 != 0.0:
            x[i] = magnitude

    return x


def _delta_t(t):

    t = np.array(t)
    return t[1:] - t[0:-1]


def _avg_t(t):

    t = np.array(t)
    return (t[0:-1] + t[1:]) / 2
