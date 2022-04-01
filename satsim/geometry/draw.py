from __future__ import division, print_function, absolute_import

import logging

import numpy as np
from skimage.draw import line, bezier_curve

logger = logging.getLogger(__name__)


def gen_line(height, width, origin, velocity, pe, t_start, t_end):
    """Generates a line segment in pixel coordinates based on a starting point
    (`t = 0`), velocity, start time, and end time. Counts, `pe`, is
    spread evenly across each pixel. Typically used to inject a moving target
    onto an oversampled image.

    Examples::

        (rr, cc, pe, t) = gen_line(5120, 5120, [.5,.5], [10,5], 1000, 1.0, 1.5)

    Args:
        height: `int`, image height in number of pixels.
        width: `int`, image width in number of pixels.
        origin: `[float, float]`, normalized starting point on image in
            [row,col] where `[0,0]` represents the top left corner and `[1,1]`
            the bottom right corner
        velocity: `[float,float]`, velocity in pixel/sec in `[row,col]` order
        pe: `float`, brightness in pe/sec
        t_start: `float`, start time in seconds from epoch (`t=0`)
        t_end: `float`, end time in seconds from epoch (`t=0`)

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            pe: `list`, list of counts (e.g. photoelectrons)
            t: `list`, list of start and stop times. length is +1 larger than `rr`, `cc`, and `pe`.
    """
    r0 = height * origin[0]
    c0 = width * origin[1]

    r1 = r0 + velocity[0] * t_start
    c1 = c0 + velocity[1] * t_start

    r2 = r0 + velocity[0] * t_end
    c2 = c0 + velocity[1] * t_end

    rr, cc = line(int(r1), int(c1), int(r2), int(c2))
    n = len(rr)
    t = np.linspace(t_start, t_end, n + 1)

    pe = (t_end - t_start) * pe / n

    return (rr, cc, np.asarray([pe for i in range(len(rr))]), t)


def gen_line_from_endpoints(r0, c0, r1, c1, pe, t_start, t_end):
    """Generates a line segment in pixel coordinates based on a starting point
    at `t_start` and an ending point at `t_end`. Counts, `pe`, is
    spread evenly across each pixel. Typically used to inject a moving target
    onto an oversampled image.

    Args:
        r0: `float`, starting row.
        c0: `float`, starting column.
        r1: `float`, ending row.
        c1: `float`, ending column.
        pe: `float`, brightness in pe/sec
        t_start: `float`, start time
        t_end: `float`, end time

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            pe: `list`, list of counts (e.g. photoelectrons)
            t: `list`, list of start and stop times. length is +1 larger than `rr`, `cc`, and `pe`.
    """
    rr, cc = line(int(r0), int(c0), int(r1), int(c1))
    n = len(rr)
    t = np.linspace(t_start, t_end, n + 1)

    pe = (t_end - t_start) * pe / n

    return (rr, cc, np.asarray([pe for i in range(len(rr))]), t)


def gen_curve_from_points(r0, c0, r1, c1, r2, c2, pe, t_start, t_end):
    """Generates a bezier curve in pixel coordinates based on a starting point
    at `t_start`, an ending point at `t_end`, and a mid point at `(t_start + t_end) / 2`.
    Counts, `pe`, is spread evenly across each pixel. Typically used to inject a moving target
    onto an oversampled image.

    Args:
        r0: `float`, starting row.
        c0: `float`, starting column.
        r1: `float`, mid row.
        c1: `float`, mid column.
        r2: `float`, ending row.
        c2: `float`, ending column.
        pe: `float`, brightness in pe/sec
        t_start: `float`, start time
        t_end: `float`, end time

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            pe: `list`, list of counts (e.g. photoelectrons)
            t: `list`, list of start and stop times. length is +1 larger than `rr`, `cc`, and `pe`.
    """
    rb = 2 * r1 - r0 / 2 - r2 / 2
    cb = 2 * c1 - c0 / 2 - c2 / 2
    r0, c0, rb, cb, r2, c2 = int(r0), int(c0), int(rb), int(cb), int(r2), int(c2)
    rr, cc = bezier_curve(r0, c0, rb, cb, r2, c2, 1.0, None)
    n = len(rr)
    t = np.linspace(t_start, t_end, n + 1)

    pe = (t_end - t_start) * pe / n

    # sort
    n2 = int(n / 2)
    if r0 == rr[0] and c0 == cc[0] and r2 == rr[-1] and c2 == cc[-1]:
        pass
    elif r0 == rr[-1] and c0 == cc[-1] and r2 == rr[0] and c2 == cc[0]:
        rr = np.flip(rr)
        cc = np.flip(cc)
    elif r0 == rr[0] and c0 == cc[0]:
        n2 = np.where(((r2, c2) == np.stack((rr,cc), axis=-1)).all(axis=-1))[0][0]
        rr = np.concatenate((rr[0:n2], np.flip(rr[n2:])))
        cc = np.concatenate((cc[0:n2], np.flip(cc[n2:])))
    elif r2 == rr[0] and c2 == cc[0]:
        rr = np.flip(rr)
        cc = np.flip(cc)
        n2 = np.where(((r0, c0) == np.stack((rr,cc), axis=-1)).all(axis=-1))[0][0]
        rr[0:n2 + 1] = np.flip(rr[0:n2 + 1])
        cc[0:n2 + 1] = np.flip(cc[0:n2 + 1])
    else:
        logger.debug('Bezier curve order unknown for: {}, {}, {}, {}, {}, {}'.format(r0, c0, r1, c1, r2, c2))

    return (rr, cc, np.asarray([pe for i in range(len(rr))]), t)
