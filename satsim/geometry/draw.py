from __future__ import division, print_function, absolute_import

import numpy as np
from skimage.draw import line, bezier_curve

from satsim.image.coordinates import normalized_to_oversampled

def _rasterize_position(value):
    """Return the pixel center containing a continuous center coordinate."""
    return int(np.floor(float(value) + 0.5))


def gen_line(height, width, origin, velocity, pe, t_start, t_end):
    """Generates a line segment in pixel coordinates based on a starting point
    (`t = 0`), velocity, start time, and end time. Counts, `pe`, are
    spread evenly across continuous time bins. Typically used to inject a
    moving target onto an oversampled image.

    Examples::

        (rr, cc, pe, t) = gen_line(5120, 5120, [.5,.5], [10,5], 1000, 1.0, 1.5)

    Args:
        height: `int`, image height in number of pixels.
        width: `int`, image width in number of pixels.
        origin: `[float, float]`, normalized starting point on image in
            [row,col]. Values are normalized image-edge coordinates: `[0,0]`
            is the top-left edge, `[1,1]` is the bottom-right edge, and `0.5`
            is the geometric center. The corresponding zero-based pixel-center
            coordinate is ``origin * size - 0.5``.
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
    r0 = normalized_to_oversampled(origin[0], height)
    c0 = normalized_to_oversampled(origin[1], width)

    r1 = r0 + velocity[0] * t_start
    c1 = c0 + velocity[1] * t_start

    r2 = r0 + velocity[0] * t_end
    c2 = c0 + velocity[1] * t_end

    return gen_line_from_endpoints(r1, c1, r2, c2, pe, t_start, t_end)


def gen_line_from_endpoints(r0, c0, r1, c1, pe, t_start, t_end):
    """Generates a line segment in pixel coordinates based on a starting point
    at `t_start` and an ending point at `t_end`. Counts, `pe`, are spread
    evenly across continuous time bins. Typically used to inject a moving
    target onto an oversampled image.

    Flux-bearing positions are the mean continuous position in each time bin,
    so their weighted centroid is the exact exposure midpoint. Exact start and
    end positions are included as zero-duration, zero-flux control samples for
    trajectory annotations.

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
    r0i = _rasterize_position(r0)
    c0i = _rasterize_position(c0)
    r1i = _rasterize_position(r1)
    c1i = _rasterize_position(c1)
    rr_pixels, _ = line(r0i, c0i, r1i, c1i)
    n = len(rr_pixels)

    if r0 == r1 and c0 == c1:
        return (
            np.asarray([r0], dtype=np.float64),
            np.asarray([c0], dtype=np.float64),
            np.asarray([(t_end - t_start) * pe]),
            np.asarray([t_start, t_end]),
        )

    # Use the rasterized line only to choose a sampling density. Depositing
    # its integer pixel centers would quantize the trajectory and can move its
    # centroid by almost a full oversampled pixel. Each returned point is the
    # mean continuous position over the matching time bin, so equal-flux bins
    # reproduce the exact exposure centroid even when the complete motion is
    # subpixel.
    u = (np.arange(n, dtype=np.float64) + 0.5) / n
    rr_flux = r0 + (r1 - r0) * u
    cc_flux = c0 + (c1 - c0) * u

    # Retain exact zero-flux endpoints for trajectory annotations. Duplicate
    # time boundaries keep callable brightness models from assigning flux to
    # those control samples.
    rr = np.concatenate(([r0], rr_flux, [r1]))
    cc = np.concatenate(([c0], cc_flux, [c1]))
    counts = np.concatenate((
        [0.0],
        np.full(n, (t_end - t_start) * pe / n),
        [0.0],
    ))
    t_flux = np.linspace(t_start, t_end, n + 1)
    t = np.concatenate(([t_start], t_flux, [t_end]))

    return (rr, cc, counts, t)


def gen_curve_from_points(r0, c0, r1, c1, r2, c2, pe, t_start, t_end):
    """Generates a bezier curve in pixel coordinates based on a starting point
    at `t_start`, an ending point at `t_end`, and a mid point at `(t_start + t_end) / 2`.
    Counts, `pe`, are spread evenly across continuous time bins. Typically used
    to inject a moving target onto an oversampled image.

    Flux-bearing positions are the mean continuous position in each time bin,
    which preserves the exact quadratic exposure centroid. Exact start, middle,
    and end positions are included as zero-duration, zero-flux control samples
    for trajectory annotations.

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
    r0i = _rasterize_position(r0)
    c0i = _rasterize_position(c0)
    rbi = _rasterize_position(rb)
    cbi = _rasterize_position(cb)
    r2i = _rasterize_position(r2)
    c2i = _rasterize_position(c2)
    rr_pixels, _ = bezier_curve(r0i, c0i, rbi, cbi, r2i, c2i, 1.0, None)
    n = len(rr_pixels)

    if r0 == r1 == r2 and c0 == c1 == c2:
        return (
            np.asarray([r0], dtype=np.float64),
            np.asarray([c0], dtype=np.float64),
            np.asarray([(t_end - t_start) * pe]),
            np.asarray([t_start, t_end]),
        )

    # Use an even number of flux bins so the exact mid-exposure control point
    # can be retained as a zero-duration annotation sample between bins.
    if n % 2 != 0:
        n += 1

    # As with lines, rasterization determines only how many samples are
    # needed. Integrate the original continuous quadratic over each matching
    # time bin instead of adding fractional offsets back to rasterized pixels:
    # the latter loses fractional curvature and can move its centroid in the
    # wrong direction.
    u0 = np.arange(n, dtype=np.float64) / n
    u1 = np.arange(1, n + 1, dtype=np.float64) / n
    mean_u = (u0 + u1) / 2.0
    mean_u2 = (u0**2 + u0 * u1 + u1**2) / 3.0

    ar = r0 - 2.0 * rb + r2
    br = 2.0 * (rb - r0)
    ac = c0 - 2.0 * cb + c2
    bc = 2.0 * (cb - c0)
    rr_flux = ar * mean_u2 + br * mean_u + r0
    cc_flux = ac * mean_u2 + bc * mean_u + c0

    mid = n // 2
    rr = np.concatenate(([r0], rr_flux[:mid], [r1], rr_flux[mid:], [r2]))
    cc = np.concatenate(([c0], cc_flux[:mid], [c1], cc_flux[mid:], [c2]))
    counts_flux = np.full(n, (t_end - t_start) * pe / n)
    counts = np.concatenate(([0.0], counts_flux[:mid], [0.0], counts_flux[mid:], [0.0]))

    t_flux = np.linspace(t_start, t_end, n + 1)
    t = np.concatenate((
        [t_start],
        t_flux[:mid + 1],
        [t_flux[mid]],
        t_flux[mid + 1:],
        [t_end],
    ))

    return (rr, cc, counts, t)
