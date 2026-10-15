import numpy as np
from satsim.geometry.astrometric import angle_between, distance_between
from satsim.geometry.astrometric import load_sun


def interpolate_magnitudes(evaluate, offsets, max_interval_seconds=1.0, tolerance_mag=1e-4):
    """Sample a smooth magnitude model with checked piecewise quadratics.

    ``evaluate`` accepts second offsets and returns magnitudes. Quarter-point
    residuals guide subdivision; they are an accuracy check, not a rigorous
    bound for arbitrary functions. Fall back to exact samples for nonfinite
    knots, exhausted sample budgets, or intervals that cannot be refined.
    """
    offsets = np.asarray(offsets, dtype=float)
    n = len(offsets)
    if n <= 5 or offsets[-1] == offsets[0]:
        return evaluate(offsets)
    span = offsets[-1] - offsets[0]
    # Avoid allocating a knot grid larger than the requested sample array.
    if span / max_interval_seconds >= (n - 1) / 4:
        return evaluate(offsets)
    segments = max(1, int(np.ceil(span / max_interval_seconds)))
    nodes = np.linspace(offsets[0], offsets[-1], 4 * segments + 1)
    values = np.asarray(evaluate(nodes))
    used = len(nodes)
    if not np.all(np.isfinite(values)):
        return evaluate(offsets)
    left, right = nodes[:-1:4], nodes[4::4]
    y0, ym, y1 = values[:-1:4], values[2::4], values[4::4]
    quarter = np.stack([values[1::4], values[3::4]], axis=1)
    depth = 0
    result = np.empty(n, dtype=float)
    while len(left):
        predicted = np.stack([.375 * y0 + .75 * ym - .125 * y1,
                              -.125 * y0 + .75 * ym + .375 * y1], axis=1)
        accepted = np.max(np.abs(quarter - predicted), axis=1) <= tolerance_mag
        for i in np.flatnonzero(accepted):
            start = np.searchsorted(offsets, left[i], side='left')
            stop = np.searchsorted(offsets, right[i], side='right')
            u = (offsets[start:stop] - left[i]) / (right[i] - left[i])
            result[start:stop] = (
                2 * (y0[i] - 2 * ym[i] + y1[i]) * u**2
                + (4 * ym[i] - 3 * y0[i] - y1[i]) * u + y0[i])
        if np.all(accepted):
            return result
        bad = ~accepted
        midpoint = (left[bad] + right[bad]) / 2
        if depth >= 12 or used + 4 * np.count_nonzero(bad) >= n or np.any(
                (midpoint == left[bad]) | (midpoint == right[bad])):
            return evaluate(offsets)
        # The checked quarter points become the two child midpoints.
        left, right = np.concatenate([left[bad], midpoint]), np.concatenate([midpoint, right[bad]])
        y0, ym, y1 = (np.concatenate([y0[bad], ym[bad]]),
                      np.concatenate([quarter[bad, 0], quarter[bad, 1]]),
                      np.concatenate([ym[bad], y1[bad]]))
        check_times = left[:, None] + (right - left)[:, None] * np.array([.25, .75])
        quarter = np.asarray(evaluate(check_times.ravel())).reshape(-1, 2)
        used += check_times.size
        if not np.all(np.isfinite(quarter)):
            return evaluate(offsets)
        depth += 1
    return result


def lambertian_sphere_to_mv(albedo, distance, radius, phase_angle):
    """Applies lambertian sphere approximation to convert target brightness
    to visual magnitudes based on sun brightness of -26.74.

    Args:
        albedo: `float`, The ratio of reflected light to incident light
            off of the object's surface
        distance: `float`, distance to object in meters
        radius: `float`, radius of sphere in meters
        phase_angle: `float`, the angle between observer, object, and sun in degrees

    Returns:
        A `float`, calculated visual magnitude
    """
    phase_angle = np.deg2rad(phase_angle)

    mv_sun = -26.74

    # Lambertian sphere approximation.
    phase_factor = np.sin(phase_angle) + (np.pi - phase_angle) * np.cos(phase_angle)
    intensity = phase_factor * (2 * albedo * (radius * radius)) / (3 * np.pi * (distance * distance))

    # Convert intensities to magnitudes
    mvVector = mv_sun - 2.5 * np.log10(intensity)

    return mvVector


def model_to_mv(observer, target, model, time):
    """Calculates the phase angle between the observer, target, and sun.

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        time: `object`, Skyfield time

    Returns:
        A `float`, the phase angle in degrees
    """
    if model['mode'] == 'lambertian_sphere':
        return lambertian_sphere_model_to_mv(observer, target, model, time)
    else:
        return None


def lambertian_sphere_model_to_mv(observer, target, model, time):
    """Calculates the phase angle between the observer, target, and sun.

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        sun: `object`, Skyfield sun object
        time: `object`, Skyfield time

    Returns:
        A `float`, the phase angle in degrees
    """
    sun = load_sun()
    distance = model['distance'] * 1000.0 if 'distance' in model else distance_between(target, observer, time) * 1000.0
    phase_angle = model['phase_angle'] if 'phase_angle' in model else angle_between(target, observer, sun, time)
    diameter = model['diameter'] if 'diameter' in model else model['size']

    return lambertian_sphere_to_mv(model['albedo'], distance, diameter * 0.5, phase_angle)
