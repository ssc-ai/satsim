import numpy as np

from satsim.geometry.astrometric import load_earth, load_sun


R_EARTH_KM = 6378.137
R_SUN_KM = 695700.0


def _as_matrix(v):
    """Ensure vectors have shape (3, n)."""
    v = np.asarray(v)
    if v.ndim == 1:
        return v.reshape(3, 1)
    return v


def _shadow_params_from_vectors(v_ts, v_te):
    """Compute angular parameters for shadow classification from vectors.

    Args:
        v_ts: array-like, vectors from target to Sun (km), shape (3,) or (3,n)
        v_te: array-like, vectors from target to Earth center (km), shape (3,) or (3,n)

    Returns:
        (alpha, beta, theta): tuple of numpy arrays (radians), each shape (n,)
    """
    v_ts = _as_matrix(v_ts)
    v_te = _as_matrix(v_te)

    d_ts = np.linalg.norm(v_ts, axis=0)
    d_te = np.linalg.norm(v_te, axis=0)

    # Angular radii as seen from target
    alpha = np.arcsin(np.clip(R_SUN_KM / d_ts, -1.0, 1.0))
    beta = np.arcsin(np.clip(R_EARTH_KM / d_te, -1.0, 1.0))

    # Angle between Sun and Earth directions as seen from target
    dot = np.sum(v_ts * v_te, axis=0)
    cos_theta = np.clip(dot / (d_ts * d_te), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    return alpha, beta, theta


def earth_shadow_umbra_mask(target, t):
    """Return a mask (1 = lit, 0 = umbra) for target at time(s) t.

    Uses Sun–Earth–target geometry. A target is in umbra if the Earth's
    apparent angular radius (beta) is greater than the Sun's (alpha) and the
    angular separation between their directions (theta) satisfies theta <= beta - alpha.

    Args:
        target: Skyfield body for the target (e.g., SGP4, ephemeris, etc.)
        t: Skyfield Time (scalar or vector)

    Returns:
        numpy.ndarray of shape (n,) with values 0.0 (umbra) or 1.0 (lit)
    """
    earth = load_earth()
    sun = load_sun()

    # Vectors from target to Sun and Earth
    v_ts = (sun - target).at(t).position.km
    v_te = (earth - target).at(t).position.km

    alpha, beta, theta = _shadow_params_from_vectors(v_ts, v_te)

    in_umbra = (beta > alpha) & (theta <= (beta - alpha))
    mask = np.where(in_umbra, 0.0, 1.0)

    return mask
