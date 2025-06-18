import numpy as np

from astropy import units as u
from astropy.time import Time
from skyfield.api import EarthSatellite
from skyfield.constants import AU_KM, DAY_S
from skyfield.vectorlib import VectorFunction

from satsim.geometry.astrometric import load_earth


DEFAULT_METHOD = 'vallado'

MU_EARTH = 398600.4418  # km^3 / s^2


def _stumpff_C(z):
    if z > 0:
        s = np.sqrt(z)
        return (1 - np.cos(s)) / z
    elif z < 0:
        s = np.sqrt(-z)
        return (np.cosh(s) - 1) / (-z)
    else:
        return 0.5


def _stumpff_S(z):
    if z > 0:
        s = np.sqrt(z)
        return (s - np.sin(s)) / s ** 3
    elif z < 0:
        s = np.sqrt(-z)
        return (np.sinh(s) - s) / s ** 3
    else:
        return 1.0 / 6.0


def _propagate_kepler(r0, v0, dt, mu=MU_EARTH, tol=1e-8, max_iter=100):
    r0 = np.asarray(r0, dtype=float)
    v0 = np.asarray(v0, dtype=float)

    r0_norm = np.linalg.norm(r0)
    v0_norm = np.linalg.norm(v0)
    vr0 = np.dot(r0, v0) / r0_norm

    alpha = 2.0 / r0_norm - v0_norm ** 2 / mu

    if abs(alpha) > tol:
        x = np.sqrt(mu) * dt * alpha
    else:
        x = np.sqrt(mu) * dt / r0_norm

    for _ in range(max_iter):
        z = alpha * x * x
        C = _stumpff_C(z)
        S = _stumpff_S(z)

        F = (
            r0_norm * vr0 / np.sqrt(mu) * x * x * C
            + (1 - alpha * r0_norm) * x ** 3 * S
            + r0_norm * x
            - np.sqrt(mu) * dt
        )

        dF = (
            r0_norm * vr0 / np.sqrt(mu) * x * (1 - z * S)
            + (1 - alpha * r0_norm) * x * x * C
            + r0_norm
        )

        ratio = F / dF
        x -= ratio
        if abs(ratio) < tol:
            break

    z = alpha * x * x
    C = _stumpff_C(z)
    S = _stumpff_S(z)

    f = 1 - x * x / r0_norm * C
    g = dt - x ** 3 * S / np.sqrt(mu)
    r = f * r0 + g * v0
    r_norm = np.linalg.norm(r)

    fdot = np.sqrt(mu) / (r_norm * r0_norm) * (z * S - 1) * x
    gdot = 1 - x * x / r_norm * C
    v = fdot * r0 + gdot * v0

    return r, v


def _to_astropy(t):

    return Time(t.tt, scale='tt', format='jd')


class EarthTwoBodySatellite(VectorFunction):
    """ Skyfield Two-body propagator Satellite type """

    center = 399

    def __init__(self, position, velocity, epoch, name='EarthTwoBodySatellite', method=DEFAULT_METHOD):
        """Construct a Two-body propagator satellite.

        Args:
            position: `array`, AstroPy position in ITRF. example: ``[-6045, -3490, 2500] * u.km``
            velocity: `array`, AstroPy velocity in ITRF. example: ``[-3.457, 6.618, 2.534] * u.km / u.s``
            epoch: `Time`, epoch state vector time as Skyfield ``Time``
            name: `string`, object name
            method: `string`, two body propagator algorithm type.
                The value is currently ignored but kept for API compatibility.
        """
        self.epoch = {
            'position': position,
            'velocity': velocity,
            'ts': epoch,
            'time': _to_astropy(epoch)
        }
        self.name = name

        self.method = method
        self.target = -500000
        # self.target_name = name

    def _at(self, t):
        """Compute this satellite's GCRS position and velocity at time `t`."""
        tt = _to_astropy(t)
        td = (tt - self.epoch['time']).to(u.s).value

        if np.ndim(td) > 0:
            rGCRS = np.zeros((3, len(td)))
            vGCRS = np.zeros((3, len(td)))
            for i, d in enumerate(td):
                r, v = _propagate_kepler(
                    self.epoch['position'].to_value(u.km),
                    self.epoch['velocity'].to_value(u.km / u.s),
                    d,
                )
                rGCRS[:, i] = r
                vGCRS[:, i] = v
        else:
            r, v = _propagate_kepler(
                self.epoch['position'].to_value(u.km),
                self.epoch['velocity'].to_value(u.km / u.s),
                td,
            )
            rGCRS = r
            vGCRS = v

        rGCRS /= AU_KM
        vGCRS /= AU_KM
        vGCRS *= DAY_S

        size = np.size(td)
        return rGCRS, vGCRS, rGCRS, [None] * size

    def __str__(self):
        return 'EarthTwoBodySatellite {0} epoch={1}'.format(
            '' if self.name is None else self.name,
            self.epoch['ts'],
        )


def create_twobody_from_tle(tle1, tle2, epoch=None, method=DEFAULT_METHOD):
    """Create a Skyfield EarthTwoBodySatellite object centered about the planet Earth
    using a two-line element (TLE) set.

    Args:
        tle1: `string`, first line of the TLE
        tle2: `string`, second line of the TLE
        epoch: `Time`, time to generate epoch state vector for two body
        method: `string`, two body propagator algorithm type.
            Valid types are: 'vallado' (default), 'farnocchia', 'cowell`.

    Returns:
        A `EarthTwoBodySatellite` Skyfield object centered about the planet Earth
    """
    sat = EarthSatellite(tle1, tle2)

    if epoch is None:
        epoch = sat.epoch

    position, velocity, _, _ = sat._at(epoch)

    position = position * u.au
    velocity = velocity * u.au / u.d

    return create_twobody(position.to(u.km), velocity.to(u.km / u.s), epoch, name=sat.name, method=method)


def create_twobody(position, velocity, epoch, name='EarthTwoBodySatellite', method=DEFAULT_METHOD):
    """Create a `EarthTwoBodySatellite` object centered about the planet Earth
    initialized with a two-line element (TLE) set at `epoch`.

    Args:
        position: `array`, AstroPy position in ITRF. example: `[-6045, -3490, 2500] * u.km`
        velocity: `array`, AstroPy velocity in ITRF. example: `[-3.457, 6.618, 2.534] * u.km / u.s`
        epoch: `Time`, epoch time
        method: `string`, two body propagator algorithm type.
            Valid types are: 'vallado' (default), 'farnocchia', 'cowell`.

    Returns:
        A `EarthTwoBodySatellite` Skyfield object centered about the planet Earth
    """
    earth = load_earth()

    return earth + EarthTwoBodySatellite(position, velocity, epoch, name, method=method)
