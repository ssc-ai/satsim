import numpy as np
from satsim.geometry.astrometric import load_earth

from skyfield.constants import AU_KM, DAY_S
from skyfield.vectorlib import VectorFunction
from skyfield.api import EarthSatellite

from poliastro.twobody.propagation import farnocchia, vallado, cowell
from poliastro.bodies import Earth
from poliastro.twobody import Orbit

from astropy.time import Time
from astropy import units as u


DEFAULT_METHOD = 'vallado'


def _to_astropy(t):

    return Time(t.tt, scale='tt', format='jd')


class EarthTwoBodySatellite(VectorFunction):
    """ Skyfield Two-body propagator Satellite type """

    center = 399

    def __init__(self, position, velocity, epoch, name='EarthTwoBodySatellite', method=DEFAULT_METHOD):
        """ Constructs a Two-body propagator satellite.

        Args:
            position: `array`, AstroPy position in ITRF. example: `[-6045, -3490, 2500] * u.km`
            velocity: `array`, AstroPy velocity in ITRF. example: `[-3.457, 6.618, 2.534] * u.km / u.s`
            epoch: `Time`, epoch state vector time as Skyfield `Time`
            name: `string`, object name
            method: `string`, two body propagator algorithm type.
                Valid types are: 'vallado' (default), 'farnocchia', 'cowell`.
        """
        self.epoch = {
            'position': position,
            'velocity': velocity,
            'ts': epoch,
            'time': _to_astropy(epoch)
        }
        self.name = name

        if method == 'vallado':
            self.method = vallado.ValladoPropagator()
        elif method == 'farnocchia':
            self.method = farnocchia.FarnocchiaPropagator()
        else:
            self.method = cowell.CowellPropagator()

        self.orbit = Orbit.from_vectors(Earth, position, velocity, self.epoch['time'])
        self.target = -500000
        # self.target_name = name

    def _at(self, t):
        """Compute this satellite's GCRS position and velocity at time `t`."""
        tt = _to_astropy(t)
        td = tt - self.epoch['time']

        if td.size > 1:
            rGCRS = np.zeros((3, td.size))
            vGCRS = np.zeros((3, td.size))
            for i in range(td.size):
                o = self.orbit.propagate(td[i], method=self.method)
                r, v = o.rv()
                rGCRS[:, i] = r.to(u.km).value
                vGCRS[:, i] = v.to(u.km / u.s).value
        else:
            o = self.orbit.propagate(td, method=self.method)
            r, v = o.rv()
            rGCRS = r.to(u.km).value
            vGCRS = v.to(u.km / u.s).value

        rGCRS /= AU_KM
        vGCRS /= AU_KM
        vGCRS *= DAY_S

        return rGCRS, vGCRS, rGCRS, [None] * td.size

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
