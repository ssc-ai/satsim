import numpy as np
from skyfield.constants import AU_KM, DAY_S
from skyfield.vectorlib import VectorFunction
from astropy import units as u
from satsim.math.interpolate import lagrange
from satsim.time import to_astropy
from satsim.geometry.astrometric import load_earth


class EphemerisObject(VectorFunction):
    """ Skyfield Ephemeris type """

    center = 399

    def __init__(self, positions, velocities, times, epoch, name='EphemerisObject', order=3):
        """ Constructs an Ephemeris object.

        Args:
            positions: `list`, ITRF positions in km.
            velocities: `list`, ITRF velocities in km/s.
            times: `array`, Times from epoch in seconds.
            epoch: `Time`, epoch time as Skyfield `Time`.
            name: `string`, object name.
            order: `int`, Lagrange interpolation order.
        """
        self.epoch = to_astropy(epoch)

        # no segment boundaries
        if np.ndim(times[0]) == 0:
            self.positions = np.array(positions)
            self.velocities = np.array(velocities)
            self.times = np.array(times)
        # with segment boundaries
        elif np.ndim(times[0]) == 1:
            self.positions = [np.array(p) for p in positions]
            self.velocities = [np.array(v) for v in velocities]
            self.times = [np.array(t) for t in times]

        self.order = order
        self.name = name
        self.target = -500000

    def _at(self, t):
        """Compute this satellite's GCRS position and velocity at time `t`."""

        td = to_astropy(t)

        if td.size > 1:

            rGCRS = []
            vGCRS = []
            for i in range(td.size):

                delta_sec = (td[i] - self.epoch).to(u.s).value

                r = lagrange(self.times, self.positions, delta_sec, order=self.order)
                v = lagrange(self.times, self.velocities, delta_sec, order=self.order)

                rGCRS.append(r)
                vGCRS.append(v)

            rGCRS = np.squeeze(np.split(np.array(rGCRS), 3, axis=1))
            vGCRS = np.squeeze(np.split(np.array(vGCRS), 3, axis=1))

        else:

            delta_sec = (td - self.epoch).to(u.s).value

            rGCRS = lagrange(self.times, self.positions, delta_sec, order=self.order)
            vGCRS = lagrange(self.times, self.velocities, delta_sec, order=self.order)

        rGCRS /= AU_KM
        vGCRS /= AU_KM
        vGCRS *= DAY_S

        return rGCRS, vGCRS, rGCRS, [None] * td.size

    def __str__(self):
        return 'EphemerisObject {0} epoch={1}'.format(
            '' if self.name is None else self.name,
            self.epoch,
        )


def create_ephemeris_object(positions, velocities, times, epoch, name='EphemerisObject', order=3):
    """Create a `EphemerisObject` object centered about the planet Earth.

    Args:
        position: `array`, positions in ITRF. example: [[-6045, -3490, 2500], ...]
        velocity: `array`, velocities in ITRF. example: [[-3.457, 6.618, 2.534], ...]
        times: `array`, seconds from epoch.
        epoch: `Time`, epoch time as Skyfield `Time`.
        name: `string`, object name.
        order: `int`, Lagrange interpolation order.

    Returns:
        A `EphemerisObject` Skyfield object centered about the planet Earth.
    """
    earth = load_earth()

    return earth + EphemerisObject(positions, velocities, times, epoch, name, order)
