import numpy as np
from satsim.geometry.astrometric import load_earth, radec_to_eci

from skyfield.constants import AU_KM, DAY_S
from skyfield.vectorlib import VectorFunction

from astropy import units as u


class EarthObservation(VectorFunction):
    """ EarthObservation type """

    def __init__(self, ra, dec, t, observer, target, d=25000 * u.km, name='EarthObservation'):
        """ Creates an EarthObservation object.

        Args:
            ra: `float`, right ascension in degrees
            dec: `float`, declination in degrees
            t: `Time`, observation time as Skyfield `Time`
            observer: `VectorFunction`, Skyfield observer
            target: `VectorFunction`, Skyfield target
            d: `float`, distance to target in km if `target` is None
            name: `string`, object name
        """
        self.center = 399
        self.name = name
        self.target = -500000

        p0 = load_earth().at(t)
        p2 = observer.at(t) - p0

        if target is not None:
            p1 = target.at(t) - p0
            p3 = p1 - p2
            ra0, dec0, d0 = p3.radec()
            d0 = d0.km
        else:
            d0 = d

        # use the input ra and dec as the target ra and dec and distance to target from observer
        x, y, z = radec_to_eci(ra, dec, d0)

        self.r = (p2.xyz.km + np.array([x, y, z])) * u.km
        self.v = np.array([0, 0, 0]) * u.km / u.s

    def _at(self, t):

        if len(t.shape) == 0:
            size = 0
        else:
            size = t.shape[0]

        if size > 0:
            rGCRS = np.zeros((3, size))
            vGCRS = np.zeros((3, size))
            for i in range(size):
                rGCRS[:, i] = self.r.to(u.km).value
                vGCRS[:, i] = self.v.to(u.km / u.s).value
        else:
            rGCRS = self.r.to(u.km).value
            vGCRS = self.v.to(u.km / u.s).value

        rGCRS /= AU_KM
        vGCRS /= AU_KM
        vGCRS *= DAY_S

        return rGCRS, vGCRS, rGCRS, [None] * size

    def __str__(self):
        return 'EarthObservation {0} epoch={1}'.format(
            '' if self.name is None else self.name,
            self.epoch['ts'],
        )


def create_observation(ra, dec, t, observer, target, d=None, name='EarthObservation'):
    """ Creates an EarthObservation object.

    Args:
        ra: `float`, right ascension in degrees
        dec: `float`, declination in degrees
        t: `Time`, observation time as Skyfield `Time`
        observer: `VectorFunction`, Skyfield observer
        target: `VectorFunction`, Skyfield target
        d: `float`, distance to target in km if `target` is None
        name: `string`, object name

    Returns:
        An `EarthObservation` object
    """

    earth = load_earth()

    if d is not None:
        return earth + EarthObservation(ra, dec, t, observer, None, d, name=name)
    else:
        return earth + EarthObservation(ra, dec, t, observer, target, None, name=name)
