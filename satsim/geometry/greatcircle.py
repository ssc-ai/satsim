import numpy as np
from pygc import great_circle
from skyfield.api import Star, Angle
from skyfield.vectorlib import VectorFunction

from .astrometric import get_los_azel


class GreatCircle(VectorFunction):
    """A great circle propagated Skyfield object."""

    center = 0

    def __init__(self, az, el, heading, velocity, t, observer):
        """Create a GreatCircle propagator.

        Parameters
        ----------
        az : float
            Starting azimuth or longitude in degrees.
        el : float
            Starting elevation or latitude in degrees.
        heading : float
            Direction in degrees; 0 degrees is north heading.
        velocity : float
            Velocity in degrees per second.
        t : skyfield.time.Time
            Epoch of the starting position.
        observer : object
            Frame of reference. If ``None`` the reference is barycentric.
        """
        self.az = az
        self.el = el
        self.heading = heading
        self.velocity = velocity
        self.t = t
        self.R = 57.29577951308232
        self.target = -500000
        self.observer = observer

    def _observe_from_bcrs(self, observer):
        dt = (observer.t - self.t) * 86400
        d = dt * self.velocity
        gc = great_circle(
            distance=d,
            azimuth=self.heading,
            latitude=self.el,
            longitude=self.az,
            rmajor=self.R,
            rminor=self.R,
        )
        if self.observer is not None:
            ra, dec, _, _, _, _ = get_los_azel(
                self.observer, gc["longitude"], gc["latitude"], observer.t
            )
        else:
            ra = gc["longitude"]
            dec = gc["latitude"]

        star = Star(ra=Angle(degrees=ra), dec=Angle(degrees=dec), parallax_mas=1e-16)
        vector, vel, t, _ = star._observe_from_bcrs(observer)
        return vector, vel, t, None

    def _at(self, t):
        dt = (t - self.t) * 86400
        d = dt * self.velocity
        gc = great_circle(
            distance=d,
            azimuth=self.heading,
            latitude=self.el,
            longitude=self.az,
            rmajor=self.R,
            rminor=self.R,
        )

        lon = gc["longitude"]
        lat = gc["latitude"]

        if np.ndim(dt) == 0:
            lon = float(lon)
            lat = float(lat)

        if self.observer is not None:
            if np.ndim(dt) == 0:
                ra, dec, _, _, _, _ = get_los_azel(self.observer, lon, lat, t)
            else:
                ra = np.zeros_like(lon, dtype=float)
                dec = np.zeros_like(lat, dtype=float)
                for i, (lo, la) in enumerate(zip(lon, lat)):
                    r, d0, _, _, _, _ = get_los_azel(self.observer, lo, la, t[i])
                    ra[i] = r
                    dec[i] = d0
        else:
            ra = lon
            dec = lat

        if np.ndim(dt) == 0:
            ra = float(ra)
            dec = float(dec)

        if np.ndim(dt) == 0:
            star = Star(ra=Angle(degrees=ra), dec=Angle(degrees=dec), parallax_mas=1e-16)
            rGCRS = star._position_au
            vGCRS = star._velocity_au_per_d
            size = 0
        else:
            r_list = []
            v_list = []
            for r_deg, d_deg in zip(ra, dec):
                star = Star(ra=Angle(degrees=r_deg), dec=Angle(degrees=d_deg), parallax_mas=1e-16)
                r_list.append(star._position_au)
                v_list.append(star._velocity_au_per_d)
            rGCRS = np.stack(r_list, axis=1)
            vGCRS = np.stack(v_list, axis=1)
            size = len(r_list)

        return rGCRS, vGCRS, None, [None] * size
