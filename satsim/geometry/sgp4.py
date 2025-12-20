import numpy as np
from sgp4.api import Satrec, SatrecArray
from skyfield.api import load, EarthSatellite
from skyfield.constants import DAY_S
from skyfield.sgp4lib import TEME, _T, mxv

from satsim.geometry.astrometric import load_earth

SATSIM_TS = None


def load_tle(urls):
    """Loads a list of URLs containing a list of two-line element (TLE) sets.
    TLE is a data format encoding a list of orbital elements of an
    Earth-orbiting object for a given point in time, the epoch.

    Args:
        urls: `list`, a list of strings

    Returns:
        A `list` of Skyfield `EarthSatellites`
    """

    earth = load_earth()

    satellites = []
    for url in urls:
        next_satellites = load.tle(url).values()
        satellites.extend(x for x in next_satellites if x not in satellites)

    satellites = list(map(lambda s: earth + s, satellites))

    return satellites


def create_sgp4(tle1, tle2):
    """Create a Skyfield EarthSatellite object centered about the planet Earth
    using a two-line element (TLE) set.

    Args:
        tle1: `string`, first line of the TLE
        tle2: 'string', second line of the TLE

    Returns:
        A `EarthSatellite` Skyfield object centered about the planet Earth
    """
    earth = load_earth()

    return earth + EarthSatellite(tle1, tle2)


def load_timescale():
    """Load a cached Skyfield timescale."""
    global SATSIM_TS
    if SATSIM_TS is None:
        SATSIM_TS = load.timescale()
    return SATSIM_TS


def create_satrec(tle1, tle2):
    """Create an sgp4 Satrec from a TLE pair."""
    return Satrec.twoline2rv(tle1, tle2)


def create_sgp4_from_satrec(satrec, name=None):
    """Create a Skyfield EarthSatellite object from a Satrec."""
    earth = load_earth()
    sat = EarthSatellite.from_satrec(satrec, load_timescale())
    if name:
        sat.name = name
    return earth + sat


def batch_sgp4_position_gcrs_km(satrecs, t):
    """Compute GCRS positions in km for a list of sgp4 Satrec objects.

    Args:
        satrecs: `list`, list of sgp4 Satrec objects
        t: Skyfield Time object

    Returns:
        Tuple of (positions_km, errors) where positions_km is an Nx3 array and
        errors is an N-length uint8 array of sgp4 error codes.
    """
    if isinstance(satrecs, SatrecArray):
        sat_array = satrecs
        if len(sat_array) == 0:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.uint8)
    else:
        if not satrecs:
            return np.empty((0, 3), dtype=np.float64), np.empty((0,), dtype=np.uint8)
        sat_array = SatrecArray(satrecs)
    jd = np.array([t.whole], dtype=np.float64)
    fr = np.array([t.tai_fraction - t._leap_seconds() / DAY_S], dtype=np.float64)
    errors, positions_teme, _ = sat_array.sgp4(jd, fr)

    rotation = _T(TEME.rotation_at(t))
    positions_teme = positions_teme[:, 0, :].T
    positions_gcrs = mxv(rotation, positions_teme).T

    return positions_gcrs, errors[:, 0]
