from skyfield.api import load, EarthSatellite

from satsim.geometry.astrometric import load_earth


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
