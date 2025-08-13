"""Tests for `satsim.geometry.astrometric` package."""
import os
from functools import lru_cache

import numpy as np
from skyfield.api import Star, Angle, iers2010
from skyfield.toposlib import _ltude
from skyfield.relativity import add_aberration, add_deflection
from skyfield.earthlib import compute_limb_angle
from skyfield.positionlib import Apparent

if 'SATSIM_SKYFIELD_LOAD_DIR' in os.environ:
    from skyfield.api import Loader
    load = Loader(os.environ['SATSIM_SKYFIELD_LOAD_DIR'])
else:
    from skyfield.api import load

from astropy.coordinates import SkyCoord, ICRS, ITRS
from astropy.time import Time
from astropy.coordinates import cartesian_to_spherical
from astropy import units as u

from satsim.geometry.wcs import get_min_max_ra_dec, get_wcs


SATSIM_EARTH = None
SATSIM_MOON = None
SATSIM_SUN = None


def load_earth():
    """Loads a Skyfield Earth object using the planetary and lunar ephemeris,
    DE 421. The Earth object is loaded once and cached as a singleton in the
    variable `SATSIM_EARTH`.

    Returns:
        The `planet` Earth Skyfield object
    """

    global SATSIM_EARTH

    if SATSIM_EARTH is None:
        planets = load('de421.bsp')
        SATSIM_EARTH = planets['earth']

    return SATSIM_EARTH


def load_moon():
    """Loads a Skyfield Moon object using the planetary and lunar ephemeris,
    DE 421. The Mon object is loaded once and cached as a singleton in the
    variable `SATSIM_MOON`.

    Returns:
        The Moon Skyfield object
    """

    global SATSIM_MOON

    if SATSIM_MOON is None:
        planets = load('de421.bsp')
        SATSIM_MOON = planets['moon']

    return SATSIM_MOON


def load_sun():
    """Loads a Skyfield Sun object using the planetary and lunar ephemeris,
    DE 421. The Sun object is loaded once and cached as a singleton in the
    variable `SATSIM_SUN`.

    Returns:
        The Sun Skyfield object
    """

    global SATSIM_SUN

    if SATSIM_SUN is None:
        planets = load('de421.bsp')
        SATSIM_SUN = planets['sun']

    return SATSIM_SUN


def create_topocentric(lat, lon, alt=0):
    """Create a Skyfield topocentric object which represents the location of a
    place on the planet Earth.

    Args:
        lat: `string or float`, latitude in degrees
        lon: `string or float`, longitude in degrees
        alt: `float`, altitude in km

    Returns:
        A `Topos` Skyfield object on the planet Earth
    """

    earth = load_earth()

    if isinstance(lat, str):
        lat = _ltude(lat, 'latitude', 'N', 'S')

    if isinstance(lon, str):
        lon = _ltude(lon, 'longitude', 'E', 'W')

    topo = iers2010.latlon(latitude_degrees=lat,
                           longitude_degrees=lon,
                           elevation_m=alt * 1000)

    return earth + topo


def apparent(p, deflection=False, aberration=True):
    """Compute an :class:`Apparent` position for this body.
    This applies two effects to the position that arise from
    relativity and shift slightly where the other body will appear
    in the sky: the deflection that the image will experience if its
    light passes close to large masses in the Solar System, and the
    aberration of light caused by the observer's own velocity.
    Note: This algorithm is copied from skyfield/positionlib.py and
    modified to disable deflection and aberration calculation.
    """

    t = p.t
    target_au = p.position.au.copy()

    cb = p.center_barycentric

    if cb is None:
        observer_gcrs_au = None
        deflection = False
        aberration = False
    else:
        bcrs_position = cb.position.au
        bcrs_velocity = cb.velocity.au_per_d
        observer_gcrs_au = cb._observer_gcrs_au

    # If a single observer position (3,) is observing an array of
    # targets (3,n), then deflection and aberration will complain
    # that "operands could not be broadcast together" unless we give
    # the observer another dimension too.
    # if len(bcrs_position.shape) < len(target_au.shape):
    #     shape = bcrs_position.shape + (1,)
    #     bcrs_position = bcrs_position.reshape(shape)
    #     bcrs_velocity = bcrs_velocity.reshape(shape)
    #     if observer_gcrs_au is not None:
    #         observer_gcrs_au = observer_gcrs_au.reshape(shape)

    if observer_gcrs_au is None:
        include_earth_deflection = np.array((False,))
    else:
        limb_angle, nadir_angle = compute_limb_angle(
            target_au, observer_gcrs_au)
        include_earth_deflection = nadir_angle >= 0.8

    if deflection:
        add_deflection(target_au, bcrs_position,
                       p._ephemeris, t, include_earth_deflection)

    if aberration and p.light_time is not None:
        add_aberration(target_au, bcrs_velocity, p.light_time)

    apparent = Apparent(target_au, None, t, p.center, p.target)
    apparent.center_barycentric = p.center_barycentric
    apparent._observer_gcrs_au = observer_gcrs_au
    return apparent


def _apply_stellar_aberration(observer, ra, dec, az, el, t, deflection):
    """Apply stellar aberration to RA/Dec coordinates.

    Args:
        observer: Skyfield observer object
        ra: Right ascension angle object
        dec: Declination angle object
        az: Azimuth angle object
        el: Elevation angle object
        t: Skyfield time object
        deflection: Boolean for deflection correction

    Returns:
        Tuple of corrected (ra, dec, el, az) values
    """
    star = Star(ra=ra, dec=dec)
    icrf_los = apparent(observer.at(t).observe(star), deflection, True)
    sa_ra, sa_dec, sa_d = icrf_los.radec()

    try:
        sa_el, sa_az, sa_d = icrf_los.altaz()
    except Exception:
        sa_az = Angle(degrees=0.0)
        sa_el = Angle(degrees=0.0)

    # Adjust the apparent RA and Dec based on stellar aberration
    # Convert to degrees for arithmetic operations
    apparent_ra = Angle(degrees=2 * ra.degrees - sa_ra.degrees)
    apparent_dec = Angle(degrees=2 * dec.degrees - sa_dec.degrees)
    apparent_el = Angle(degrees=2 * el.degrees - sa_el.degrees)
    apparent_az = Angle(degrees=2 * az.degrees - sa_az.degrees)

    return apparent_ra, apparent_dec, apparent_el, apparent_az


@lru_cache(maxsize=32)
def get_los(observer, target, t, deflection=False, aberration=True, stellar_aberration=False):
    """Get the apparent line of sight vector from an observer and target in
    right ascension (RA) and declination (Dec).

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        t: `object`, skyfield time
        deflection: `boolean`, enable deflection adjustment
        aberration: `boolean`, enable aberration of light adjustment for the target
        stellar_aberration: `boolean`, enable stellar aberration adjustment (apparent)

    Returns:
        A `tuple`, containing:
            ra: right ascension in degrees
            dec: declination in degrees
            d: distance between observer and target in km
            az: azimuth angle in degrees
            el: elevation angle in degrees
            icrf_los: LOS as skyfield ICRF object
    """

    if deflection or aberration:
        icrf_los = apparent(observer.at(t).observe(target), deflection, aberration)
    else:
        icrf_los = (target - observer).at(t)

    ra, dec, d = icrf_los.radec()
    try:
        el, az, d = icrf_los.altaz()
    except Exception:
        az = Angle(degrees=0.0)
        el = Angle(degrees=0.0)

    if stellar_aberration:
        ra, dec, el, az = _apply_stellar_aberration(observer, ra, dec, az, el, t, deflection)

    return ra._degrees, dec._degrees, d.km, az._degrees, el._degrees, icrf_los


@lru_cache(maxsize=32)
def get_los_azel(observer, az, el, t, deflection=False, aberration=True, stellar_aberration=False):
    """Get the apparent line of sight vector from an observer based on topocentric
    az and el.

    Args:
        observer: `object`, observer as a Skyfield object
        az: `float`, azimuth
        el: `float`, elevation
        t: `object`, skyfield time

    Returns:
        A `tuple`, containing:
            ra: right ascension in degrees
            dec: declination in degrees
            d: distance between observer and target in km
            az: azimuth angle in degrees
            el: elevation angle in degrees
            icrf_los: LOS as skyfield ICRF object
    """

    icrf_los = apparent(observer.at(t).from_altaz(alt_degrees=el, az_degrees=az), deflection, aberration)
    ra, dec, d = icrf_los.radec()

    if stellar_aberration:
        # Convert float values to Angle objects for stellar aberration calculation
        from skyfield.units import Angle
        el_angle = Angle(degrees=el)
        az_angle = Angle(degrees=az)
        ra, dec, el_angle, az_angle = _apply_stellar_aberration(observer, ra, dec, az_angle, el_angle, t, deflection)
        el = el_angle.degrees
        az = az_angle.degrees

    return ra._degrees, dec._degrees, d.km, az, el, icrf_los


def get_analytical_los(observer, target, t, frame="observer"):
    """Return a line of sight in the specified frame for analytical observations.

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        t: `object`, skyfield time
        frame: `string`, one of ``barycentric``, ``geocentric`` or ``observer``

    Returns:
        A `tuple`, containing:
            ra: right ascension in degrees
            dec: declination in degrees
            d: distance between observer and target in km
    """

    frame = frame.lower()

    # compute light time using apparent LOS from observer to target
    icrf_los_abr = observer.at(t).observe(target)
    lt = icrf_los_abr.light_time

    try:
        if frame == "barycentric":
            icrf_los = target.at(t - lt) - observer.at(t)
        elif frame == "geocentric":
            earth = load_earth()
            observer_gc = observer - earth
            target_gc = target - earth
            icrf_los = target_gc.at(t - lt) - observer_gc.at(t)
        elif frame == "observer":  # observer frame
            observer_oc = observer - observer
            target_oc = target - observer
            icrf_los = target_oc.at(t - lt) - observer_oc.at(t)
        else:
            raise ValueError(f"Unknown frame: {frame}")
    except Exception:
        # Handle cases where the observer or target does not support the requested frame
        icrf_los = icrf_los_abr

    ra, dec, d = icrf_los.radec()

    return ra._degrees, dec._degrees, d.km


def query_by_los(height, width, y_fov, x_fov, ra, dec, t0, observer, targets=[], rot=0, pad_mult=0, origin='center', offset=[0,0]):
    """Return objects that are within the minimum and maximum RA and declination
    bounds of the observer's focal plan array with padding, `pad_mult`.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        ra: `float`, right ascension of origin (center or corner [0,0])
        dec: `float`, declination of origin (center or corner [0,0])
        t0: `Time`, Skyfield `Time` representing the time
        observer: `object`, the observer as a Skyfield object
        targets: `list`, list of objects to test if in bounds
        rot: `float`, rotation of the focal plane in degrees
        pad_mult: `float`, padding multiplier
        origin: `string`, corner or center
        offset: `float`, array specifying [row, col] offset in pixels

    Returns:
        A `tuple`, containing:
            visible: `list`, list of Skyfield objects that are in bounds
            idx: indices of visible objects in `targets`
    """

    [cmin, cmax, wcs] = get_min_max_ra_dec(height, width, y_fov / height, x_fov / width, ra, dec, rot, pad_mult, origin, offset=offset)

    visible = []
    idx = []
    ii = 0
    for s in targets:
        [sra, sdec, dist, az, el, los] = get_los(observer, s, t0)

        if sra >= cmin[0] and sra <= cmax[0] and sdec >= cmin[1] and sdec <= cmax[1]:
            visible.append(s)
            idx.append(ii)

        ii += 1

    return visible, idx


def gen_track_from_wcs(height, width, wcs, observer, targets, t0, tt, origin='center',
                       offset=[0,0], flipud=False, fliplr=False):
    """Generates a list of pixel coordinates on the observing focal plane
    array for each object in the list, `target`. Target is tracked based on
    `wcs0` and `wcs1`.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        wcs: `list`, list of AstroPy world coordinate system object used to
            transform world to pixel coordinates at `t_start`
        observer: `object`, the observer as a Skyfield object
        targets: `list`, list of objects to test if in bounds
        t0: `Time`, Skyfield `Time` representing the track start time
        tt: `list`, list of Skyfield `Time` representing observation times
        origin: `string`, corner or center
        offset: `float`, array specifying [row, col] offset in pixels
        flipud: `boolean`, flip array in up/down direction
        fliplr: `boolean`, flip array in left/right direction

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row coordinates for each `target` at each time
            cc: `list`, list of column coordinates for each `target` at each time
            dr: `float`, star motion along the rows in pixel per second
            dc: `float`, star motion along the columns in pixel per second
    """
    rr, cc = [], []
    ra, dec = [], []
    for t,w in zip(tt, wcs):

        rra0, ddec0, = [], []
        for s in targets:
            [sra, sdec, dist, az, el, los] = get_los(observer, s, t)
            rra0.append(sra)
            ddec0.append(sdec)

        cc0, rr0 = w.wcs_world2pix(rra0, ddec0, 0)

        if origin == 'center':
            rr0 += height / 2.0 + offset[0]
            cc0 += width / 2.0 + offset[1]

        los_ra0, los_dec0 = w.wcs_pix2world(0, 0, 0)

        if flipud:
            rr0 = height - rr0

        if fliplr:
            cc0 = width - cc0

        ra.append(los_ra0)
        dec.append(los_dec0)
        rr.append(rr0)
        cc.append(cc0)

    # find the velocity in the initial wcs frame
    c0, r0 = wcs[0].wcs_world2pix(ra[0], dec[0], 0)
    c1, r1 = wcs[0].wcs_world2pix(ra[-1], dec[-1], 0)
    if flipud:
        r0 = height - r0
        r1 = height - r1

    if fliplr:
        c0 = width - c0
        c1 = width - c1

    exposure_time = (tt[-1] - tt[0]) * 86400  # convert days to seconds

    return rr, cc, (r1 - r0) / exposure_time, (c1 - c0) / exposure_time


def wcs_from_observer_rate(height, width, y_fov, x_fov, observer, t0, tt, rot, track,
                           deflection=False, aberration=True, stellar_aberration=False):
    """Calculate the world coordinate system (WCS) transform from the `observer`
    at times `tt` while rate tracking the object, `track`.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        observer: `object`, the observer as a Skyfield object
        t0: `Time`, Skyfield `Time` representing the track start time
        tt: `list`, list of Skyfield `Time` representing observation times
        rot: `float`, rotation of the focal plane in degrees
        track: `object`, the target to rate track as a Skyfield object

    Returns:
        A `object`: WCS transform
    """
    wsc = []
    for t in tt:
        [ra0, dec0, d0, az0, el0, los0] = get_los(
            observer,
            track,
            t,
            deflection=deflection,
            aberration=aberration,
            stellar_aberration=stellar_aberration,
        )
        wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
        wsc.append(wcs0)
    return wsc


def wcs_from_observer_fixed(height, width, y_fov, x_fov, observer, tt, rot, az, el,
                            deflection=False, aberration=True, stellar_aberration=False):
    """Calculate the world coordinate system (WCS) transform from the `observer`
    at a fixed pointing position based on azimuth and elevation.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        observer: `object`, the observer as a Skyfield object
        tt: `list`, list of Skyfield `Time` representing observation times
        rot: `float`, rotation of the focal plane in degrees
        az: `float`, azimuth
        el: `float`, elevation

    Returns:
        A `object`: WCS transform
    """
    wsc = []
    for t, a, e in zip(tt, az, el):
        [ra0, dec0, d0, az0, el0, los0] = get_los_azel(
            observer,
            a,
            e,
            t,
            deflection=deflection,
            aberration=aberration,
            stellar_aberration=stellar_aberration,
        )
        wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
        wsc.append(wcs0)
    return wsc


def wcs_from_observer_sidereal(height, width, y_fov, x_fov, observer, t0, tt, rot, track,
                               deflection=False, aberration=True, stellar_aberration=False):
    """Calculate the world coordinate system (WCS) transform from the `observer`
    at times `tt` while sidereal tracking the object, `track`.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        observer: `object`, the observer as a Skyfield object
        t0: `Time`, Skyfield `Time` representing the track start time
        tt: `list`, list of Skyfield `Time` representing observation times
        rot: `float`, rotation of the focal plane in degrees
        track: `object`, the target to rate track as a Skyfield object

    Returns:
        A `object`: WCS transform
    """
    wsc = []
    [ra0, dec0, d0, az0, el0, los0] = get_los(
        observer,
        track,
        t0,
        deflection=deflection,
        aberration=aberration,
        stellar_aberration=stellar_aberration,
    )
    wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
    for t in tt:
        wsc.append(wcs0)
    return wsc


def gen_track(height, width, y_fov, x_fov, observer, track, satellites, brightnesses, t0, tt, rot=0, pad_mult=0, track_type='rate',
              offset=[0, 0], flipud=False, fliplr=False, az=None, el=None,
              deflection=False, aberration=True, stellar_aberration=False):
    """Generates a list of pixel coordinates from the observing focal plane
    array to each satellite in the list, `satellites`. Track mode can be either
    `rate` or `sidereal`.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        observer: `object`, the observer as a Skyfield object
        track: `object`, the target to rate track as a Skyfield object
        satellites: `list`, list of targets to calculate to pixel coordinates
        brightnesses: `float`, list of brightnesses of each target in Mv
        t0: `Time`, Skyfield `Time` representing the track start time
        tt: `list`, list of Skyfield `Time` representing observation times
        rot: `float`, rotation of the focal plane in degrees
        pad_mult: `float`, padding multiplier (unused)
        track_type: `string`, `rate` or `sidereal`
        offset: `float`, array specifying [row, col] offset in pixels
        flipud: `boolean`, flip array in up/down direction
        fliplr: `boolean`, flip array in left/right direction
        az: `float`, azimuth in degrees for `fixed` tracking
        el: `float`, elevation in degrees for `fixed` tracking
        deflection: `bool`, include gravitational deflection in astrometry
        aberration: `bool`, include light transit time effects in astrometry
        stellar_aberration: `bool`, include stellar aberration in astrometry

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row coordinates for each `target` at each time
            cc: `list`, list of column coordinates for each `target` at each time
            dr: `float`, star motion along the rows in pixel per second
            dc: `float`, star motion along the columns in pixel per second
            b: `float`, list of brightnesses of each target in Mv
    """
    # note wcs are based on apparent position of the tracked target
    if track_type == 'rate':
        wcs = wcs_from_observer_rate(
            height,
            width,
            y_fov,
            x_fov,
            observer,
            t0,
            tt,
            rot,
            track,
            deflection=deflection,
            aberration=aberration,
            stellar_aberration=stellar_aberration,
        )
    elif track_type == 'fixed':
        wcs = wcs_from_observer_fixed(
            height,
            width,
            y_fov,
            x_fov,
            observer,
            tt,
            rot,
            az,
            el,
            deflection=deflection,
            aberration=aberration,
            stellar_aberration=stellar_aberration,
        )
    else:
        wcs = wcs_from_observer_sidereal(
            height,
            width,
            y_fov,
            x_fov,
            observer,
            t0,
            tt,
            rot,
            track,
            deflection=deflection,
            aberration=aberration,
            stellar_aberration=stellar_aberration,
        )

    b = np.asarray(brightnesses)
    visible = satellites

    # TODO add option to calc visible  obs
    # [cmin, cmax, wcs] = get_min_max_ra_dec(height, width, y_fov / height, x_fov / width, ra, dec, rot, pad_mult, 'center', offset)
    # visible, idx, wcs = query_by_los(height, width, y_fov, x_fov, ra, dec, t_start, observer, satellites, rot, pad_mult)

    return gen_track_from_wcs(height, width, wcs, observer, visible, t0, tt, 'center', offset=offset, flipud=flipud, fliplr=fliplr) + (b,)


def angle_between(observer, object_a, object_b, t):
    """Calculate the angle between `observer` to `object_a` and `observer`
    and `object_b` at time `t`. Angle is returned in degrees.

    Args:
        observer: `object`, Skyfield object
        object_a: `object`, Skyfield object
        object_b: `object`, Skyfield object
        t: `Time`, Skyfield `Time`

    Returns:
        A `float`, angle in degrees
    """
    o = observer.at(t)
    a = o.observe(object_a).apparent()
    b = o.observe(object_b).apparent()

    return a.separation_from(b).degrees


def distance_between(object_a, object_b, t):
    """Calculate the distance between `object_a` and `object_b` at time `t`.

    Args:
        object_a: `object`, Skyfield object
        object_b: `object`, Skyfield object
        t: `Time`, Skyfield `Time`

    Returns:
        A `float`, distance in km
    """
    a = object_a.at(t)
    b = object_b.at(t)

    return (a - b).distance().km


def angle_from_los(observer, object_a, ra, dec, t):
    """Calculate the angle between `observer` to `object_a` and `observer`
    to a line of sight in `ra` and `dec`. Angle is returned in degrees.

    Args:
        observer: `object`, Skyfield object
        object_a: `object`, Skyfield object
        ra: `float`, right ascension in degrees
        dec: `float`, declination in degrees
        t: `Time`, Skyfield `Time`

    Returns:
        A `float`, angle in degrees
    """
    s = Star(ra=Angle(degrees=ra), dec=Angle(degrees=dec))

    return angle_between(observer, s, object_a, t)


def eci_to_ecr(time, ra, dec, roll=0):
    """Covert an Earth centered fixed sky coordinates to Earth centered rotating.

    Args:
        ra: `float`, right ascension in degrees
        dec: `float`, declination in degrees
        roll: `float`, field rotation (ignored)

    Returns:
        A `float`, ra, dec and roll in degrees
    """
    sc = SkyCoord(ra=ra, dec=dec, unit='deg', frame=ICRS)
    sc2 = sc.transform_to(ITRS(obstime=Time(time)))
    sc3 = SkyCoord(sc2)

    _, dec, ra = cartesian_to_spherical(sc3.x, sc3.y, sc3.z)
    dec = dec + 270 * u.deg
    ra = -ra

    return ra, dec, roll


def eci_to_radec(x, y, z):
    """Covert ECI coordinates to ra, dec, and distance.

    Args:
        x: `float`, x coordinate
        y: `float`, y coordinate
        z: `float`, z coordinate

    Returns:
        A `float`, ra, dec and distance in degrees
    """

    rho2 = x * x + y * y
    rho = np.sqrt(rho2)

    r = np.sqrt(rho2 + z * z)
    dec = 0.0 if r == 0.0 else np.arctan2(z, rho)
    ra = 0.0 if rho == 0.0 else np.arctan2(y, x)
    if ra < 0.0:
        ra += np.pi * 2

    return ra * 180 / np.pi, dec * 180 / np.pi, r


def radec_to_eci(ra, dec, d=0):
    """Covert ra, dec, and distance to ECI coordinates.

    Args:
        ra: `float`, right ascension in degrees
        dec: `float`, declination in degrees
        d: `float`, distance

    Returns:
        A `float`, x, y, and z coordinates
    """

    ra = ra * np.pi / 180
    dec = dec * np.pi / 180

    r = 1.0 if d is None or d <= 0.0 else d

    rCosTheta = r * np.cos(dec)

    x = rCosTheta * np.cos(ra)
    y = rCosTheta * np.sin(ra)
    z = r * np.sin(dec)

    return x, y, z


def optimized_angle_from_los_cosine(observer, target, ra_c, dec_c, ts_mid):
    """Calculate the cosine of the angle between observer-to-target and observer-to-RA/Dec lines of sight.

    This is a highly optimized version that directly computes position vectors and avoids
    expensive Skyfield operations like observe() and separation_from(). Returns cosine
    for efficient threshold comparisons without arccos operations.

    Args:
        observer: Skyfield object representing the observer
        target: Skyfield object representing the target
        ra_c: float, right ascension in degrees
        dec_c: float, declination in degrees
        ts_mid: Skyfield Time object

    Returns:
        float: cosine of the angle (range [-1, 1])
    """
    try:
        earth = load_earth()
        observer_pos = (observer - earth).at(ts_mid).position.km
        target_pos = (target - earth).at(ts_mid).position.km

        los_to_target = target_pos - observer_pos
        los_to_target_norm = los_to_target / np.linalg.norm(los_to_target)

    except (TypeError, AttributeError):
        # Fallback for Star objects or other types that don't support position arithmetic
        observer_at_time = observer.at(ts_mid)
        target_apparent = observer_at_time.observe(target)
        target_ra, target_dec, _ = target_apparent.radec()

        target_x, target_y, target_z = radec_to_eci(target_ra.degrees, target_dec.degrees, 1.0)
        los_to_target_norm = np.array([target_x, target_y, target_z])

    los_x, los_y, los_z = radec_to_eci(ra_c, dec_c, 1.0)
    los_radec_norm = np.array([los_x, los_y, los_z])

    cos_angle = np.clip(np.dot(los_to_target_norm, los_radec_norm), -1.0, 1.0)

    return cos_angle
