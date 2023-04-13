"""Tests for `satsim.geometry.astrometric` package."""
import os
from functools import lru_cache

import numpy as np
from pygc import great_circle
from skyfield.api import Topos, Star, Angle
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


class GreatCircle(object):
    """ A great circle propagated skyfield object."""

    def __init__(self, az, el, heading, velocity, t, observer):
        """ Constructor.

        Args:
            az: `float`, starting azimuth or longitude in degrees
            el: `float`, starting elevation or latitude in degrees
            heading: `float`, direction in degrees. 0 degrees is north heading
            velocity: `float`, velocity in degrees per second
            t: `object`, skyfield time
            observer: `object`, frame of reference, if `None`, reference is barycenter

        Returns:
            A `Topos` Skyfield object on the planet Earth
        """
        self.az = az
        self.el = el
        self.heading = heading
        self.velocity = velocity
        self.t = t
        self.R = 57.29577951308232
        self.target = None
        self.observer = observer

    def _observe_from_bcrs(self, observer):
        t = (observer.t - self.t) * 86400
        d = t * self.velocity
        gc = great_circle(distance=d, azimuth=self.heading, latitude=self.el, longitude=self.az, rmajor=self.R, rminor=self.R)
        if self.observer is not None:
            ra, dec, _, _, _, _ = get_los_azel(self.observer, gc['longitude'], gc['latitude'], observer.t)
        else:
            ra = gc['longitude']
            dec = gc['latitude']

        s = Star(ra=Angle(degrees=ra), dec=Angle(degrees=dec), parallax_mas=1e-16)
        vector, vel, t, light_time = s._observe_from_bcrs(observer)
        return vector, vel, t, None


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


def create_topocentric(lat, lon):
    """Create a Skyfield topocentric object which represents the location of a
    place on the planet Earth.

    Args:
        lat: `string or float`, latitude in degrees
        lon: `string or float`, longitude in degrees

    Returns:
        A `Topos` Skyfield object on the planet Earth
    """

    earth = load_earth()

    return earth + Topos(lat, lon)


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


@lru_cache(maxsize=32)
def get_los(observer, target, t, deflection=False, aberration=True):
    """Get the apparent line of sight vector from an observer and target in
    right ascension (RA) and declination (Dec).

    Args:
        observer: `object`, observer as a Skyfield object
        target: `object`, target as a Skyfield object
        t: `object`, skyfield time
        deflection: `boolean`, enable deflection adjustment
        aberration: `boolean`, enable aberration of light adjustment

    Returns:
        A `tuple`, containing:
            ra: right ascension in degrees
            dec: declination in degrees
            d: distance between observer and target in km
            az: azimuth angle in degrees
            el: elevation angle in degrees
            icrf_los: LOS as skyfield ICRF object
    """

    icrf_los = apparent(observer.at(t).observe(target), deflection, aberration)
    ra, dec, d = icrf_los.radec()
    el, az, d = icrf_los.altaz()

    return ra._degrees, dec._degrees, d.km, az._degrees, el._degrees, icrf_los


@lru_cache(maxsize=32)
def get_los_azel(observer, az, el, t, deflection=False, aberration=True):
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

    return ra._degrees, dec._degrees, d.km, az, el, icrf_los


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


def wcs_from_observer_rate(height, width, y_fov, x_fov, observer, t0, tt, rot, track):
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
        [ra0,dec0,d0,az0,el0,los0] = get_los(observer, track, t)
        wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
        wsc.append(wcs0)
    return wsc


def wcs_from_observer_fixed(height, width, y_fov, x_fov, observer, tt, rot, az, el):
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
    for t,a,e in zip(tt,az,el):
        [ra0,dec0,d0,az0,el0,los0] = get_los_azel(observer, a, e, t)
        wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
        wsc.append(wcs0)
    return wsc


def wcs_from_observer_sidereal(height, width, y_fov, x_fov, observer, t0, tt, rot, track):
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
    [ra0,dec0,d0,az0,el0,los0] = get_los(observer, track, t0)
    wcs0 = get_wcs(height, width, y_fov / height, x_fov / width, ra0, dec0, rot)
    for t in tt:
        wsc.append(wcs0)
    return wsc


def gen_track(height, width, y_fov, x_fov, observer, track, satellites, brightnesses, t0, tt, rot=0, pad_mult=0, track_type='rate',
              offset=[0,0], flipud=False, fliplr=False, az=None, el=None):
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

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row coordinates for each `target` at each time
            cc: `list`, list of column coordinates for each `target` at each time
            dr: `float`, star motion along the rows in pixel per second
            dc: `float`, star motion along the columns in pixel per second
            b: `float`, list of brightnesses of each target in Mv
    """
    if track_type == 'rate':
        wcs = wcs_from_observer_rate(height, width, y_fov, x_fov, observer, t0, tt, rot, track)
    elif track_type == 'fixed':
        wcs = wcs_from_observer_fixed(height, width, y_fov, x_fov, observer, tt, rot, az, el)
    else:
        wcs = wcs_from_observer_sidereal(height, width, y_fov, x_fov, observer, t0, tt, rot, track)

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
