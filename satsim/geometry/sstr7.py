import math
import os
import struct
from functools import lru_cache

import numpy as np

from satsim.geometry.wcs import get_min_max_ra_dec

DEFAULT_SSTR7_PATH = os.environ['SATSIM_SSTR7_PATH'] if 'SATSIM_SSTR7_PATH' in os.environ else '/workspace/share/sstrc7'
RECORD_LEN = 30
RECORD_LEN_BYTES = RECORD_LEN * 2

# read_star and _decode_records multiply by these same Python floats, which
# keeps the two parsers bit-identical.
_MILLIARCSEC = 4.84813681109535993589914102358e-9
_YEAR = 3.1556952e7
_ANGLE_SCALE = _MILLIARCSEC  # to radians
_PROPER_MOTION_SCALE = 0.32 * _MILLIARCSEC / _YEAR  # to radians / sec
_PARALLAX_SCALE = 0.032 * _MILLIARCSEC
_MV_SCALE = 1.0e-3

_MV_PRIORITY = (
    0,  # Open
    5,  # Johnson_R
    8,  # Sloan_r
    4,  # Johnson_V
    3,  # Johnson_B
)
_MV_MISSING = 32

# One on-disk record, in read_star's native byte order. np.frombuffer raises
# only when a buffer is not a whole number of records; _load_zone_stars checks
# the byte count to catch truncation.
_RECORD_DTYPE = np.dtype([
    ('ra', '=i4'),
    ('dec', '=i4'),
    ('ra_pm', '=i2'),
    ('dec_pm', '=i2'),
    ('parallax', '=i2'),
    ('mv', '=i2', (18,)),
    ('_unused', 'V{}'.format(RECORD_LEN_BYTES - 50)),
])

# One decoded star, 48 bytes; field names are the read_star dict keys.
_STAR_DTYPE = np.dtype([(name, np.float64) for name in ('ra', 'dec', 'ra_pm', 'dec_pm', 'parallax', 'mv')])


@lru_cache(maxsize=10)
def load_index(filename, numRaZones=60, numDecZones=1800):
    """Map SSTRC index file entries to index vector and RA and Dec index maps.
    The SSTRC index files differ from the SSTRC catalog accelerator (index)
    files in that the only the zone position and length are stored as binary
    unsigned integers in a single file.

    Args:
        filename: `int`, index filename.
        numRaZones: `int`, number of RA zones. default: 60
        numDecZones: `int`, number of Dec zones. default 1800

    Returns:
        A `list`, list of dictionaries with zone position and length
    """

    zoneIndex =  [[{}] * numRaZones for i in range(numDecZones)]

    with open(filename, mode='rb') as f:
        data = np.fromfile(f, dtype=np.dtype('<u4'))
        i = 0
        for decIndex in range(numDecZones):
            for raIndex in range(numRaZones):
                zoneIndex[decIndex][raIndex] = {'pos': data[i], 'length': data[i + 1]}
                i = i + 2

    return zoneIndex


def select_zone(ra_min, ra_max, dec_min, dec_max, zoneIndex, numRaZones=60, numDecZones=1800):
    """Select a list of regions that intersect with the rectangular coordinate
    bounds

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        zoneIndex: `list`, list of zones
        numRaZones: `int`, number of RA zones. default: 60
        numDecZones: `int`, number of Dec zones. default 1800

    Returns:
        A `list`, list of dictionaries with zone position and length that
            encompass the ra/dec min max
    """

    decZoneLimit = numDecZones - 1
    raZoneLimit = numRaZones - 1

    zoneHeight = math.pi / numDecZones
    zoneWidth = 2.0 * math.pi / numRaZones

    selectZoneList = []

    spd_min = math.pi / 2.0 + dec_min
    spd_max = math.pi / 2.0 + dec_max

    minSPDIndex = max(int(spd_min / zoneHeight), 0)
    maxSPDIndex = min(int(spd_max / zoneHeight), decZoneLimit)

    minRAIndex = 0
    maxRAIndex = 0

    def append_zones(minRAIndex, maxRAIndex, bound_func):

        for spd in range(minSPDIndex, maxSPDIndex + 1):
            for ra in range(minRAIndex, maxRAIndex + 1):
                selectZone = {
                    'id': int(spd),
                    'pos': zoneIndex[spd][ra]['pos'],
                    'length': zoneIndex[spd][ra]['length']
                }
                if bound_func == 0:
                    selectZone['bound'] = 'maxRA' if ((ra + 1) * zoneWidth > ra_max) else ('minRA' if (ra * zoneWidth < ra_min) else 'inside')
                elif bound_func == 1:
                    selectZone['bound'] = 'minRA' if (ra * zoneWidth < ra_max) else 'inside'
                elif bound_func == 2:
                    selectZone['bound'] = 'maxRA' if (ra * zoneWidth > ra_min) else 'inside'

                if selectZone['length'] > 0:
                    selectZoneList.append(selectZone)

    # Check to see if catalog search bounds crosses 0 deg Right Ascension and
    # if so, query from the minimum RA coordinate to the end of the SPD band
    # (360 deg RA) and from the start of the SPD band (0 deg RA) to the maximum
    # RA coordinate
    if ra_min <= ra_max:
        minRAIndex = max(int(ra_min / zoneWidth), 0)
        maxRAIndex = min(int(ra_max / zoneWidth), raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 0)

    # Search bounds cross 0 deg RA
    else:
        minRAIndex = max(int(ra_min / zoneWidth) - 1, 0)
        maxRAIndex = raZoneLimit
        append_zones(minRAIndex, maxRAIndex, 1)

        minRAIndex = 0
        maxRAIndex = min(int(ra_max / zoneWidth) + 1, raZoneLimit)
        append_zones(minRAIndex, maxRAIndex, 2)

    return selectZoneList


def load_zone(currentZone, rootPath):
    """Load the region with the records from the SSTRC catalog data file
    between the beginning and ending records for the specified index entry.
    """
    # Next, assure the proper zone catalog file is opened. Seek to the proper
    # file offset and read the entire zone region into the zone buffer
    filename = os.path.join(rootPath, 's{:04d}.cat'.format(currentZone['id']))

    with open(filename, 'rb') as file:

        zoneBufferFilePos = currentZone['pos']
        zoneBufferOffset = zoneBufferFilePos * RECORD_LEN_BYTES
        file.seek(zoneBufferOffset)

        # Read the region star data into memory
        zoneBufferLen = currentZone['length'] * RECORD_LEN_BYTES
        zoneBuffer = file.read(zoneBufferLen)

        zoneBufferPos = zoneBufferOffset
        zoneBufferEnd = zoneBufferOffset + zoneBufferLen

        return zoneBuffer, zoneBufferPos, zoneBufferEnd


@lru_cache(maxsize=1024)
def _load_zone_stars(id, pos, length, rootPath):
    """Load and decode every star in one catalog zone.

    Args:
        id: `int`, zone (SPD band) number, selects s{id:04d}.cat
        pos: `int`, index of the zone's first record in that file
        length: `int`, number of records in the zone
        rootPath: `string`, path to root directory

    Returns:
        A read-only `numpy.ndarray` of `_STAR_DTYPE`, one element per record
        in file order, shared by every later query of this zone
    """
    buffer, _, _ = load_zone({'id': id, 'pos': pos, 'length': length}, rootPath)
    if len(buffer) != length * RECORD_LEN_BYTES:
        raise ValueError('truncated zone in s{:04d}.cat: expected {} bytes at record {}, read {}'.format(
            id, length * RECORD_LEN_BYTES, pos, len(buffer)))
    stars = _decode_records(np.frombuffer(buffer, dtype=_RECORD_DTYPE))
    stars.flags.writeable = False
    return stars


def _decode_records(records):
    """Vectorized `read_star` over a zone.

    Args:
        records: `numpy.ndarray` of `_RECORD_DTYPE`, raw catalog records

    Returns:
        A `numpy.ndarray` of `_STAR_DTYPE`; element i equals `read_star` of
        record i bit for bit
    """
    # read_star multiplies a Python int by a Python float. Casting to float64
    # first (exact for int32 and int16) and multiplying by the same constant
    # gives the same bits under any numpy scalar promotion rules.
    stars = np.empty(len(records), dtype=_STAR_DTYPE)
    stars['ra'] = records['ra'].astype(np.float64) * _ANGLE_SCALE
    stars['dec'] = records['dec'].astype(np.float64) * _ANGLE_SCALE
    stars['ra_pm'] = records['ra_pm'].astype(np.float64) * _PROPER_MOTION_SCALE
    stars['dec_pm'] = records['dec_pm'].astype(np.float64) * _PROPER_MOTION_SCALE
    stars['parallax'] = records['parallax'].astype(np.float64) * _PARALLAX_SCALE
    mv = records['mv'].astype(np.float64) * _MV_SCALE
    best = np.full(len(records), float(_MV_MISSING))
    # walk the priority list backwards so the first band below 32 is written last
    for band in reversed(_MV_PRIORITY):
        best = np.where(mv[:, band] < _MV_MISSING, mv[:, band], best)
    stars['mv'] = best
    return stars


def _query_stars(ra_min, ra_max, dec_min, dec_max, rootPath, clip_min_max):
    """Query the catalog for the stars within minimum and maximum right
    ascension and declination.

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        rootPath: `string`, path to root directory
        clip_min_max: `boolean`, clip stars outside of `ra_min` and `ra_max`

    Returns:
        A new `numpy.ndarray` of `_STAR_DTYPE`: zones in `select_zone` order,
        each in reversed record order for `minRA` zones and file order
        otherwise
    """
    zoneIndex = load_index(os.path.join(rootPath, 'sstrc.acc'), numRaZones=60, numDecZones=1800)

    parts = []
    for z in select_zone(ra_min, ra_max, dec_min, dec_max, zoneIndex, numRaZones=60, numDecZones=1800):
        stars = _load_zone_stars(z['id'], z['pos'], z['length'], rootPath)
        if z['bound'] == 'minRA':
            # a view, so the cache holds each zone once in file order
            stars = stars[::-1]
        if clip_min_max and z['bound'] != 'inside':
            # records within a zone are in RA order, so cutting at the first
            # star past the edge drops every star beyond it
            past_edge = stars['ra'] < ra_min if z['bound'] == 'minRA' else stars['ra'] > ra_max
            first = np.flatnonzero(past_edge)
            if first.size:
                stars = stars[:first[0]]
        parts.append(stars)

    if not parts:
        return np.empty(0, dtype=_STAR_DTYPE)
    return np.concatenate(parts)


def clear_star_cache():
    """Release every cached catalog zone.

    Queries cache each zone they read (up to 1024 zones, about 48 bytes per
    star, or about 2 GB for the densest SSTRC7 zones) so later queries near
    the same pointing skip the disk. A long-running process that renders many
    pointings can call this between renders to return that memory; the next
    query re-reads its zones from disk. Arrays already returned by
    `query_by_los` or `query_by_min_max` stay valid.

    The zone index read from `sstrc.acc` stays cached: it has a fixed size
    per catalog path and does not grow with the number of pointings.
    """
    _load_zone_stars.cache_clear()


def query_by_los(height, width, y_fov, x_fov, ra, dec, rot=0, rootPath=DEFAULT_SSTR7_PATH, pad_mult=0,
                 origin='center', filter_ob=True, flipud=False, fliplr=False, sky_bounds=None):
    """Query the catalog based on focal plane parameters and ra and dec line
    of sight vector. Line of sight vector is defined as the top left corner
    of the focal plane array.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
        sky_bounds: optional ``(minimum, maximum)`` RA/Dec bounds in degrees;
            replaces the rectangular query footprint, e.g. for polynomial distortion.
        ra: `float`, right ascension of top left corner of array, [0,0]
        dec: `float`, declination of top left corner of array, [0,0]
        rot: `float`, focal plane rotation
        rootPath: path to root directory. default: environment variable SATSIM_SSTR7_PATH
        pad_mult: `float`, padding multiplier
        origin: `string`, if `center`, rr and cc will be defined where the line of sight is at the center of
            the focal plane array. default='center'
        filter_ob: `boolean`, remove stars outside pad
        flipud: `boolean`, flip row coordinates
        fliplr: `boolean`, flip column coordinates

    Returns:
        A `tuple`, containing:
            rr: `list`, list of row pixel locations
            cc: `list`, list of column pixel locations
            mv: `list`, list of visual magnitudes
            rra: `list`, list of RA positions in degrees
            ddec: `list`, list of declination positions in degrees
    """

    cmin, cmax, w = get_min_max_ra_dec(height, width, y_fov / height, x_fov / width, ra, dec, rot, pad_mult, origin)
    if sky_bounds is not None:
        cmin, cmax = sky_bounds

    cmin = np.radians(cmin)
    cmax = np.radians(cmax)

    stars = _query_stars(cmin[0], cmax[0], cmin[1], cmax[1], rootPath, clip_min_max=True)

    rra = np.degrees(stars['ra'])
    ddec = np.degrees(stars['dec'])
    mm = stars['mv'].copy()

    cc, rr = w.wcs_world2pix(rra, ddec, 0)

    if filter_ob:
        hp = height * (1 + pad_mult)
        wp = width * (1 + pad_mult)
        in_bounds = np.logical_and.reduce([rr <= hp, rr >= -hp, cc <= wp, cc >= -wp])
        rr = rr[in_bounds]
        cc = cc[in_bounds]
        mm = mm[in_bounds]
        rra = rra[in_bounds]
        ddec = ddec[in_bounds]

    if origin == 'center':
        rr += height / 2.0
        cc += width / 2.0

    if flipud:
        rr = height - rr

    if fliplr:
        cc = width - cc

    return rr, cc, mm, rra, ddec


def query_by_min_max(ra_min, ra_max, dec_min, dec_max, rootPath=DEFAULT_SSTR7_PATH, clip_min_max=True):
    """Query the catalog based on focal plane parameters and minimum and
    maximum right ascension and declination.

    Args:
        ra_min: `float`, min RA bounds
        ra_max: `float`, max RA bounds
        dec_min: `float`, min dec bounds
        dec_max: `float`, max dec bounds
        rootPath: `string`, path to root directory. default: environment
            variable SATSIM_SSTR7_PATH
        clip_min_max: `boolean`, clip stars outsize of `ra_min` and `ra_max`

    Returns:
        A `list`, stars within the bounds of input parameters
    """

    stars = _query_stars(ra_min, ra_max, dec_min, dec_max, rootPath, clip_min_max)
    return [dict(zip(_STAR_DTYPE.names, values)) for values in stars.tolist()]


def read_star(buffer):
    """Reads a byte buffer and parses star parameters.

    Args:
        buffer: `list`, byte array of length 60 bytes

    Returns:
        A `dict`, the star position and magnitudes
    """

    raw = []
    raw = struct.unpack('=iihhh', buffer[0:14])

    raw_mv = []
    raw_mv = struct.unpack('=hhhhhhhhhhhhhhhhhh', buffer[14:14 + 18 * 2])

    ra, dec, ra_pm, dec_pm, parallax = raw[0:5]

    return {
        'ra': ra * _ANGLE_SCALE,
        'dec': dec * _ANGLE_SCALE,
        'ra_pm': ra_pm * _PROPER_MOTION_SCALE,
        'dec_pm': dec_pm * _PROPER_MOTION_SCALE,
        'parallax': parallax * _PARALLAX_SCALE,
        'mv': get_star_mv(np.asarray(raw_mv) * _MV_SCALE)
    }


def get_star_mv(mv):
    """Gets the best visual magnitude available to be used for simulation.

    Args:
        star: `list`, list of star magnitudes, see `read_star`

    Returns:
        A `float`, the visual magnitude
    """
    for band in _MV_PRIORITY:
        if mv[band] < _MV_MISSING:
            return mv[band]

    return _MV_MISSING
