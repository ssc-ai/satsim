import math
import os
import struct
from functools import lru_cache

import numpy as np

from satsim.geometry.wcs import get_min_max_ra_dec

DEFAULT_SSTR7_PATH = os.environ['SATSIM_SSTR7_PATH'] if 'SATSIM_SSTR7_PATH' in os.environ else '/workspace/share/sstrc7'
RECORD_LEN = 30
RECORD_LEN_BYTES = RECORD_LEN * 2


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
def load_stars_for_zone(id, pos, length, bound, rootPath):  # ra_min, ra_max):
    """Load stars for specified zone.
    """
    buffer, start, end = load_zone({"id": id, "pos": pos, "length": length}, rootPath)
    stars = []
    if bound == 'minRA':
        for s in [i * RECORD_LEN_BYTES for i in reversed(range((end - start) // RECORD_LEN_BYTES))]:
            star = read_star(buffer[s:s + RECORD_LEN_BYTES])
            stars.append(star)
            # if star['ra'] < ra_min:
            #     print('load minRA:', len(stars))
            #     break
    else:
        for s in [i * RECORD_LEN_BYTES for i in range((end - start) // RECORD_LEN_BYTES)]:
            star = read_star(buffer[s:s + RECORD_LEN_BYTES])
            stars.append(star)
            # if bound == 'maxRA' and star['ra'] > ra_max:
            #     print('load maxRA:', len(stars))
            #     break

    return stars


def query_by_los(height, width, y_fov, x_fov, ra, dec, rot=0, rootPath=DEFAULT_SSTR7_PATH, pad_mult=0,
                 origin='center', filter_ob=True, flipud=False, fliplr=False):
    """Query the catalog based on focal plane parameters and ra and dec line
    of sight vector. Line of sight vector is defined as the top left corner
    of the focal plane array.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_fov: `float`, y fov in degrees
        x_fov: `float`, x fov in degrees
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

    cmin = np.radians(cmin)
    cmax = np.radians(cmax)

    stars = query_by_min_max(cmin[0], cmax[0], cmin[1], cmax[1], rootPath)

    rra = np.degrees(np.array([s['ra'] for s in stars]))
    ddec = np.degrees(np.array([s['dec'] for s in stars]))
    mm = np.array([s['mv'] for s in stars])

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

    zoneIndex = load_index(os.path.join(rootPath, 'sstrc.acc'), numRaZones=60, numDecZones=1800)

    zones = select_zone(ra_min, ra_max, dec_min, dec_max, zoneIndex, numRaZones=60, numDecZones=1800)

    stars = []
    for z in zones:
        ss = load_stars_for_zone(
            z["id"],
            z["pos"],
            z["length"],
            z["bound"],
            rootPath
        )

        if clip_min_max:

            def clip_stars():
                if z['bound'] == 'minRA':
                    for i in range(len(ss)):
                        if ss[i]['ra'] < ra_min:
                            return ss[0:i]
                else:
                    # break for `maxRA` and not `inside`
                    for i in range(len(ss)):
                        if z['bound'] == 'maxRA' and ss[i]['ra'] > ra_max:
                            return ss[0:i]
                return ss

            ss = clip_stars()

        stars += ss

    return stars


def read_star(buffer):
    """Reads a byte buffer and parses star parameters.

    Args:
        buffer: `list`, byte array of length 60 bytes

    Returns:
        A `dict`, the star position and magnitudes
    """

    milliarcsec = 4.84813681109535993589914102358e-9
    year = 3.1556952e7

    angleScale = milliarcsec  # to radians
    properMotionScale = 0.32 * milliarcsec / year  # to radians / sec
    parallaxScale = 0.032 * milliarcsec

    raw = []
    raw = struct.unpack('=iihhh', buffer[0:14])

    raw_mv = []
    raw_mv = struct.unpack('=hhhhhhhhhhhhhhhhhh', buffer[14:14 + 18 * 2])

    ra, dec, ra_pm, dec_pm, parallax = raw[0:5]

    return {
        'ra': ra * angleScale,
        'dec': dec * angleScale,
        'ra_pm': ra_pm * properMotionScale,
        'dec_pm': dec_pm * properMotionScale,
        'parallax': parallax * parallaxScale,
        'mv': get_star_mv(np.asarray(raw_mv) * 1.0e-3)
    }


def get_star_mv(mv):
    """Gets the best visual magnitude available to be used for simulation.

    Args:
        star: `list`, list of star magnitudes, see `read_star`

    Returns:
        A `float`, the visual magnitude
    """
    if mv[0] < 32:    # Open
        return mv[0]
    elif mv[5] < 32:  # Johnson_R
        return mv[5]
    elif mv[8] < 32:  # Sloan_r
        return mv[8]
    elif mv[4] < 32:  # Johnson_V
        return mv[4]
    elif mv[3] < 32:  # Johnson_B
        return mv[3]

    return 32
