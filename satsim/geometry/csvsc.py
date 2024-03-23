import csv
import numpy as np
from functools import lru_cache

from satsim.geometry.wcs import get_min_max_ra_dec


@lru_cache(maxsize=1024)
def _load(rootPath):
    """ Helper function to load data from a csv file."""

    file = open(rootPath, "r")
    data = list(csv.reader(file, delimiter=","))
    file.close()

    def m(x):
        return [int(x[0]), float(x[1]), float(x[2]), float(x[3])]

    def f(x):
        if x[2].strip() == '':
            return False
        return True

    data = np.array(list(map(m, filter(f, data))))

    rra = data[:,2]
    ddec = data[:,3]
    mm = data[:,1]

    return rra, ddec, mm


def query_by_los(height, width, y_fov, x_fov, ra, dec, rot=0, rootPath="hip_main.txt",
                 origin='center', flipud=False, fliplr=False):
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
        rootPath: path to root directory. default: "hip_main.txt"
        origin: `string`, if `center`, rr and cc will be defined where the line of sight is at the center of
            the focal plane array. default='center'
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

    cmin, cmax, w = get_min_max_ra_dec(height, width, y_fov / height, x_fov / width, ra, dec, rot, 0, origin)

    rra, ddec, mm = _load(rootPath)

    cc, rr = w.wcs_world2pix(rra, ddec, 0)

    if origin == 'center':
        rr += height / 2.0
        cc += width / 2.0

    if flipud:
        rr = height - rr

    if fliplr:
        cc = width - cc

    return rr, cc, mm, rra, ddec
