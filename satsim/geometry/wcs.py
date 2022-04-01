import numpy as np
import astropy.wcs as wcs


def get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot=0):
    """Get an AstroPy world coordinate system (WCS) object used to transform
    RA, Dec coordinates to focal plane array pixel coordinates.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_ifov: `float`, y i-fov in degrees
        x_ifov: `float`, x i-fov in degrees
        ra: `float`, right ascension at pixel 0,0
        dec: `float`, declination of at pixel 0,0
        rot: `float`, rotation of the focal plane in degrees

    Returns:
        A `WCS`, used to transform RA, Dec to pixel coordinates
    """

    # TODO move to center
    crpix = [1, 1]

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = crpix
    w.wcs.cdelt = np.array([x_ifov, y_ifov])
    w.wcs.crval = np.array([ra, dec])
    w.wcs.crota = [rot, rot]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]

    return w


def get_min_max_ra_dec(height, width, y_ifov, x_ifov, ra, dec, rot=0, pad_mult=0, origin='center', offset=[0,0]):
    """Get the min and max RA and Dec bounds based on focal plane parameters.

    Args:
        height: `int`, height in number of pixels
        width: `int`, width in number of pixels
        y_ifov: `float`, y ifov in degrees
        x_ifov: `float`, x ifov in degrees
        ra: `float`, right ascension of `origin`
        dec: `float`, declination of `origin`
        rot: `float`, focal plane rotation
        pad_mult: `float`, padding multiplier
        origin: `string`, corner or center
        offset: `float`, array specifying [row, col] offset in pixels

    Returns:
        A `tuple`, containing:
            cmin: `array`, minimum ra and dec
            cmax: `array`, maximum ra and dec
            wcs: `WCS`, used to transform RA, Dec to pixel coordinates

    """

    crpix = 0

    w = get_wcs(height, width, y_ifov, x_ifov, ra, dec, rot)

    # pixcrd = np.array([[1,1],[1,height+1],[width+1,1],[width+1,height+1]], np.float_)
    hp = height * pad_mult
    wp = width * pad_mult
    pixcrd = np.array([
        [crpix - wp, crpix - hp],
        [crpix - wp, crpix + height * 0.5],
        [crpix - wp, crpix + height + hp],
        [crpix + width * 0.5, crpix + height + hp],
        [crpix + width + wp, crpix + height + hp],
        [crpix + width + wp, crpix + height * 0.5],
        [crpix + width + wp, crpix - hp],
        [crpix + width * 0.5, crpix - hp]], np.float_)

    center = np.array([[width / 2.0, height / 2.0]])

    if origin == 'center':
        pixcrd[:,0] -= width / 2.0 + offset[1]
        pixcrd[:,1] -= height / 2.0 + offset[0]
        center[:,0] -= width / 2.0 + offset[1]
        center[:,1] -= height / 2.0 + offset[0]

    # Convert pixel coordinates to world coordinates
    world = w.wcs_pix2world(pixcrd, 1)
    cworld = w.wcs_pix2world(center, 1)

    [cra, cdec] = cworld[0]

    [minTheta, minPhi] = np.min(world, axis=0)
    [maxTheta, maxPhi] = np.max(world, axis=0)

    northpole = w.wcs_world2pix([[0,89.99999]], 1)[0]
    southpole = w.wcs_world2pix([[0,-89.99999]], 1)[0]

    if not np.any(np.isnan(northpole)) and northpole[0] > 0 and northpole[0] < width and northpole[1] > 0 and northpole[1] < height:
        cmin = [0, minPhi]
        cmax = [360.0, 90.0]
    elif not np.any(np.isnan(southpole)) and southpole[0] > 0 and southpole[0] < width and southpole[1] > 0 and southpole[1] < height:
        cmin = [0, -90.0]
        cmax = [360.0, maxPhi]
    elif cra > maxTheta or cra < minTheta or (maxTheta - minTheta) > 180:
        # Including theta meridian crossing
        cmin = [maxTheta, minPhi]
        cmax = [minTheta, maxPhi]
    else:
        cmin = [minTheta, minPhi]
        cmax = [maxTheta, maxPhi]

    return cmin, cmax, w
