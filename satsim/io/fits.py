from __future__ import division, print_function, absolute_import

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from datetime import datetime, timedelta

from satsim import __version__


def save(filename, fpa, exposure_time=0, dt_start=datetime.now(), header={}, overwrite=False, dtype='uint16', astrometrics=None):
    """Save a SatNet compatible FITS file.

    Args:
        filename: `string`, the FITS filename.
        fpa: `np.array`, input image as a 2D numpy array.
        exposure_time: `float`, exposure time for header.
        dt_start: `datetime`, datetime of start of exposure.
        header: `dict`, placeholder, not implemented
        overwrite: `boolean`, if True overwrite file if it exists
        dtype: `string`, 'int16', 'uint16', 'int32', 'uint32', or 'float32'. default: 'int16'
    """
    
    ver = 'SATSIM {}'.format(__version__)

    # copy data to be thread safe
    fpa_copy = np.copy(fpa)

    hdu = fits.PrimaryHDU(fpa_copy)
    hdr = hdu.header

    if astrometrics is not None and 'time' in astrometrics:
        dt_start = astrometrics['time']

    def get_or_default(key, default=''):
        if astrometrics is None or key not in astrometrics:
            return default
        else:
            return astrometrics[key]

    rd = SkyCoord(ra=get_or_default('ra', 0) * u.degree, dec=get_or_default('dec', 0) * u.degree, frame='icrs')
    (ra, dec) = rd.to_string('hmsdms').split(' ')
    ra = ra.replace('h', ' ').replace('m', ' ').replace('s', ' ')
    dec = dec.replace('d', ' ').replace('m', ' ').replace('s', ' ')

    orchcomm = '{}#[{}:{}]@[{}]%[filter wheel disabled by user]'.format(
        get_or_default('track_mode'), get_or_default('frame_num'), get_or_default('num_frames'), get_or_default('site', 'sim'))

    # flake8: noqa
    hdr['BIAS']     = get_or_default('bias', 0)
    hdr['FOCALLEN'] = 0
    hdr['APTAREA']  = 0
    hdr['APTDIA']   = 0
    hdr['OBSERVER'] = get_or_default('observer')
    hdr['DATE-OBS'] = dt_start.isoformat()
    hdr['TIME-OBS'] = str(dt_start.time())
    hdr['SWCREATE'] = ver
    hdr['SET-TEMP'] = 0
    hdr['COLORCCD'] = 0
    hdr['DISPCOLR'] = 1
    hdr['IMAGETYP'] = 'Light Frame'
    hdr['CCDSFPT']  = 1
    hdr['XORGSUBF'] = 0
    hdr['YORGSUBF'] = 0
    hdr['CCDSUBFL'] = 0
    hdr['CCDSUBFT'] = 0
    hdr['XBINNING'] = 2
    hdr['CCDXBIN']  = 2
    hdr['YBINNING'] = 2
    hdr['CCDYBIN']  = 2
    hdr['CCD-TEMP'] = 0
    hdr['TEMPERAT'] = 0
    hdr['OBJECT']   = get_or_default('object')
    hdr['OBJCTRA']  = ra
    hdr['OBJCTDEC'] = dec
    hdr['TELTKRA']  = get_or_default('ra_rate')
    hdr['TELTKDEC'] = get_or_default('dec_rate')
    az_val = get_or_default('az')
    az_val = 0 if az_val != az_val else az_val
    el_val = get_or_default('el')
    el_val = 0 if el_val != el_val else el_val
    hdr['CENTAZ']   = az_val
    hdr['CENTALT']  = el_val
    hdr['TELHA']    = ''
    hdr['LST']      = ''
    hdr['AIRMASS']  = 0
    if 'x' in astrometrics if astrometrics is not None else False:
        hdr['SITEX'] = get_or_default('x', 0)
        hdr['SITEY'] = get_or_default('y', 0)
        hdr['SITEZ'] = get_or_default('z', 0)
        hdr['SITEVX'] = get_or_default('vx', 0)
        hdr['SITEVY'] = get_or_default('vy', 0)
        hdr['SITEVZ'] = get_or_default('vz', 0)
    else:
        hdr['SITELAT']  = get_or_default('lat')
        hdr['SITELONG'] = get_or_default('lon')
        hdr['SITEALT']  = get_or_default('alt')
    hdr['ORCHCOMM'] = orchcomm
    hdr['INSTRUME'] = ''
    hdr['FILTER']   = ''
    hdr['EXPTIME']  = exposure_time
    hdr['EXPOSURE'] = exposure_time
    hdr['CBLACK']   = int(np.amin(fpa.flatten()))
    hdr['CWHITE']   = int(np.median(fpa.flatten())*4)
    hdr['SH-UT001'] = dt_start.isoformat()
    hdr['SH-UT002'] = (dt_start + timedelta(seconds=exposure_time)).isoformat()
    hdr['CTYPE1']   = 'RA---TAN'
    hdr['CTYPE2']   = 'DEC--TAN'
    hdr['CRVAL1']   = get_or_default('ra', 0)
    hdr['CRVAL2']   = get_or_default('dec', 0)
    hdr['CRPIX1']   = fpa.shape[0] / 2.0
    hdr['CRPIX2']   = fpa.shape[1] / 2.0
    hdr['CDELT1']   = get_or_default('x_ifov', 0)
    hdr['CDELT2']   = get_or_default('y_ifov', 0)
    hdr['CD1_1']    = get_or_default('x_ifov', 0)
    hdr['CD1_2']    = 0.0
    hdr['CD2_1']    = 0.0
    hdr['CD2_2']    = get_or_default('y_ifov', 0)
    hdr['CROTA1']   = 0.0
    hdr['CROTA2']   = 0.0
    hdr['EQUINOX']  = 2000.0
    hdr["TRKMODE"]  = get_or_default('track_mode')

    if dtype == 'uint16':
        hdu.scale('int16', bzero=32768)
        hdr['BSCALE'] = 1
    elif dtype == 'int16':
        hdu.scale('int16')
    elif dtype == 'uint32':
        hdu.scale('int32', bzero=2147483648)
        hdr['BSCALE'] = 1
    elif dtype == 'int32':
        hdu.scale('int32')
    elif dtype == 'float32':
        pass

    hdu.writeto(filename, overwrite=overwrite)

    return hdu, hdr
