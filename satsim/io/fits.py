from __future__ import division, print_function, absolute_import

import numpy as np

from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time, TimeDelta
from datetime import datetime, timezone
from copy import deepcopy

from satsim import __version__

_COMPRESSION_TYPES = {
    'none': None,
    'gzip': 'GZIP_1',
    'gzip2': 'GZIP_2',
    'rice': 'RICE_1',
    'hcompress': 'HCOMPRESS_1',
    'plio': 'PLIO_1',
}


def _normalize_compression(fits_compression):
    if fits_compression is None or fits_compression is False:
        return None
    if not isinstance(fits_compression, str):
        raise ValueError('fits_compression must be a string or None.')

    value = fits_compression.strip()
    if not value:
        return None

    key = value.lower()
    if key in _COMPRESSION_TYPES:
        return _COMPRESSION_TYPES[key]

    valid = ', '.join(sorted(_COMPRESSION_TYPES.keys()))
    raise ValueError('Unknown FITS compression type: {}. Valid values: {}.'.format(
        fits_compression, valid))


def _utc_time(value, name):
    """Read a scalar epoch; naive datetimes and strings denote UTC."""
    try:
        if value is None:
            raise ValueError('missing time')
        if isinstance(value, datetime) and value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        epoch = Time(value).utc if isinstance(value, Time) else Time(value, scale='utc')
        if not epoch.isscalar or not np.isfinite(epoch.jd1) or not np.isfinite(epoch.jd2):
            raise ValueError('not a finite scalar time')
    except (TypeError, ValueError) as exc:
        raise ValueError('{} must be a valid scalar datetime, Astropy Time or UTC date string.'.format(name)) from exc
    return epoch


def _exposure_timing(dt_start, exposure_time, exposure_end, astrometrics):
    """Resolve acquisition timing independently of the astrometric/WCS epoch.

    Explicit arguments and per-frame exposure metadata must agree to one
    microsecond (the serialized date precision). Legacy astrometric-only
    callers supply a midpoint, from which start is derived using duration.
    A legacy WCS DATE-OBS is a last timing reference before a call-time UTC
    default; it never overrides explicit timing or an astrometric midpoint.
    """
    try:
        if isinstance(exposure_time, (bool, str)) or np.ndim(exposure_time) != 0:
            raise ValueError('not a numeric scalar')
        duration = float(exposure_time)
        if not np.isfinite(duration) or duration < 0:
            raise ValueError('not a finite nonnegative duration')
    except (TypeError, ValueError) as exc:
        raise ValueError('exposure_time must be a finite nonnegative number of seconds.') from exc

    metadata = astrometrics if astrometrics is not None else {}

    def explicit_time(argument, key):
        result = _utc_time(argument, key) if argument is not None else None
        if key in metadata:
            other = _utc_time(metadata[key], key)
            if result is not None and abs((result - other).to_value('sec')) > 1e-6:
                raise ValueError('Conflicting explicit {} values.'.format(key))
            if result is None:
                result = other
        return result

    start = explicit_time(dt_start, 'exposure_start')
    end = explicit_time(exposure_end, 'exposure_end')
    if start is None:
        if 'time' in metadata:
            start = _utc_time(metadata['time'], "astrometrics['time']") - TimeDelta(duration / 2, format='sec')
        elif 'DATE-OBS' in metadata.get('wcs_header', {}):
            wcs = metadata['wcs_header']
            start = Time(wcs['DATE-OBS'], format='fits', scale=wcs.get('TIMESYS', 'UTC').lower()).utc
            start = _utc_time(start, 'WCS DATE-OBS')
        elif end is not None:
            raise ValueError('exposure_end requires an exposure start or legacy timing reference.')
        else:
            start = _utc_time(datetime.now(timezone.utc), 'current time')
    expected_end = start + TimeDelta(duration, format='sec')
    if end is None:
        end = expected_end
    elif abs((end - expected_end).to_value('sec')) > 1e-6:
        raise ValueError('Exposure boundaries conflict with exposure_time.')
    if end < start:
        raise ValueError('Exposure end precedes exposure start.')
    midpoint = start + (end - start) / 2
    return start, midpoint, end, duration


def _write_exposure_timing(hdr, start, midpoint, end, duration):
    """Apply timing last so WCS metadata cannot replace acquisition epochs."""
    dates = []
    for epoch in (start, midpoint, end):
        epoch = epoch.replicate()
        epoch.precision = 6
        dates.append(epoch.to_value('fits', subfmt='date_hms'))
    hdr['TIMESYS'] = ('UTC', 'Time scale of acquisition timestamps')
    hdr['DATE-OBS'] = (dates[0], 'Exposure start (UTC)')
    hdr['DATE-BEG'] = (dates[0], 'Exposure start (UTC)')
    hdr['DATE-AVG'] = (dates[1], 'Arithmetic midpoint (UTC); unweighted')
    hdr['DATE-END'] = (dates[2], 'Exposure end (UTC)')
    hdr['TIME-OBS'] = (dates[0].split('T')[1], 'Time of day at exposure start (UTC)')
    hdr['SH-UT001'] = (dates[0], 'Shutter exposure start (UTC)')
    hdr['SH-UT002'] = (dates[2], 'Shutter exposure end (UTC)')
    hdr['EXPTIME'] = (duration, 'Exposure duration in seconds; excludes gaps')
    hdr['EXPOSURE'] = (duration, 'Exposure duration in seconds; excludes gaps')
    hdr['MJD-OBS'] = (float(start.utc.mjd), 'MJD of exposure start (UTC)')
    for key, epoch in zip(('MJD-BEG', 'MJD-AVG', 'MJD-END'), (start, midpoint, end)):
        if key in hdr:
            hdr[key] = (float(epoch.utc.mjd), 'MJD of {} (UTC)'.format(key[4:].lower()))


def save(filename, fpa, exposure_time=0, dt_start=None, header={}, overwrite=False, dtype='uint16', astrometrics=None, fits_compression=None, exposure_end=None):
    """Save a SatNet compatible FITS file.

    Args:
        filename: `string`, the FITS filename.
        fpa: `np.array`, input image as a 2D numpy array.
        exposure_time: `float`, finite nonnegative exposure duration in seconds.
            Zero writes identical start, midpoint and end timestamps.
        dt_start: `datetime`, Astropy `Time` or UTC date string, exposure start.
            Aware datetimes are converted to UTC; naive datetimes mean UTC.
            If omitted, use astrometrics['exposure_start'], otherwise derive
            start from the legacy astrometrics['time'] midpoint and duration,
            otherwise use legacy WCS DATE-OBS, or current UTC at call time.
            Explicit timing sources must agree; WCS dates cannot override them.
        header: `dict`, placeholder, not implemented
        overwrite: `boolean`, if True overwrite file if it exists
        dtype: `string`, 'int16', 'uint16', 'int32', 'uint32', or 'float32'. default: 'int16'
        fits_compression: `string`, FITS compression type. Valid values: 'none', 'gzip',
            'gzip2', 'rice', 'hcompress', 'plio'.
        exposure_end: optional explicit end, in the same formats as dt_start.
            Also read from astrometrics['exposure_end']. Boundaries must match
            exposure_time within one microsecond. Dates are serialized as UTC
            FITS strings to six fractional digits, without timezone suffixes.
    """
    
    ver = 'SATSIM {}'.format(__version__)
    timing = _exposure_timing(dt_start, exposure_time, exposure_end, astrometrics)

    # copy data to be thread safe
    fpa_copy = np.copy(fpa)

    compression_type = _normalize_compression(fits_compression)
    if compression_type:
        hdu = fits.CompImageHDU(fpa_copy, compression_type=compression_type)
    else:
        hdu = fits.PrimaryHDU(fpa_copy)
    hdr = hdu.header

    def get_or_default(key, default=''):
        if astrometrics is None or key not in astrometrics:
            return default
        else:
            return astrometrics[key]

    def has_astrometric(key):
        return astrometrics is not None and key in astrometrics

    has_apparent_wcs = has_astrometric('ra_apparent') and has_astrometric('dec_apparent')

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
    hdr['CBLACK']   = int(np.amin(fpa.flatten()))
    hdr['CWHITE']   = int(np.median(fpa.flatten())*4)
    hdr['CTYPE1']   = 'RA---TAN'
    hdr['CTYPE2']   = 'DEC--TAN'
    hdr['CRVAL1']   = get_or_default('ra_apparent' if has_apparent_wcs else 'ra', 0)
    hdr['CRVAL2']   = get_or_default('dec_apparent' if has_apparent_wcs else 'dec', 0)
    hdr['RADESYS']  = 'GAPPT' if has_apparent_wcs else 'ICRS'
    if has_astrometric('ra'):
        hdr['RA-OBJ'] = get_or_default('ra')
    if has_astrometric('dec'):
        hdr['DEC-OBJ'] = get_or_default('dec')
    if has_apparent_wcs:
        hdr['RA-APP'] = get_or_default('ra_apparent')
        hdr['DEC-APP'] = get_or_default('dec_apparent')
    hdr['CRPIX1']   = fpa.shape[1] / 2.0
    hdr['CRPIX2']   = fpa.shape[0] / 2.0
    x_ifov = get_or_default('x_ifov', 0)
    y_ifov = get_or_default('y_ifov', 0)
    roll = float(get_or_default('roll', 0.0))
    theta = np.deg2rad(roll)
    hdr['CDELT1']   = x_ifov
    hdr['CDELT2']   = y_ifov
    # Apply focal plane roll to CD matrix (matches astropy CROTA handling).
    hdr['CD1_1']    = x_ifov * np.cos(theta)
    hdr['CD1_2']    = -y_ifov * np.sin(theta)
    hdr['CD2_1']    = x_ifov * np.sin(theta)
    hdr['CD2_2']    = y_ifov * np.cos(theta)
    hdr['CROTA1']   = roll
    hdr['CROTA2']   = roll
    if not has_apparent_wcs:
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

    if astrometrics is not None and 'wcs_header' in astrometrics:
        # Replace the independently synthesized legacy TAN header with the
        # validated sky-coordinate approximation of the detector mapping.
        for key in list(hdr):
            if key.startswith(('CTYPE', 'CRPIX', 'CRVAL', 'CDELT', 'CROTA', 'CD1_', 'CD2_', 'PC1_', 'PC2_')):
                del hdr[key]
        hdr.update(deepcopy(astrometrics['wcs_header']))

    _write_exposure_timing(hdr, *timing)
    hdu.writeto(filename, overwrite=overwrite)

    return hdu, hdr
