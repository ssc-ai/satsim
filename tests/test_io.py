"""Tests for `satsim.io.fits` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os

import numpy as np
import pytest
from astropy.io import fits as afits

from satsim.io import fits, image


def test_save_fits(dtype='uint16', nptype=np.uint16):

    a = np.zeros([5,5])

    # top right corner of image
    a[0,4] = 1

    filename = './fits_file_for_test.fits'
    hdu, hdr = fits.save(filename, a, overwrite=True, dtype=dtype)

    assert(os.path.exists(filename))

    hdul = afits.open(filename)
    hdulhdr = hdul[0].header
    hduldata = hdul[0].data

    # check a few headers
    assert(hduldata.dtype == np.dtype(nptype))
    assert(hdulhdr['SIMPLE'] == hdr['SIMPLE'])
    assert(hdulhdr['EXPTIME'] == hdr['EXPTIME'])
    assert(hdulhdr['DATE-OBS'] == hdr['DATE-OBS'])

    # check images match
    np.testing.assert_array_equal(a, hduldata)

    os.remove(filename)


def test_save_fits_float32():

    test_save_fits('float32', '>f8')


def test_save_fits_int16():

    test_save_fits('int16', '>i2')


def test_save_fits_uint32():

    test_save_fits('uint32', np.uint32)


def test_save_fits_int32():

    test_save_fits('int32', '>i4')


def test_save_fits_compressed():

    a = np.zeros([5,5])
    a[0,4] = 1

    filename = './fits_file_for_test_compressed.fits'
    if os.path.exists(filename):
        os.remove(filename)

    hdu, hdr = fits.save(filename, a, overwrite=True, fits_compression='gzip')

    assert(os.path.exists(filename))

    with afits.open(filename) as hdul:
        hdulhdr = hdul[1].header
        hduldata = hdul[1].data
        assert(isinstance(hdul[1], afits.CompImageHDU))
        assert(hdul[1].compression_type == 'GZIP_1')
        assert(hdulhdr['EXPTIME'] == hdr['EXPTIME'])
        np.testing.assert_array_equal(a, hduldata)

    os.remove(filename)


def test_save_fits_invalid_compression():

    a = np.zeros([5,5])
    filename = './fits_file_for_test_invalid_compress.fits'
    if os.path.exists(filename):
        os.remove(filename)

    with pytest.raises(ValueError, match='Unknown FITS compression type'):
        fits.save(filename, a, overwrite=True, fits_compression='gzip_1')

    assert(not os.path.exists(filename))


def test_save_fits_labels_apparent_wcs(tmp_path):

    filename = tmp_path / 'apparent_wcs.fits'
    astrometrics = {
        'ra': 12.5,
        'dec': -4.25,
        'ra_apparent': 12.75,
        'dec_apparent': -4.0,
    }

    _, hdr = fits.save(
        str(filename),
        np.zeros([5, 5]),
        overwrite=True,
        dtype='float32',
        astrometrics=astrometrics,
    )

    assert hdr['CRVAL1'] == pytest.approx(astrometrics['ra_apparent'])
    assert hdr['CRVAL2'] == pytest.approx(astrometrics['dec_apparent'])
    assert hdr['RADESYS'] == 'GAPPT'
    assert 'EQUINOX' not in hdr
    assert hdr['RA-OBJ'] == pytest.approx(astrometrics['ra'])
    assert hdr['DEC-OBJ'] == pytest.approx(astrometrics['dec'])
    assert hdr['RA-APP'] == pytest.approx(astrometrics['ra_apparent'])
    assert hdr['DEC-APP'] == pytest.approx(astrometrics['dec_apparent'])

    with afits.open(filename) as hdul:
        saved = hdul[0].header
        assert saved['RADESYS'] == 'GAPPT'
        assert 'EQUINOX' not in saved
        assert saved['CRVAL1'] == pytest.approx(astrometrics['ra_apparent'])
        assert saved['CRVAL2'] == pytest.approx(astrometrics['dec_apparent'])


def test_save_fits_labels_icrs_wcs_without_apparent_coordinates(tmp_path):

    filename = tmp_path / 'icrs_wcs.fits'
    astrometrics = {
        'ra': 12.5,
        'dec': -4.25,
    }

    _, hdr = fits.save(
        str(filename),
        np.zeros([5, 5]),
        overwrite=True,
        dtype='float32',
        astrometrics=astrometrics,
    )

    assert hdr['CRVAL1'] == pytest.approx(astrometrics['ra'])
    assert hdr['CRVAL2'] == pytest.approx(astrometrics['dec'])
    assert hdr['RADESYS'] == 'ICRS'
    assert hdr['EQUINOX'] == pytest.approx(2000.0)
    assert hdr['RA-OBJ'] == pytest.approx(astrometrics['ra'])
    assert hdr['DEC-OBJ'] == pytest.approx(astrometrics['dec'])
    assert 'RA-APP' not in hdr
    assert 'DEC-APP' not in hdr

    with afits.open(filename) as hdul:
        saved = hdul[0].header
        assert saved['RADESYS'] == 'ICRS'
        assert saved['EQUINOX'] == pytest.approx(2000.0)
        assert saved['CRVAL1'] == pytest.approx(astrometrics['ra'])
        assert saved['CRVAL2'] == pytest.approx(astrometrics['dec'])


def test_save_image():

    a = np.zeros([5,5])

    # top right corner of image
    a[0,4] = 1
    b = {
        'y_min': 1,
        'y_max': 2,
        'x_min': 1,
        'x_max': 2,
        'class_name': 'Satellite'
    }

    filename = './png_file_for_test.png'
    image.save(filename, a, vauto=True, annotation=[b])

    assert(os.path.exists(filename))

    os.remove(filename)


def test_save_apng():

    a = np.zeros([5,5])

    a[0,4] = 1

    image.save('./png_file_for_test_0.png', a)
    image.save('./png_file_for_test_1.png', a)

    image.save_apng('./', 'movie.png')

    assert(os.path.exists('movie.png'))

    os.remove('./movie.png')
    os.remove('./png_file_for_test_0.png')
    os.remove('./png_file_for_test_1.png')
