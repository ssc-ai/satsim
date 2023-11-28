"""Tests for `satsim.io.fits` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os

import numpy as np
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
