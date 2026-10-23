"""Tests for `satsim.io.fits` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import os
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from astropy.io import fits as afits
from astropy.time import Time, TimeDelta

from satsim.io import fits, image


def _assert_exposure_header(header, start, duration):
    start = Time(start, scale='utc')
    for keyword, seconds in [('DATE-OBS', 0), ('DATE-BEG', 0),
                             ('DATE-AVG', duration / 2), ('DATE-END', duration)]:
        value = header[keyword]
        assert len(value.split('.')[1]) == 6
        assert not value.endswith('Z') and '+' not in value
        actual = Time(value, format='fits', scale='utc')
        # Date strings have microsecond precision; compare split-JD times.
        assert abs((actual - (start + TimeDelta(seconds, format='sec'))).sec) < 1.1e-6
    assert header['DATE-BEG'] == header['DATE-OBS'] == header['SH-UT001']
    assert header['DATE-END'] == header['SH-UT002']
    assert header['TIME-OBS'] == header['DATE-OBS'].split('T')[1]
    assert header['TIMESYS'] == 'UTC'
    assert header['EXPTIME'] == header['EXPOSURE'] == duration
    # Scalar MJD cards have lower precision than split-JD time arithmetic.
    assert abs((header['MJD-OBS'] - start.mjd) * 86400) < 1e-5
    assert 'start' in header.comments['DATE-OBS'].lower()
    assert 'arithmetic' in header.comments['DATE-AVG'].lower()
    assert 'unweighted' in header.comments['DATE-AVG'].lower()


@pytest.mark.parametrize('compression', [None, 'gzip'])
@pytest.mark.parametrize('kind', ['naive', 'utc', 'offset', 'astropy_tt'])
def test_exposure_dates_are_utc_and_roll_over_midnight(tmp_path, compression, kind):
    start = datetime(2026, 10, 22, 23, 59, 59, 123456, tzinfo=timezone.utc)
    value = {'naive': start.replace(tzinfo=None), 'utc': start,
             'offset': start.astimezone(timezone(timedelta(hours=-10))),
             'astropy_tt': Time(start).tt}[kind]
    filename = tmp_path / 'timing.fits'
    fits.save(filename, np.arange(20).reshape(4, 5), 3.125, value,
              fits_compression=compression)
    with afits.open(filename) as hdus:
        science = hdus[1] if compression else hdus[0]
        _assert_exposure_header(science.header, start, 3.125)
        assert science.header['DATE-END'] == '2026-10-23T00:00:02.248456'
        np.testing.assert_array_equal(science.data, np.arange(20).reshape(4, 5))


@pytest.mark.parametrize('compression', [None, 'gzip'])
def test_acquisition_timing_survives_stale_wcs_merge(tmp_path, compression):
    start = Time('2026-10-22T10:00:00.125000', scale='utc')
    wcs = afits.Header()
    for key in ('DATE-OBS', 'DATE-BEG', 'DATE-AVG', 'DATE-END', 'SH-UT001', 'SH-UT002'):
        wcs[key] = ('2000-01-01T00:00:00.000000', 'stale WCS timing')
    wcs['TIMESYS'] = 'TAI'
    wcs['TIME-OBS'] = '00:00:00'
    for key in ('MJD-OBS', 'MJD-BEG', 'MJD-AVG', 'MJD-END', 'EXPTIME', 'EXPOSURE'):
        wcs[key] = -123.0
    wcs['WCSMJD'] = (float(start.tt.mjd), 'WCS reference epoch (TT)')
    wcs['WCSREF'] = 'Exposure start; WCSMJD uses TT'
    wcs['CTYPE1'] = ('RA---TAN', 'Coordinate card comment')
    wcs['CTYPE2'] = 'DEC--TAN'
    wcs['CRVAL1'], wcs['CRVAL2'] = 120.0, 20.0
    metadata = {'time': (start + TimeDelta(1.75, format='sec')).to_datetime(timezone=timezone.utc),
                'exposure_start': start.fits,
                'exposure_end': (start + TimeDelta(3.5, format='sec')).fits,
                'wcs_header': wcs}
    original_midpoint = metadata['time']
    filename = tmp_path / 'wcs-timing.fits'
    fits.save(filename, np.ones((3, 4)), 3.5, start, astrometrics=metadata,
              fits_compression=compression)
    with afits.open(filename) as hdus:
        hdr = hdus[1 if compression else 0].header
        _assert_exposure_header(hdr, start, 3.5)
        assert hdr['WCSMJD'] == wcs['WCSMJD']
        assert hdr.comments['WCSMJD'] == 'WCS reference epoch (TT)'
        assert hdr['WCSREF'] == wcs['WCSREF']
        assert hdr['CRVAL1'] == 120.0 and hdr['CRVAL2'] == 20.0
        assert hdr.comments['CTYPE1'] == 'Coordinate card comment'
        for key, seconds in [('MJD-BEG', 0), ('MJD-AVG', 1.75), ('MJD-END', 3.5)]:
            assert abs((hdr[key] - (start + TimeDelta(seconds, format='sec')).mjd) * 86400) < 1e-5
    assert metadata['time'] == original_midpoint
    assert wcs['TIMESYS'] == 'TAI'  # Caller-owned metadata is not mutated.


def test_legacy_astrometric_time_is_a_midpoint_not_a_start(tmp_path):
    start = Time('2026-10-22T10:00:00', scale='utc')
    midpoint = start + TimeDelta(2.25, format='sec')
    metadata = {'time': midpoint, 'wcs_header': {'DATE-OBS': '2000-01-01T00:00:00'}}
    filename = tmp_path / 'legacy.fits'
    fits.save(filename, np.zeros((2, 2)), exposure_time=4.5, astrometrics=metadata)
    with afits.open(filename) as hdus:
        _assert_exposure_header(hdus[0].header, start, 4.5)
    assert metadata['time'] == midpoint


def test_explicit_end_and_submicrosecond_inputs_round_only_at_serialization(tmp_path):
    start = Time('2026-10-22T10:00:00.123456789', scale='utc', precision=9)
    duration = 0.246913578
    end = start + TimeDelta(duration, format='sec')
    filename = tmp_path / 'precise.fits'
    fits.save(filename, np.zeros((2, 2)), duration, start,
              exposure_end=end.tt,
              astrometrics={'exposure_start': start, 'exposure_end': end})
    with afits.open(filename) as hdus:
        hdr = hdus[0].header
        _assert_exposure_header(hdr, start, duration)
        assert hdr['DATE-OBS'] == '2026-10-22T10:00:00.123457'
        assert hdr['DATE-AVG'] == '2026-10-22T10:00:00.246914'
        assert hdr['DATE-END'] == '2026-10-22T10:00:00.370370'


def test_explicit_start_wins_over_generic_astrometric_time(tmp_path):
    start = '2026-10-22T10:00:00.000000'
    filename = tmp_path / 'explicit.fits'
    fits.save(filename, np.zeros((2, 2)), 2.0, start,
              astrometrics={'time': Time('2000-01-01'),
                            'wcs_header': {'DATE-OBS': '2001-01-01T00:00:00'}})
    with afits.open(filename) as hdus:
        _assert_exposure_header(hdus[0].header, start, 2.0)


def test_legacy_wcs_start_fallback_converts_its_time_scale(tmp_path):
    start = Time('2026-10-22T10:00:00', scale='utc')
    filename = tmp_path / 'legacy-wcs.fits'
    fits.save(filename, np.zeros((2, 2)), 2.0,
              astrometrics={'wcs_header': {'DATE-OBS': start.tt.fits, 'TIMESYS': 'TT'}})
    with afits.open(filename) as hdus:
        _assert_exposure_header(hdus[0].header, start, 2.0)


@pytest.mark.parametrize('compression', [None, 'gzip'])
def test_writer_preserves_leap_seconds(tmp_path, compression):
    start = Time('2016-12-31T23:59:59.500000', scale='utc')
    filename = tmp_path / 'leap.fits'
    fits.save(filename, np.zeros((2, 2)), 1.5, start, fits_compression=compression)
    with afits.open(filename) as hdus:
        hdr = hdus[1 if compression else 0].header
        _assert_exposure_header(hdr, start, 1.5)
        assert hdr['DATE-AVG'] == '2016-12-31T23:59:60.250000'
        assert hdr['DATE-END'] == '2017-01-01T00:00:00.000000'


def test_zero_duration_and_call_time_default(tmp_path, monkeypatch):
    class Clock(datetime):
        next_second = 0

        @classmethod
        def now(cls, tz=None):
            value = cls(2026, 10, 23, 0, 45, cls.next_second, 999999, tzinfo=timezone.utc)
            cls.next_second += 10
            return value

    monkeypatch.setattr(fits, 'datetime', Clock)
    for index in range(2):
        filename = tmp_path / '{}.fits'.format(index)
        fits.save(filename, np.zeros((2, 2)))
        with afits.open(filename) as hdus:
            hdr = hdus[0].header
            _assert_exposure_header(hdr, '2026-10-23T00:45:{:02d}.999999'.format(index * 10), 0)
            assert hdr['DATE-OBS'] == hdr['DATE-AVG'] == hdr['DATE-END']


@pytest.mark.parametrize('compression', [None, 'gzip'])
@pytest.mark.parametrize('source', ['explicit', 'clock', 'metadata', 'midpoint', 'wcs'])
def test_zero_duration_fractional_timing_sources(tmp_path, monkeypatch, compression, source):
    start = datetime(2026, 10, 23, 0, 45, 0, 999999, tzinfo=timezone.utc)
    kwargs = {}
    if source == 'explicit':
        kwargs.update(dt_start=start, exposure_time=0)
    elif source == 'clock':
        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(2026, 10, 23, 0, 45, 0, 999999, tzinfo=timezone.utc)
        monkeypatch.setattr(fits, 'datetime', Clock)
    elif source == 'metadata':
        kwargs['astrometrics'] = {'exposure_start': start, 'exposure_end': start}
    elif source == 'midpoint':
        kwargs['astrometrics'] = {'time': start}
    else:
        kwargs['astrometrics'] = {'wcs_header': {'DATE-OBS': '2026-10-23T00:45:00.999999'}}
    filename = tmp_path / 'zero.fits'
    fits.save(filename, np.zeros((2, 2)), fits_compression=compression, **kwargs)
    with afits.open(filename) as hdus:
        hdr = hdus[1 if compression else 0].header
        _assert_exposure_header(hdr, start, 0)
        assert hdr['DATE-OBS'] == hdr['DATE-BEG'] == hdr['DATE-AVG'] == hdr['DATE-END']
        assert hdr['DATE-OBS'] == '2026-10-23T00:45:00.999999'


@pytest.mark.parametrize('duration', [0.0, 0.25e-6])
@pytest.mark.parametrize('offset', [-2e-6, -1e-6, -0.75e-6, 0.75e-6, 1e-6, 2e-6])
def test_exposure_ordering_uses_boundary_tolerance(tmp_path, duration, offset):
    start = Time('2026-10-23T00:45:00.999999', scale='utc', precision=9)
    end = start + TimeDelta(duration + offset, format='sec')
    filename = tmp_path / 'boundary.fits'
    if abs(offset) > 1e-6:
        with pytest.raises(ValueError, match='boundaries conflict'):
            fits.save(filename, np.zeros((2, 2)), duration, start, exposure_end=end)
        assert not filename.exists()
    else:
        fits.save(filename, np.zeros((2, 2)), duration, start, exposure_end=end)
        with afits.open(filename) as hdus:
            hdr = hdus[0].header
            _assert_exposure_header(hdr, start, duration)
            dates = [Time(hdr[key], format='fits', scale='utc')
                     for key in ['DATE-OBS', 'DATE-AVG', 'DATE-END']]
            assert dates[0] <= dates[1] <= dates[2]
            if duration == 0:
                assert hdr['DATE-OBS'] == hdr['DATE-AVG'] == hdr['DATE-END']


@pytest.mark.parametrize('kwargs', [
    {'exposure_time': -1}, {'exposure_time': float('nan')}, {'exposure_time': float('inf')},
    {'exposure_time': True}, {'exposure_time': '2'}, {'exposure_time': [2]},
    {'dt_start': 'not a date'}, {'dt_start': Time(['2026-01-01', '2026-01-02'])},
    {'exposure_end': '2026-01-01T00:00:02'},
    {'dt_start': '2026-01-01', 'exposure_time': 2, 'exposure_end': '2026-01-01T00:00:03'},
    {'dt_start': '2026-01-01', 'exposure_time': 2, 'exposure_end': '2025-12-31T23:59:59'},
    {'dt_start': '2026-01-01', 'astrometrics': {'exposure_start': '2026-01-02'}},
    {'dt_start': '2026-01-01', 'exposure_time': 2, 'exposure_end': '2026-01-01T00:00:02',
     'astrometrics': {'exposure_end': '2026-01-01T00:00:03'}},
    {'astrometrics': {'exposure_start': None}},
    {'astrometrics': {'exposure_start': '2026-01-01', 'exposure_end': '2026-01-01T00:00:02'}},
])
def test_invalid_or_conflicting_exposure_timing_is_rejected(tmp_path, kwargs):
    filename = tmp_path / 'invalid.fits'
    with pytest.raises(ValueError):
        fits.save(filename, np.zeros((2, 2)), **kwargs)
    assert not filename.exists()


def test_unequal_exposures_have_correct_midpoint_separation(tmp_path):
    dates = []
    starts = [Time('2026-10-22T10:00:00'), Time('2026-10-22T10:00:07')]
    for index, duration in enumerate([1.0, 5.0]):
        filename = tmp_path / '{}.fits'.format(index)
        fits.save(filename, np.zeros((2, 2)), duration, starts[index])
        with afits.open(filename) as hdus:
            hdr = hdus[0].header
            dates.append((Time(hdr['DATE-AVG'], format='fits', scale='utc'),
                          Time(hdr['DATE-OBS'], format='fits', scale='utc') +
                          TimeDelta(hdr['EXPTIME'] / 2, format='sec')))
    assert abs((dates[1][0] - dates[0][0]).sec - 9.0) < 1e-6
    assert abs((dates[1][1] - dates[0][1]).sec - 9.0) < 1e-6


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
