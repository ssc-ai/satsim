"""Tests for `satsim.geometry.sstr7` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import gc
import os
import struct
import tracemalloc

import numpy as np
import pytest

from satsim.geometry import sstr7
from satsim.geometry.sstr7 import DEFAULT_SSTR7_PATH, RECORD_LEN_BYTES, get_star_mv, load_index, load_zone, query_by_los, query_by_min_max, read_star, select_zone
from satsim.geometry.wcs import get_min_max_ra_dec

MAS_PER_DEG = 3.6e6
RADIANS_PER_MAS = 4.84813681109535993589914102358e-9
MISSING = 32767

# Hand-picked records keyed by (SPD zone, RA zone): RA in deg, then raw dec
# (mas), RA and dec proper motion, parallax, and magnitudes by band (milli-mag,
# missing bands written as 32767). SPD 900 and 901 hold dec 0 to 0.2 deg; each
# RA zone is 6 deg wide. Zones (900, 0) and (900, 1) are unsorted so that
# clipping at the first star past the edge differs from filtering every star
# past the edge.
SYNTHETIC_ZONES = {
    (900, 0): [
        (1.0, 180000, 0, 0, 0, {0: 5000}),
        (5.0, 180000, 10, -10, 1, {5: 6000}),
        (2.0, 180000, 0, 0, 0, {8: 7000}),
        (4.5, 180000, -1, 1, 2, {4: 8000}),
    ],
    (900, 1): [
        (6.5, 72000, 0, 0, 0, {3: 9000}),
        (7.0, 108000, 0, 0, 0, {1: 1000}),
        (9.0, 144000, 0, 0, 0, {0: -1500}),
        (7.5, 180000, 0, 0, 0, {0: 31999, 5: 4000}),
    ],
    (901, 1): [
        (6.1, -2 ** 31, -2 ** 15, 2 ** 15 - 1, -2 ** 15, {0: 32000, 5: 32000, 8: 32000, 4: 32000, 3: 31999}),
        (6.2, 2 ** 31 - 1, 2 ** 15 - 1, -2 ** 15, 2 ** 15 - 1, {0: -2 ** 15}),
        (6.3, 540000, 0, 0, 0, {0: 2 ** 15 - 1}),
    ],
    (900, 58): [
        (353.0, 180000, 0, 0, 0, {4: 12345}),
    ],
    (900, 59): [
        (358.0, 180000, 0, 0, 0, {0: 11000}),
        (355.0, 216000, 0, 0, 0, {5: 12000}),
    ],
    (900, 30): [
        (182.0, 180000, 0, 0, 0, {}),
        (184.0, 216000, 0, 0, 0, {1: 1000}),
        (183.0, 252000, 0, 0, 0, {0: 32000}),
    ],
}

# RA min, RA max, dec min, dec max in degrees.
SYNTHETIC_QUERIES = {
    'min_and_max_edges': (4.0, 8.0, 0.01, 0.15),
    'crosses_ra_zero': (357.0, 3.0, 0.01, 0.09),
    'missing_mv_only': (181.0, 185.0, 0.01, 0.09),
}

# height, width, y_fov, x_fov, ra, dec
SYNTHETIC_POINTINGS = {
    'min_and_max_edges': (14, 400, 0.14, 4.0, 6.0, 0.08),
    'crosses_ra_zero': (14, 400, 0.14, 4.0, 359.5, 0.05),
    'missing_mv_only': (14, 400, 0.14, 4.0, 183.0, 0.05),
}

# ra, dec, rot, y_fov, x_fov, origin; the pointings of test_query_by_los
REAL_POINTINGS = {
    'vega_corner': (279.2358441666667, 38.78492111111112, 0.0, 0.2, 0.1, 'corner'),
    'vega_center': (279.2358441666667, 38.78492111111112, 0.0, 0.2, 0.1, 'center'),
    'vega_rot90_corner': (279.2358441666667, 38.78492111111112, 90.0, 0.2, 0.1, 'corner'),
    'vega_rot90_center': (279.2358441666667, 38.78492111111112, 90.0, 0.2, 0.1, 'center'),
    'vega_rot-45_corner': (279.2358441666667, 38.78492111111112, -45.0, 0.2, 0.1, 'corner'),
    'vega_rot-45_center': (279.2358441666667, 38.78492111111112, -45.0, 0.2, 0.1, 'center'),
    'north_pole': (268, 89.9, 0, 0.2, 0.1, 'corner'),
    'south_pole': (359.9, -89.99, 190, 0.2, 0.1, 'corner'),
    'meridian_inside': (359.9, -85, 0, 0.2, 0.1, 'corner'),
    'meridian_min_ra': (1.0, 0, 0, 10, 10, 'corner'),
}


def _write_catalog(root, zones):
    index = np.zeros(1800 * 60 * 2, dtype='<u4')
    for spd in sorted({spd for spd, _ in zones}):
        records = []
        for ra_zone in range(60):
            stars = zones.get((spd, ra_zone), [])
            index[2 * (spd * 60 + ra_zone)] = len(records)
            index[2 * (spd * 60 + ra_zone) + 1] = len(stars)
            for ra, dec, ra_pm, dec_pm, parallax, mv in stars:
                raw_mv = [mv.get(band, MISSING) for band in range(18)]
                records.append(
                    struct.pack('=iihhh', round(ra * MAS_PER_DEG), dec, ra_pm, dec_pm, parallax)
                    + struct.pack('=18h', *raw_mv)
                    + bytes(RECORD_LEN_BYTES - 50))
        with open(os.path.join(root, 's{:04d}.cat'.format(spd)), 'wb') as f:
            f.write(b''.join(records))
    index.tofile(os.path.join(root, 'sstrc.acc'))
    return str(root)


@pytest.fixture
def synthetic_catalog(tmp_path):
    return _write_catalog(tmp_path, SYNTHETIC_ZONES)


@pytest.fixture
def bulk_catalog(tmp_path):
    ras = np.linspace(6.1, 11.9, 20000)
    return _write_catalog(tmp_path, {(900, 1): [(ra, 180000, 0, 0, 0, {0: 10000}) for ra in ras]})


def _reference_query_by_min_max(ra_min, ra_max, dec_min, dec_max, rootPath, clip_min_max=True):
    """`query_by_min_max` with one `read_star` per record, the reference for parity tests."""
    zoneIndex = load_index(os.path.join(rootPath, 'sstrc.acc'), numRaZones=60, numDecZones=1800)
    stars = []
    for z in select_zone(ra_min, ra_max, dec_min, dec_max, zoneIndex, numRaZones=60, numDecZones=1800):
        buffer, start, end = load_zone(z, rootPath)
        ss = [read_star(buffer[s:s + RECORD_LEN_BYTES]) for s in range(0, end - start, RECORD_LEN_BYTES)]
        if z['bound'] == 'minRA':
            ss.reverse()
        if clip_min_max and z['bound'] != 'inside':
            for i, star in enumerate(ss):
                if (star['ra'] < ra_min) if z['bound'] == 'minRA' else (star['ra'] > ra_max):
                    ss = ss[0:i]
                    break
        stars += ss
    return stars


def _radians(bounds):
    return tuple(np.radians(b) for b in bounds)


def _assert_same_dicts(actual, expected):
    assert [list(s.keys()) for s in actual] == [list(s.keys()) for s in expected]
    assert actual == expected


def test_star_mv():

    star = np.full((18,), 32.0)

    mv = get_star_mv(star)
    assert(mv == 32.0)

    star[3] = 1.0
    mv = get_star_mv(star)
    assert(mv == 1.0)

    star[4] = 2.0
    mv = get_star_mv(star)
    assert(mv == 2.0)

    star[8] = 3.0
    mv = get_star_mv(star)
    assert(mv == 3.0)

    star[5] = 4.0
    mv = get_star_mv(star)
    assert(mv == 4.0)

    star[0] = 4.0
    mv = get_star_mv(star)
    assert(mv == 4.0)


def _synthetic_star(catalog, ra):
    stars = query_by_min_max(*_radians((ra - 0.01, ra + 0.01, 0.01, 0.15)), catalog, clip_min_max=False)
    star, = [s for s in stars if s['ra'] == round(ra * MAS_PER_DEG) * RADIANS_PER_MAS]
    return star


# RA of a SYNTHETIC_ZONES record and its expected magnitude. A raw 31999
# milli-mag decodes to 31999 * 1.0e-3, one ulp above the literal 31.999.
MV_ANCHORS = {
    'open': (1.0, 5.0),
    'johnson_r': (5.0, 6.0),
    'sloan_r': (2.0, 7.0),
    'johnson_v': (4.5, 8.0),
    'johnson_b': (6.5, 9.0),
    'open_before_johnson_r': (7.5, 31999 * 1.0e-3),
    'above_31999_falls_through': (6.1, 31999 * 1.0e-3),
    'all_missing': (182.0, 32),
}


@pytest.mark.parametrize('ra, expected_mv', MV_ANCHORS.values(), ids=MV_ANCHORS.keys())
def test_decoded_mv_matches_band_priority(synthetic_catalog, ra, expected_mv):

    assert _synthetic_star(synthetic_catalog, ra)['mv'] == expected_mv


def test_decoded_position_matches_raw_record(synthetic_catalog):

    # SYNTHETIC_ZONES[(901, 1)][0]: the int32 and int16 extremes
    star = _synthetic_star(synthetic_catalog, 6.1)

    assert star['ra'] == 21960000 * RADIANS_PER_MAS
    assert star['dec'] == -2 ** 31 * RADIANS_PER_MAS
    assert star['ra_pm'] == -2 ** 15 * (0.32 * RADIANS_PER_MAS / 3.1556952e7)
    assert star['dec_pm'] == (2 ** 15 - 1) * (0.32 * RADIANS_PER_MAS / 3.1556952e7)
    assert star['parallax'] == -2 ** 15 * (0.032 * RADIANS_PER_MAS)


def test_query_by_los():

    height = 512
    width = 256

    y_fov = 0.2
    x_fov = 0.1

    # vega coordinates
    ra = 279.2358441666667
    dec = 38.78492111111112

    def test_it(expected_num_stars, origin='corner', filter_ob=False, flipud=False, fliplr=False):

        yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin=origin, filter_ob=filter_ob, flipud=flipud, fliplr=fliplr)

        assert(len(yy) == expected_num_stars)
        assert(len(xx) == expected_num_stars)
        assert(len(mm) == expected_num_stars)

        vega = np.where(mm == 0.07)[0]

        assert(len(vega) == 1)

        if origin == 'center':
            np.testing.assert_almost_equal(yy[vega], height / 2.0)
            np.testing.assert_almost_equal(xx[vega], width / 2.0)
            np.testing.assert_almost_equal(mm[vega], 0.07)
        else:
            if flipud:
                np.testing.assert_almost_equal(yy[vega], height)
            else:
                np.testing.assert_almost_equal(yy[vega], 0)

            if fliplr:
                np.testing.assert_almost_equal(xx[vega], width)
            else:
                np.testing.assert_almost_equal(xx[vega], 0)
            np.testing.assert_almost_equal(mm[vega], 0.07)

    rot = 0.0
    test_it(4532)
    test_it(372, 'corner', True)
    test_it(372, 'corner', True, True, True)
    test_it(4426, 'center')
    test_it(285, 'center', True)
    test_it(285, 'center', True, True, True)

    rot = 90.0
    test_it(2852)
    test_it(2980, 'center')

    rot = -45.0
    test_it(4730)
    test_it(4532, 'center')

    # north pole
    ra = 268
    dec = 89.9
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 242)

    # south pole
    ra = 359.9
    dec = -89.99
    rot = 190
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 976)

    # meridian crossing + inside
    ra = 359.9
    dec = -85
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 2434)

    # meridian crossing + minRA
    ra = 1.0
    dec = 0
    rot = 0
    yy, xx, mm, rra, ddec = query_by_los(height, width, 10, 10, ra, dec, rot, origin='corner', filter_ob=False)
    assert(len(yy) == 83612)


@pytest.mark.parametrize('clip_min_max', [True, False], ids=['clip', 'no_clip'])
@pytest.mark.parametrize('query', SYNTHETIC_QUERIES.values(), ids=SYNTHETIC_QUERIES.keys())
def test_query_by_min_max_matches_per_record_parse(synthetic_catalog, query, clip_min_max):

    expected = _reference_query_by_min_max(*_radians(query), synthetic_catalog, clip_min_max)

    assert len(expected) > 0
    _assert_same_dicts(query_by_min_max(*_radians(query), synthetic_catalog, clip_min_max), expected)


@pytest.mark.parametrize('pointing', REAL_POINTINGS.values(), ids=REAL_POINTINGS.keys())
def test_query_by_min_max_matches_per_record_parse_on_catalog(pointing):

    ra, dec, rot, y_fov, x_fov, origin = pointing
    cmin, cmax, _ = get_min_max_ra_dec(512, 256, y_fov / 512, x_fov / 256, ra, dec, rot, 0, origin)
    cmin = np.radians(cmin)
    cmax = np.radians(cmax)
    bounds = (cmin[0], cmax[0], cmin[1], cmax[1], DEFAULT_SSTR7_PATH)

    _assert_same_dicts(query_by_min_max(*bounds), _reference_query_by_min_max(*bounds))


@pytest.mark.parametrize('name', SYNTHETIC_POINTINGS.keys())
def test_query_by_los_matches_per_record_parse(synthetic_catalog, name):

    height, width, y_fov, x_fov, ra, dec = SYNTHETIC_POINTINGS[name]
    cmin, cmax, _ = get_min_max_ra_dec(height, width, y_fov / height, x_fov / width, ra, dec, 0, 0, 'center')
    cmin = np.radians(cmin)
    cmax = np.radians(cmax)
    stars = _reference_query_by_min_max(cmin[0], cmax[0], cmin[1], cmax[1], synthetic_catalog)
    expected = {
        'rra': np.degrees(np.array([s['ra'] for s in stars])),
        'ddec': np.degrees(np.array([s['dec'] for s in stars])),
        'mm': np.array([s['mv'] for s in stars]),
    }
    assert len(stars) > 0

    def check():
        _, _, mm, rra, ddec = query_by_los(height, width, y_fov, x_fov, ra, dec, rootPath=synthetic_catalog, filter_ob=False)
        for key, actual in (('rra', rra), ('ddec', ddec), ('mm', mm)):
            if key == 'mm' and name == 'missing_mv_only':
                # every star lacks a magnitude: the per-record reference builds an int64 array of 32
                np.testing.assert_array_equal(actual, expected[key])
                continue
            assert actual.dtype == expected[key].dtype
            assert actual.shape == expected[key].shape
            assert actual.tobytes() == expected[key].tobytes()

    check()
    check()
    sstr7.clear_star_cache()
    check()


def test_query_by_min_max_returns_fresh_dicts(synthetic_catalog):

    query = _radians(SYNTHETIC_QUERIES['min_and_max_edges'])
    first = query_by_min_max(*query, synthetic_catalog)
    expected = [dict(s) for s in first]

    first[0]['mv'] = 99.0

    _assert_same_dicts(query_by_min_max(*query, synthetic_catalog), expected)


def _traced_bytes():
    gc.collect()
    return tracemalloc.get_traced_memory()[0]


def test_cached_zone_memory_per_star(bulk_catalog):

    # an empty region loads the zone index without caching any zone
    query_by_min_max(*_radians((100.0, 101.0, 0.01, 0.09)), bulk_catalog)
    tracemalloc.start()
    try:
        before = _traced_bytes()
        num_stars = len(query_by_min_max(*_radians((4.0, 14.0, 0.01, 0.09)), bulk_catalog, clip_min_max=False))
        held = _traced_bytes() - before
    finally:
        tracemalloc.stop()

    assert num_stars == 20000
    assert held / num_stars <= 60


def test_clear_star_cache_releases_zone_memory(bulk_catalog):

    query = _radians((4.0, 14.0, 0.01, 0.09))
    expected = query_by_min_max(*query, bulk_catalog)
    sstr7.clear_star_cache()
    tracemalloc.start()
    try:
        before = _traced_bytes()
        query_by_los(14, 800, 0.14, 8.0, 9.0, 0.05, rootPath=bulk_catalog)
        held = _traced_bytes() - before
        sstr7.clear_star_cache()
        cleared = _traced_bytes() - before
        sstr7.clear_star_cache()
        cleared_twice = _traced_bytes() - before
    finally:
        tracemalloc.stop()

    assert held >= 48 * 20000
    assert cleared < held / 50
    assert cleared_twice < held / 50
    _assert_same_dicts(query_by_min_max(*query, bulk_catalog), expected)
