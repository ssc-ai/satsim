"""Tests for `satsim.image.` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

from collections import OrderedDict
from datetime import datetime

import numpy as np
import pytest
import tifffile
import tensorflow as tf

from satsim.geometry.draw import gen_line
from satsim.io.satnet import init_annotation, set_frame_annotation, write_frame


def test_annotation_pixels_use_containing_detector_pixel():
    h = 10
    w = 10
    snr = tf.reshape(tf.range(h * w, dtype=tf.float32), [h, w])
    pix = {
        'rr': np.array([4.0]),
        'cc': np.array([6.0]),
        'rrr': np.array([3.6]),
        'rcc': np.array([5.6]),
        'mv': 15,
        'pe': 100,
    }

    annotation = init_annotation('./', 0, h, w, 2.0, 3.0)
    result = set_frame_annotation(annotation, 0, h, w, [pix], snr=snr)
    ob = result['data']['objects'][0]

    assert ob['pixels'] == [[4, 6]]
    assert ob['snr'] == [46.0]


def test_annotation():

    h = 10
    w = 20

    (orr, occ, opp, ott) = gen_line(h, w, [0.5, 0.5], [2., 1.], 100., 0., 1.)

    pix = {
        'rr': orr,
        'cc': occ,
        'mv': 15,
        'pe': 100,
        'ra_obs': 1.0,
        'dec_obs': -1.0,
        'ra': 0.5,
        'dec': -0.5,
    }

    a = init_annotation('./', 0, h, w, 2., 3.)

    assert(a['data']['sensor']['height'] == h)
    assert(a['data']['sensor']['width'] == w)

    b = set_frame_annotation(a, 0, h, w, [pix])

    c = b['data']['objects'][0]
    assert(c['y_min'] == 0.5)
    assert(c['x_min'] == 0.5)
    assert(c['y_max'] == 0.7)
    assert(c['x_max'] == 0.55)

    assert(c['y_center'] == (0.5 + 0.7) / 2.)
    assert(c['x_center'] == (0.5 + 0.55) / 2.)

    assert c['ra_obs'] == 1.0
    assert c['dec_obs'] == -1.0
    assert c['ra'] == 0.5
    assert c['dec'] == -0.5

    assert(c['seg_id'] == -1)

    b = set_frame_annotation(a, 0, h, w, [pix], box_size=[3,3], box_pad=1)  # total box size will be 5

    c = b['data']['objects'][0]
    assert(c['y_min'] == 0.5)
    assert(c['x_min'] == 0.5)
    assert(c['y_max'] == 0.7)
    assert(c['x_max'] == 0.55)

    assert(c['y_center'] == (0.5 + 0.7) / 2.)
    assert(c['x_center'] == (0.5 + 0.55) / 2.)

    assert(c['bbox_height'] == 0.5)
    assert(c['bbox_width'] == 0.25)

    assert(c['seg_id'] == -1)

    b = set_frame_annotation(a, 0, h, w, [pix], box_size=None, box_pad=1)  # total box size will be 2+max-min

    c = b['data']['objects'][0]
    assert(c['y_min'] == 0.5)
    assert(c['x_min'] == 0.5)
    assert(c['y_max'] == 0.7)
    assert(c['x_max'] == 0.55)

    assert(c['y_center'] == (0.5 + 0.7) / 2.)
    assert(c['x_center'] == (0.5 + 0.55) / 2.)

    assert(c['seg_id'] == -1)

    np.testing.assert_almost_equal(c['bbox_height'], 0.4)
    np.testing.assert_almost_equal(c['bbox_width'], 0.15)


def test_annotation_odd():

    h = 11
    w = 21

    (orr, occ, opp, ott) = gen_line(h, w, [0.5, 0.5], [0., 0.], 100., 0., 1.)

    pix = {
        'rr': orr,
        'cc': occ,
        'mv': 15,
        'pe': 100,
        'id': 1
    }

    a = init_annotation('./', 0, h, w, 2., 3.)

    assert(a['data']['sensor']['height'] == h)
    assert(a['data']['sensor']['width'] == w)

    b = set_frame_annotation(a, 0, h, w, [pix],[11,21])

    c = b['data']['objects'][0]

    # Canonical origin [0.5, 0.5] maps to pixel coordinates (5, 10).
    # Annotation normalization maps those pixel centers back to [0.5, 0.5].
    assert(c['y_min'] == 0.5)
    assert(c['x_min'] == 0.5)
    assert(c['y_max'] == 0.5)
    assert(c['x_max'] == 0.5)

    assert(c['y_center'] == 0.5)
    assert(c['x_center'] == 0.5)

    assert(c['bbox_height'] == 1)
    assert(c['bbox_width'] == 1)

    assert(c['seg_id'] == 1)


def test_annotation_ob():

    h = 11
    w = 21

    (orr, occ, opp, ott) = gen_line(h, w, [-0.5, 0.5], [0., 0.], 100., 0., 1.)

    pix = {
        'rr': orr,
        'cc': occ,
        'mv': 15,
        'pe': 100
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [pix],[5,5], filter_ob=True)

    assert(len(b['data']['objects']) == 0)

    (orr, occ, opp, ott) = gen_line(h, w, [0.5, 1.5], [0., 0.], 100., 0., 1.)

    pix = {
        'rr': orr,
        'cc': occ,
        'mv': 15,
        'pe': 100
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [pix],[5,5], filter_ob=True)

    assert(len(b['data']['objects']) == 0)

    (orr, occ, opp, ott) = gen_line(h, w, [-0.5, -0.5], [11., 21.], 100., 0., 1.)

    pix = {
        'rr': orr,
        'cc': occ,
        'mv': 15,
        'pe': 100
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [pix],[5,5], filter_ob=True)

    assert(len(b['data']['objects']) == 1)

    c = b['data']['objects'][0]

    # Canonical origin [-0.5, -0.5] maps to pixel coordinates (-6, -11).
    # Applying velocity [11, 21] produces endpoint (5, 10), so annotation
    # normalization produces endpoints -0.5 through 0.5.
    assert(c['class_name'] == 'Satellite')
    assert(c['y_min'] == -0.5)
    assert(c['x_min'] == -0.5)
    assert(c['y_center'] == 0.0)
    assert(c['x_center'] == 0.0)
    assert(c['y_max'] == 0.5)
    assert(c['x_max'] == 0.5)

    assert(c['y_start'] == -0.5)
    assert(c['x_start'] == -0.5)

    assert(c['y_mid'] == 0.0)
    assert(c['x_mid'] == 0.0)

    assert(c['y_end'] == 0.5)
    assert(c['x_end'] == 0.5)


def test_annotation_star():

    h = 11
    w = 21

    pix = {
        'h': h,
        'w': w,
        'h_pad': 0,
        'w_pad': 0,
        'rr': [5.0],
        'cc': [9.0],
        'pe': [100.0],
        'mv': [13.0],
        'ra': [0.0],
        'dec': [0.0],
        'seg_id': [1],
        't_start': 0.0,
        't_end': 1.0,
        'rot': 0.0,
        'tran': [0.0, 1.0],
        'min_mv': 15,
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [], [5,5], filter_ob=True, star_os_pix=pix)

    c = b['data']['objects'][0]

    assert(c['class_name'] == 'Star')
    assert(c['y_min'] == 0.5)
    assert(c['x_min'] == 0.4523809552192688)
    assert(c['y_center'] == 0.5)
    assert(c['x_center'] == 0.4761904776096344)
    assert(c['y_max'] == 0.5)
    assert(c['x_max'] == 0.5)

    assert(c['y_start'] == 0.5)
    assert(c['x_start'] == 0.4523809552192688)
    assert(c['y_mid'] == 0.5)
    assert(c['x_mid'] == 0.4761904776096344)
    assert(c['y_end'] == 0.5)
    assert(c['x_end'] == 0.5)

    assert(c['pe_per_sec'] == 100.0)
    assert(c['magnitude'] == 13.0)

    assert(c['ra'] == 0.0)
    assert(c['dec'] == 0.0)
    assert(c['seg_id'] == 1)


def test_annotation_star_cloud_transmission_samples_detector_plane():

    h = 22
    w = 42
    cloud_transmission = np.arange(11 * 21, dtype=np.float32).reshape(11, 21) / 1000.0

    pix = {
        'h': h,
        'w': w,
        'h_pad': 0,
        'w_pad': 0,
        'rr': [10.0],
        'cc': [18.0],
        'pe': [100.0],
        'mv': [13.0],
        'ra': [0.0],
        'dec': [0.0],
        'seg_id': [1],
        't_start': 0.0,
        't_end': 1.0,
        'rot': 0.0,
        'tran': [0.0, 2.0],
        'min_mv': 15,
        'cloud_transmission': cloud_transmission,
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [], [5, 5], filter_ob=True, star_os_pix=pix)

    c = b['data']['objects'][0]
    assert c['cloud_sample_row'] == 4.75
    assert c['cloud_sample_col'] == 9.25
    expected = 0.109
    np.testing.assert_allclose(c['cloud_transmission'], expected)


def test_write_frame_ground_truth_min_exempts_cloud_transmission(tmp_path):

    h = 2
    w = 2
    fpa_digital = np.zeros((h, w), dtype=np.uint16)
    meta_data = init_annotation(str(tmp_path), [], h, w, 1.0, 1.0)
    ground_truth = OrderedDict([
        ('background_pe', np.full((h, w), 0.25, dtype=np.float32)),
        ('cloud_transmission', np.full((h, w), 0.25, dtype=np.float32)),
    ])

    write_frame(
        str(tmp_path),
        'sat',
        fpa_digital,
        meta_data,
        0,
        1.0,
        datetime(2026, 1, 1),
        {},
        save_jpeg=False,
        ground_truth=ground_truth,
        ground_truth_min=0.5,
    )

    truth = tifffile.imread(tmp_path / 'Annotations' / 'sat.0000.tiff')
    assert truth.shape == (2, h, w)
    np.testing.assert_allclose(truth[0], 0.0)
    np.testing.assert_allclose(truth[1], 0.25)


def test_annotation_empty():

    h = 11
    w = 21

    pix = {
        'h': h,
        'w': w,
        'h_pad': 0,
        'w_pad': 0,
        'rr': [],
        'cc': [],
        'pe': [],
        'mv': [],
        'ra': [],
        'dec': [],
        'seg_id': [],
        't_start': 0.0,
        't_end': 1.0,
        'rot': 0.0,
        'tran': [0.0, 1.0],
        'min_mv': 15,
    }

    a = init_annotation('./', 0, h, w, 2., 3.)
    b = set_frame_annotation(a, 0, h, w, [], [5,5], filter_ob=True, star_os_pix=pix)

    assert(len(b['data']['objects']) == 0)


def _f32(x):
    return float(np.float32(x))


def _star_pix(size, rr, cc, mv, **fields):
    pix = {
        'h': size,
        'w': size,
        'h_pad': 0,
        'w_pad': 0,
        'rr': rr,
        'cc': cc,
        'pe': [100.0] * len(rr),
        'mv': mv,
        'ra': [0.0] * len(rr),
        'dec': [0.0] * len(rr),
        'seg_id': None,
        't_start': 0.0,
        't_end': 1.0,
        'rot': 0.0,
        'tran': [0.0, 0.0],
        'min_mv': 15,
    }
    pix.update(fields)
    return pix


def _star_objects(pix):
    size = pix['h']
    a = init_annotation('./', 0, size, size, 2., 3.)
    return set_frame_annotation(a, 0, size, size, [], filter_ob=True, star_os_pix=pix)['data']['objects']


def _expected_star(size, rows, cols, magnitude, pe_per_sec, seg_id, ra, dec, **fields):
    ys = [(r + 0.5) / size for r in rows]
    xs = [(c + 0.5) / size for c in cols]
    star = {
        'class_name': 'Star',
        'class_id': 2,
        'y_min': min(ys),
        'x_min': min(xs),
        'y_max': max(ys),
        'x_max': max(xs),
        'y_center': (min(ys) + max(ys)) / 2,
        'x_center': (min(xs) + max(xs)) / 2,
        'bbox_height': max(ys) - min(ys),
        'bbox_width': max(xs) - min(xs),
        'source': 'satsim',
        'magnitude': magnitude,
        'pe_per_sec': pe_per_sec,
        'y_start': ys[0],
        'x_start': xs[0],
        'y_mid': ys[1],
        'x_mid': xs[1],
        'y_end': ys[2],
        'x_end': xs[2],
        'seg_id': seg_id,
        'object_name': '',
        'object_id': '',
        'ra': ra,
        'dec': dec,
        'variable': False,
    }
    star.update(fields)
    return star


def _types(obj):
    return {k: type(v) for k, v in obj.items()}


@pytest.mark.parametrize('seg_id, expected_seg_ids', [
    (None, [-1, -1]),
    ([7, 8, 9, 10, 11, 3], [7, 3]),
], ids=['default_seg_id', 'explicit_seg_id'])
def test_annotation_star_filters_and_keeps_order(seg_id, expected_seg_ids):

    nan = float('nan')
    pix = _star_pix(
        16,
        rr=[3.5, 5.5, nan, -4.5, 7.5, 11.5],
        cc=[7.5, 5.5, 5.5, 7.5, 23.5, 1.5],
        mv=[9.0, 16.0, 9.0, 9.0, 9.0, 15.0],
        pe=[100.1, 1.0, 1.0, 1.0, 1.0, 250.0],
        ra=[0.1, 1.0, 1.0, 1.0, 1.0, 359.5],
        dec=[-0.1, 1.0, 1.0, 1.0, 1.0, 89.75],
        seg_id=seg_id,
    )

    objs = _star_objects(pix)

    expected = [
        _expected_star(16, [3.5] * 3, [7.5] * 3, 9.0, _f32(100.1), expected_seg_ids[0], _f32(0.1), _f32(-0.1)),
        _expected_star(16, [11.5] * 3, [1.5] * 3, 15.0, 250.0, expected_seg_ids[1], 359.5, 89.75),
    ]
    assert objs == expected
    assert [_types(o) for o in objs] == [_types(o) for o in expected]


@pytest.mark.parametrize('rr, cc, mv', [
    ([], [], []),
    ([3.5, -4.5, float('nan')], [7.5, 7.5, 5.5], [16.0, 9.0, 9.0]),
], ids=['empty', 'all_filtered'])
def test_annotation_star_none_annotated(rr, cc, mv):

    pix = _star_pix(
        16,
        rr=rr,
        cc=cc,
        mv=mv,
        seg_id=[1] * len(rr),
        variable=[True] * len(rr),
        snr_aperture=[5.0] * len(rr),
        cloud_transmission=np.ones((4, 4), dtype=np.float32),
    )

    assert _star_objects(pix) == []


def test_annotation_star_positions_keep_float64_sky():

    pix = _star_pix(
        16,
        rr=[3.5],
        cc=[7.5],
        mv=[9.0],
        ra=[0.1],
        dec=[-0.1],
        annotation_positions=([[3.5] * 3], [[7.5] * 3]),
    )

    objs = _star_objects(pix)

    expected = [_expected_star(16, [3.5] * 3, [7.5] * 3, 9.0, 100.0, -1, 0.1, -0.1)]
    assert objs == expected
    assert [_types(o) for o in objs] == [_types(o) for o in expected]


def test_annotation_star_optional_fields():

    cloud_transmission = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    pix = _star_pix(
        16,
        rr=[3.5, 5.5, 11.5, 13.5],
        cc=[7.5, 5.5, 1.5, 13.5],
        mv=[9.0, 16.0, 10.0, 11.0],
        seg_id=[1, 2, 3, 4],
        variable=[True, True, False, True],
        snr_aperture=[5.25, 99.0, None, 6.5],
        cloud_transmission=cloud_transmission,
    )

    objs = _star_objects(pix)

    def expected(row, col, magnitude, seg_id, variable, snr_aperture, cloud_row, cloud_col):
        return _expected_star(
            16, [row] * 3, [col] * 3, magnitude, 100.0, seg_id, 0.0, 0.0,
            variable=variable,
            snr_aperture=snr_aperture,
            cloud_sample_row=cloud_row,
            cloud_sample_col=cloud_col,
            cloud_transmission=(4 * cloud_row + cloud_col) / 16,
        )

    expected_objs = [
        expected(3.5, 7.5, 9.0, 1, True, 5.25, 0.5, 1.5),
        expected(11.5, 1.5, 10.0, 3, False, None, 2.5, 0.0),
        expected(13.5, 13.5, 11.0, 4, True, 6.5, 3.0, 3.0),
    ]
    assert objs == expected_objs
    assert [_types(o) for o in objs] == [_types(o) for o in expected_objs]
    assert list(objs[0])[-7:] == ['ra', 'dec', 'variable', 'snr_aperture', 'cloud_sample_row', 'cloud_sample_col', 'cloud_transmission']
