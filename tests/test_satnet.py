"""Tests for `satsim.image.` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.draw import gen_line
from satsim.io.satnet import init_annotation, set_frame_annotation


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
    assert(c['y_min'] == 0.55)
    assert(c['x_min'] == 0.525)
    assert(c['y_max'] == 0.75)
    assert(c['x_max'] == 0.575)

    assert(c['y_center'] == (0.55 + 0.75) / 2.)
    assert(c['x_center'] == (0.525 + 0.575) / 2.)

    assert c['ra_obs'] == 1.0
    assert c['dec_obs'] == -1.0
    assert c['ra'] == 0.5
    assert c['dec'] == -0.5

    assert(c['seg_id'] == -1)

    b = set_frame_annotation(a, 0, h, w, [pix], box_size=[3,3], box_pad=1)  # total box size will be 5

    c = b['data']['objects'][0]
    assert(c['y_min'] == 0.55)
    assert(c['x_min'] == 0.525)
    assert(c['y_max'] == 0.75)
    assert(c['x_max'] == 0.575)

    assert(c['y_center'] == (0.55 + 0.75) / 2.)
    assert(c['x_center'] == (0.525 + 0.575) / 2.)

    assert(c['bbox_height'] == 0.5)
    assert(c['bbox_width'] == 0.25)

    assert(c['seg_id'] == -1)

    b = set_frame_annotation(a, 0, h, w, [pix], box_size=None, box_pad=1)  # total box size will be 2+max-min

    c = b['data']['objects'][0]
    assert(c['y_min'] == 0.55)
    assert(c['x_min'] == 0.525)
    assert(c['y_max'] == 0.75)
    assert(c['x_max'] == 0.575)

    assert(c['y_center'] == (0.55 + 0.75) / 2.)
    assert(c['x_center'] == (0.525 + 0.575) / 2.)

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

    assert(c['class_name'] == 'Satellite')
    assert(c['y_min'] == -0.4090909090909091)
    assert(c['x_min'] == -0.4523809523809524)
    assert(c['y_center'] == 0.04545454545454544)
    assert(c['x_center'] == 0.023809523809523808)
    assert(c['y_max'] == 0.5)
    assert(c['x_max'] == 0.5)

    assert(c['y_start'] == -4.5 / h)
    assert(c['x_start'] == -9.5 / w)

    np.testing.assert_almost_equal(c['y_mid'], (0.5 - 4.5 / h) / 2)
    np.testing.assert_almost_equal(c['x_mid'], (0.5 - 9.5 / w) / 2)

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
