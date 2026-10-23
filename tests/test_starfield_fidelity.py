"""Native ePSF star fidelity against an independent float64 reference.

The reference computes motion, phase lookup, bilinear deposition, clipping and
flux-tier selection in NumPy without calling production helpers. Render cases
run once per star accumulator and check which one ran; the ``cpu`` cases render
under a CPU device scope so they also run on GPU hosts.
"""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.epsf import add_epsf_counts, build_epsf_lut, transform_and_add_epsf
from satsim.image.render import render_epsf
from tests.epsf_helpers import assert_equivalent, float64_midpoint_bilinear_reference, quadratic_warp, routed


OSF = 3


@pytest.fixture(scope='module')
def optics():
    result = {}
    for parity, size in [('odd', 33), ('even', 36)]:
        y, x = np.indices((size, size), dtype=np.float64)
        y -= (size - 1) / 2
        x -= (size - 1) / 2
        psf = np.exp(-((y / 2.1) ** 2 + (x / 2.7) ** 2) / 2)
        psf += 0.13 * np.exp(-(((y - 3.2) / 1.8) ** 2 + ((x + 2.3) / 2.2) ** 2) / 2)
        with tf.device('/CPU:0'):
            psf = tf.constant((psf / psf.sum()).astype(np.float32))
            result[parity] = (psf, build_epsf_lut(psf, OSF, 7))
    return result


@pytest.fixture(params=['transform', 'streak'], ids=['transform', 'streak'])
def star_mode(request):
    return request.param


@pytest.fixture(params=['tensorflow', 'cpu'], ids=['tensorflow', 'cpu'])
def accumulator(request):
    return request.param


def test_direct_bilinear_lookup_matches_independent_reference():
    rng = np.random.default_rng(50)
    lut = rng.uniform(0.01, 1, (OSF, OSF, 5, 5)).astype(np.float32)
    lut /= lut.sum(axis=(2, 3), keepdims=True)
    rows = np.array([-1.25, -0.001, 2.9997, 3.0003, 14.125, 27.2], np.float32)
    cols = np.array([5.8, -0.25, 14.9997, 15.0003, 13.875, 24.1], np.float32)
    flux = np.array([0.2, 2, 20, 200, 2e3, 2e4], np.float32)

    expected = float64_midpoint_bilinear_reference(
        lut, rows, cols, flux, shape=(9, 9), pad=0, samples=1, start=0, end=0, velocity=(0, 0))
    actual = add_epsf_counts(tf.zeros((9, 9)), rows, cols, flux, tf.constant(lut), OSF).numpy()

    assert_equivalent(actual, expected, float(flux.sum()))


@pytest.mark.parametrize('parity', ['odd', 'even'], ids=['odd-support', 'even-support'])
def test_stars_match_reference_across_phase_carry(optics, star_mode, accumulator, parity):
    psf, lut = optics[parity]
    _check_scene(psf, lut, star_mode, accumulator, [28.9997, 49.31, 73.94], [35.8, 54.0003, 69.125], [10, 1000, 1e5],
                 velocity=(0.41, -0.73), start=2, end=5, samples=37)


@pytest.mark.parametrize(
    'start, end, samples, velocity',
    [
        (2.0, 5.0, 37, (-6.4, 1.7)),
        (-1.0, 3.0, 37, (9.4, -4.3)),
        (-2.0, 1.0, 1, (2.7, -1.3)),
        (2.0, 5.0, 37, (0.0, 0.0)),
    ],
    ids=['signed-absolute-time', 'long-one-sided-trail', 'single-midpoint', 'static'],
)
def test_star_motion_quadrature_matches_reference(optics, star_mode, accumulator, start, end, samples, velocity):
    psf, lut = optics['even']
    _check_scene(psf, lut, star_mode, accumulator, [58.23], [70.77], [1200],
                 start=start, end=end, samples=samples, velocity=velocity)


def test_total_flux_tiers_in_bright_faint_crowding(optics, star_mode, accumulator):
    psf, lut = optics['odd']
    _check_scene(psf, lut, star_mode, accumulator, [52.2, 53.25, 52.7], [50.37, 49.8, 50.9], [10, 100, 1e5],
                 velocity=(-1.3, 0.8), crop={'mode': 'manual', 'threshold_pe': 100, 'kernel_size': 3})


@pytest.mark.parametrize(
    'rotation, warp',
    [(0.055, False), (0.0, True)],
    ids=['field-rotation', 'position-transform'],
)
def test_nonuniform_star_motion_matches_reference(optics, star_mode, accumulator, rotation, warp):
    psf, lut = optics['odd']
    _check_scene(psf, lut, star_mode, accumulator, [30.2, 78.7], [38.9, 83.25], [40, 3000],
                 rotation=rotation, warp=warp, velocity=(0.3, -0.2), start=2, end=5)


@pytest.mark.parametrize('row', [-0.75, 72.25], ids=['negative-center-wing', 'bottom-center-wing'])
def test_off_frame_optical_wings_are_deposited(optics, star_mode, accumulator, row):
    psf, lut = optics['odd']
    scene = dict(shape=(24, 24), pad=0, start=2, end=5, samples=37, velocity=(0, 0.2))
    expected = float64_midpoint_bilinear_reference(lut.numpy(), [row], [30.75], [200], **scene)
    assert expected.sum() > 0

    with routed(accumulator):
        filtered = transform_and_add_epsf(
            tf.zeros((24, 24)), [row], [30.75], [200], 2., 5., 37, 0., [0., 0.2], lut, OSF,
            accumulator=accumulator).numpy()
    rendered, _ = _render(psf, lut, [row], [30.75], [200], star_mode, accumulator, **scene)

    assert_equivalent(filtered, expected, 200)
    assert_equivalent(rendered, expected, 200)


def test_rotating_arc_enters_frame_between_outside_endpoints(optics, star_mode, accumulator):
    psf, lut = optics['odd']
    center = (24 * OSF - 1) / 2
    radius = 55.0
    start_angle = np.deg2rad(-8.0)
    sweep = np.deg2rad(344.0)
    row = center + radius * np.cos(start_angle)
    col = center + radius * np.sin(start_angle)
    scene = dict(shape=(24, 24), pad=0, start=0, end=1, samples=37, velocity=(0, 0), rotation=sweep)
    expected = float64_midpoint_bilinear_reference(lut.numpy(), [row], [col], [500], **scene)
    assert expected.sum() > 0

    actual, _ = _render(psf, lut, [row], [col], [500], star_mode, accumulator, **scene)

    assert_equivalent(actual, expected, 500)


@pytest.mark.parametrize('parity', ['odd', 'even'], ids=['odd-support', 'even-support'])
def test_warm_cache_support_shape_registers_like_psf(optics, star_mode, accumulator, parity):
    psf, lut = optics[parity]
    scene = dict(velocity=(-1.1, 0.7), start=2, end=5)
    cold, _ = _render(psf, lut, [55.23, 41.6], [60.77, 32.1], [2000, 50], star_mode, accumulator, **scene)
    warm, _ = _render(None, lut, [55.23, 41.6], [60.77, 32.1], [2000, 50], star_mode, accumulator,
                      psf_support_shape=tuple(psf.shape), **scene)

    np.testing.assert_array_equal(warm, cold)


@pytest.mark.parametrize('parity', ['odd', 'even'], ids=['odd-support', 'even-support'])
def test_target_pixels_are_bitwise_independent_of_star_mode(optics, accumulator, parity):
    psf, lut = optics[parity]
    scene = dict(targets=([14.2, 39.7, 70.9], [20.8, 42.1, 61.4], [0.1, 2e3, 4e5]),
                 velocity=(-1.1, 0.7), start=2, end=5)
    _, transform_targets = _render(psf, lut, [55.23], [60.77], [2000], 'transform', accumulator, **scene)
    _, streak_targets = _render(psf, lut, [55.23], [60.77], [2000], 'streak', accumulator, **scene)

    np.testing.assert_array_equal(streak_targets, transform_targets)


def test_graph_rendering_uses_tensorflow(optics, star_mode):
    psf, lut = optics['odd']
    rows, cols, flux = [36.2, 70.9], [40.7, 82.1], [20., 2000.]

    def graph_image(star_accumulator):
        return tf.function(lambda: render_epsf(
            96, 96, 120, 120, 12, 12, OSF, lut, [], [], [], rows, cols, flux,
            2., 5., 7, 0.013, [0.41, -0.73], psf_os=psf, star_render_mode=star_mode,
            star_accumulator=star_accumulator)[0])()

    expected = float64_midpoint_bilinear_reference(
        lut.numpy(), rows, cols, flux, start=2, end=5, samples=7, rotation=0.013)
    with routed('tensorflow'):
        actual = graph_image('auto').numpy()
    assert_equivalent(actual, expected, sum(flux))
    with pytest.raises(ValueError, match='eager execution'):
        graph_image('cpu')


def _check_scene(psf, lut, star_mode, accumulator, rows, cols, flux, **scene):
    offset = _support_offset(psf)
    expected = float64_midpoint_bilinear_reference(lut.numpy(), rows, cols, flux, offset=offset, **scene)
    actual, _ = _render(psf, lut, rows, cols, flux, star_mode, accumulator, **scene)
    assert_equivalent(actual, expected, float(np.sum(flux)))
    if len(rows) > 1:
        for row, col, count in zip(rows, cols, flux):
            isolated = float64_midpoint_bilinear_reference(lut.numpy(), [row], [col], [count], offset=offset, **scene)
            measured, _ = _render(psf, lut, [row], [col], [count], star_mode, accumulator, **scene)
            assert_equivalent(measured, isolated, float(count))


def _support_offset(psf):
    return tuple(-0.5 if int(size) % 2 == 0 else 0.0 for size in psf.shape)


def _render(psf, lut, rows, cols, flux, star_mode, accumulator, shape=(32, 32), pad=4, start=0, end=3,
            samples=37, velocity=(0.41, -0.73), rotation=0, warp=False, crop=None, targets=([], [], []),
            psf_support_shape=None):
    with routed(accumulator):
        result = render_epsf(
            shape[0] * OSF, shape[1] * OSF, (shape[0] + 2 * pad) * OSF, (shape[1] + 2 * pad) * OSF,
            pad * OSF, pad * OSF, OSF, lut, *targets,
            np.asarray(rows, np.float32), np.asarray(cols, np.float32), np.asarray(flux, np.float32),
            float(start), float(end), samples, float(rotation), list(velocity),
            psf_os=psf, psf_support_shape=psf_support_shape, star_render_mode=star_mode,
            epsf_crop=crop, star_position_transform=quadratic_warp if warp else None,
            star_accumulator=accumulator)
    return result[0].numpy(), result[1].numpy()
