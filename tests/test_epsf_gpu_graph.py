"""GPU deposition remains accurate as successive calls change inputs."""
import numpy as np
import pytest
import tensorflow as tf

import satsim.image.epsf as epsf
from satsim.image.epsf import transform_and_add_epsf
from satsim.util import configure_eager
from tests.epsf_helpers import assert_equivalent, deterministic_ops, float64_midpoint_bilinear_reference, quadratic_warp


@pytest.mark.skipif(not tf.config.get_visible_devices('GPU'), reason='GPU required')
@pytest.mark.parametrize('dtype', [tf.float32, tf.float64], ids=['float32', 'float64'])
@pytest.mark.parametrize('warp', [False, True], ids=['affine', 'nonlinear'])
def test_gpu_deposition_varying_shapes_values_and_optics(dtype, warp):
    configure_eager()
    rng = np.random.default_rng(65)
    pad, osf = 2, 3
    # Reuse shapes after intervening calls, changing LUT values as well as stars.
    for count, shape in [(1, (12, 15)), (7, (14, 17)), (0, (12, 15)), (3, (12, 15)), (7, (14, 17))]:
        lut = rng.uniform(.01, 1, (osf, osf, 5, 5)).astype(dtype.as_numpy_dtype)
        lut /= lut.sum(axis=(2, 3), keepdims=True)
        rows = rng.uniform(-1, (shape[0] + 2 * pad) * osf + 1, count).astype(np.float32)
        cols = rng.uniform(-1, (shape[1] + 2 * pad) * osf + 1, count).astype(np.float32)
        if count == 1:
            rows[0], cols[0] = (shape[0] / 2 + pad) * osf, (shape[1] / 2 + pad) * osf
        flux = np.geomspace(1, 10000, count).astype(np.float32)
        crop = dict(threshold_pe=100, kernel_size=3)
        expected = float64_midpoint_bilinear_reference(
            lut, rows, cols, flux, shape=shape, pad=pad, samples=7,
            start=.2, end=1.7, velocity=(.4, -.7), rotation=.013, warp=warp, crop=crop)
        with tf.device('/GPU:0'):
            image = transform_and_add_epsf(
                tf.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad), dtype),
                rows, cols, flux, .2, 1.7, 7, .013, [.4, -.7], tf.constant(lut), osf,
                epsf_tiers=[dict(flux_max=100, kernel_size=3), dict(flux_max=float('inf'), kernel_size=5)],
                batch_size_cap=2, position_transform=quadratic_warp if warp else None,
                accumulator='tensorflow')
        assert image.dtype == dtype and 'GPU' in image.device
        actual = image.numpy()[pad:-pad, pad:-pad]
        if count:
            assert_equivalent(actual, expected, float(flux.sum()))
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(not tf.config.get_visible_devices('GPU'), reason='GPU required')
@pytest.mark.parametrize('point_rendering', ['floor', 'bilinear', 'phase_nearest'])
@pytest.mark.parametrize('dtype', [tf.float32, tf.float64], ids=['float32', 'float64'])
@pytest.mark.parametrize('motion', ['rotation', 'nonlinear', 'static-callback', 'cpu-callback'])
@pytest.mark.parametrize('deterministic', [False, True], ids=['normal', 'deterministic'])
def test_gpu_star_graph_matches_eager_deposition(monkeypatch, point_rendering, dtype, motion, deterministic):
    """Check graph accumulation against the pre-optimization GPU calculation."""
    configure_eager()
    rng = np.random.default_rng(650)
    phases = 6 if point_rendering == 'phase_nearest' else 3
    lut = rng.uniform(.01, 1, (phases, phases, 7, 7)).astype(dtype.as_numpy_dtype)
    lut /= lut.sum(axis=(2, 3), keepdims=True)
    rows = [-4.25, -.001, .1, 2.9999, 15.125, 15.1251, 35.999, 38.25]
    cols = [9.25, 2.9999, .001, 33.1, 15.3, 15.299, 35.9, 9.7]
    flux = [.001, 10., 100., 1., 100000., .01, 500., 50.]
    graph = epsf._add_epsf_counts_graph
    graph_calls = []

    def graph_deposit(*args, **kwargs):
        graph_calls.append(args[0].device)
        return graph(*args, **kwargs)

    def render(deposit):
        callbacks = []

        def warp(rr, cc):
            callbacks.append((rr.numpy(), cc.numpy()))
            if motion == 'cpu-callback':
                with tf.device('/CPU:0'):
                    return quadratic_warp(rr, cc)
            return quadratic_warp(rr, cc)

        monkeypatch.setattr(epsf, '_add_epsf_counts_graph', deposit)
        with tf.device('/GPU:0'):
            image = transform_and_add_epsf(
                tf.ones((12, 12), dtype), rows, cols, flux, .2, 1.7, 7,
                .03 if motion == 'rotation' else 0.,
                [0., 0.] if motion == 'static-callback' else [.4, -.7], tf.constant(lut), 3,
                epsf_tiers=[dict(flux_max=100, kernel_size=3), dict(flux_max=float('inf'), kernel_size=7)],
                batch_size_cap=2, position_transform=warp if motion != 'rotation' else None,
                point_rendering=point_rendering, accumulator='tensorflow')
        assert 'GPU' in image.device and image.dtype == dtype
        return image.numpy(), callbacks

    def check():
        actual, actual_calls = render(graph_deposit)
        expected, expected_calls = render(epsf.add_epsf_counts)
        assert graph_calls and all('GPU' in device for device in graph_calls)
        assert len(actual_calls) == len(expected_calls)
        for actual_call, expected_call in zip(actual_calls, expected_calls):
            for a, b in zip(actual_call, expected_call):
                np.testing.assert_array_equal(a, b)
        assert_equivalent(actual, expected.astype(np.float64), sum(flux))
        if deterministic:
            repeated, _ = render(graph_deposit)
            np.testing.assert_array_equal(actual, repeated)

    if deterministic:
        with deterministic_ops():
            check()
    else:
        check()


@pytest.mark.parametrize('route', ['cpu', 'graph', 'static', 'single-midpoint'])
def test_other_deposition_routes_do_not_use_gpu_star_graph(monkeypatch, route):
    if route != 'cpu' and not tf.config.get_visible_devices('GPU'):
        pytest.skip('GPU required')
    configure_eager()

    def forbidden(*args, **kwargs):
        raise AssertionError('This route must retain its existing deposition path')

    monkeypatch.setattr(epsf, '_add_epsf_counts_graph', forbidden)

    def render():
        return transform_and_add_epsf(
            tf.zeros((9, 9)), [12.2], [12.7], [100.], 0., 1., 1 if route == 'single-midpoint' else 7,
            0., [0., 0.] if route == 'static' else [.4, -.7],
            tf.ones((3, 3, 3, 3)) / 9, 3, accumulator='tensorflow')

    with tf.device('/CPU:0' if route == 'cpu' else '/GPU:0'):
        image = tf.function(render)() if route == 'graph' else render()
    assert float(tf.reduce_sum(image)) == pytest.approx(100., rel=1e-6)


def test_gpu_graph_constructor_supports_legacy_tensorflow_keyword(monkeypatch):
    """Exercise the compatibility branch without requiring an old CUDA runtime."""
    compile_function = tf.function

    def legacy_function(function, **kwargs):
        if 'reduce_retracing' in kwargs:
            raise TypeError('unexpected keyword argument: reduce_retracing')
        return compile_function(function, **kwargs)

    monkeypatch.setattr(tf, 'function', legacy_function)
    graph = epsf._compile_gpu_epsf_deposition()
    with tf.device('/CPU:0'):
        result = graph(tf.zeros((3, 3)), [1.], [1.], [10.], tf.ones((1, 1, 1, 1)), 1)
    assert float(tf.reduce_sum(result)) == pytest.approx(10.)


def test_gpu_deposition_respects_each_explicit_gpu_scope():
    if len(tf.config.get_visible_devices('GPU')) < 2:
        pytest.skip('Two visible GPUs required')
    configure_eager()
    lut = np.ones((3, 3, 5, 5), np.float32) / 25
    rows, cols, flux = [12.2, 24.7], [15.3, 27.8], [100., 200.]
    expected = float64_midpoint_bilinear_reference(
        lut, rows, cols, flux, shape=(12, 15), pad=0, samples=7,
        start=.2, end=1.7, velocity=(.4, -.7), rotation=.03)
    for device in ['/GPU:0', '/GPU:1', '/GPU:0']:
        with tf.device(device):
            image = transform_and_add_epsf(
                tf.zeros((12, 15)), rows, cols, flux, .2, 1.7, 7, .03, [.4, -.7],
                tf.constant(lut), 3, accumulator='tensorflow')
        assert device[1:] in image.device
        assert_equivalent(image.numpy(), expected, sum(flux))
