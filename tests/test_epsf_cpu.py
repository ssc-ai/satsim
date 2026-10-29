"""Tests for the compiled CPU ePSF star accumulator and its routing."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

import satsim.image.epsf as epsf
from satsim.image.epsf import _choose_star_accumulator, build_epsf_lut, transform_and_add_epsf
from satsim.image.render import render_epsf
from satsim.util import configure_eager
from tests.epsf_helpers import assert_equivalent, routed


configure_eager()

CPU = '/job:localhost/replica:0/task:0/device:CPU:0'
GPU = '/job:localhost/replica:0/task:0/device:GPU:0'
requires_gpu = pytest.mark.skipif(not tf.config.get_visible_devices('GPU'), reason='GPU required')

ROWS = [-4.25, -0.001, 0.1, 2.9999, 15.125, 15.1251, 35.999, 38.25]
COLS = [9.25, 2.9999, 0.001, 33.1, 15.3, 15.299, 35.9, 9.7]
COUNTS = [0.001, 10., 100., 1., 100000., 0.01, 500., 50.]


def _lut(asymmetric, phases=3):
    y, x = np.indices((7, 7))
    stamps = np.empty((phases, phases, 7, 7), np.float32)
    for r in range(phases):
        for c in range(phases):
            stamp = np.exp(-((y - 3 - r / phases) ** 2 + (x - 3 - c / phases) ** 2) / 2)
            if asymmetric:
                stamp += 0.2 * np.exp(-((y - 1) ** 2 + (x - 4) ** 2) / 3)
            stamps[r, c] = stamp / stamp.sum()
    return tf.constant(stamps)


def _warp(rows, cols):
    return rows + 0.003 * cols * cols, cols - 0.01 * rows


@pytest.mark.parametrize('point_rendering', ['floor', 'bilinear', 'phase_nearest'], ids=['floor', 'bilinear', 'phase'])
@pytest.mark.parametrize('motion', ['static', 'translation', 'rotation', 'callback', 'static-callback'])
@pytest.mark.parametrize('asymmetric', [False, True], ids=['gaussian', 'asymmetric'])
def test_cpu_accumulator_matches_tensorflow(point_rendering, motion, asymmetric):
    with tf.device('/CPU:0'):
        start = tf.ones([12, 12], tf.float32)
        kwargs = dict(
            t_start=0.2, t_end=1.7, t_osf=7,
            rotation=0.03 if motion == 'rotation' else 0.,
            translation=[0., 0.] if motion in ('static', 'static-callback') else [1.3, -0.7],
            epsf_lut=_lut(asymmetric, 6 if point_rendering == 'phase_nearest' else 3), s_osf=3,
            point_rendering=point_rendering, position_offset_os=(-0.5, 0.25), batch_size_cap=2,
            epsf_tiers=[{'flux_max': 10., 'kernel_size': 3}, {'flux_max': float('inf'), 'kernel_size': 7}],
            position_transform=_warp if 'callback' in motion else None)

    with routed('tensorflow'):
        expected = transform_and_add_epsf(start, ROWS, COLS, COUNTS, accumulator='tensorflow', **kwargs)
    with routed('cpu'):
        actual = transform_and_add_epsf(start, ROWS, COLS, COUNTS, accumulator='cpu', **kwargs)

    assert actual.dtype == tf.float32
    np.testing.assert_allclose(actual.numpy(), expected.numpy(), rtol=1e-6, atol=1e-6)
    assert_equivalent((actual - start).numpy(), (expected - start).numpy().astype(np.float64), sum(COUNTS))


@pytest.mark.parametrize('dtype', [tf.float32, tf.float64], ids=['float32', 'float64'])
def test_cpu_accumulator_keeps_image_dtype_and_static_flux(dtype):
    with tf.device('/CPU:0'):
        kwargs = dict(t_start=10., t_end=13., t_osf=37, rotation=0., translation=[-0., -0.],
                      epsf_lut=_lut(True), s_osf=3)
        fpa = tf.zeros([12, 12], dtype)
        expected = transform_and_add_epsf(fpa, [15.2], [15.7], [12345.], accumulator='tensorflow', **kwargs)
    with routed('cpu'):
        actual = transform_and_add_epsf(fpa, [15.2], [15.7], [12345.], accumulator='cpu', **kwargs)

    assert actual.dtype == dtype
    assert_equivalent(actual.numpy(), expected.numpy().astype(np.float64), 12345.)
    assert float(tf.reduce_sum(tf.cast(actual, tf.float64))) == pytest.approx(12345., rel=1e-6)


def test_cpu_accumulator_returns_input_image_for_empty_stars():
    with routed('cpu'):
        start = tf.reshape(tf.range(144, dtype=tf.float32), [12, 12])
        actual = transform_and_add_epsf(
            start, [], [], [], 0., 1., 5, 0., [1., 0.], _lut(False), 3,
            epsf_tiers=[{'flux_max': 10., 'kernel_size': 3}, {'flux_max': float('inf'), 'kernel_size': 7}],
            accumulator='cpu')

    np.testing.assert_array_equal(actual.numpy(), start.numpy())


@pytest.mark.parametrize('dtype', [np.float32, np.float64], ids=['float32', 'float64'])
def test_numpy_image_renders_like_tensor_image(dtype):
    with routed('cpu'):
        args = ([15.2, 4.1], [15.7, 30.3], [100., 7.], 0., 1., 4, 0., [1., -0.5], _lut(True), 3)
        expected = transform_and_add_epsf(tf.zeros([12, 12], dtype), *args)
        actual = transform_and_add_epsf(np.zeros([12, 12], dtype), *args)

    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual.numpy(), expected.numpy())


@pytest.mark.parametrize(
    'requested, eager, devices, dtype, expected',
    [
        ('auto', True, (CPU, CPU), tf.float32, 'cpu'),
        ('auto', True, (CPU,), tf.float64, 'cpu'),
        ('cpu', True, (CPU, CPU), tf.float32, 'cpu'),
        ('auto', False, ('', ''), tf.float32, 'tensorflow'),
        ('auto', True, (GPU, GPU), tf.float32, 'tensorflow'),
        ('auto', True, (CPU, GPU), tf.float32, 'tensorflow'),
        ('auto', True, (CPU, ''), tf.float32, 'tensorflow'),
        ('auto', True, (CPU,), tf.float16, 'tensorflow'),
        ('tensorflow', True, (CPU,), tf.float32, 'tensorflow'),
    ],
    ids=['eager-cpu', 'eager-cpu-float64', 'forced-cpu', 'graph', 'gpu', 'mixed', 'unknown-input', 'float16',
         'forced-tensorflow'],
)
def test_choose_star_accumulator_routes_by_placement(requested, eager, devices, dtype, expected):
    assert _choose_star_accumulator(requested, eager, devices, dtype) == expected


@pytest.mark.parametrize(
    'requested, eager, devices, dtype',
    [
        ('cpu', False, ('',), tf.float32),
        ('cpu', True, (GPU,), tf.float32),
        ('cpu', True, (CPU, GPU), tf.float32),
        ('cpu', True, (CPU,), tf.float16),
        ('numba', True, (CPU,), tf.float32),
    ],
    ids=['graph', 'gpu', 'mixed', 'float16', 'unknown-request'],
)
def test_choose_star_accumulator_rejects_ineligible_cpu_request(requested, eager, devices, dtype):
    with pytest.raises(ValueError):
        _choose_star_accumulator(requested, eager, devices, dtype)


def _render_stars(lut, translation, star_accumulator, rows=(30.2, 61.7), cols=(38.9, 70.25)):
    return render_epsf(
        72, 96, 96, 120, 12, 12, 3, lut, [], [], [],
        tf.constant(rows), tf.constant(cols), tf.constant([400., 9000.]),
        0., 3., 9, 0.004, translation, star_accumulator=star_accumulator)[0]


@requires_gpu
def test_gpu_render_never_enters_cpu_accumulator():
    with tf.device('/GPU:0'), routed('tensorflow'):
        image = _render_stars(build_epsf_lut(None, 3, 7), [0.4, -0.7], 'auto')

    assert 'GPU' in image.device


@requires_gpu
def test_mixed_device_inputs_keep_tensorflow():
    with tf.device('/GPU:0'):
        translation = [tf.constant(0.4), tf.constant(-0.7)]
    with tf.device('/CPU:0'), routed('tensorflow'):
        lut = build_epsf_lut(None, 3, 7)
        _render_stars(lut, translation, 'auto')
        with pytest.raises(ValueError, match='every render tensor on the CPU'):
            _render_stars(lut, translation, 'cpu')


@requires_gpu
def test_cpu_scope_on_gpu_host_matches_gpu_render():
    with tf.device('/GPU:0'):
        expected = _render_stars(build_epsf_lut(None, 3, 7), [0.4, -0.7], 'tensorflow')
    with routed('cpu'):
        actual = _render_stars(build_epsf_lut(None, 3, 7), [0.4, -0.7], 'cpu')

    assert 'GPU' in expected.device
    assert_equivalent(actual.numpy(), expected.numpy().astype(np.float64), 9400.)


@requires_gpu
@pytest.mark.parametrize('cpu_batches', [0, 1], ids=['first-batch', 'later-batch'])
def test_gpu_position_transform_outputs_never_reach_cpu_kernel(monkeypatch, cpu_batches):
    kernel_batches = []
    add_stamps = epsf.add_epsf_stamps

    def counted_add_stamps(*args):
        kernel_batches.append(len(args[1]))
        add_stamps(*args)

    monkeypatch.setattr(epsf, 'add_epsf_stamps', counted_add_stamps)

    def render(accumulator):
        calls = []

        def position_transform(rows, cols):
            calls.append(int(tf.size(rows)))
            rows, cols = _warp(rows, cols)
            if len(calls) <= cpu_batches:
                return rows, cols
            with tf.device('/GPU:0'):
                return tf.identity(rows), tf.identity(cols)

        with tf.device('/CPU:0'):
            image = transform_and_add_epsf(
                tf.zeros([12, 12]), ROWS, COLS, COUNTS, 0., 1., 4, 0.002, [0.3, -0.2], _lut(True), 3,
                batch_size_cap=2, position_transform=position_transform, accumulator=accumulator)
        return image, calls

    expected, expected_calls = render('tensorflow')
    actual, auto_calls = render('auto')

    assert len(kernel_batches) == cpu_batches
    assert auto_calls == expected_calls
    assert_equivalent(actual.numpy(), expected.numpy().astype(np.float64), sum(COUNTS))
    with pytest.raises(ValueError, match='position_transform outputs on the CPU'):
        render('cpu')
    assert len(kernel_batches) == 2 * cpu_batches
