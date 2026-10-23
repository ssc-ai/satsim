"""Tests for `satsim.image.epsf`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.epsf import (
    add_epsf_counts,
    build_epsf_lut,
    crop_epsf_lut,
    effective_epsf_batch_size,
    epsf_wing_profile,
    filter_epsf_sources_in_bounds,
    phase_nearest_error_bound_px,
    resolve_epsf_tiers,
    resolve_phase_oversample,
    summarize_epsf_tiers,
    transform_and_add_epsf,
)
from satsim.image.psf import gen_gaussian
from satsim.util import configure_eager
from tests.epsf_helpers import assert_equivalent, float64_midpoint_bilinear_reference


configure_eager()


def _centroid(image):
    image = np.asarray(image, dtype=float)
    yy, xx = np.indices(image.shape)
    total = np.sum(image)
    return np.sum(yy * image) / total, np.sum(xx * image) / total


def test_build_epsf_lut_delta_shape_and_normalization():
    lut = build_epsf_lut(None, 3, 5)

    assert(lut.shape == (3, 3, 5, 5))
    np.testing.assert_allclose(tf.reduce_sum(lut, axis=[2, 3]).numpy(), np.ones([3, 3]))
    np.testing.assert_array_equal(lut[:, :, 2, 2].numpy(), np.ones([3, 3]))


def test_resolve_phase_oversample_uses_centroid_error_target():
    assert(resolve_phase_oversample(1) == 25)
    assert(resolve_phase_oversample(3) == 9)
    assert(resolve_phase_oversample(5) == 5)
    assert(resolve_phase_oversample(3, 4) == 4)
    assert(phase_nearest_error_bound_px(3, 4) == pytest.approx(1.0 / 24.0))


def test_effective_epsf_batch_size_honors_budget_and_expansion():
    floor_batch = effective_epsf_batch_size(
        5,
        point_rendering='floor',
        batch_element_budget=32000000,
    ).numpy()
    bilinear_batch = effective_epsf_batch_size(
        5,
        point_rendering='bilinear',
        batch_element_budget=32000000,
    ).numpy()
    transform_batch = effective_epsf_batch_size(
        5,
        point_rendering='floor',
        batch_element_budget=32000000,
        temporal_osf=8,
    ).numpy()

    assert(floor_batch == 262144)
    assert(bilinear_batch == 262144)
    assert(transform_batch == 160000)


def test_effective_epsf_batch_size_uses_cap_only_when_present():
    uncapped = effective_epsf_batch_size(
        5,
        point_rendering='floor',
        batch_element_budget=32000000,
    ).numpy()
    capped = effective_epsf_batch_size(
        5,
        point_rendering='floor',
        batch_element_budget=32000000,
        batch_size_cap=7,
    ).numpy()

    assert(uncapped > 1024)
    assert(capped == 7)


def test_effective_epsf_batch_size_large_budget_does_not_overflow_int32():
    batch = effective_epsf_batch_size(
        5,
        point_rendering='floor',
        batch_element_budget=10_000_000_000,
    ).numpy()

    assert(batch == 262144)


def test_build_epsf_lut_does_not_normalize_cropped_kernel_by_default():
    psf_os = np.ones([21, 21], dtype=np.float32)
    psf_os /= np.sum(psf_os)

    default_lut = build_epsf_lut(psf_os, 1, 3)
    normalized_lut = build_epsf_lut(psf_os, 1, 3, normalize=True)

    default_sum = tf.reduce_sum(default_lut, axis=[2, 3]).numpy()
    normalized_sum = tf.reduce_sum(normalized_lut, axis=[2, 3]).numpy()

    assert(np.all(default_sum < 1.0))
    np.testing.assert_allclose(normalized_sum, np.ones([1, 1]), atol=1e-6)


def test_build_epsf_lut_rejects_invalid_kernel_size():
    with pytest.raises(ValueError):
        build_epsf_lut(None, 3, 0)

    with pytest.raises(ValueError):
        build_epsf_lut(None, 3, 4)


def test_add_epsf_counts_bilinear_weights_and_centroid():
    lut = build_epsf_lut(None, 2, 1)
    img = add_epsf_counts(
        tf.zeros([3, 3], tf.float32),
        [1.5],
        [1.5],
        [100.0],
        lut,
        2,
        point_rendering='bilinear',
    ).numpy()

    expected = np.zeros([3, 3])
    expected[0, 0] = 25.0
    expected[0, 1] = 25.0
    expected[1, 0] = 25.0
    expected[1, 1] = 25.0

    np.testing.assert_allclose(img, expected)
    np.testing.assert_allclose(_centroid(img), [0.5, 0.5])


def test_add_epsf_counts_phase_nearest_centroid_within_quantization_bound():
    s_osf = 3
    phase_oversample = 4
    psf_os = gen_gaussian(93, 93, 1.5 * s_osf).numpy()
    bilinear_lut = build_epsf_lut(psf_os, s_osf, 31, normalize=True)
    phase_lut = build_epsf_lut(psf_os, s_osf, 31, normalize=True, phase_oversample=phase_oversample)
    row_det = 24.37
    col_det = 25.81
    r_os = (row_det + 0.5) * s_osf - 0.5
    c_os = (col_det + 0.5) * s_osf - 0.5

    expected = add_epsf_counts(
        tf.zeros([64, 64], tf.float32),
        [r_os],
        [c_os],
        [100.0],
        bilinear_lut,
        s_osf,
        point_rendering='bilinear',
    ).numpy()
    img = add_epsf_counts(
        tf.zeros([64, 64], tf.float32),
        [r_os],
        [c_os],
        [100.0],
        phase_lut,
        s_osf,
        point_rendering='phase_nearest',
    ).numpy()

    np.testing.assert_allclose(np.sum(img), 100.0, atol=1e-4)
    np.testing.assert_allclose(
        _centroid(img),
        _centroid(expected),
        atol=phase_nearest_error_bound_px(s_osf, phase_oversample) + 1e-6,
    )


def test_add_epsf_counts_edge_clipping_does_not_renormalize():
    lut = build_epsf_lut(None, 2, 1)
    img = add_epsf_counts(
        tf.zeros([2, 2], tf.float32),
        [-0.5],
        [-0.5],
        [100.0],
        lut,
        2,
        point_rendering='bilinear',
    ).numpy()

    expected = np.zeros([2, 2])
    expected[0, 0] = 25.0
    np.testing.assert_allclose(img, expected)
    np.testing.assert_allclose(np.sum(img), 25.0)


def test_add_epsf_counts_padded_scatter_clips_all_edges():
    lut = tf.ones([1, 1, 3, 3], tf.float32)
    cases = [
        ((-1, 1), np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=np.float32)),
        ((3, 1), np.array([[0, 0, 0], [0, 0, 0], [1, 1, 1]], dtype=np.float32)),
        ((1, -1), np.array([[1, 0, 0], [1, 0, 0], [1, 0, 0]], dtype=np.float32)),
        ((1, 3), np.array([[0, 0, 1], [0, 0, 1], [0, 0, 1]], dtype=np.float32)),
    ]

    for (row, col), expected in cases:
        img = add_epsf_counts(
            tf.zeros([3, 3], tf.float32),
            [row],
            [col],
            [1.0],
            lut,
            1,
            point_rendering='floor',
        ).numpy()
        np.testing.assert_allclose(img, expected)


def test_add_epsf_counts_matches_across_batch_sizes():
    lut = build_epsf_lut(None, 3, 5)
    r = [3.5, 7.25, 11.75, 15.10]
    c = [4.25, 8.75, 12.50, 16.90]
    cnt = [10.0, 20.0, 30.0, 40.0]

    expected = add_epsf_counts(
        tf.zeros([12, 12], tf.float32),
        r,
        c,
        cnt,
        lut,
        3,
        batch_size=1,
        point_rendering='bilinear',
    ).numpy()
    for batch_size in (2, 7, None):
        actual = add_epsf_counts(
            tf.zeros([12, 12], tf.float32),
            r,
            c,
            cnt,
            lut,
            3,
            batch_size=batch_size,
            point_rendering='bilinear',
        ).numpy()
        np.testing.assert_allclose(actual, expected, atol=1e-6)


def _random_epsf_case(height, width, s_osf, kernel_size, count, seed):
    rng = np.random.default_rng(seed)
    half = kernel_size // 2
    # Centers reach one pixel past half a kernel beyond every frame edge.
    r_os = rng.uniform(-(half + 1) * s_osf, (height + half + 1) * s_osf, count).astype(np.float32)
    c_os = rng.uniform(-(half + 1) * s_osf, (width + half + 1) * s_osf, count).astype(np.float32)
    cnt = rng.uniform(1.0, 100.0, count).astype(np.float32)
    lut = rng.random([s_osf, s_osf, kernel_size, kernel_size], dtype=np.float32)
    fpa = rng.random([height, width], dtype=np.float32)
    return fpa, r_os, c_os, cnt, lut


def _reference_epsf_deposit(fpa, r_os, c_os, cnt, lut, s_osf, point_rendering):
    r = np.asarray(r_os, np.float64)
    c = np.asarray(c_os, np.float64)
    cnt = np.asarray(cnt, np.float64)
    if point_rendering == 'bilinear':
        r0 = np.floor(r)
        c0 = np.floor(c)
        dr = r - r0
        dc = c - c0
        rr = np.concatenate([r0, r0, r0 + 1, r0 + 1])
        cc = np.concatenate([c0, c0 + 1, c0, c0 + 1])
        weights = np.concatenate([(1 - dr) * (1 - dc), (1 - dr) * dc, dr * (1 - dc), dr * dc])
        values = np.tile(cnt, 4) * weights
    else:
        rr = np.floor(r)
        cc = np.floor(c)
        values = cnt
    rr = rr.astype(np.int64)
    cc = cc.astype(np.int64)

    kernel_size = lut.shape[2]
    pad = kernel_size + 1
    height, width = fpa.shape
    offsets = np.arange(kernel_size) - kernel_size // 2
    stamps = lut[rr % s_osf, cc % s_osf].astype(np.float64) * values[:, None, None]
    rows = np.broadcast_to((rr // s_osf + pad)[:, None, None] + offsets[None, :, None], stamps.shape)
    cols = np.broadcast_to((cc // s_osf + pad)[:, None, None] + offsets[None, None, :], stamps.shape)
    image = np.zeros([height + 2 * pad, width + 2 * pad])
    np.add.at(image, (rows, cols), stamps)
    return fpa + image[pad:pad + height, pad:pad + width]


@pytest.mark.parametrize('point_rendering', ['bilinear', 'floor'], ids=['bilinear', 'floor'])
def test_add_epsf_counts_matches_numpy_reference_deposit_with_off_frame_centers(point_rendering):
    s_osf = 3
    fpa, r_os, c_os, cnt, lut = _random_epsf_case(13, 17, s_osf, 7, 300, seed=11)

    actual = add_epsf_counts(
        tf.constant(fpa),
        r_os,
        c_os,
        cnt,
        lut,
        s_osf,
        point_rendering=point_rendering,
    ).numpy()

    expected = _reference_epsf_deposit(fpa, r_os, c_os, cnt, lut, s_osf, point_rendering)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)


def test_add_epsf_counts_is_bit_identical_across_repeated_cpu_calls():
    s_osf = 3
    fpa, r_os, c_os, cnt, lut = _random_epsf_case(96, 96, s_osf, 7, 4000, seed=5)

    with tf.device('/CPU:0'):
        first = add_epsf_counts(tf.constant(fpa), r_os, c_os, cnt, lut, s_osf).numpy()
        second = add_epsf_counts(tf.constant(fpa), r_os, c_os, cnt, lut, s_osf).numpy()

    np.testing.assert_array_equal(first, second)


@pytest.mark.skipif(not tf.config.get_visible_devices('GPU'), reason='GPU required')
def test_add_epsf_counts_gpu_matches_reference_across_repeated_calls():
    s_osf = 3
    fpa, r_os, c_os, cnt, lut = _random_epsf_case(96, 96, s_osf, 7, 4000, seed=5)
    expected = _reference_epsf_deposit(fpa, r_os, c_os, cnt, lut, s_osf, 'bilinear')

    # Parallel GPU accumulation can change summation order between calls.
    # Check each call against the float64 reference instead of requiring
    # bit identity; use the same tolerance as the off-frame deposit test.
    with tf.device('/GPU:0'):
        for _ in range(3):
            actual = add_epsf_counts(tf.constant(fpa), r_os, c_os, cnt, lut, s_osf)
            assert 'GPU:0' in actual.device
            np.testing.assert_allclose(actual.numpy(), expected, rtol=1e-5, atol=1e-4)


def test_transform_and_add_epsf_preserves_flux_for_static_source():
    lut = build_epsf_lut(None, 3, 1)
    img = transform_and_add_epsf(
        tf.zeros([8, 8], tf.float32),
        [12.5],
        [13.5],
        [90.0],
        0.0,
        1.0,
        3,
        0.0,
        [0.0, 0.0],
        lut,
        3,
        point_rendering='bilinear',
    ).numpy()

    np.testing.assert_allclose(np.sum(img), 90.0, atol=1e-5)


def test_transform_and_add_epsf_t_osf_one_samples_mid_exposure():
    lut = build_epsf_lut(None, 1, 3)
    img = transform_and_add_epsf(
        tf.zeros([12, 12], tf.float32),
        [5.0],
        [5.0],
        [100.0],
        0.0,
        1.0,
        1,
        0.0,
        [2.0, 0.0],
        lut,
        1,
        point_rendering='floor',
    ).numpy()

    np.testing.assert_allclose(np.sum(img), 100.0, atol=1e-5)
    np.testing.assert_allclose(_centroid(img), [6.0, 5.0], atol=1e-6)


@pytest.mark.parametrize(
    't_start, t_end, rotation, translation',
    [
        (0.0, 1.0, 0.0, [0.0, 0.0]),
        (0.0, 1.0, 0.0, [7.0, -5.0]),
        (-2.0, 1.5, 0.0, [-4.0, 3.0]),
        (0.0, 1.0, 0.03, [0.0, 0.0]),
        (0.0, 1.0, -0.02, [5.0, 2.0]),
        (-1.0, 2.0, 0.3, [3.0, -6.0]),
    ],
    ids=['static', 'translation', 'signed-time', 'rotation', 'rotation-translation', 'wide-sweep'],
)
def test_default_filter_keeps_every_contributing_source(t_start, t_end, rotation, translation):
    s_osf = 3
    kernel_size = 5
    halo = (kernel_size // 2 + 1) * s_osf
    fpa, _, _, cnt, lut = _random_epsf_case(12, 14, s_osf, kernel_size, 600, seed=7)
    rng = np.random.default_rng(3)
    r_os = rng.uniform(-3 * halo, 12 * s_osf + 3 * halo, 600).astype(np.float32)
    c_os = rng.uniform(-3 * halo, 14 * s_osf + 3 * halo, 600).astype(np.float32)

    def render(filter_out_of_bounds):
        return transform_and_add_epsf(
            tf.constant(fpa), r_os, c_os, cnt, t_start, t_end, 8, rotation, translation, lut, s_osf,
            filter_out_of_bounds=filter_out_of_bounds,
        ).numpy()

    kept, _, _ = filter_epsf_sources_in_bounds(
        fpa, r_os, c_os, cnt, t_start, t_end, rotation, translation, s_osf, kernel_size)
    assert 0 < kept.shape[0] < r_os.shape[0]
    np.testing.assert_array_equal(render(True), render(False))


@pytest.mark.parametrize('rotation', [0.0, 1e-6, -0.02], ids=['static', 'slow-rotation', 'rotation'])
def test_filter_culls_beyond_support_halo(rotation):
    # 12 detector rows at s_osf=3 span 36 oversampled rows; a 5-pixel kernel adds a 9-row halo.
    rows = np.array([-9.5, -8.5, 44.5, 45.5], np.float32)

    kept, _, _ = filter_epsf_sources_in_bounds(
        tf.zeros([12, 12]), rows, np.full(4, 18.0, np.float32), np.ones(4, np.float32),
        0.0, 1.0, rotation, [0.0, 0.0], 3, 5,
    )

    np.testing.assert_array_equal(kept.numpy(), [-8.5, 44.5])


@pytest.mark.parametrize('point_rendering', ['floor', 'bilinear'])
@pytest.mark.parametrize(
    's_osf, offset',
    [(1, 0.99), (1, 2.5), (1, -3.0), (3, 2.5), (3, (-4.0, 3.5))],
    ids=['s1-just-over-one', 's1-positive', 's1-negative', 's3-positive', 's3-per-axis'],
)
def test_filter_keeps_every_contributing_source_for_any_offset(s_osf, offset, point_rendering):
    halo = (5 // 2 + 1) * s_osf
    edges = np.concatenate([
        np.linspace(-halo - 4 * s_osf, -halo + 1, 200),
        np.linspace(12 * s_osf + halo - 1, 12 * s_osf + halo + 4 * s_osf, 200),
    ]).astype(np.float32)
    middle = np.full(edges.shape, 6.0 * s_osf, np.float32)
    rows = np.concatenate([edges, middle])
    cols = np.concatenate([middle, edges])

    def render(filter_out_of_bounds):
        return transform_and_add_epsf(
            tf.zeros([12, 12]), rows, cols, np.ones(rows.shape, np.float32), 0.0, 1.0, 1, 0.0, [0.0, 0.0],
            tf.ones([s_osf, s_osf, 5, 5]), s_osf, filter_out_of_bounds=filter_out_of_bounds,
            point_rendering=point_rendering, position_offset_os=offset,
        ).numpy()

    unfiltered = render(False)
    assert unfiltered.sum() > 0
    np.testing.assert_array_equal(render(True), unfiltered)


def test_graph_render_accepts_lut_with_dynamic_kernel_size():
    rows, cols, counts = [-30.0, -3.5, 4.25, 14.0], [6.0, 6.0, 5.5, 6.25], [5.0, 10.0, 20.0, 30.0]
    lut = tf.ones([1, 1, 5, 5]) / 25.0

    def render(epsf_lut):
        return transform_and_add_epsf(tf.zeros([12, 12]), rows, cols, counts, 0.0, 1.0, 4, 0.0, [1.0, 0.0], epsf_lut, 1)

    graph = tf.function(render, input_signature=[tf.TensorSpec([1, 1, None, None], tf.float32)])

    np.testing.assert_allclose(graph(lut).numpy(), render(lut).numpy(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    'rotation, translation',
    [(0.0, [0.0, 0.0]), (-0.0, [-0.0, 0.0]), (tf.constant(0.0), [tf.constant(0.0), tf.constant(0.0)])],
    ids=['static', 'negative-zero', 'static-tensors'],
)
def test_exact_zero_motion_renders_identically_for_any_t_osf(rotation, translation):
    def render(t_osf):
        return transform_and_add_epsf(
            tf.zeros([8, 8]), [12.3, 7.6], [11.7, 15.2], [10.0, 3.0], 0.0, 1.0, t_osf, rotation, translation,
            build_epsf_lut(None, 3, 3), 3,
        ).numpy()

    np.testing.assert_array_equal(render(5), render(1))


@pytest.mark.parametrize(
    'rotation, translation, graph',
    [(0.0, [3e-7, 0.0], False), (3e-8, [0.0, 0.0], False), (3e-8, [0.0, 0.0], True)],
    ids=['slow-translation', 'slow-rotation', 'graph-rotation'],
)
def test_slow_motion_samples_every_midpoint(rotation, translation, graph):
    lut = tf.constant(np.random.default_rng(5).uniform(0.1, 1.0, [3, 3, 5, 5]).astype(np.float32))

    def render(rotation):
        return transform_and_add_epsf(
            tf.zeros([12, 12]), [6.4], [30.1], [10.0], 0.0, 1e7, 5, rotation, translation, lut, 3)

    if graph:
        render = tf.function(render, input_signature=[tf.TensorSpec([], tf.float32)])
    expected = float64_midpoint_bilinear_reference(
        lut.numpy(), [6.4], [30.1], [10.0], shape=(12, 12), pad=0, start=0, end=1e7, samples=5,
        velocity=translation, rotation=rotation)

    assert_equivalent(render(rotation).numpy(), expected, 10.0)


def test_position_transform_sees_every_sample_without_motion():
    seen = []

    def transform(rr, cc):
        seen.append(int(tf.size(rr)))
        return rr, cc

    transform_and_add_epsf(
        tf.zeros([8, 8]), [12.0, 13.0], [12.0, 9.0], [10.0, 5.0], 0.0, 1.0, 5, 0.0, [0.0, 0.0],
        build_epsf_lut(None, 3, 3), 3, position_transform=transform)

    assert sum(seen) == 2 * 5


def test_crop_epsf_lut_center_slice_and_validation():
    lut = tf.reshape(tf.range(49, dtype=tf.float32), [1, 1, 7, 7])

    np.testing.assert_array_equal(
        crop_epsf_lut(lut, 3).numpy(),
        lut.numpy()[:, :, 2:5, 2:5],
    )
    np.testing.assert_array_equal(
        crop_epsf_lut(lut.numpy(), 3),
        lut.numpy()[:, :, 2:5, 2:5],
    )
    np.testing.assert_array_equal(crop_epsf_lut(lut, 7).numpy(), lut.numpy())

    with pytest.raises(ValueError):
        crop_epsf_lut(lut, 4)

    with pytest.raises(ValueError):
        crop_epsf_lut(lut, 9)


def test_epsf_wing_profile_reports_wing_and_lost_energy():
    lut = tf.ones([1, 1, 5, 5], tf.float32)

    profile = epsf_wing_profile(lut, [3])

    assert(profile[3]['wing_max'] == 1.0)
    assert(profile[3]['enclosed_energy_min'] == 9.0)
    assert(profile[3]['enclosed_energy_max'] == 9.0)
    assert(profile[3]['full_energy_min'] == 25.0)
    assert(profile[3]['full_energy_max'] == 25.0)
    assert(profile[3]['energy_loss_max'] == 16.0)


def test_resolve_epsf_tiers_auto_monotonic_and_summarizes_counts():
    lut = tf.ones([1, 1, 9, 9], tf.float32) / 81.0

    tiers = resolve_epsf_tiers(
        lut,
        {
            'mode': 'auto',
            'noise_fraction': 0.1,
            'sigma_pix': 100.0,
            'sizes': [3, 5, 7],
        },
    )

    assert(tiers is not None)
    assert([tier['kernel_size'] for tier in tiers] == [3, 5, 7, 9])
    flux_max = [tier['flux_max'] for tier in tiers]
    assert(all(a < b for a, b in zip(flux_max[:-1], flux_max[1:])))

    summary = summarize_epsf_tiers(tiers, [0.01, 0.03, 0.10, 100.0])
    assert(summary['source_count'] == 4)
    assert(summary['full_stamp_elements'] == 4 * 9 * 9)
    assert(sum(tier['count'] for tier in summary['tiers']) == 4)
    for tier in summary['tiers'][:-1]:
        assert(tier['binding'] in ('wing', 'loss'))
        assert(tier['wing_flux_max_pe'] is not None)
        assert(tier['loss_flux_max_pe'] is not None)
    assert(summary['tiers'][-1]['binding'] is None)
    assert(summary['tiers'][-1]['wing_flux_max_pe'] is None)
    assert(summary['tiers'][-1]['loss_flux_max_pe'] is None)


def test_resolve_epsf_tiers_manual_threshold_is_inclusive():
    lut = tf.ones([1, 1, 5, 5], tf.float32)

    tiers = resolve_epsf_tiers(
        lut,
        {
            'mode': 'manual',
            'threshold_pe': 10.0,
            'kernel_size': 3,
        },
    )

    summary = summarize_epsf_tiers(tiers, [9.99, 10.0, 10.01])
    assert(summary['tiers'][0]['kernel_size'] == 3)
    assert(summary['tiers'][0]['count'] == 2)
    assert(summary['tiers'][1]['kernel_size'] == 5)
    assert(summary['tiers'][1]['count'] == 1)


def test_transform_and_add_epsf_tiers_by_total_flux_not_temporal_sample():
    lut = tf.ones([1, 1, 5, 5], tf.float32)
    tiers = [
        {'kernel_size': 1, 'flux_max': 50.0},
        {'kernel_size': 5, 'flux_max': np.inf},
    ]

    actual = transform_and_add_epsf(
        tf.zeros([11, 11], tf.float32),
        [5],
        [5],
        [100.0],
        0.0,
        1.0,
        10,
        0.0,
        [0.0, 0.0],
        lut,
        1,
        point_rendering='floor',
        epsf_tiers=tiers,
    ).numpy()
    expected = transform_and_add_epsf(
        tf.zeros([11, 11], tf.float32),
        [5],
        [5],
        [100.0],
        0.0,
        1.0,
        10,
        0.0,
        [0.0, 0.0],
        lut,
        1,
        point_rendering='floor',
    ).numpy()

    np.testing.assert_allclose(actual, expected)


def test_auto_crop_criterion_bounds_pixel_and_flux_error():
    y, x = np.indices([9, 9])
    stamp = np.exp(-((y - 4) ** 2 + (x - 4) ** 2) / (2 * 1.4 ** 2))
    stamp = stamp / np.sum(stamp)
    lut = tf.reshape(tf.cast(stamp, tf.float32), [1, 1, 9, 9])
    noise_fraction = 0.25
    sigma_pix = 5.0

    tiers = resolve_epsf_tiers(
        lut,
        {
            'mode': 'auto',
            'noise_fraction': noise_fraction,
            'sigma_pix': sigma_pix,
            'sizes': [3, 5, 7],
        },
    )

    for tier in tiers:
        if tier.get('mode') == 'full':
            continue

        flux = 0.5 * tier['flux_max']
        if not np.isfinite(flux) or flux <= 0:
            continue

        full = add_epsf_counts(
            tf.zeros([21, 21], tf.float32),
            [10],
            [10],
            [flux],
            lut,
            1,
            point_rendering='floor',
        ).numpy()
        cropped = add_epsf_counts(
            tf.zeros([21, 21], tf.float32),
            [10],
            [10],
            [flux],
            crop_epsf_lut(lut, tier['kernel_size']),
            1,
            point_rendering='floor',
        ).numpy()
        diff = full - cropped

        assert(np.max(np.abs(diff)) <= noise_fraction * sigma_pix + 1e-6)
        assert(abs(np.sum(diff)) <= noise_fraction * np.sqrt(flux) + 1e-6)


def test_auto_crop_dense_dim_field_reduces_stamp_elements_by_more_than_4x():
    y, x = np.indices([51, 51])
    stamp = np.exp(-((y - 25) ** 2 + (x - 25) ** 2) / (2 * 1.5 ** 2))
    stamp = stamp / np.sum(stamp)
    lut = tf.reshape(tf.cast(stamp, tf.float32), [1, 1, 51, 51])
    tiers = resolve_epsf_tiers(
        lut,
        {
            'mode': 'auto',
            'noise_fraction': 0.1,
            'sigma_pix': 1000.0,
            'sizes': [5, 9, 17, 33],
        },
    )
    crop_tiers = [tier for tier in tiers if tier.get('mode') != 'full']
    assert(crop_tiers)

    flux_max = crop_tiers[0]['flux_max']
    dim_flux = 1.0 if not np.isfinite(flux_max) else max(1e-6, 0.5 * flux_max)
    summary = summarize_epsf_tiers(tiers, np.full(154000, dim_flux, dtype=np.float32))

    assert(summary['source_count'] == 154000)
    assert(summary['stamp_elements'] * 4 <= summary['full_stamp_elements'])
