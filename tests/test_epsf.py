"""Tests for `satsim.image.epsf`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.epsf import (
    add_epsf_counts,
    add_epsf_counts_tiered,
    build_epsf_lut,
    build_trailed_epsf_lut,
    crop_epsf_lut,
    effective_epsf_batch_size,
    epsf_wing_profile,
    phase_nearest_error_bound_px,
    resolve_epsf_tiers,
    resolve_phase_oversample,
    summarize_epsf_tiers,
    transform_and_add_epsf,
)
from satsim.image.psf import gen_gaussian
from satsim.util import configure_eager


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


def test_build_trailed_epsf_lut_kernel_growth_and_static_degenerate_case():
    static_lut = build_epsf_lut(None, 3, 5)
    trailed_static_lut, effective_kernel = build_trailed_epsf_lut(
        None,
        3,
        5,
        0.0,
        1.0,
        5,
        0.0,
        [0.0, 0.0],
    )

    assert(effective_kernel == 5)
    np.testing.assert_allclose(trailed_static_lut.numpy(), static_lut.numpy(), atol=1e-6)

    _, effective_kernel = build_trailed_epsf_lut(
        None,
        3,
        5,
        0.0,
        1.0,
        20,
        0.0,
        [0.0, 7.0],
    )

    assert(effective_kernel == 9)


def test_build_trailed_epsf_lut_preserves_flux_inside_effective_kernel():
    lut, _ = build_trailed_epsf_lut(
        None,
        3,
        5,
        0.0,
        1.0,
        20,
        0.0,
        [3.0, 6.0],
        normalize=False,
    )

    np.testing.assert_allclose(
        tf.reduce_sum(lut, axis=[2, 3]).numpy(),
        np.ones([3, 3]),
        atol=2e-3,
    )


def test_build_trailed_epsf_lut_rebases_absolute_frame_times():
    lut, _ = build_trailed_epsf_lut(
        None,
        1,
        5,
        5.0,
        6.0,
        20,
        0.0,
        [3.0, 0.0],
        normalize=False,
    )

    np.testing.assert_allclose(
        tf.reduce_sum(lut, axis=[2, 3]).numpy(),
        np.ones([1, 1]),
        atol=2e-3,
    )


def test_build_trailed_epsf_lut_rejects_absurd_kernel_growth():
    with pytest.raises(ValueError, match='trailed ePSF effective kernel_size'):
        build_trailed_epsf_lut(
            None,
            3,
            5,
            0.0,
            1.0,
            20,
            0.0,
            [300.0, 0.0],
            max_kernel_size=21,
        )


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


def test_add_epsf_counts_tiered_matches_manual_crop_composition():
    lut = tf.ones([1, 1, 5, 5], tf.float32)
    tiers = [
        {'kernel_size': 1, 'flux_max': 10.0},
        {'kernel_size': 5, 'flux_max': np.inf},
    ]

    actual = add_epsf_counts_tiered(
        tf.zeros([11, 11], tf.float32),
        [4, 6],
        [4, 6],
        [5.0, 20.0],
        lut,
        1,
        tiers,
        point_rendering='floor',
    ).numpy()

    expected = add_epsf_counts(
        tf.zeros([11, 11], tf.float32),
        [4],
        [4],
        [5.0],
        crop_epsf_lut(lut, 1),
        1,
        point_rendering='floor',
    )
    expected = add_epsf_counts(
        expected,
        [6],
        [6],
        [20.0],
        lut,
        1,
        point_rendering='floor',
    ).numpy()

    np.testing.assert_allclose(actual, expected)


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


def test_trailed_auto_crop_respects_trail_extent_and_error_bounds():
    osf = 3
    base_kernel = 15
    noise_fraction = 0.25
    sigma_pix = 20.0
    psf_os = gen_gaussian(63, 63, 1.5 * osf).numpy()
    lut, _, info = build_trailed_epsf_lut(
        psf_os,
        osf,
        base_kernel,
        0.0,
        1.0,
        20,
        0.0,
        [6.0, 0.0],
        return_info=True,
    )
    tiers = resolve_epsf_tiers(
        lut,
        {
            'mode': 'auto',
            'noise_fraction': noise_fraction,
            'sigma_pix': sigma_pix,
            'sizes': [3, 5, 9, 13],
        },
        trail_det=info['trail_det'],
    )
    crop_tiers = [tier for tier in tiers if tier.get('mode') != 'full']
    assert(crop_tiers)

    for tier in crop_tiers:
        assert(tier['kernel_size'] >= tier['core_size'] + info['trail_det'])

    finite_crop_tiers = [
        tier for tier in crop_tiers
        if np.isfinite(tier['flux_max']) and tier['flux_max'] > 0
    ]
    tier = finite_crop_tiers[0] if finite_crop_tiers else crop_tiers[0]
    flux = 1.0 if not np.isfinite(tier['flux_max']) else max(1e-6, 0.5 * tier['flux_max'])

    full = add_epsf_counts(
        tf.zeros([41, 41], tf.float32),
        [20 * osf],
        [20 * osf],
        [flux],
        lut,
        osf,
        point_rendering='floor',
    ).numpy()
    cropped = add_epsf_counts(
        tf.zeros([41, 41], tf.float32),
        [20 * osf],
        [20 * osf],
        [flux],
        crop_epsf_lut(lut, tier['kernel_size']),
        osf,
        point_rendering='floor',
    ).numpy()
    diff = full - cropped

    assert(np.max(np.abs(diff)) <= noise_fraction * sigma_pix + 1e-6)
    assert(abs(np.sum(diff)) <= noise_fraction * np.sqrt(flux) + 1e-6)
