"""Tests for `satsim.image.epsf`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import pytest
import tensorflow as tf

from satsim.image.epsf import build_epsf_lut, add_epsf_counts, transform_and_add_epsf
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
