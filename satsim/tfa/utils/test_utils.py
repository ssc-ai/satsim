"""Utilities for testing Addons."""
import numpy as np
import tensorflow as tf


def assert_not_allclose(a, b, **kwargs):  # pragma: no cover
    """Assert that two numpy arrays, do not have near values.

    Args:
      a: the first value to compare.
      b: the second value to compare.
      **kwargs: additional keyword arguments to be passed to the underlying
        `np.testing.assert_allclose` call.

    Raises:
      AssertionError: If `a` and `b` are unexpectedly close at all elements.
    """
    try:
        np.testing.assert_allclose(a, b, **kwargs)
    except AssertionError:
        return
    raise AssertionError("The two values are close at all elements")


def assert_allclose_according_to_type(
    a,
    b,
    rtol=1e-6,
    atol=1e-6,
    float_rtol=1e-6,
    float_atol=1e-6,
    half_rtol=1e-3,
    half_atol=1e-3,
    bfloat16_rtol=1e-2,
    bfloat16_atol=1e-2,
):  # pragma: no cover
    """
    Similar to tf.test.TestCase.assertAllCloseAccordingToType()
    but this doesn't need a subclassing to run.
    """
    a = np.array(a)
    b = np.array(b)
    # types with lower tol are put later to overwrite previous ones.
    if (
        a.dtype == np.float32
        or b.dtype == np.float32
        or a.dtype == np.complex64
        or b.dtype == np.complex64
    ):
        rtol = max(rtol, float_rtol)
        atol = max(atol, float_atol)
    if a.dtype == np.float16 or b.dtype == np.float16:
        rtol = max(rtol, half_rtol)
        atol = max(atol, half_atol)
    if a.dtype == tf.bfloat16.as_numpy_dtype or b.dtype == tf.bfloat16.as_numpy_dtype:
        rtol = max(rtol, bfloat16_rtol)
        atol = max(atol, bfloat16_atol)

    np.testing.assert_allclose(a, b, rtol=rtol, atol=atol)
