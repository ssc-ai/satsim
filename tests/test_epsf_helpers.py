"""Acceptance gates for faint flux inside otherwise empty apertures."""
import numpy as np
import pytest

from tests.epsf_helpers import assert_equivalent


@pytest.mark.parametrize('sign', [-1, 1])
def test_empty_aperture_uses_documented_source_scaled_absolute_floor(sign):
    reference = np.zeros((21, 21), dtype=np.float64)
    reference[3, 3] = reference[17, 17] = 50000.0
    source_flux = reference.sum()
    floor = source_flux * 1e-8

    actual = reference.copy()
    actual[10, 10] = sign * floor / 2
    assert_equivalent(actual, reference, source_flux)

    # The image and total-flux errors still satisfy their relative gates,
    # but the empty central apertures must reject a larger perturbation.
    actual[10, 10] = sign * floor * 2
    with pytest.raises(AssertionError):
        assert_equivalent(actual, reference, source_flux)
