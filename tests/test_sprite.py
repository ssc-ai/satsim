"""Tests for `sprites` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
from satsim.geometry.sprite import load_sprite_from_file


def test_load_sprite():

    img = load_sprite_from_file('./tests/sprite.fits', normalize=False)

    expected = np.zeros([8,8])
    expected[3,3] = 65535
    expected[4,3] = 65535
    expected[4,4] = 65535

    np.testing.assert_array_equal(img, expected)

    img = load_sprite_from_file('./tests/sprite.fits', normalize=True)

    expected = np.zeros([8,8])
    expected[3,3] = 1.0 / 3.0
    expected[4,3] = 1.0 / 3.0
    expected[4,4] = 1.0 / 3.0

    np.testing.assert_array_almost_equal_nulp(img, expected)

    img = load_sprite_from_file('./tests/sprite.nosupport', normalize=True)

    assert(img is None)
