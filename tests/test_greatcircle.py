import numpy as np
import pytest

from satsim import time
from satsim.geometry.astrometric import create_topocentric
from satsim.geometry.greatcircle import GreatCircle


@pytest.mark.parametrize('seconds', [[10.0], [10.0, 20.0, 30.0]])
@pytest.mark.parametrize('topocentric', [False, True])
@pytest.mark.filterwarnings(
    'error:Conversion of an array with ndim > 0 to a scalar is deprecated:DeprecationWarning'
)
def test_scalar_times_match_vector_times(seconds, topocentric):
    epoch = time.utc(2021, 3, 18, 0, 0, 0.0)
    observer = create_topocentric('20.746111 N', '156.431667 W') if topocentric else None
    target = GreatCircle(0, 0, 90, 1.0, epoch, observer)
    times = time.utc(2021, 3, 18, 0, 0, np.asarray(seconds))

    vector = target.at(times)
    assert vector.position.au.shape == (3, len(seconds))
    assert vector.velocity.au_per_d.shape == (3, len(seconds))
    for index in range(len(seconds)):
        scalar = target.at(times[index])
        assert scalar.position.au.shape == (3,)
        assert scalar.velocity.au_per_d.shape == (3,)
        np.testing.assert_allclose(scalar.position.au, vector.position.au[:, index], rtol=1e-12)
        np.testing.assert_allclose(scalar.velocity.au_per_d, vector.velocity.au_per_d[:, index], rtol=1e-12)
