"""Independent sky projection -> direct polynomial -> rendered pixels."""

import numpy as np
import pytest
from astropy.wcs import WCS
from satsim.geometry.wcs import get_wcs
from satsim.satsim import image_generator
from tests.test_distortion import small_scene, centroid


@pytest.mark.parametrize("order", [2, 3])
@pytest.mark.parametrize("osf", [1, 2, 5])
@pytest.mark.parametrize("flipud,fliplr", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("mode", ["fftconv2p", "epsf"])
def test_repointed_render_matches_direct_formula(tmp_path, order, osf, flipud, fliplr, mode):
    scene = small_scene(tmp_path, mode, crop=True)
    block = scene["fpa"]["distortion"]
    block["order"] = order
    if order == 2:
        block["dx"].pop("3_0", None)
    scene["sim"]["spacial_osf"] = osf
    scene["fpa"].update(flip_up_down=flipud, flip_left_right=fliplr)
    ideal = np.array([35.3, 27.2])
    u, v = (ideal - [39.5, 31.5]) / [40.0, 32.0]
    expected = ideal + [1.2 * u * u + (0.05 * u**3 if order == 3 else 0), -0.6 * v * v]
    ra, dec, roll = 359.9, -63.0, 27.0
    sky = get_wcs(64, 80, 0.01, 0.01, ra, dec, roll).wcs_pix2world([ideal - [39.5, 31.5]], 0)[0]
    catalog = tmp_path / "stars.csv"
    catalog.write_text(f"1,10,{sky[0]:.14f},{sky[1]:.14f}\n")
    scene["geometry"]["stars"] = {
        "mode": "csv",
        "path": str(catalog),
        "ra": ra,
        "dec": dec,
        "rotation": roll,
        "motion": {"mode": "affine", "rotation": 0.0, "translation": [0.0, 0.0]},
    }
    scene["geometry"]["obs"]["list"] = []
    frame = next(image_generator(scene, with_meta=True, num_sets=1))
    if fliplr:
        expected[0] = 79 - expected[0]
    if flipud:
        expected[1] = 63 - expected[1]
    expected -= [5, 3]
    np.testing.assert_allclose(centroid(frame[4])[::-1], expected, atol=0.035, rtol=0)
    header = frame[2]["wcs_header"]
    np.testing.assert_allclose(WCS(header).all_world2pix([sky], 0)[0], expected, atol=1e-3, rtol=0)
    assert header["WCSERR"] <= 0.001
