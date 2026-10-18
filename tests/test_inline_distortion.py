"""Public polynomial convention, domains, validation and portability."""

import json
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from satsim.geometry.distortion import DetectorDistortion, DistortionError
from satsim.image.coordinates import detector_to_oversampled, oversampled_to_detector
from tests.test_distortion import polynomial


@pytest.mark.parametrize("order", [2, 3])
def test_polynomial_is_direct_pixel_displacement(order):
    block = polynomial(order)
    block["dx"].update({"0_0": 2.0, "1_0": 0.2, "1_1": -0.3})
    block["dy"].update({"0_0": -1.0, "0_1": 0.1})
    m = DetectorDistortion(json.loads(json.dumps(block)), 64, 80)
    x = np.linspace(-20, 100, 10001)
    y = np.linspace(-20, 90, 10001)
    u, v = (x - 39.5) / 40, (y - 31.5) / 32
    expected_x = x + 2 + 0.2 * u - 0.3 * u * v + 1.2 * u * u + (0.05 * u**3 if order == 3 else 0)
    expected_y = y - 1 + 0.1 * v - 0.6 * v * v
    actual = m.map_xy(x, y)
    np.testing.assert_allclose(actual, [expected_x, expected_y], atol=1e-12, rtol=0)
    np.testing.assert_allclose(m.ideal_to_detector(y, x), actual[::-1], atol=0, rtol=0)
    assert m.map_xy([], [])[0].size == 0


@pytest.mark.parametrize("osf", [1, 2, 3, 5])
@pytest.mark.parametrize("flipud,fliplr", [(False, False), (True, False), (False, True), (True, True)])
def test_identity_pixel_centers_and_legacy_boresight(osf, flipud, fliplr):
    block = polynomial()
    block.update(dx={}, dy={})
    m = DetectorDistortion(block, 64, 80)
    m.flipud, m.fliplr = flipud, fliplr
    rows, cols = np.array([0.0, 31.5, 63.0]), np.array([0.0, 39.5, 79.0])
    for celestial in (False, True):
        r = detector_to_oversampled(rows, osf)
        c = detector_to_oversampled(cols, osf)
        if celestial:
            r, c = r + 0.5, c + 0.5
        if flipud:
            r = 64 * osf - (0 if celestial else 1) - r
        if fliplr:
            c = 80 * osf - (0 if celestial else 1) - c
        mapped = m.map_oversampled(r + 17, c + 17, osf, padding=17, celestial=celestial)
        actual = [oversampled_to_detector(a - 17, osf) for a in mapped]
        np.testing.assert_allclose(
            actual, [63 - rows if flipud else rows, 79 - cols if fliplr else cols], atol=1e-12, rtol=0
        )
    # Normalization reference changes coordinates of the polynomial only;
    # it must never move the projection's boresight.
    block["reference_pixel"] = [7.0, 12.0]
    offcenter = DetectorDistortion(block, 64, 80)
    r, c = offcenter.map_oversampled([32 * osf], [40 * osf], osf, celestial=True)
    np.testing.assert_allclose([oversampled_to_detector(r, osf), oversampled_to_detector(c, osf)], [[31.5], [39.5]])


@pytest.mark.parametrize("legacy", [{"fits": "x.fits"}, {"sip": {}}, {"hdu": 0}, {}])
def test_no_calibration_formats_in_simulator(legacy):
    with pytest.raises(DistortionError, match="FITS/SIP"):
        DetectorDistortion.from_config({"sim": {}, "fpa": {"height": 64, "width": 80, "distortion": legacy}})


@pytest.mark.parametrize(
    "key,value",
    [
        ("order", 1),
        ("order", True),
        ("order", 8),
        ("model", "sip"),
        ("reference_pixel", [True, 0]),
        ("reference_pixel", [0, float("nan")]),
        ("normalization_pixels", [0, 1]),
        ("dx", {"4_0": 1}),
        ("dx", {"2_0": True}),
        ("dy", {"1_1": float("inf")}),
        ("dx", []),
        ("valid_domain", {"x": [1, 0], "y": [0, 1]}),
        ("image_plane", "bad"),
    ],
)
def test_invalid_polynomials_fail_before_render(key, value):
    block = polynomial()
    block[key] = value
    with pytest.raises(DistortionError):
        DetectorDistortion(block, 64, 80)


def test_domain_and_folds_are_explicit():
    m = DetectorDistortion(polynomial(), 64, 80)
    for x, y in [(-32.6, 0), (0, 96), (float("nan"), 0)]:
        with pytest.raises(DistortionError, match="valid_domain"):
            m.map_xy(x, y)
    block = polynomial()
    block["dx"] = {"2_0": 100}
    with pytest.raises(DistortionError, match="folds"):
        DetectorDistortion(block, 64, 80)


def test_schema_and_json_only_config(tmp_path):
    block = polynomial()
    schema = json.loads((Path(__file__).parents[1] / "schema/v1/types/Distortion.json").read_text())
    jsonschema.validate(block, schema)
    cfg = {"sim": {"mode": "epsf"}, "fpa": {"height": 64, "width": 80, "distortion": block}}
    path = tmp_path / "only.json"
    path.write_text(json.dumps(cfg))
    loaded = json.loads(path.read_text())
    loaded["_input_dir"] = "/does/not/exist"
    m = DetectorDistortion.from_config(loaded)
    assert m.map_xy(39.5, 31.5) == (39.5, 31.5)
    assert list(tmp_path.iterdir()) == [path]
    for invalid in (dict(block, fits="x.fits"), dict(block, sip={}), dict(block, order=1), dict(block, order=2)):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(invalid, schema)
    loaded["fpa"]["linear_wcs"] = {"matrix": [[1, 0], [0, 1]]}
    with pytest.raises(DistortionError, match="linear_wcs"):
        DetectorDistortion.from_config(loaded)
