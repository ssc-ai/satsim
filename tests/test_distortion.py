"""Direct polynomial rendering, motion, annotations and output integration checks."""

import copy
import importlib

import numpy as np
import pytest
import tensorflow as tf
from astropy.wcs import WCS
from astropy.wcs.utils import wcs_to_celestial_frame
from astropy.coordinates import ICRS

from satsim import config
from satsim.geometry.distortion import (
    DetectorDistortion,
    DistortionError,
    sample_distorted_track,
    distorted_star_motion,
)
from satsim.geometry.draw import TrackSamplingError
from satsim.geometry.wcs import get_wcs
from satsim.image.coordinates import detector_to_oversampled, oversampled_to_detector
from satsim.image.render import render_full, render_epsf
from satsim.image.epsf import build_epsf_lut
from satsim.image.psf import gen_gaussian
from satsim.image.fpa import transform_and_add_counts
from satsim.io import fits as output_fits
from satsim.io.satnet import set_frame_annotation, init_annotation
from satsim.satsim import image_generator
from satsim.pipeline import _avg_t


def polynomial(order=3):
    return {
        "model": "polynomial",
        "order": order,
        "reference_pixel": [39.5, 31.5],
        "normalization_pixels": [40.0, 32.0],
        "dx": {"2_0": 1.2, **({"3_0": 0.05} if order >= 3 else {})},
        "dy": {"0_2": -0.6},
        "valid_domain": {"x": [-32.5, 111.5], "y": [-32.5, 95.5]},
    }


@pytest.fixture
def distortion():
    return DetectorDistortion(polynomial(), 64, 80)


def centroid(image):
    im = np.asarray(image, float)
    y, x = np.indices(im.shape)
    return np.array([(y * im).sum(), (x * im).sum()]) / im.sum()


def test_mapped_track_curvature_flux_and_budget(distortion):
    rr, cc, pe, edges = sample_distorted_track([12.0, 49.0], [8.0, 70.0], 300.0, 0.0, 2.0, distortion.ideal_to_detector)
    assert len(rr) == len(cc) == len(pe) == len(edges) - 1
    np.testing.assert_allclose(pe.sum(), 600.0)
    np.testing.assert_allclose(pe, np.diff(edges) * 300.0)
    # Nonlinear mapping creates a curve from a straight ideal path.
    assert abs(cc[len(cc) // 2] - (cc[0] + cc[-1]) / 2) > 0.2
    t = np.linspace(0, 1, 20001)
    fine = distortion.ideal_to_detector(12 + 37 * t, 8 + 62 * t)
    expected = [np.trapz(a, t) for a in fine]
    np.testing.assert_allclose([np.average(rr, weights=pe), np.average(cc, weights=pe)], expected, atol=1e-5)
    with pytest.raises(TrackSamplingError):
        sample_distorted_track([12.0, 49.0], [8.0, 70.0], 300.0, 0.0, 2.0, distortion.ideal_to_detector, max_samples=10)


@pytest.mark.parametrize("epsf", [False, True])
def test_renderer_maps_each_star_sample_and_conserves_flux(distortion, epsf):
    osf, pad = 2, 8
    r, c = detector_to_oversampled(np.array([26.0]), osf) + pad, detector_to_oversampled(np.array([18.0]), osf) + pad
    mapping = distortion.transform(osf, pad)
    psf = gen_gaussian(80 * osf, 96 * osf, 1.5 * osf)
    args = (64 * osf, 80 * osf, 80 * osf, 96 * osf, pad, pad, osf)
    # Use a 10-detector-pixel trail, with midpoint time quadrature in both paths.
    common = ([], [], [], r, c, [1000.0], 0.0, 1.0, 80, 0.0, [0.0, 20.0])
    if epsf:
        lut = build_epsf_lut(psf, osf, 21)
        stars = render_epsf(
            *args, lut, *common, psf_os=psf, star_render_mode="streak", star_position_transform=mapping
        )[0]
    else:
        stars = render_full(*args, psf, *common, star_render_mode="streak", star_position_transform=mapping)[0]
    times = (np.arange(80) + 0.5) / 80
    expected = distortion.ideal_to_detector(np.full(80, 26.0), 18 + 10 * times)
    np.testing.assert_allclose(centroid(stars), np.mean(expected, axis=1), atol=0.03)
    np.testing.assert_allclose(np.asarray(stars).sum(), 1000.0, rtol=3e-3)


@pytest.mark.parametrize("graph", [False, True])
def test_fft_padding_does_not_undergo_distortion(distortion, graph):
    # The real source moves into the detector. The batch's zero-coordinate
    # padding would move past the SIP fold and has no valid inverse.
    mapping = distortion.transform(1)

    def render():
        return transform_and_add_counts(
            tf.zeros([64, 80]),
            [30.0],
            [380.0],
            [100.0],
            1.0,
            1.001,
            2,
            0.0,
            [0.0, -350.0],
            batch_size=1,
            interpolation="bilinear",
            position_transform=mapping,
        )

    image = tf.function(render)() if graph else render()
    np.testing.assert_allclose(np.sum(image), 100.0, rtol=1e-6)
    expected = distortion.ideal_to_detector([30.0, 30.0], [29.9125, 29.7375])
    np.testing.assert_allclose(centroid(image), np.mean(expected, axis=1), atol=1e-4)


def small_scene(tmp_path, mode="fftconv2p", crop=False):
    ssp = config.load_json("tests/config_static.json")
    ssp["sim"].update(
        mode=mode,
        spacial_osf=2,
        temporal_osf=16,
        padding=8,
        enable_shot_noise=False,
        save_segmentation=False,
        star_annotation_threshold=20,
    )
    ssp["fpa"].update(
        height=64,
        width=80,
        num_frames=1,
        y_fov=0.64,
        x_fov=0.8,
        dark_current=0,
        bias=0,
        distortion=dict(polynomial(), image_plane="ideal"),
    )
    ssp["fpa"]["noise"] = {"read": 0.0, "electronic": 0.0}
    ssp["fpa"]["time"] = {"exposure": 1.0, "gap": 0.0}
    ssp["background"]["galactic"] = 30.0
    ssp["geometry"]["stars"]["mv"] = {"bins": [], "density": []}
    ssp["geometry"]["stars"]["motion"].update(rotation=0.0, translation=[0.0, 0.0])
    ssp["geometry"]["obs"]["list"] = [{"mode": "line", "origin": [0.3, 0.3], "velocity": [9.0, 12.0], "mv": 12.0}]
    if crop:
        ssp["fpa"]["crop"] = {"height_offset": 3, "width_offset": 5, "height": 55, "width": 68}
    if mode == "epsf":
        ssp["sim"]["epsf"] = {"kernel_size": 21}
    return ssp


@pytest.mark.parametrize("catalog_mode", ["csv", "sstr7"])
def test_catalog_star_can_enter_later_frames(tmp_path, monkeypatch, catalog_mode):
    ssp = small_scene(tmp_path)
    ssp["fpa"]["distortion"].update(dx={}, dy={}, valid_domain={"x": [-600, 680], "y": [-600, 664]})
    sky = get_wcs(64, 80, 0.01, 0.01, 120.0, 20.0).wcs_pix2world([[-240.0, 0.0]], 0)[0]
    catalog = tmp_path / "entering.csv"
    catalog.write_text("1,12,{:.12f},{:.12f}\n".format(*sky))
    ssp["fpa"]["num_frames"] = 6
    ssp["sim"]["apply_star_wrap_around"] = False
    ssp["geometry"]["stars"] = {
        "mode": "csv",
        "path": str(catalog),
        "ra": 120.0,
        "dec": 20.0,
        "motion": {"mode": "affine", "rotation": 0.0, "translation": [0.0, 50.0]},
    }
    if catalog_mode == "sstr7":
        sim = importlib.import_module("satsim.satsim")
        csv_query = sim.csvsc_query_by_los

        def query(h, w, yfov, xfov, ra, dec, rot, rootPath, pad_mult, filter_ob, flipud, fliplr, sky_bounds):
            if sky_bounds is not None:
                # A catalog loaded once must cover stars that enter across
                # the whole collect, including its later affine drift.
                low, high = sky_bounds
                assert low[0] <= sky[0] <= high[0]
                assert low[1] <= sky[1] <= high[1]
            return csv_query(h, w, yfov, xfov, ra, dec, rot=rot, rootPath=rootPath, flipud=flipud, fliplr=fliplr)

        monkeypatch.setattr(sim, "query_by_los", query)
        ssp["geometry"]["stars"].update(mode="sstr7", gcvs5={"enabled": True, "path": "tests/data/gcvs5_fixture.txt"})
    ssp["geometry"]["obs"]["list"] = []
    plain = copy.deepcopy(ssp)
    del plain["fpa"]["distortion"]
    reference = list(image_generator(plain, with_meta=True, num_sets=1))
    frames = list(image_generator(ssp, with_meta=True, num_sets=1))
    assert len(frames[0][12]["ra"]) == 0
    for i in [4, 5]:
        assert np.asarray(reference[i][4]).sum() > 100.0
        assert np.asarray(frames[i][4]).sum() > 100.0
        np.testing.assert_allclose(frames[i][12]["ra"], [sky[0]])
        np.testing.assert_allclose(frames[i][12]["mv"], [12.0])
        assert len(frames[i][12]["variable"]) == 1


@pytest.mark.parametrize("rotation,translation", [(0.0, [0.0, 0.0]), (0.07, [1.5, -2.0])])
@pytest.mark.parametrize("flipud,fliplr", [(False, False), (True, False), (False, True), (True, True)])
def test_no_site_catalog_wcs_at_exposure_start(tmp_path, rotation, translation, flipud, fliplr):
    ssp = small_scene(tmp_path, crop=True)
    catalog = tmp_path / "pointing.csv"
    sky = np.array([[120.0, 20.0], [120.05, 20.03], [119.96, 19.95]])
    catalog.write_text("".join("{},12,{},{}\n".format(i + 1, *p) for i, p in enumerate(sky)))
    ssp["fpa"].update(num_frames=3, flip_up_down=flipud, flip_left_right=fliplr)
    ssp["fpa"]["time"]["gap"] = 0.25
    ssp["geometry"]["stars"] = {
        "mode": "csv",
        "path": str(catalog),
        "ra": 120.0,
        "dec": 20.0,
        "rotation": 27.0,
        "motion": {"mode": "affine", "rotation": rotation, "translation": translation},
    }
    ssp["geometry"]["obs"]["list"] = []
    frames = list(image_generator(ssp, with_meta=True, num_sets=1))
    for i, frame in enumerate(frames):
        # Annotation controls are instantaneous start positions, whereas the
        # rendered image integrates the entire exposure's motion.
        paths = frame[12]["annotation_positions"]
        expected = np.column_stack(
            [
                oversampled_to_detector(paths[1, :, 0] - 16, 2),
                oversampled_to_detector(paths[0, :, 0] - 16, 2),
            ]
        )
        # Reopen the independently qualified output WCS.
        _, header = output_fits.save(
            str(tmp_path / "frame{}.fits".format(i)), np.asarray(frame[4]), astrometrics=frame[2]
        )
        wcs = WCS(header)
        assert header["RADESYS"] == "ICRS"
        assert isinstance(wcs_to_celestial_frame(wcs), ICRS)
        np.testing.assert_allclose(wcs.all_world2pix(sky, 0), expected, atol=1e-3)
        np.testing.assert_allclose(
            (header["WCSMJD"] - frames[0][2]["wcs_header"]["WCSMJD"]) * 86400, i * 1.25, atol=5e-5
        )  # Scalar Julian dates resolve ~40 microseconds.
    # The configured boresight uses the calibrated optical reference pixel.
    expected_center = [39.5, 31.5]
    if fliplr:
        expected_center[0] = 79 - expected_center[0]
    if flipud:
        expected_center[1] = 63 - expected_center[1]
    np.testing.assert_allclose(
        WCS(frames[0][2]["wcs_header"]).all_world2pix(sky[:1], 0)[0], np.array(expected_center) - [5, 3], atol=1e-3
    )


@pytest.mark.parametrize("mode", ["line", "line-polar"])
@pytest.mark.parametrize("epoch", [10.0, -2.0])
@pytest.mark.parametrize("brightness", ["pe", "mv"])
def test_distorted_synthetic_target_preserves_epoch(tmp_path, mode, epoch, brightness):
    ssp = small_scene(tmp_path)
    ssp["fpa"]["num_frames"] = 2
    seen = []

    def lightcurve(x, t):
        seen.append((float(t[0]), float(t[-1])))
        rate = 20.0 + _avg_t(t)
        return rate if brightness == "pe" else ssp["fpa"]["zeropoint"] - 2.5 * np.log10(rate)

    ssp["geometry"]["obs"]["list"] = [
        {
            "mode": mode,
            "origin": [0.5, 0.5],
            "velocity": [0.0, 0.1],
            "epoch": epoch,
            brightness: lightcurve,
        }
    ]
    plain = copy.deepcopy(ssp)
    del plain["fpa"]["distortion"]
    reference = list(image_generator(plain, with_meta=True, num_sets=1))
    seen.clear()
    frames = list(image_generator(ssp, with_meta=True, num_sets=1))
    np.testing.assert_allclose(seen, [(epoch, epoch + 1), (epoch + 1, epoch + 2)])
    for i, frame in enumerate(frames):
        expected_flux = 20.0 + epoch + i + 0.5
        np.testing.assert_allclose(np.sum(reference[i][3][0]["pp"]), expected_flux)
        np.testing.assert_allclose(np.sum(frame[3][0]["pp"]), expected_flux)
        np.testing.assert_allclose(np.sum(frame[5]), expected_flux, rtol=3e-3)


@pytest.mark.parametrize("mode", ["fftconv2p", "epsf"])
@pytest.mark.parametrize("crop", [False, True])
def test_generator_target_truth_annotation_and_fits(tmp_path, mode, crop):
    ssp = small_scene(tmp_path, mode, crop)
    frame = next(image_generator(ssp, output_dir=str(tmp_path), with_meta=True, num_sets=1))
    meta, objects, target = frame[2], frame[3], frame[5]
    assert len(objects) == 1
    ob = objects[0]
    expected = [np.average(ob["rrr"], weights=ob["pp"]), np.average(ob["rcc"], weights=ob["pp"])]
    np.testing.assert_allclose(centroid(target), expected, atol=0.04)
    np.testing.assert_allclose(np.sum(target), np.sum(ob["pp"]), rtol=3e-3)
    _, header = output_fits.save(str(tmp_path / "output.fits"), np.asarray(target), astrometrics=meta)
    assert header["CTYPE1"] == "RA---TPV"
    assert "PV1_1" in header
    assert "WCSMJD" in header
    written = WCS(header)
    points = np.array([[15.0, 15.0], [25.0, 35.0]])
    np.testing.assert_allclose(written.all_world2pix(written.all_pix2world(points, 0), 0), points, atol=1e-5)


def test_config_rejects_unknown_keys_and_piecewise(tmp_path):
    ssp = small_scene(tmp_path)
    ssp["fpa"]["distortion"]["unknown"] = 1
    with pytest.raises(DistortionError):
        DetectorDistortion.from_config(ssp)
    del ssp["fpa"]["distortion"]["unknown"]
    ssp["sim"]["render_size"] = 32
    with pytest.raises(DistortionError, match="full-frame"):
        DetectorDistortion.from_config(ssp)


def test_star_annotation_paths_match_curved_motion(distortion):
    n, paths = distorted_star_motion(distortion, [22.0], [18.0], 1, 0, 0.0, 1.0, 0.0, [8.0, 20.0], 1, False)
    assert n > 20
    stars = {
        "h": 64,
        "w": 80,
        "h_pad": 0,
        "w_pad": 0,
        "rr": [22.0],
        "cc": [18.0],
        "pe": [100.0],
        "mv": [10.0],
        "t_start": 0.0,
        "t_end": 1.0,
        "rot": 0.0,
        "tran": [8.0, 20.0],
        "ra": [15.0],
        "dec": [25.0],
        "seg_id": None,
        "min_mv": 20,
        "annotation_positions": paths,
    }
    data = set_frame_annotation(init_annotation("test", 0, 64, 80, 0.01, 0.01), 0, 64, 80, [], star_os_pix=stars)
    ob = data["data"]["objects"][0]
    expected = distortion.ideal_to_detector([22.0, 26.0, 30.0], [18.0, 28.0, 38.0])
    for i, label in enumerate(["start", "mid", "end"]):
        np.testing.assert_allclose(
            [ob["y_" + label] * 64 - 0.5, ob["x_" + label] * 80 - 0.5], np.asarray(expected)[:, i], atol=1e-6
        )


@pytest.mark.parametrize("mode", ["fftconv2p", "epsf"])
@pytest.mark.parametrize("flipud,fliplr", [(False, False), (True, False), (False, True), (True, True)])
def test_catalog_star_and_orbit_share_exported_wcs(tmp_path, mode, flipud, fliplr):
    ssp = small_scene(tmp_path, mode, crop=True)
    catalog = tmp_path / "stars.csv"
    catalog.write_text("1,10.0,91.79056434,-3.41884296\n2,10,270,70\n3,10,nan,nan\n")
    ssp["geometry"]["time"] = [2025, 12, 29, 10, 0, 0.0]
    ssp["sim"].update(enable_deflection=False, enable_light_transit=False, enable_stellar_aberration=False)
    ssp["fpa"].update(flip_up_down=flipud, flip_left_right=fliplr)
    ssp["fpa"]["time"]["exposure"] = 0.01
    target = {
        "mode": "twobody",
        "position": [-1311.059106, 42143.611807, 1.961689],
        "velocity": [-3.073150753, -0.095603993, 0.007795326],
        "epoch": [2025, 12, 29, 10, 0, 0.0],
        "mv": 10.0,
    }
    ssp["geometry"]["site"] = {
        "lat": "20 N",
        "lon": "156 W",
        "alt": 0.0,
        "gimbal": {"mode": "wcs", "rotation": 23.0},
        "track": dict(target, mode="rate", track_apparent=False),
    }
    ssp["geometry"]["stars"] = {"mode": "csv", "path": str(catalog), "motion": {"mode": "none"}}
    ssp["geometry"]["obs"]["list"] = [target]
    frame = next(image_generator(ssp, output_dir=str(tmp_path), with_meta=True, num_sets=1))
    assert len(frame[3]) == 1
    _, h = output_fits.save(str(tmp_path / "sky.fits"), np.asarray(frame[4]), astrometrics=frame[2])
    w = WCS(h)
    expected = w.all_world2pix([[91.79056434, -3.41884296]], 0)[0][::-1]
    np.testing.assert_allclose(centroid(frame[4]), expected, atol=0.01)
    # The tracked target is at the projection reference pixel, transformed by
    # image reflection/cropping. Sky truth is kept independently in the object.
    ob = frame[3][0]
    expected_target = w.all_world2pix([[ob["ra"], ob["dec"]]], 0)[0][::-1]
    np.testing.assert_allclose(centroid(frame[5]), expected_target, atol=0.01)
    annotation = set_frame_annotation(
        init_annotation("test", 0, 55 * 2, 68 * 2, 0.01, 0.01), 0, 55 * 2, 68 * 2, frame[3], star_os_pix=frame[12]
    )
    star = [o for o in annotation["data"]["objects"] if o["class_name"] == "Star"][0]
    np.testing.assert_allclose([star["y_start"] * 55 - 0.5, star["x_start"] * 68 - 0.5], expected, atol=1e-3)
    np.testing.assert_allclose(w.all_world2pix([[star["ra"], star["dec"]]], 0)[0][::-1], expected, atol=1e-7)


def test_disabled_distortion_preserves_rendered_scene(tmp_path):
    ssp = small_scene(tmp_path)
    del ssp["fpa"]["distortion"]
    plain = next(image_generator(copy.deepcopy(ssp), with_meta=True, num_sets=1))
    ssp["fpa"]["distortion"] = None
    disabled = next(image_generator(ssp, with_meta=True, num_sets=1))
    np.testing.assert_array_equal(plain[5], disabled[5])
    assert "wcs_header" not in disabled[2]


def test_tensorflow_graph_mapping(distortion):
    mapping = distortion.transform(1)

    @tf.function
    def mapped(r, c):
        return mapping(r, c)

    r, c = tf.constant([20.0, 30.0]), tf.constant([15.0, 40.0])
    actual = mapped(r, c)
    expected = distortion.ideal_to_detector(r, c)
    np.testing.assert_allclose(actual, expected, atol=2e-6)


@pytest.mark.parametrize("ra,dec", [(359.99, 0.0), (0.01, 40.0), (120.0, 89.9), (250.0, -89.9)])
def test_catalog_bounds_include_distorted_detector_and_poles(distortion, ra, dec):
    low, high = distortion.catalog_bounds(ra, dec, 0.01, 0.012, padding=8)
    x, y = np.meshgrid(np.linspace(-8, 87, 31), np.linspace(-8, 71, 31))
    for roll in [0.0, 31.0, 87.0]:
        sky = distortion.sky_wcs(ra, dec, 0.01, 0.012, roll).all_pix2world(np.column_stack([x.ravel(), y.ravel()]), 0)
        assert np.all((sky[:, 1] >= low[1]) & (sky[:, 1] <= high[1]))
        if low[0] > high[0]:
            assert np.all((sky[:, 0] >= low[0]) | (sky[:, 0] <= high[0]))
        else:
            assert np.all((sky[:, 0] >= low[0]) & (sky[:, 0] <= high[0]))


def test_star_sampling_budget_and_nonprojectable_candidates(distortion):
    mask = distortion.star_candidates([20.0, np.nan, 1e9], [30.0, np.nan, 1e9], 1, 0, 0.0, 1.0, [0.0, 0.0])
    np.testing.assert_array_equal(mask, [True, False, False])
    with pytest.raises(TrackSamplingError, match="max_track_samples"):
        distorted_star_motion(distortion, [20.0], [10.0], 1, 0, 0.0, 1.0, 0.0, [0.0, 30.0], 100, False, max_samples=10)


def test_generated_ideal_stars_are_selected_per_frame(tmp_path, monkeypatch):
    """Broad synthetic populations must not be extrapolated or lose later entrants."""
    ssp = small_scene(tmp_path)
    ssp["fpa"]["distortion"].update(dx={}, dy={}, valid_domain={"x": [-600, 680], "y": [-600, 664]})
    ssp["fpa"]["num_frames"] = 6
    ssp["geometry"]["obs"]["list"] = []
    ssp["geometry"]["stars"]["motion"].update(rotation=0.0, translation=[0.0, 50.0])
    sim = importlib.import_module("satsim.satsim")

    def generated(*args, **kwargs):
        # Oversampled, padded synthetic coordinates; first star enters later,
        # second is irrelevant and outside the calibrated domain throughout.
        return (
            detector_to_oversampled(np.array([30.0, 30.0]), 2) + 8,
            detector_to_oversampled(np.array([-240.0, 5000.0]), 2) + 8,
            np.array([1000.0, 1000.0]),
        )

    monkeypatch.setattr(sim, "gen_random_points", generated)
    frames = list(image_generator(ssp, with_meta=True, num_sets=1))
    assert len(frames[0][12]["ra"]) == 0
    assert len(frames[5][12]["ra"]) == 1
    np.testing.assert_array_equal(frames[5][12]["ra"], [0])
    assert np.asarray(frames[5][4]).sum() > 900


@pytest.mark.parametrize("order", range(2, 8))
@pytest.mark.parametrize("flipud,fliplr", [(False, False), (True, False), (False, True), (True, True)])
@pytest.mark.parametrize("ra,dec,cropped", [(359.8, 63.0, True), (12.0, -80.0, False)])
def test_polynomial_tpv_with_long_drift_and_high_orders(order, flipud, fliplr, ra, dec, cropped):
    block = polynomial(order)
    block["dx"] = {"2_0": 0.1, f"{order}_0": 0.001}
    block["dy"] = {"1_1": -0.03, f"0_{order}": -0.001}
    model = DetectorDistortion(block, 64, 80)
    model.flipud, model.fliplr = flipud, fliplr
    flip = np.diag([-1 if fliplr else 1, -1 if flipud else 1])
    angle, elapsed = 0.43, 10.0
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    affine = flip @ rotate @ flip
    offset = np.array([-450.0, 125.0])
    translation = (flip @ offset - (rotate - np.eye(2)) @ np.full(2, 0.5)) / elapsed
    crop = {"height_offset": 3, "width_offset": 5, "height": 55, "width": 68} if cropped else None
    try:
        w = model.sky_wcs(
            ra,
            dec,
            0.01,
            0.012,
            27.0,
            crop,
            motion_time=elapsed,
            rotation=angle / elapsed,
            translation=translation[::-1],
        )
    except DistortionError as exc:
        # High-order TPV can be numerically unreliable under large drift.
        # Such a header must be rejected, never accepted in one direction only.
        assert order >= 5
        assert "no qualified TPV representation" in str(exc)
        return
    reopened = WCS(w.to_header(relax=True))
    x, y = np.meshgrid(np.linspace(0, 79, 41), np.linspace(0, 63, 39))
    ideal = np.column_stack([x.ravel(), y.ravel()])
    physical = np.column_stack(model.map_xy(*ideal.T))
    pixels = (physical - model.center) @ flip + model.center - ([5, 3] if cropped else [0, 0])
    origin = np.linalg.solve(affine, (ideal - model.center - offset).T).T
    nominal = get_wcs(64, 80, 0.01, 0.012, ra, dec, 27.0)
    sky = nominal.wcs_pix2world(origin, 0)
    assert np.max(np.linalg.norm(reopened.all_world2pix(sky, 0) - pixels, axis=1)) < 0.001
    recovered_ideal = nominal.wcs_world2pix(reopened.all_pix2world(pixels, 0), 0) @ affine.T + model.center + offset
    assert np.max(np.linalg.norm(np.column_stack(model.map_xy(*recovered_ideal.T)) - physical, axis=1)) < 0.001


def test_valid_polynomial_can_reject_unqualified_sky_export():
    block = polynomial()
    block["dx"] = {"3_0": 20.0}
    model = DetectorDistortion(block, 64, 80)
    np.testing.assert_array_equal(model.map_xy(39.5, 31.5), [39.5, 31.5])
    with pytest.raises(DistortionError, match="Cannot export polynomial distortion"):
        model.sky_wcs(120.0, 20.0, 0.01, 0.01)


def test_stationary_catalog_halo_does_not_extrapolate(tmp_path):
    ssp = small_scene(tmp_path)
    block = ssp['fpa']['distortion']
    block['valid_domain'] = {'x': [-8.5, 87.5], 'y': [-8.5, 71.5]}
    ssp['geometry']['obs']['list'] = []
    nominal = get_wcs(64, 80, .01, .01, 120., 20.)
    # Second star is in the conservative displacement halo but outside the
    # calibrated ideal footprint. It is beyond the padded detector support.
    sky = nominal.wcs_pix2world(np.array([[20., 30.], [-9., 30.]]) - [39.5, 31.5], 0)
    catalog = tmp_path/'halo.csv'
    catalog.write_text(''.join(f'{i},12,{ra:.12f},{dec:.12f}\n' for i,(ra,dec) in enumerate(sky)))
    ssp['geometry']['stars'] = {'mode':'csv', 'path':str(catalog), 'ra':120., 'dec':20.,
                                'motion':{'mode':'none'}}
    frame = next(image_generator(ssp,with_meta=True,num_sets=1))
    assert len(frame[12]['ra']) == 1
    np.testing.assert_allclose(frame[12]['ra'],[sky[0,0]])
    assert np.asarray(frame[4]).sum() > 0
