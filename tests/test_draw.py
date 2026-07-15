"""Tests for :mod:`satsim.geometry.draw`."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np

from satsim.geometry.draw import gen_line, gen_line_from_endpoints, gen_curve_from_points


def _centroid(rr, cc, weights):
    return np.array([
        np.average(rr, weights=weights),
        np.average(cc, weights=weights),
    ])


def _curve_centroid(r0, c0, r1, c1, r2, c2):
    """Integral mean of a quadratic constrained at start, middle, and end."""
    return np.array([
        (r0 + 4.0 * r1 + r2) / 6.0,
        (c0 + 4.0 * c1 + c2) / 6.0,
    ])


def test_gen_line_uses_canonical_normalized_coordinates():
    rr, cc, counts, _ = gen_line(500, 500, [.5, .5], [0., 0.], 100, 0, 0.5)
    np.testing.assert_allclose([rr[0], cc[0]], [249.5, 249.5])
    np.testing.assert_allclose(np.sum(counts), 50.0)

    cases = [
        # Normalized zero is the top/left image edge at pixel coordinate -0.5.
        ([0., 0.], [4., 0.], 0.0, 1.0, [1.5, -0.5]),
        # Pixel-center zero maps to normalized coordinate 0.5 / size.
        ([0.001, 0.001], [4., 4.], 0.0, 1.0, [2.0, 2.0]),
        ([0.00199, 0.00199], [4., 4.], 0.0, 1.0, [2.495, 2.495]),
        ([0.002, 0.002], [2., 2.], 0.0, 2.0, [2.5, 2.5]),
        ([1. - 1. / 500., 1. - 1. / 500.], [-4., -4.], 0.0, 1.0, [496.5, 496.5]),
    ]
    for origin, velocity, t_start, t_end, expected in cases:
        rr, cc, counts, times = gen_line(
            500, 500, origin, velocity, 100, t_start, t_end
        )
        np.testing.assert_allclose(_centroid(rr, cc, counts), expected)
        np.testing.assert_allclose(np.sum(counts), (t_end - t_start) * 100)
        np.testing.assert_allclose(counts, np.diff(times) * 100)
        assert len(times) == len(rr) + 1

    # Rows and columns use their own dimensions and velocity components.
    rr, cc, counts, _ = gen_line(
        500, 1000, [0.004, 0.001], [4., 2.], 90, 0, 2.0
    )
    np.testing.assert_allclose(_centroid(rr, cc, counts), [5.5, 2.5])
    np.testing.assert_allclose(np.sum(counts), 180.0)


def test_gen_line_from_endpoints_preserves_continuous_centroid():
    cases = [
        (250., 250., 250., 250., 0.0, 0.5),
        (0., 0., 4., 0., 0.0, 1.0),
        (0.5, 0.5, 4., 4., 0.0, 1.0),
        (0.99, 0.99, 4., 4., 0.0, 1.0),
        (1., 1., 5., 5., 0.0, 2.0),
        (499., 499., 495., 495., 0.0, 1.0),
        (2., 1., 10., 5., 0.0, 2.0),
    ]
    for r0, c0, r1, c1, t_start, t_end in cases:
        rr, cc, counts, times = gen_line_from_endpoints(
            r0, c0, r1, c1, 100, t_start, t_end
        )
        np.testing.assert_allclose(
            _centroid(rr, cc, counts),
            [(r0 + r1) / 2.0, (c0 + c1) / 2.0],
        )
        np.testing.assert_allclose(np.sum(counts), (t_end - t_start) * 100)
        np.testing.assert_allclose(counts, np.diff(times) * 100)
        assert len(times) == len(rr) + 1
        if r0 != r1 or c0 != c1:
            np.testing.assert_allclose([rr[0], cc[0]], [r0, c0])
            np.testing.assert_allclose([rr[-1], cc[-1]], [r1, c1])
            np.testing.assert_allclose([counts[0], counts[-1]], [0.0, 0.0])

    # A motion wholly inside one raster cell must not snap to either endpoint.
    rr, cc, counts, _ = gen_line_from_endpoints(
        0.1, 0.2, 0.2, 0.3, 100, 0.0, 1.0
    )
    np.testing.assert_allclose([rr[0], cc[0]], [0.1, 0.2])
    np.testing.assert_allclose([rr[-1], cc[-1]], [0.2, 0.3])
    np.testing.assert_allclose(
        [rr[counts > 0][0], cc[counts > 0][0]],
        [0.15, 0.25],
    )
    np.testing.assert_allclose(counts, [0.0, 100.0, 0.0])


def test_gen_curve_from_points_preserves_continuous_centroid():
    cases = [
        (250., 250., 250., 250., 250., 250., 0.5),
        (250., 250., 251., 251., 252., 252., 0.5),
        (0., 0., 2., 3., 0., 6., 0.5),
        (0., 0., 2., 3., 6., 6., 0.5),
        (250., 250., 255., 255., 250., 260., 0.5),
        (94., 6., 128., 187., 44., 182., 0.5),
        (31., 240., 54., 155., 41., 27., 0.5),
    ]
    for r0, c0, r1, c1, r2, c2, duration in cases:
        rr, cc, counts, times = gen_curve_from_points(
            r0, c0, r1, c1, r2, c2, 90, 0, duration
        )
        np.testing.assert_allclose(
            _centroid(rr, cc, counts),
            _curve_centroid(r0, c0, r1, c1, r2, c2),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(np.sum(counts), duration * 90)
        np.testing.assert_allclose(counts, np.diff(times) * 90)
        assert len(times) == len(rr) + 1
        if r0 != r2 or c0 != c2 or r0 != r1 or c0 != c1:
            mid = len(rr) // 2
            np.testing.assert_allclose([rr[0], cc[0]], [r0, c0])
            np.testing.assert_allclose([rr[mid], cc[mid]], [r1, c1])
            np.testing.assert_allclose([rr[-1], cc[-1]], [r2, c2])
            np.testing.assert_allclose(
                [counts[0], counts[mid], counts[-1]],
                [0.0, 0.0, 0.0],
            )


# A rate-tracked target sits at the frame-center row, height / 2 in
# oversampled pixels (768.0 for a 512-pixel detector at spacial_osf 3), but
# floating point error in the WCS round trip can return it epsilon below the
# exact integer. int() truncation used to move the target a full oversampled
# pixel while stars kept exact coordinates.
RATE_TRACK_CENTER_ROW = 768.0 - 1.75e-11
RATE_TRACK_CENTER_COL = 768.0


def test_stationary_subpixel_trajectory_is_not_truncated():
    for generator in (gen_line_from_endpoints, gen_curve_from_points):
        if generator is gen_line_from_endpoints:
            args = (
                RATE_TRACK_CENTER_ROW,
                RATE_TRACK_CENTER_COL,
                RATE_TRACK_CENTER_ROW,
                RATE_TRACK_CENTER_COL,
            )
        else:
            args = (
                RATE_TRACK_CENTER_ROW,
                RATE_TRACK_CENTER_COL,
                RATE_TRACK_CENTER_ROW,
                RATE_TRACK_CENTER_COL,
                RATE_TRACK_CENTER_ROW,
                RATE_TRACK_CENTER_COL,
            )
        rr, cc, _, _ = generator(*args, 100, 0, 0.5)
        np.testing.assert_allclose(rr, [RATE_TRACK_CENTER_ROW])
        np.testing.assert_allclose(cc, [RATE_TRACK_CENTER_COL])


def test_fractional_curve_midpoint_changes_path_and_centroid_correctly():
    low = gen_curve_from_points(0., 0., 2.1, 0., 4., 0., 100, 0, 1.0)
    high = gen_curve_from_points(0., 0., 2.4, 0., 4., 0., 100, 0, 1.0)

    low_centroid = _centroid(low[0], low[1], low[2])
    high_centroid = _centroid(high[0], high[1], high[2])
    np.testing.assert_allclose(low_centroid, _curve_centroid(0., 0., 2.1, 0., 4., 0.))
    np.testing.assert_allclose(high_centroid, _curve_centroid(0., 0., 2.4, 0., 4., 0.))
    np.testing.assert_allclose(high_centroid - low_centroid, [0.2, 0.0])

    # Even a complete fractional curve inside one pixel retains its exact
    # exposure centroid instead of collapsing to an integer pixel center.
    rr, cc, counts, _ = gen_curve_from_points(
        0.1, 0.1, 0.2, 0.3, 0.3, 0.2, 100, 0, 1.0
    )
    np.testing.assert_allclose(
        _centroid(rr, cc, counts),
        _curve_centroid(0.1, 0.1, 0.2, 0.3, 0.3, 0.2),
    )


def test_gen_line_subpixel_uses_edge_based_normalized_origin():
    rr, cc, counts, _ = gen_line(
        500, 500, [0.001, 0.001], [4., 4.], 100, 0, 1.0
    )

    # 0.001 * 500 - 0.5 is pixel-center coordinate zero; the exposure
    # centroid of the four-pixel motion is therefore coordinate two.
    np.testing.assert_allclose(_centroid(rr, cc, counts), [2.0, 2.0])
    np.testing.assert_allclose(np.sum(counts), 100.0)
