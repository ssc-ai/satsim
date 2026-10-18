"""Direct, scene-independent polynomial distortion in native pixel coordinates.

Integer coordinates are pixel centers. Public map_xy takes (column, row);
renderer adapters take (row, column). Calibration file formats belong upstream.
"""

import copy
import hashlib
import json
import math
import re
from numbers import Real

import numpy as np

from satsim.geometry.draw import MAX_TRACK_SAMPLES, TrackSamplingError
from satsim.image.coordinates import detector_to_oversampled, oversampled_to_detector

DISTORTION_API = 3
MAX_POLYNOMIAL_ORDER = 7


class DistortionError(ValueError):
    """Invalid polynomial, coordinate domain, or output representation."""


def polynomial_powers(order):
    return [(i, degree - i) for degree in range(order + 1) for i in range(degree + 1)]


def polynomial_design(points, reference, scale, powers):
    uv = (np.asarray(points, float) - reference) / scale
    return np.column_stack([uv[:, 0] ** i * uv[:, 1] ** j for i, j in powers])


def _number(value):
    return isinstance(value, Real) and not isinstance(value, (bool, np.bool_)) and np.isfinite(value)


class DetectorDistortion:
    """x' = x + dx(u,v), y' = y + dy(u,v), with pixel-valued corrections."""

    def __init__(self, config, height, width):
        required = {"model", "order", "reference_pixel", "normalization_pixels", "dx", "dy", "valid_domain"}
        if not isinstance(config, dict) or not required <= config.keys() or set(config) - required - {"image_plane"}:
            raise DistortionError(
                "distortion requires model, order, reference_pixel, normalization_pixels, dx, dy and valid_domain; "
                "FITS/SIP inputs are no longer supported. Convert them with satsim-calibrate."
            )
        if config["model"] != "polynomial":
            raise DistortionError("distortion.model must be polynomial.")
        order = config["order"]
        if isinstance(order, bool) or not isinstance(order, int) or not 2 <= order <= MAX_POLYNOMIAL_ORDER:
            raise DistortionError("distortion.order must be an integer from 2 to 7.")
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0 for n in (height, width)):
            raise DistortionError("Detector dimensions must be positive integers.")
        self.height, self.width, self.order = height, width, order
        self.center = np.array([(width - 1) / 2.0, (height - 1) / 2.0])
        for key in ("reference_pixel", "normalization_pixels"):
            value = config[key]
            if not isinstance(value, (list, tuple)) or len(value) != 2 or not all(_number(v) for v in value):
                raise DistortionError("{} must contain two finite numbers in (x,y) order.".format(key))
        self.reference = np.array(config["reference_pixel"], float)
        self.scale = np.array(config["normalization_pixels"], float)
        if np.any(self.scale <= 0):
            raise DistortionError("normalization_pixels must be strictly positive.")
        domain = config["valid_domain"]
        if not isinstance(domain, dict) or set(domain) != {"x", "y"}:
            raise DistortionError("valid_domain requires x and y bounds in ideal native pixels.")
        for key in ("x", "y"):
            value = domain[key]
            if (
                not isinstance(value, (list, tuple))
                or len(value) != 2
                or not all(_number(v) for v in value)
                or value[0] >= value[1]
            ):
                raise DistortionError("valid_domain bounds must be finite and strictly increasing.")
        self.domain = np.array([domain["x"], domain["y"]], float)
        terms = {}
        for axis, key in enumerate(("dx", "dy")):
            if not isinstance(config[key], dict):
                raise DistortionError("{} must be a coefficient dictionary.".format(key))
            for power, coefficient in config[key].items():
                if not isinstance(power, str) or not re.fullmatch(r"[0-7]_[0-7]", power) or not _number(coefficient):
                    raise DistortionError("Polynomial coefficients must have i_j keys and finite numeric pixel values.")
                i, j = map(int, power.split("_"))
                if i + j > order:
                    raise DistortionError("Coefficient {} exceeds polynomial order {}.".format(power, order))
                if coefficient:
                    terms.setdefault((i, j), [0.0, 0.0])[axis] = float(coefficient)
        self._terms = tuple((i, j, *v) for (i, j), v in sorted(terms.items()))
        self.image_plane = config.get("image_plane", "detector")
        if self.image_plane not in ("ideal", "detector"):
            raise DistortionError("image_plane must be ideal or detector.")
        self.flipud = self.fliplr = False
        self.config = copy.deepcopy(config)
        self.metadata = {
            "mode": "polynomial",
            "direction": "ideal_to_detector",
            "pixel_centers": "integer",
            "height": height,
            "width": width,
            "order": order,
            "image_plane": self.image_plane,
            "parameters": self.config,
            "parameters_sha256": hashlib.sha256(
                json.dumps(config, sort_keys=True, allow_nan=False).encode()
            ).hexdigest(),
        }
        # Reject sampled folds/singularities before rendering. This bounded check
        # does not prove global invertibility; also check every evaluated point.
        x, y = np.meshgrid(np.linspace(*self.domain[0], 33), np.linspace(*self.domain[1], 33))
        jac = self.jacobian_determinant(x, y)
        if not np.isfinite(jac).all() or not (np.all(jac > 1e-10) or np.all(jac < -1e-10)):
            raise DistortionError("Polynomial folds or is singular over its sampled valid_domain.")
        self.orientation = np.sign(jac.flat[0])

    @classmethod
    def from_config(cls, ssp):
        fpa = ssp["fpa"]
        if fpa.get("linear_wcs") is not None:
            raise DistortionError(
                "linear_wcs is no longer a simulation input; include its affine geometry in the pixel polynomial."
            )
        config = fpa.get("distortion")
        if config is None:
            return None
        result = cls(config, fpa["height"], fpa["width"])
        result.flipud, result.fliplr = fpa.get("flip_up_down", False), fpa.get("flip_left_right", False)
        if ssp["sim"].get("render_size") is not None and ssp["sim"].get("mode") != "epsf":
            raise DistortionError("Polynomial distortion requires full-frame FFT rendering or native ePSF.")
        if ssp["sim"].get("mode") == "none":
            raise DistortionError("Detector distortion is an optical image feature; sim.mode=none is unsupported.")
        return result

    @staticmethod
    def _evaluate(terms, x, y, derivatives=False):
        xp, yp = [np.ones_like(x), x], [np.ones_like(y), y]
        for _ in range(2, max((t[0] for t in terms), default=0) + 1):
            xp.append(xp[-1] * x)
        for _ in range(2, max((t[1] for t in terms), default=0) + 1):
            yp.append(yp[-1] * y)
        values = [np.zeros_like(x) for _ in range(6 if derivatives else 2)]
        for i, j, a, b in terms:
            values[0] += a * xp[i] * yp[j]
            values[1] += b * xp[i] * yp[j]
            if derivatives:
                if i:
                    values[2] += a * i * xp[i - 1] * yp[j]
                    values[4] += b * i * xp[i - 1] * yp[j]
                if j:
                    values[3] += a * j * xp[i] * yp[j - 1]
                    values[5] += b * j * xp[i] * yp[j - 1]
        return values

    def jacobian_determinant(self, x, y):
        u, v = (np.asarray(x) - self.reference[0]) / self.scale[0], (np.asarray(y) - self.reference[1]) / self.scale[1]
        _, _, ax, ay, bx, by = self._evaluate(self._terms, u, v, True)
        return (1 + ax / self.scale[0]) * (1 + by / self.scale[1]) - ay * bx / np.prod(self.scale)

    def contains(self, x, y):
        eps = 1e-9 * max(self.width, self.height)
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= self.domain[0, 0] - eps)
            & (x <= self.domain[0, 1] + eps)
            & (y >= self.domain[1, 0] - eps)
            & (y <= self.domain[1, 1] + eps)
        )

    def map_xy(self, x, y):
        """Map ideal (x,y) to distorted (x,y), with no WCS or iterative inverse."""
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if not np.all(self.contains(x, y)):
            raise DistortionError(
                "Distortion input is nonfinite or outside valid_domain; recalibrate for the required detector/padding/motion."
            )
        out_x, out_y = np.empty(x.size), np.empty(y.size)
        for first in range(0, x.size, 8192):
            last = min(first + 8192, x.size)
            xx, yy = x.flat[first:last], y.flat[first:last]
            u, v = (xx - self.reference[0]) / self.scale[0], (yy - self.reference[1]) / self.scale[1]
            with np.errstate(over="ignore", invalid="ignore"):
                a, b = self._evaluate(self._terms, u, v)
                jac = self.jacobian_determinant(xx, yy)
            if not np.isfinite([a, b, jac]).all() or np.any(jac * self.orientation <= 1e-10):
                raise DistortionError("Polynomial is nonfinite, folded or singular at an evaluated position.")
            out_x[first:last], out_y[first:last] = xx + a, yy + b
        return out_x.reshape(x.shape), out_y.reshape(y.shape)

    def ideal_to_detector(self, rows, cols):
        """Row/column adapter for the existing renderer and track sampler."""
        x, y = self.map_xy(cols, rows)
        return y, x

    def _ideal_native(self, rows, cols, osf=1, padding=0, celestial=False):
        r, c = np.asarray(rows, float) - padding, np.asarray(cols, float) - padding
        if self.flipud:
            r = self.height * osf - r if celestial else self.height * osf - 1 - r
        if self.fliplr:
            c = self.width * osf - c if celestial else self.width * osf - 1 - c
        r, c = oversampled_to_detector(r, osf), oversampled_to_detector(c, osf)
        if celestial:
            # Legacy projections center at size_os/2, not (size_os-1)/2.
            # Correct before the polynomial, independently of its reference.
            r -= 0.5 / osf
            c -= 0.5 / osf
        return r, c

    def map_oversampled(self, rows, cols, osf=1, padding=0, celestial=False):
        r, c = self._ideal_native(rows, cols, osf, padding, celestial)
        r, c = self.ideal_to_detector(r, c)
        if self.flipud:
            r = self.height - 1 - r
        if self.fliplr:
            c = self.width - 1 - c
        return detector_to_oversampled(r, osf) + padding, detector_to_oversampled(c, osf) + padding

    def transform(self, osf, padding=0, celestial=False):
        import tensorflow as tf

        def callback(rows, cols):
            r, c = tf.numpy_function(
                lambda r, c: self.map_oversampled(r, c, osf, padding, celestial), [rows, cols], [tf.float64, tf.float64]
            )
            r.set_shape(rows.shape)
            c.set_shape(cols.shape)
            return tf.cast(r, rows.dtype), tf.cast(c, cols.dtype)

        return callback

    def sky_wcs(self, ra, dec, y_ifov, x_ifov, roll=0, crop=None, motion_time=0, rotation=0, translation=(0, 0), osf=1):
        # Output serialization is separate from the detector mapping. No sky
        # pointing, FITS file or WCS coefficients are inputs to the polynomial.
        from satsim.geometry.distortion_wcs import export_wcs

        return export_wcs(self, ra, dec, y_ifov, x_ifov, roll, crop, motion_time, rotation, translation, osf)

    def query_padding(self, padding=0):
        extent = np.max(np.abs((self.domain - self.reference[:, None]) / self.scale[:, None]), axis=1)
        bounds = np.zeros(2)
        for i, j, a, b in self._terms:
            bounds += np.abs([a, b]) * extent[0] ** i * extent[1] ** j
        return float(max(bounds)) + padding + 2

    def catalog_bounds(self, ra, dec, y_ifov, x_ifov, padding=0, motion_pixels=0):
        # The admissible ideal rectangle is known directly; no inverse SIP
        # or distorted-corner approximation is required for catalog selection.
        extent = np.max(np.abs(self.domain - self.center[:, None]), axis=1)
        radius_pixels = np.linalg.norm(extent) + float(motion_pixels)
        radius = np.degrees(np.arctan(np.radians(radius_pixels * max(abs(x_ifov), abs(y_ifov))))) + 1e-8
        low, high = max(-90.0, dec - radius), min(90.0, dec + radius)
        if low <= -90 or high >= 90:
            return [0.0, low], [360.0, high]
        span = np.degrees(np.arcsin(np.clip(np.sin(np.radians(radius)) / np.cos(np.radians(dec)), -1.0, 1.0)))
        return [(ra - span) % 360.0, low], [(ra + span) % 360.0, high]

    def star_candidates(self, rows, cols, osf, padding, t_start, t_end, translation, rotation=0, celestial=True):
        r, c = self._ideal_native(rows, cols, osf, padding, celestial=celestial)
        halo = self.query_padding(padding / osf)
        drift = np.abs(translation) * max(abs(float(t_start)), abs(float(t_end))) / osf
        if rotation == 0:
            candidates = (
                np.isfinite(r)
                & np.isfinite(c)
                & (r >= -0.5 - halo - drift[0])
                & (r <= self.height - 0.5 + halo + drift[0])
                & (c >= -0.5 - halo - drift[1])
                & (c <= self.width - 0.5 + halo + drift[1])
            )
            if not np.any(translation):
                # Catalog queries conservatively cover more than the calibrated
                # ideal footprint. Stationary sources outside that footprint
                # are not supported detector sources. Moving sources retain the
                # existing full-trajectory domain check so entrants are not lost.
                candidates &= self.contains(c, r)
            return candidates
        radius = np.hypot(self.width / 2 + halo, self.height / 2 + halo) + np.linalg.norm(drift)
        return np.isfinite(r) & np.isfinite(c) & (np.hypot(c - self.center[0], r - self.center[1]) <= radius)


def sample_distorted_track(rows, cols, pe, t_start, t_end, mapping, max_samples=MAX_TRACK_SAMPLES):
    """Sample a mapped line/quadratic in time with bounded spatial resolution.

    Spatial intervals are at most one oversampled pixel, with midpoint chord
    residual <= .01 oversampled pixels. Two-point Gaussian quadrature locates
    each time bin's flux centroid. Controls have zero duration and zero flux.
    """
    rows, cols = np.asarray(rows, float), np.asarray(cols, float)
    if len(rows) not in (2, 3) or not np.all(np.isfinite([rows, cols])):
        raise TrackSamplingError("Distorted track controls must be finite lines or quadratics.")

    def ideal(u):
        if len(rows) == 2:
            return rows[0] + u * (rows[1] - rows[0]), cols[0] + u * (cols[1] - cols[0])

        def quadratic(a):
            return (2 * a[0] - 4 * a[1] + 2 * a[2]) * u**2 + (-3 * a[0] + 4 * a[1] - a[2]) * u + a[0]

        return quadratic(rows), quadratic(cols)

    def mapped(u):
        return np.asarray(mapping(*ideal(u)))

    pilot = mapped(np.linspace(0, 1, 65))
    n = max(2, int(math.ceil(np.max(np.abs(np.diff(pilot)), initial=0) * 64)))
    n += n % 2
    endpoints = None
    while True:
        if n + 3 > max_samples:
            raise TrackSamplingError("Distorted trajectory exceeds max_track_samples.")
        edges = np.linspace(0, 1, n + 1)
        if endpoints is None:
            endpoints = mapped(edges)
        mid = (edges[:-1] + edges[1:]) * 0.5
        centers = mapped(mid)
        step = np.max(np.abs(np.diff(endpoints)), initial=0)
        error = np.max(np.abs(centers - (endpoints[:, :-1] + endpoints[:, 1:]) / 2), initial=0)
        if step <= 1 and error <= 0.01:
            break
        if 2 * n + 3 > max_samples:
            raise TrackSamplingError("Distorted trajectory exceeds max_track_samples.")
        # The doubled grid contains the old edges and centers exactly.
        refined = np.empty((2, 2 * n + 1))
        refined[:, ::2], refined[:, 1::2] = endpoints, centers
        endpoints = refined
        n *= 2
    delta = 0.5 / (n * math.sqrt(3))
    mean = (mapped(mid - delta) + mapped(mid + delta)) / 2
    half = n // 2
    controls = endpoints[:, [0, half, -1]]
    rr, cc = [np.concatenate(([a[0]], m[:half], [a[1]], m[half:], [a[2]])) for a, m in zip(controls, mean)]
    counts = np.concatenate(
        ([0.0], np.full(half, (t_end - t_start) * pe / n), [0.0], np.full(half, (t_end - t_start) * pe / n), [0.0])
    )
    times = t_start + edges * (t_end - t_start)
    times = np.concatenate(([t_start], times[: half + 1], [times[half]], times[half + 1:], [t_end]))
    return rr, cc, counts, times


def distorted_star_motion(
    distortion,
    rows,
    cols,
    osf,
    padding,
    t_start,
    t_end,
    rotation,
    translation,
    samples,
    celestial,
    max_samples=MAX_TRACK_SAMPLES,
):
    """Choose temporal resolution and retain start/mid/end plus bounding extrema.

    Coordinates are evaluated in bounded time/source blocks. Annotation slots
    are start, minimum, midpoint, maximum, end; the two extrema are bounds, not
    a second physical motion model.
    """
    r, c = np.asarray(rows, float), np.asarray(cols, float)
    h, w = distortion.height * osf + 2 * padding, distortion.width * osf + 2 * padding
    rotation, translation = float(rotation), np.asarray(translation, float)

    def at(times, first=0, last=None):
        t = np.asarray(times, float)[:, None]
        s, co = np.sin(t * rotation), np.cos(t * rotation)
        rr, cc = r[first:last][None, :], c[first:last][None, :]
        rr, cc = (
            s * cc + co * rr + ((h - 1) - (s * (w - 1) + co * (h - 1))) / 2 + translation[0] * t,
            co * cc - s * rr + ((w - 1) - (co * (w - 1) - s * (h - 1))) / 2 + translation[1] * t,
        )
        return np.asarray(distortion.map_oversampled(rr, cc, osf, padding, celestial))

    if not r.size:
        return 1, np.empty((2, 0, 5))
    if rotation == 0 and not np.any(translation):
        # Five annotation slots share one immutable position for a static
        # field; evaluate the mapping once instead of at every pilot time.
        if max_samples < 1:
            raise TrackSamplingError("Distorted star motion exceeds max_track_samples.")
        mapped = np.asarray(distortion.map_oversampled(r, c, osf, padding, celestial))
        return 1, np.repeat(mapped[:, :, None], 5, axis=2)
    speed = 0.0
    paths = np.empty((2, len(r), 5))
    for first in range(0, len(r), 512):
        last = min(first + 512, len(r))
        pilot_times = np.linspace(t_start, t_end, 17)
        pilot_times[8] = (t_start + t_end) / 2
        pilot = at(pilot_times, first, last)
        speed = max(speed, float(np.max(np.abs(np.diff(pilot, axis=1)))))
        paths[:, first:last, 0] = pilot[:, 0]
        paths[:, first:last, 2] = pilot[:, 8]
        paths[:, first:last, 4] = pilot[:, -1]
    n = max(int(samples), int(math.ceil(speed * 16 * 1.1)), 2)
    while True:
        if n > max_samples:
            raise TrackSamplingError("Distorted star motion exceeds max_track_samples.")
        max_step = 0.0
        for first in range(0, len(r), 512):
            last = min(first + 512, len(r))
            controls = paths[:, first:last, :][:, :, [0, 2, 4]].transpose(0, 2, 1)
            low, high = controls.min(axis=1), controls.max(axis=1)
            previous = controls[:, 0]
            # At most 65,536 source/time positions per mapping call.
            time_block = min(128, 65536 // (last - first))
            for start in range(0, n + 1, time_block):
                times = t_start + np.arange(start, min(start + time_block, n + 1)) * (t_end - t_start) / n
                p = at(times, first, last)
                max_step = max(
                    max_step, np.max(np.abs(p[:, 0] - previous)), np.max(np.abs(np.diff(p, axis=1)), initial=0)
                )
                previous = p[:, -1]
                low, high = np.minimum(low, p.min(axis=1)), np.maximum(high, p.max(axis=1))
            paths[:, first:last, 1] = low
            paths[:, first:last, 3] = high
        if max_step <= 1:
            return n, paths
        n *= 2
