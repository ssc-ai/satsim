"""Validated FITS output for a direct pixel polynomial; no calibration loading.

FITS describes detector->sky whereas rendering uses ideal->detector. Fit the
inverse only for serialization, and qualify both serialized mapping directions.
"""

import math
import numpy as np
from astropy.wcs import WCS, NoConvergence

from satsim.geometry.distortion import DistortionError, polynomial_design, polynomial_powers
from satsim.geometry.wcs import get_wcs

EXPORT_TOLERANCE_PX = 1e-3


def _grid(bounds, n):
    x, y = np.meshgrid(np.linspace(*bounds[0], n), np.linspace(*bounds[1], n))
    return np.column_stack([x.ravel(), y.ravel()])


def _tpv(coefficients, powers, center, scale, matrix, affine, offset, flip, ra, dec, crop):
    plane = matrix @ np.linalg.inv(affine)
    # Put the inverse model's local linear geometry (including parity/skew)
    # in CD. Leaving a reflection in PV makes wcslib's inverse iteration
    # unreliable even for an otherwise exactly affine model.
    linear = np.column_stack(
        [coefficients[powers.index((1, 0))] / scale[0], coefficients[powers.index((0, 1))] / scale[1]]
    )
    cd = plane @ linear @ flip
    # The fitted polynomials give ideal absolute pixels as functions of
    # normalized pre-reflection detector positions.
    values = coefficients.copy()
    values[0] -= center + offset
    values = values @ plane.T
    transform = np.diag(1 / scale) @ flip @ np.linalg.inv(cd)
    expanded = {}
    for (i, j), vector in zip(powers, values):
        for k in range(i + 1):
            for m in range(j + 1):
                power = (k + m, i + j - k - m)
                factor = (
                    math.comb(i, k)
                    * math.comb(j, m)
                    * transform[0, 0] ** k
                    * transform[0, 1] ** (i - k)
                    * transform[1, 0] ** m
                    * transform[1, 1] ** (j - m)
                )
                expanded[power] = expanded.get(power, np.zeros(2)) + factor * vector
    indices = {(0, 0): 0, (1, 0): 1, (0, 1): 2}
    index = 4
    for degree in range(2, 8):
        for j in range(degree + 1):
            indices[(degree - j, j)] = index
            index += 1
        if degree % 2:
            index += 1
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TPV", "DEC--TPV"]
    w.wcs.crval = [ra, dec]
    w.wcs.crpix = center + 1
    w.wcs.cd = cd
    pv = []
    for (i, j), vector in expanded.items():
        pv.extend([(1, indices[(i, j)], vector[0]), (2, indices[(j, i)], vector[1])])
    w.wcs.set_pv(pv)
    if crop:
        y, x = crop["height_offset"], crop["width_offset"]
        w = w.slice((slice(y, y + crop["height"]), slice(x, x + crop["width"])))
    return w


def export_wcs(model, ra, dec, y_ifov, x_ifov, roll, crop, motion_time, rotation, translation, osf):
    reference = get_wcs(model.height, model.width, y_ifov, x_ifov, ra, dec, roll)
    matrix = reference.pixel_scale_matrix
    angle = rotation * motion_time
    rotate = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    flip = np.diag([-1 if model.fliplr else 1, -1 if model.flipud else 1])
    affine = flip @ rotate @ flip
    offset = flip @ ((rotate - np.eye(2)) @ np.full(2, 0.5 / osf) + np.asarray(translation)[::-1] * motion_time)
    center, scale = model.center, np.array([model.width, model.height]) / 2
    train = _grid(model.domain, 33)
    detector = np.column_stack(model.map_xy(*train.T))
    rng = np.random.default_rng(1701)
    independent = np.vstack([_grid(model.domain, 42), rng.uniform(model.domain[:, 0], model.domain[:, 1], (1024, 2))])
    expected = np.column_stack(model.map_xy(*independent.T))
    crop_offset = np.array([crop["width_offset"], crop["height_offset"]]) if crop else np.zeros(2)
    output = (expected - center) @ flip + center - crop_offset
    sky = reference.all_pix2world((independent - center - offset) @ np.linalg.inv(affine).T, 0)
    physical = _grid(np.array([[-0.5, model.width - 0.5], [-0.5, model.height - 0.5]]), 33)
    output_physical = (physical - center) @ flip + center - crop_offset
    for order in range(2, 8):
        powers = polynomial_powers(order)
        design = polynomial_design(detector, center, scale, powers)
        coefficients, _, rank, _ = np.linalg.lstsq(design, train, rcond=None)
        if rank != len(powers):
            continue
        check = polynomial_design(expected, center, scale, powers) @ coefficients
        if np.max(np.linalg.norm(check - independent, axis=1)) > EXPORT_TOLERANCE_PX:
            continue
        w = _tpv(coefficients, powers, center, scale, matrix, affine, offset, flip, ra, dec, crop)
        try:
            serialized = WCS(w.to_header(relax=True))
            recovered = serialized.all_world2pix(sky, 0)
            ideal_check = reference.all_world2pix(serialized.all_pix2world(output_physical, 0), 0)
            ideal_check = ideal_check @ affine.T + center + offset
            detector_check = np.column_stack(model.map_xy(*ideal_check.T))
            errors = [
                np.max(np.linalg.norm(recovered - output, axis=1)),
                np.max(np.linalg.norm(detector_check - physical, axis=1)),
            ]
            if np.isfinite(errors).all() and max(errors) <= EXPORT_TOLERANCE_PX:
                model.metadata["wcs_export"] = {
                    "convention": "TPV",
                    "inverse_fit_order": order,
                    "maximum_checked_error_px": float(max(errors)),
                    "tolerance_px": EXPORT_TOLERANCE_PX,
                }
                return w
        except (ValueError, RuntimeError, NoConvergence):
            continue
    raise DistortionError(
        "Cannot export polynomial distortion within 0.001 pixel in both WCS directions; "
        "the model/domain/affine motion has no qualified TPV representation."
    )
