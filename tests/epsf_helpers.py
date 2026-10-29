"""Shared ePSF test helpers: the float64 star reference, its acceptance gates and accumulator routing."""
import contextlib

import numpy as np
import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import config as tf_config

import satsim.image.epsf as epsf


def quadratic_warp(rows, cols):
    return rows + 0.0009 * (cols - 60) ** 2, cols - 0.0007 * (rows - 60) ** 2


def float64_midpoint_bilinear_reference(lut, rows, cols, flux, shape=(32, 32), pad=4, start=0, end=3,
                                        samples=37, velocity=(0.41, -0.73), rotation=0, warp=False,
                                        crop=None, offset=(0, 0)):
    """Deposit ``samples`` exposure-bin midpoints per star in float64.

    Rows and columns are padded-frame oversampled positions. Rotation turns
    about the padded-frame center, like the production frame transform; the
    optional ``crop`` applies one manual flux tier by total star count.
    """
    rows, cols, flux = [np.asarray(v, np.float32).astype(np.float64) for v in (rows, cols, flux)]
    lut = np.asarray(lut, np.float64)
    osf = lut.shape[0]
    image = np.zeros((shape[0] + 2 * pad, shape[1] + 2 * pad), np.float64)
    center = (np.array(image.shape) * osf - 1) / 2
    t0, t1 = float(np.float32(start)), float(np.float32(end))
    times = t0 + (np.arange(samples) + 0.5) * (t1 - t0) / samples
    for row, col, count in zip(rows, cols, flux):
        active_lut = lut
        if crop and count <= crop['threshold_pe']:
            margin = (lut.shape[2] - crop['kernel_size']) // 2
            active_lut = lut[:, :, margin:margin + crop['kernel_size'], margin:margin + crop['kernel_size']]
        for time in times:
            angle = float(np.float32(rotation)) * time
            sine, cosine = np.sin(angle), np.cos(angle)
            rr = sine * (col - center[1]) + cosine * (row - center[0]) + center[0] + velocity[0] * time
            cc = cosine * (col - center[1]) - sine * (row - center[0]) + center[1] + velocity[1] * time
            if warp:
                rr, cc = quadratic_warp(rr, cc)
            rr, cc = rr + offset[0], cc + offset[1]
            ir, ic = int(np.floor(rr)), int(np.floor(cc))
            fr, fc = rr - ir, cc - ic
            for dr, wr in [(0, 1 - fr), (1, fr)]:
                for dc, wc in [(0, 1 - fc), (1, fc)]:
                    site_r, site_c = ir + dr, ic + dc
                    stamp = active_lut[site_r % osf, site_c % osf]
                    top, left = site_r // osf - stamp.shape[0] // 2, site_c // osf - stamp.shape[1] // 2
                    r0, c0 = max(top, 0), max(left, 0)
                    r1, c1 = min(top + stamp.shape[0], image.shape[0]), min(left + stamp.shape[1], image.shape[1])
                    if r1 > r0 and c1 > c0:
                        image[r0:r1, c0:c1] += stamp[r0 - top:r1 - top, c0 - left:c1 - left] * (count / samples * wr * wc)
    return image[pad:pad + shape[0], pad:pad + shape[1]]


def _centroid_moments(image):
    image = np.asarray(image, np.float64)
    y, x = np.indices(image.shape, dtype=np.float64)
    flux = image.sum()
    center = np.array([(image * y).sum(), (image * x).sum()]) / flux
    dy, dx = y - center[0], x - center[1]
    moments = np.array([(image * dy ** 2).sum(), (image * dy * dx).sum(), (image * dx ** 2).sum()]) / flux
    return center, moments


def _metrics(actual, reference):
    actual, reference = np.asarray(actual, np.float64), np.asarray(reference, np.float64)
    actual_center, actual_moments = _centroid_moments(actual)
    reference_center, reference_moments = _centroid_moments(reference)
    return {
        'l2_relative': float(np.linalg.norm(actual - reference) / np.linalg.norm(reference)),
        'centroid_detector_pixels': float(np.linalg.norm(actual_center - reference_center)),
        'moment_relative': float(np.linalg.norm(actual_moments - reference_moments) / np.linalg.norm(reference_moments)),
    }


def assert_equivalent(actual, reference, source_flux):
    """Apply the acceptance gates: relative L2, total and aperture flux 1e-5;
    centroid 1e-4 detector px; relative second moment 1e-4.

    Flux tolerances have an absolute floor of ``source_flux * 1e-8`` so tiny
    clipped apertures do not demand more than float32 resolves.
    """
    metrics = _metrics(actual, reference)
    assert metrics['l2_relative'] <= 1e-5, metrics
    flux_floor = source_flux * 1e-8
    tolerance = max(1e-5 * abs(float(reference.sum())), flux_floor)
    assert abs(np.sum(actual, dtype=np.float64) - reference.sum()) <= tolerance, metrics
    center, _ = _centroid_moments(reference)
    y, x = np.indices(reference.shape)
    for radius in (1.5, 3.0, 6.0):
        aperture = (y - center[0]) ** 2 + (x - center[1]) ** 2 <= radius ** 2
        aperture_flux = reference[aperture].sum()
        tolerance = max(1e-5 * abs(float(aperture_flux)), flux_floor)
        assert abs(np.sum(actual[aperture], dtype=np.float64) - aperture_flux) <= tolerance, metrics
    assert metrics['centroid_detector_pixels'] <= 1e-4, metrics
    assert metrics['moment_relative'] <= 1e-4, metrics


@contextlib.contextmanager
def routed(accumulator):
    """Run under ``accumulator``'s placement and check that only it deposited stars."""
    runs = []
    accumulate = epsf._accumulate_stars_cpu

    def spy(fpa, *args):
        runs.append(fpa.device)
        return accumulate(fpa, *args)

    epsf._accumulate_stars_cpu = spy
    try:
        with tf.device('/CPU:0') if accumulator == 'cpu' else contextlib.nullcontext():
            yield
    finally:
        epsf._accumulate_stars_cpu = accumulate
    assert bool(runs) == (accumulator == 'cpu'), runs


@contextlib.contextmanager
def deterministic_ops(seed=42):
    """Run the block with TensorFlow op determinism and a global seed, then restore both.

    GPU scatter-adds sum in an unspecified order, so two renders of one scene
    match bitwise only under op determinism, which also rejects unseeded
    random ops.
    """
    was_enabled = tf_config.is_op_determinism_enabled()
    previous_seed = context.global_seed()
    tf_config.enable_op_determinism()
    tf.random.set_seed(seed)
    try:
        yield
    finally:
        tf.random.set_seed(previous_seed)
        if not was_enabled:
            tf_config.disable_op_determinism()


def assert_renders_match(render_pair, source_flux):
    """Check that ``render_pair()`` returns two renders of one scene that agree.

    They must be bitwise equal under ``deterministic_ops`` and pass
    ``assert_equivalent`` under default execution. Returns the default pair.
    """
    with deterministic_ops():
        first, second = render_pair()
    np.testing.assert_array_equal(first, second)
    first, second = render_pair()
    assert_equivalent(first, np.asarray(second, np.float64), source_flux)
    return first, second
