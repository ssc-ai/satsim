"""Compare observable interpolation, photometry, and memory use with a git baseline.

Run from the repository root with consistent BLAS thread settings, for example:
    OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 benchmarks/observable_performance.py \
        --baseline-ref 9fc1390 --output /tmp/observable-performance.json

Each photometry variant runs in a fresh process so ru_maxrss values are
comparable. Timings are medians, not assertions. No crowded-collect speedup is
inferred from these microbenchmarks.
"""
import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time as clock
from pathlib import Path

import numpy as np
import scipy
import skyfield

from satsim import time
from satsim.geometry.astrometric import create_topocentric
from satsim.geometry.photometric import model_to_mv
from satsim.geometry.sgp4 import create_sgp4
from satsim.image.fpa import mv_to_pe
from satsim.math.interpolate import lagrange
from satsim.pipeline import _avg_t, _delta_t
from satsim.satsim import _apply_photometry

START = [2026, 4, 7, 22, 0, 0.]
TLE = (
    '1 00005U 58002B   26097.92152118  .00000210  00000-0  25961-3 0  9992',
    '2 00005  34.2431 171.1570 1836179 181.6781 177.6859 10.85968047435646',
)
LEO = (
    '1 41788U 16059F   21260.56440839  .00022290  00000-0  23487-2 0  9990',
    '2 41788  98.2495   9.3816 0022044 246.7613 113.2488 14.89172957267561',
)
MODEL = {'mode': 'lambertian_sphere', 'albedo': .2, 'size': 1.}
ADAPTIVE = {'mode': 'adaptive', 'max_interval_seconds': 1., 'tolerance_mag': 1e-4}


def photometry_worker(variant, samples, repeats):
    observer = create_topocentric(20.7, -156.3, 3.)
    target = create_sgp4(*TLE)

    def evaluate(k):
        edges = k * 10 + np.linspace(0., 1., samples + 1)
        if variant == 'baseline':
            magnitudes = model_to_mv(observer, target, MODEL, time.utc_from_list(START, _avg_t(edges)))
            return mv_to_pe(20., magnitudes) * _delta_t(edges)
        return _apply_photometry(
            np.zeros(samples), edges, observer, target, MODEL, START, 20.,
            sampling=ADAPTIVE if variant == 'adaptive' else None)

    # Warm up without creating the large vector whose peak memory is measured.
    model_to_mv(observer, target, MODEL, time.utc_from_list(START, np.linspace(0., 1., 3)))
    durations = []
    for k in range(repeats):
        start = clock.perf_counter()
        counts = evaluate(k)
        durations.append(clock.perf_counter() - start)
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        peak_rss /= 1024
    return {'median_seconds': float(np.median(durations)), 'peak_rss_kib': peak_rss,
            'counts': counts.tolist()}


def interpolation_benchmark(baseline_ref, repeats):
    source = subprocess.check_output(
        ['git', 'show', baseline_ref + ':satsim/math/interpolate.py'], text=True)
    legacy = {}
    exec(compile(source, baseline_ref + ':interpolate.py', 'exec'), legacy)
    rng = np.random.default_rng(0)
    x = np.arange(-60., 7290., 30.)
    omega = 2 * np.pi / 5800
    positions = np.stack([7000 * np.cos(omega * x), 7000 * np.sin(omega * x), .1 * x], axis=1)
    queries = np.concatenate([rng.uniform(x[0], x[-1], 2000), x[::7]])
    outputs = {}
    results = {}
    for label, function in [('baseline', legacy['lagrange']), ('current', lagrange)]:
        durations = []
        for _ in range(repeats):
            start = clock.perf_counter()
            outputs[label] = np.array([function(x, positions, q, order=3) for q in queries])
            durations.append(clock.perf_counter() - start)
        results[label + '_seconds'] = float(np.median(durations))
    results['queries'] = len(queries)
    results['speedup'] = results['baseline_seconds'] / results['current_seconds']
    results['max_position_difference_km'] = float(np.max(np.abs(outputs['baseline'] - outputs['current'])))
    return results


def accuracy_benchmark():
    observer = create_topocentric(20.7, -156.3, 3.)
    results = []
    for label, tle, epoch in [
        ('eccentric', TLE, START),
        ('close_leo', LEO, [2021, 9, 17, 23, 55, 30.]),
    ]:
        target = create_sgp4(*tle)
        for duration in [1., 10., 60., 120.]:
            edges = np.linspace(-duration / 2, duration / 2, 1026)
            exact_mv = model_to_mv(observer, target, MODEL, time.utc_from_list(epoch, _avg_t(edges)))
            exact_flux = mv_to_pe(20., exact_mv) * _delta_t(edges)
            actual = _apply_photometry(np.zeros(1025), edges, observer, target, MODEL, epoch, 20.,
                                       sampling=ADAPTIVE)
            actual_mv = 20 - 2.5 * np.log10(actual / _delta_t(edges))
            results.append({'orbit': label, 'exposure_seconds': duration,
                            'max_magnitude_error': float(np.max(np.abs(actual_mv - exact_mv))),
                            'relative_total_flux_error': float(abs(actual.sum() - exact_flux.sum()) / exact_flux.sum())})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='9fc1390')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=['baseline', 'exact', 'adaptive'])
    parser.add_argument('--samples', type=int, default=8192)
    args = parser.parse_args()
    if args.repeats < 1 or args.samples < 1:
        parser.error('repeats and samples must be positive')
    if args.worker:
        print(json.dumps(photometry_worker(args.worker, args.samples, args.repeats)))
        return
    results = {
        'baseline_ref': args.baseline_ref,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'skyfield': skyfield.__version__,
                        'platform': platform.platform(),
                        'OPENBLAS_NUM_THREADS': os.environ.get('OPENBLAS_NUM_THREADS')},
        'repeats': args.repeats,
        'interpolation': interpolation_benchmark(args.baseline_ref, args.repeats),
        'photometry': {},
    }
    for samples in [55, 8192]:
        measurements = {}
        for variant in ['baseline', 'exact', 'adaptive']:
            raw = subprocess.check_output([
                sys.executable, str(Path(__file__).resolve()), '--worker', variant,
                '--samples', str(samples), '--repeats', str(args.repeats)], text=True)
            measurements[variant] = json.loads(raw)
        reference = np.array(measurements['baseline']['counts'])
        for variant in measurements:
            counts = np.array(measurements[variant].pop('counts'))
            measurements[variant]['max_relative_flux_error'] = float(np.max(np.abs(counts - reference) / reference))
            measurements[variant]['speedup'] = measurements['baseline']['median_seconds'] / measurements[variant]['median_seconds']
        results['photometry'][str(samples)] = measurements
    results['adaptive_accuracy'] = accuracy_benchmark()
    text = json.dumps(results, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')


if __name__ == '__main__':
    main()
