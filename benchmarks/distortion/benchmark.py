#!/usr/bin/env python3
"""Measure direct pixel-polynomial evaluation, separately from rendering/export.

PYTHONPATH=. python benchmarks/distortion/benchmark.py --output /tmp/polynomial.json
Historical SIP timings are not directly comparable to this microbenchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import tracemalloc

import numpy as np

from satsim.geometry.distortion import DetectorDistortion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=1000000)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    if args.samples <= 0 or args.repeats <= 0:
        parser.error("samples and repeats must be positive")
    points = np.random.default_rng(1701).uniform(-0.5, 2047.5, (args.samples, 2))
    results = {}
    for order in (2, 3):
        block = {
            "model": "polynomial",
            "order": order,
            "reference_pixel": [1023.5, 1023.5],
            "normalization_pixels": [1024.0, 1024.0],
            "dx": {"2_0": 1.2, "1_1": -0.3, **({"3_0": 0.05} if order == 3 else {})},
            "dy": {"0_2": -0.8, "1_1": 0.2, **({"0_3": -0.04} if order == 3 else {})},
            "valid_domain": {"x": [-32.5, 2079.5], "y": [-32.5, 2079.5]},
        }
        model = DetectorDistortion(block, 2048, 2048)
        actual = np.column_stack(model.map_xy(*points.T))
        expected = points.copy()
        u, v = ((points - 1023.5) / 1024).T
        expected[:, 0] += 1.2 * u * u - 0.3 * u * v + (0.05 * u * u * u if order == 3 else 0)
        expected[:, 1] += -0.8 * v * v + 0.2 * u * v + (-0.04 * v * v * v if order == 3 else 0)
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-10
        timings = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            model.map_xy(*points.T)
            timings.append(time.perf_counter() - start)
        tracemalloc.start()
        model.map_xy(*points.T)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[str(order)] = dict(
            config=block,
            seconds=timings,
            median_seconds=float(np.median(timings)),
            traced_peak_bytes=peak,
            max_formula_error_px=error,
        )
    source = Path(__file__).resolve().parents[2] / "satsim/geometry/distortion.py"
    report = dict(
        samples=args.samples,
        repeats=args.repeats,
        numpy=np.__version__,
        scope="CPU direct map including domain and Jacobian checks; excludes TF rendering and FITS output",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        orders=results,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {order: {k: v for k, v in item.items() if k != "config"} for order, item in results.items()}, indent=2
        )
    )


if __name__ == "__main__":
    main()
