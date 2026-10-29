# Benchmarks

| Study | Runner | Recorded report | Raw results |
| --- | --- | --- | --- |
| Observable performance | [observable_performance.py](observable_performance.py) | [Report](observable_performance_report.txt) | [JSON](observable_performance_results.json) |
| Direct pixel-polynomial distortion | [distortion/benchmark.py](distortion/benchmark.py) | [Report](distortion/report.txt) | [JSON](distortion/results.json), [test results](distortion/test_results.json) |
| Cloud transmission variation | [clouds/compare.py](clouds/compare.py) | [Report and visuals](clouds/README.md) | [JSON](clouds/results.json) |
| ePSF GPU performance and fidelity | [epsf_gpu/benchmark.py](epsf_gpu/benchmark.py) | [Report](epsf_gpu/report.md) | [JSON](epsf_gpu/results.json) |

## Run the current distortion benchmark

From the repository root:

```bash
PYTHONPATH=. python3 benchmarks/distortion/benchmark.py --output /tmp/polynomial-results.json
```

This measures quadratic and cubic CPU mapping, including domain/Jacobian checks
and formula-error checks. It excludes rendering and FITS export, so its timings
are not directly comparable to the historical SIP collect benchmarks.

The recorded report/results describe the current cleanup and qualification pass.
Source hashes, full-suite results and calibration provenance are recorded in
`distortion/test_results.json`. The current eight-sensor measurements and charts
are in `../satsim-calibrate/docs/calibration_report.md`.
