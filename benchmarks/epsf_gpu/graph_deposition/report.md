# GPU ePSF graph deposition (#65)

The corrected GPU renderer is **2.55× faster** for the dense moving renderer probe. The complete three-frame collect median falls **3.088 → 2.210 seconds**, or **28.4% less time**. Renderer allocator peak falls **680 → 515 MiB**; collect peak falls **635 → 537 MiB**. Accuracy and output checks pass.

## Implementation and reference

Baseline: `24c3bb9bfc96de81abcc4d713f0d188d13f6a438`, including the odd-width FFT correction. Candidate engine: `68af42405957918b0aed6c4f331efd576a72d97e`. The measured engine file hashes in [results.json](results.json) match that commit. Observations were collected before committing, so raw worker HEAD metadata still names the baseline commit for the dirty candidate tree; source hashes identify the measured implementation. The benchmark runner's subsequent change adds stricter analysis only.

Eager GPU stars with multiple temporal samples deposit through a reusable `tf.function`, with one in-flight stamp batch. All sample coordinates, interpolation weights, flux tiers and callback outputs are inputs to the same deposition calculation. No samples are removed and no shared-trail approximation is used. CPU accumulation, targets, single-sample scenes and existing graph execution retain their previous paths. Shape relaxation reuses the graph as tensor dimensions change. Public Python/configuration interfaces and dependencies are unchanged.

## Measurements

RTX 3090, driver 617.14; Python 3.10, TensorFlow 2.13.1, CUDA 11.8/cuDNN 8, NumPy 1.26.4. One GPU is used serially by two persistent workers, with five alternating warm measurements per cell. GPU completion is fenced; collect timing includes generation, SNR/annotations and completed FITS/JSON writes. Every GPU worker rejects the NumPy CPU accumulator. These are synthetic affine scenes: 2394 × 1595, OSF 5, optical kernel 31, 520/3,600 stars, 38/54 midpoint samples and unchanged automatic crop tiers.

Seconds: median [minimum–maximum]. Peak memory is the TensorFlow active allocator high-water mark, including retained inputs; it is not total process VRAM.

| Scope | Scene | Corrected baseline | Candidate | Speedup | Peak MiB, baseline / candidate |
| --- | --- | --- | --- | --- | --- |
| stage | sparse static | 0.189 [0.184–0.208] | 0.183 [0.181–0.206] | 1.03× | 152 / 152 |
| stage | sparse moving | 0.468 [0.428–0.498] | 0.372 [0.362–0.449] | 1.26× | 516 / 359 |
| stage | dense static | 0.274 [0.262–0.281] | 0.267 [0.262–0.278] | 1.03× | 193 / 193 |
| stage | dense moving | 3.019 [2.906–3.091] | 1.186 [1.172–1.194] | 2.55× | 680 / 515 |
| collect | sparse static | 1.302 [1.146–1.337] | 1.227 [1.156–1.345] | 1.06× | 320 / 320 |
| collect | sparse moving | 1.322 [1.298–1.364] | 1.190 [1.172–1.255] | 1.11× | 349 / 319 |
| collect | dense static | 2.209 [2.191–2.295] | 2.224 [2.212–2.259] | 0.99× | 320 / 335 |
| collect | dense moving | 3.088 [3.015–3.101] | 2.210 [2.124–2.379] | 1.40× | 635 / 537 |

Static calculations use the existing path; small timing differences have overlapping ranges and do not establish a general speedup or slowdown. Dense moving collect ranges do not overlap in this sample. Some static working-set observations include memory retained by earlier graph calls; memory is not claimed to improve in every cell.

## Accuracy and qualification

- `make coverage BROWSER=true`: **1465 passed, 1 skipped**, **95.54% statement coverage**. The skipped multi-GPU case passed separately with both 3090s visible.
- The 58 added cases cover all three interpolation modes, float32/float64, changing frame/source shapes and LUT values, empty fields, nonlinear and CPU callbacks, deterministic execution, graph/CPU/single-sample routing, both GPUs, and the legacy TensorFlow keyword fallback. Existing full-suite tests cover support parity, phase carries, off-frame wings, warm caches, modeled-target FFT fallback, distortion, annotations and photometry.
- **110 paired component images** checked. All **55 target pairs** are bitwise equal. Maximum normal relative image L2 is 1.1e-07, relative total-flux error 7.1e-09, centroid displacement 2.7e-06 detector pixel, and relative second-moment error 3.76e-09. Bright-source apertures pass the existing 1e-5 flux gate with its source-scaled absolute floor. Deterministic star and target arrays are bitwise equal across versions in the recorded scenes.
- **108 paired FITS frames** and **259,548 annotation objects** checked across all normal, deterministic, streak and profiled collect outputs. Annotations/SNR, metadata and FITS headers match exactly, with no exclusions. Across 412,390,440 exported pixel pairs, 115 differ, each by 1 ADU; deterministic FITS pixels are identical.
- Full local CPU suite: **1337 passed, 118 skipped, 11 failed**. The failure-name set exactly matches the prior [release-base qualification](https://github.com/ssc-ai/satsim/pull/66) in this TensorFlow 2.13.1 environment; no additional failing tests were observed. The base result was 1335 passed, 62 skipped, 11 failed. Failure IDs are recorded in the JSON; this is not a fully passing local CPU-suite claim.
- Lint, documentation build, wheel/source builds, `twine check`, and an installed-wheel GPU render/CLI smoke check pass. All 106 packaged Python files match the qualified engine checkout.

## Costs and limits

New shapes/settings can incur graph tracing. First sparse moving renderer observation: **0.647 → 1.426 seconds**. A separate fresh-worker three-frame dense moving streak collect, including graph tracing and first LUT-cache use, took **3.765 → 3.817 seconds**. Imports and GPU/LUT setup precede those timers. These are single observations, baseline first, without clearing OS/driver caches; they do not establish cold-start distributions.

Separate CUPTI host callback captures show unchanged CPU/GPU transfer totals: renderer D2H/H2D **1407.411 / 806.325 MiB** on both versions; collect **680.507 / 435.662 MiB** on both versions. The exported GPU kernel timeline remains unavailable on this TF/WSL setup. The measurements establish a benefit from compiled deposition and bounded batch scheduling, not removal of transfers or per-kernel attribution.

Other GPU architectures and TensorFlow GPU runtimes, long-running workloads with many distinct graph configurations, and concurrent production workers are outside this qualification. The compatibility keyword fallback is exercised in tests; an old TensorFlow runtime was not installed for this run. Sampling/optical-support convergence and downstream detector/astrometric-solver validation remain separate work. CI status is reported on the PR.

## Reproduce

Create separate baseline and candidate checkouts at the engine commits above. Run the commands from this report's checkout, which contains the updated analyzer; each worker imports the engine tree specified on the command line. Replace `/path/to/...` and the CUPTI library path with local paths. Keep profiling separate from timing.

```bash
python3 benchmarks/epsf_gpu/benchmark.py --baseline /path/to/base --candidate /path/to/candidate --output /tmp/epsf-graph/normal --gpu 0 --rounds 5
python3 benchmarks/epsf_gpu/benchmark.py --baseline /path/to/base --candidate /path/to/candidate --output /tmp/epsf-graph/extras --gpu 0 --rounds 5 --extra-only --streak-collect --profile --cupti-libdir /path/to/cupti/lib
python3 benchmarks/epsf_gpu/benchmark.py --baseline /path/to/base --candidate /path/to/candidate --output /tmp/epsf-graph/deterministic --gpu 0 --rounds 0 --deterministic
CUDA_VISIBLE_DEVICES="" OPENBLAS_NUM_THREADS=1 python3 benchmarks/epsf_gpu/benchmark.py --analyze --corrected-baseline --output /tmp/epsf-graph/normal --extra-output /tmp/epsf-graph/extras --deterministic-output /tmp/epsf-graph/deterministic
```

`--corrected-baseline` applies exact metadata and sampled-streak fidelity gates to both corrected engines. Omitting it retains the historical PR #63 analysis for comparisons with the different legacy streak approximation. Historical recorded results remain in the parent directory. [results.json](results.json) consolidates source hashes, timings, memory, accuracy, transfer records and qualification counts; raw arrays, generated catalogs and profiler files remain under the local artifact directory.
