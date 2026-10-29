# PR #63: RTX 3090 ePSF qualification

The equivalent native `transform` calculation shows no unexplained GPU slowdown in these fixtures. Dense static collects are **1.59× faster**; moving collect medians are close to the baseline. All numerical and output acceptance checks pass.

The old shared `streak` approximation is a different calculation. Replacing it with correct per-source sampling costs **6.00×** in the dense moving renderer probe, and **7.4%** in the complete-collect median (overlapping ranges). That tradeoff needs an explicit maintainer decision before merging/closing #61. This report does not waive that decision.

## Inputs and protocol

- Baseline: `8d32165042647d8e598ea6e9c12013700267698e` (`r0.27.1`, includes #62 annotation optimization).
- Candidate engine: `7996c8ee1ec1c48f178b75de627eff37f6822075` (PR #63). Engine source SHA-256 values are recorded in [results.json](results.json); the report commit changes benchmark files only.
- One NVIDIA RTX 3090, 24 GiB; driver 617.14; TensorFlow 2.13.1, CUDA 11.8, cuDNN 8, Python 3.10, NumPy 1.26.4. Linux container using the WSL NVIDIA driver. The second 3090 was not used for measured runs.
- Two persistent workers use the same GPU serially. Five alternating baseline/candidate runs per cell, reversing order each round. Native library thread counts are one. Timed runs use normal GPU accumulation settings. Separate validation workers enable deterministic operations.
- Detector: 2394 columns × 1595 rows; spatial oversampling 5; 31-pixel optical kernel; bilinear phase deposition; automatic crop tiers 5/9/17. Sparse/dense fields have 520/3,600 catalog stars; midpoint samples are 38/54. Static motion is exactly zero. Moving fields combine 0.0002 rad/s rotation with [0.7, −1.1] detector pixels/s translation.
- Renderer probes use a Gaussian PSF with sigma 1.25 detector pixels. Complete collects use the engine's Gaussian `eod: 0.5` PSF and contain three 3-second frames, 0.1-second gaps, one stationary target, magnitude-22 sky, seeded read noise, and no Poisson noise. Their optical LUTs are verified identical between versions. These are controlled synthetic affine pixel scenes, not a replay of sensorkit-sim production jobs.
- A collect measurement includes generation, SNR/annotation generation and completed FITS/JSON writes. A renderer measurement fences GPU completion before stopping its timer. Array export for analysis and profiler runs are outside the timed measurements. All proposed GPU renders fail immediately if they enter the CPU accumulator.
- Each case has a first-use observation followed by five warm observations. A fresh optical cache is used for each worker/case; later collect runs use its warm cache. Startup observations are single samples, baseline first; OS/driver caches were not cleared, so import timings are not speedup estimates. GPU setup includes CUDA/LUT initialization. No Numba stamp kernel is called or compiled in GPU rendering.
- A pilot used a magnitude-0 sky that saturated saved images. It was rejected and is excluded from all recorded results here. The published fixture asserts useful output dynamic range; corrected observed ranges include 15–12,105 ADU.

## Equivalent-calculation timings

Wall seconds: median [minimum–maximum]. Peak memory is TensorFlow's active GPU allocator high-water mark, reset per invocation; it is not total driver-reserved VRAM. The persistent workers retain their inputs, so absolute peaks include those allocations.

| Scope | Scene | Baseline seconds | PR seconds | Baseline / PR | GPU peak MiB, baseline / PR |
| --- | --- | --- | --- | --- | --- |
| stage | sparse static | 0.461 [0.408–0.481] | 0.189 [0.183–0.199] | 2.44× | 514 / 150 |
| stage | sparse moving | 0.431 [0.414–0.502] | 0.423 [0.415–0.542] | 1.02× | 514 / 514 |
| stage | dense static | 2.860 [2.804–3.075] | 0.231 [0.228–0.239] | 12.37× | 688 / 197 |
| stage | dense moving | 2.925 [2.780–3.116] | 2.868 [2.751–3.103] | 1.02× | 703 / 703 |
| collect | sparse static | 1.276 [1.253–1.386] | 1.279 [1.129–1.352] | 1.00× | 331 / 331 |
| collect | sparse moving | 1.301 [1.243–1.345] | 1.329 [1.246–1.377] | 0.98× | 353 / 330 |
| collect | dense static | 3.306 [2.891–3.396] | 2.085 [1.898–2.145] | 1.59× | 603 / 331 |
| collect | dense moving | 3.083 [3.040–3.208] | 3.003 [2.948–3.294] | 1.03× | 599 / 598 |

Static speedups come from collapsing repeated zero-motion samples. The moving comparisons use the same midpoint samples and optical operator on both versions. CPU-backend dispatch adds no measurable penalty: the dense moving PR renderer's auto/forced-TensorFlow medians are 2.925 / 2.983 seconds, respectively; these observations do not establish an auto-dispatch speedup.

## First use and startup

Single observations, seconds; import/setup and rendering are measured separately. Cold process/driver state is not statistically characterized.

| Worker | Imports | GPU/LUT setup |
| --- | --- | --- |
| baseline | 2.969 | 0.519 |
| candidate | 3.028 | 0.517 |

| First invocation | Baseline seconds | PR seconds |
| --- | --- | --- |
| stage sparse static | 0.703 | 0.237 |
| stage sparse moving | 0.416 | 0.549 |
| stage dense static | 2.940 | 0.223 |
| stage dense moving | 3.040 | 2.815 |
| collect sparse static | 1.462 | 1.587 |
| collect sparse moving | 1.581 | 1.891 |
| collect dense static | 3.758 | 2.569 |
| collect dense moving | 3.326 | 3.623 |

## Image and output verification

- Equivalent calculations pass relative image L2 and total-flux gates of 1e-5, centroid displacement ≤1e-4 detector pixel, relative second moments ≤1e-4, and bright-source aperture checks at radii 1.5/3/6 with the documented source-scaled absolute floor. Optical LUT bytes match for every saved pair.
- Maximum ordinary-execution relative image L2: **1.4843e-6**; relative total-flux error: **3.7388e-8**; centroid difference: **1.2978e-5 detector pixel**; relative second-moment error: **9.7572e-8**. The largest aperture error uses 18.55% of its tolerance budget.
- 64 ordinary and 32 deterministic star/target component images were compared. **All 48 paired isolated target arrays are bitwise equal**, including the deterministic validation.
- **84 paired FITS frames** (72 ordinary, 12 deterministic) and **173,124 annotation objects** were checked. Annotation objects, including their seeded SNR values, are exactly equal. Frame metadata matches after excluding the intentionally removed `epsf.crop.trail_det`; headers match after excluding software version `SWCREATE`.
- Across 320,748,120 compared FITS pixels, 666 pixels differ, all by **1 ADU**. All floating images are finite and exported frames have the expected shape and nonconstant dynamic range. Those ADC rounding differences are not a bitwise-image claim.
- The engine commit also passed `make coverage BROWSER=true`: 1,316 tests, 95.53% statement coverage. That is separate evidence from this benchmark.

## Cost of correcting legacy GPU streaks

| Dense moving comparison | Baseline shared streak | Correct sampled PR streak | PR / baseline |
| --- | --- | --- | --- |
| Renderer | 0.488 s [0.468–0.502] | 2.931 s [2.883–3.045] | 6.00× |
| Three-frame collect | 2.892 s [2.831–3.031] | 3.105 s [2.897–3.439] | 1.074× |

The older path deposits a precomputed shared trail once per source; the corrected renderer deposits each source at every midpoint with its own rotation. Their star images differ by 11.8% relative L2 in the renderer probe and about 25.5% in the collect frames. Those images are not equivalent references. The PR's sampled result agrees with the equivalent `transform` reference within the acceptance gates above. The 7.4% collect median increase has overlapping observed ranges; no precise universal slowdown percentage is claimed.

A follow-up GPU optimization may be useful. Keeping the faster legacy approximation would retain the fidelity defects this PR fixes. Choose explicitly whether to accept the measured correctness cost now or optimize before merge.

## GPU transfer audit

The container initially lacked CUPTI. NVIDIA CUPTI 11.8.87 was extracted into a temporary directory and loaded only for separate profiler workers through `LD_LIBRARY_PATH`; timed runs did not use it. CUDA callback copy records were collected for a dense moving renderer and collect. No PR GPU render entered the Numba accumulator.

| Scope | Device-to-host MiB, baseline / PR | Host-to-device MiB, baseline / PR |
| --- | --- | --- |
| stage | 1407.425 / 1407.411 | 806.325 / 806.325 |
| collect | 680.538 / 680.507 | 435.663 / 435.662 |

The existing TensorFlow path already performs substantial bulk transfers on this environment. The PR does not increase total transfer bytes in these matched captures. One additional 96,100-byte (optical-LUT-sized) device-to-host copy appears in the candidate collect trace, alongside fewer source-mask transfers; this is not a new image/stamp NumPy accumulator path. Treat attribution of that individual copy as an inference from its size, not a proven call-site attribution. Detailed size/count histograms are recorded in the JSON.

The exported GPU device timeline contains no device events on this TensorFlow/WSL combination, despite CUPTI collecting callback/activity records. Accordingly, this report uses synchronized wall timing and host CUDA callback copy accounting; it makes no per-kernel GPU-duration claim. Existing bulk-transfer optimization is a separate opportunity, not a newly introduced dispatch regression.

## Reproduce

Create two checkouts at the engine commits above. Run this harness from the report checkout with the same Python environment; each worker imports its requested engine tree. Replace the example paths below. Generated catalogs, images, arrays, traces and logs remain in the artifact directory rather than the repository.

```bash
python benchmarks/epsf_gpu/benchmark.py --baseline /tmp/base --candidate /tmp/pr --output /tmp/epsf-gpu/normal --gpu 0 --rounds 5
python benchmarks/epsf_gpu/benchmark.py --baseline /tmp/base --candidate /tmp/pr --output /tmp/epsf-gpu/extras --gpu 0 --rounds 5 --extra-only --streak-collect --profile --cupti-libdir /path/to/CUPTI/lib64
python benchmarks/epsf_gpu/benchmark.py --baseline /tmp/base --candidate /tmp/pr --output /tmp/epsf-gpu/deterministic --gpu 0 --rounds 0 --deterministic
OPENBLAS_NUM_THREADS=1 CUDA_VISIBLE_DEVICES="" python benchmarks/epsf_gpu/benchmark.py --analyze --output /tmp/epsf-gpu/normal --extra-output /tmp/epsf-gpu/extras --deterministic-output /tmp/epsf-gpu/deterministic
```

The consolidated JSON includes raw observations, source hashes, validation maxima, and transfer histograms. `$BASELINE`, `$CANDIDATE`, and `$ARTIFACTS` replace machine-specific paths in the recorded copy. Local raw artifacts for this run are under `internal/pr63_gpu_qualification`; the rejected pilot is not included in the published JSON.

Scientific sampling/optical-support convergence, downstream detection/astrometric-solver behavior, other GPU architectures/TensorFlow versions and concurrent production workers remain outside this controlled qualification. They are not established by these timing and sampled-operator checks.
