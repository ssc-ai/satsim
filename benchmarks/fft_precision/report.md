# Legacy CPU FFT accuracy and TensorFlow compatibility

The TensorFlow 2.13.1 CPU FFT error is fixed without relaxing existing test tolerances. The original noiseless dataset changes from **−150 DN to +106 DN**, and all **256** spurious below-bias pixels disappear. Signed inputs remain supported. Default GPU and modern CPU outputs match the baseline bit for bit in the recorded probes.

## Implementation

TensorFlow 2.14 still uses Eigen CPU FFTs; TensorFlow 2.15 switches to DUCC ([2.14 implementation](https://raw.githubusercontent.com/tensorflow/tensorflow/v2.14.0/tensorflow/core/kernels/fft_ops.cc), [2.15 FFT implementation](https://raw.githubusercontent.com/tensorflow/tensorflow/v2.15.0/tensorflow/core/kernels/fft_impl.cc)). Accordingly, the workaround applies before **2.15**, rather than before 2.19.

- `fftconv2` and `fftconv2p` use double-precision arithmetic for legacy CPU single-precision requests, then cast to the requested real output dtype. Input quantization, transform dimensions, padding and pixel registration are preserved.
- Traced functions select CPU/GPU precision at execution time, including CPU fallback when no GPU exists. One traced function can run on CPU, GPU and CPU again without retracing. Default GPU and TensorFlow 2.15+ CPU requests retain their native single-precision FFTs.
- Explicit real `float64` requests now use float64 instead of being silently reduced to float32. Cached spectra distinguish input precision, calculation precision, placed device and graph ownership.
- ePSF optical-table cache version increases from 2 to 3. Version 1 and 2 tables rebuild on every device: an inaccurate table generated on an older CPU can otherwise be loaded on a GPU or newer runtime.

This does not clamp negative convolution values or round away errors. Seventy-nine additional test combinations check SciPy float64 references, odd/even shapes, flux, pixel registration, cached/uncached calculations, signed inputs, output dtypes, gradients, reused graphs, soft placement and obsolete optical tables. The noiseless dataset additionally requires every pixel to remain at or above bias. Existing tolerance values are unchanged.

## Original dataset regression

TensorFlow 2.13.1, NumPy 1.24.3, CPU; gain 1.5 pe/DN and bias 1,000 DN. Measurements use the existing `test_dataset_augment` fixture.

| Measurement | Before | After |
| --- | ---: | ---: |
| Pre-ADC signal sum (pe) | 184.676324 | 184.722572 |
| Minimum pre-ADC signal (pe) | -0.000445153331 | -9.98300292e-13 |
| Pixels below digital bias | 256 | 0 |
| Positive digital counts above bias (DN) | 106 | 106 |
| Background deficit (DN) | -256 | 0 |
| Net digital signal above bias (DN) | -150 | 106 |

The unchanged assertion expects **108 ± 2 DN**. The remaining negative floating-point residuals are approximately 1e−12 pe and do not move any background pixels across the ADC floor boundary.

## Accuracy and execution cost

Same 1,000-count impulse and Gaussian input arrays before/after, with image sizes 192², 520² and 2136², padded by one pixel per side. The reference flux is approximately 999.999995612 counts because the normalized Gaussian is stored as float32.

| Input grid | CPU 2.13 flux before | CPU 2.13 flux after | Maximum image error before | Maximum image error after |
| --- | ---: | ---: | ---: | ---: |
| 192² | 1000.015024409 | 999.999995531 | 0.000117835 | 2.30968e-07 |
| 520² | 999.951923281 | 999.999995530 | 0.000364725 | 2.30968e-07 |
| 2136² | 999.709587175 | 999.999995530 | 0.0022921 | 2.30968e-07 |

Warm convolution medians, including synchronization and result copy:

| CPU 2.13 input | Execution | Before (ms) | After (ms) | After / before |
| --- | --- | ---: | ---: | ---: |
| 192² | eager | 11.459 | 10.360 | 0.904 |
| 192² | graph | 15.501 | 13.803 | 0.890 |
| 520² | eager | 124.945 | 114.112 | 0.913 |
| 520² | graph | 192.027 | 174.081 | 0.907 |
| 2136² | eager | 2580.509 | 2647.809 | 1.026 |
| 2136² | graph | 4075.275 | 4009.647 | 0.984 |

The measured legacy CPU medians range from about 11% faster to 3% slower. This is a small convolution study, not a complete-collect benchmark. Double-precision real/complex FFT buffers use twice the bytes per element; peak process memory was not measured. Cost depends on shape and workload, and accurate seeded legacy CPU images can differ from historical output.

| Unaffected default path | Bitwise-identical probes | Timing ratio range, after / before |
| --- | ---: | ---: |
| TensorFlow 2.19 CPU | 6/6 | 0.942–1.135 |
| TensorFlow 2.20 CPU | 6/6 | 0.803–1.134 |
| TensorFlow 2.13 GPU | 6/6 | 0.738–1.164 |

Short-call timings vary between cases and do not establish complete-collect performance equivalence. Legacy GPU graphs now include runtime precision dispatch; the native FFT arithmetic and measured outputs are unchanged. Raw per-call times and output hashes are in [results.json](results.json).

## Qualification

Full local suites, Python 3.10.12; CPU pools bounded to one native thread. CPU-only runs disable unrelated machine-wide pytest plugin autoload consistently. Modern CPU runtimes use isolated `tensorflow-cpu` wheels; the GPU run uses the existing CUDA 11.8 TensorFlow installation on an RTX 3090.

| Runtime | Device | NumPy | Passed | Skipped | Failures |
| --- | --- | --- | ---: | ---: | ---: |
| 2.13.1 | CPU | 1.24.3 | 1390 | 97 | 0 |
| 2.19.0 | CPU | 1.26.4 | 1391 | 96 | 0 |
| 2.20.0 | CPU | 1.26.4 | 1391 | 96 | 0 |
| 2.13.1 | GPU, `make coverage` | 1.26.4 | 1485 | 2 | 0 |

Modern CPU skips are all GPU-only cases. The legacy CPU run also skips the test that requires unchanged native CPU float32 results, because that is the path being corrected. The GPU skips are that same legacy CPU native-result test and the CPU-only soft-placement test. GPU coverage is **95.51%** (11,464/12,003 statements).

The fix was also applied to PR #67's exact head (`39dbde8`) in a temporary checkout. Its GPU graph-deposition tests together with the FFT/cache/dataset regressions pass: **227 passed, 3 skipped** on GPU and **93 passed, 137 skipped** with GPUs hidden. Neither PR #67 nor the original checkout's release-preparation edits were modified.

The new [CPU CI matrix](../../.github/workflows/cpu-fft-compatibility.yml) installs the package with pinned TensorFlow/NumPy pairs and checks FFT accuracy, registration, photometry and ADC behavior. [Recorded clean-install run](https://github.com/ssc-ai/satsim/actions/runs/40958925698): 2.13.1 **127 passed / 81 skipped**; 2.19.0 and 2.20.0 each **128 passed / 80 skipped**. All jobs pass. The [full Docker test run](https://github.com/ssc-ai/satsim/actions/runs/40958925671) also passes. Newer GPU TensorFlow runtimes were not exercised locally in this pass.

## Reproduction

Use clean checkouts of the release base `24c3bb9` and the candidate. [results.json](results.json) records both production-file hashes and the implementation commit. The benchmark imports each tree in a separate sequential worker, warms three calls, then records five single-call samples on an otherwise idle host. The long legacy CPU suite was suspended during timing and resumed afterwards; suite duration is not a performance measurement.

```bash
# Run with each TensorFlow environment's interpreter; use --device GPU for CUDA.
python benchmarks/fft_precision/benchmark.py \
  --before /path/to/base --after /path/to/candidate --output /tmp/fft-results.json

# Full CPU qualification (each of 2.13.1 / 2.19.0 / 2.20.0).
CUDA_VISIBLE_DEVICES='' PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
python -m pytest --junitxml=/tmp/fft-full.xml

make coverage BROWSER=true
make lint
```

No minimum TensorFlow version or scene configuration changes are required.
