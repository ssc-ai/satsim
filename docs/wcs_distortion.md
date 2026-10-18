# Pixel polynomial optical distortion

The simulator maps ideal detector coordinates to distorted coordinates directly.
It does not load optical calibration files or invert SIP at render time.
`fpa.distortion` contains the complete reusable model; sky pointing and time
remain in the scene geometry. See [the JSON example](usage.md#pixel-space-optical-distortion).

## Coordinate contract

- `(x,y)` means `(column,row)`, in full detector native pixels before flips/crop.
- Integer coordinates are pixel centers: the first is `(0,0)`, with edges at
  `-0.5` and `0.5`. The detector center is `((width-1)/2,(height-1)/2)`.
- Normalize about `reference_pixel` using positive `normalization_pixels`.
  Coefficient `i_j` is in pixel displacement units and multiplies `u^i v^j`.
- Add the evaluated `dx,dy` to the input `x,y`. Missing terms are zero.
  Orders 2–7 include constant, linear and all mixed terms up to total order.
  A constant term can move the optical axis; changing the expansion origin
  alone does not define a new scene pointing.
- Convert native to oversampled positions using `(x+0.5)*osf-0.5`, then add
  renderer padding. Reflect native positions using `size-1-coordinate` and
  subtract crop offsets only at output. Never rescale polynomial coefficients.
- Legacy celestial projections place boresight at `size_os/2`. The distortion
  adapter subtracts `0.5/osf` in native coordinates before polynomial evaluation,
  producing the canonical center for every oversampling factor. Omitted/null
  distortion retains the historical projection path. Consequently a zero
  polynomial and the legacy disabled path can differ by that half-subpixel;
  compare nonlinear on/off with the same polynomial adapter and affine terms.

`valid_domain.x/y` are ideal-coordinate bounds. Calibration must cover the
required detector, PSF halo and source motion. Outside-domain/nonfinite inputs
raise `DistortionError`. Initialization checks a 33×33 Jacobian grid; every mapped
point also checks determinant sign and nonsingularity. These sampled checks do
not constitute a mathematical proof of global invertibility. Consistent negative
orientation is permitted to represent affine parity.

`image_plane` defaults to `detector`, preserving synthetic image-plane positions;
set `ideal` to transform those too. Celestial sources always use distortion.
Stars and targets are mapped at time samples before deposition, and annotations
use the same positions. Full-frame FFT and native ePSF are supported, including
flips and crops. Piecewise FFT, sprites and `sim.mode=none` are rejected. PSF
warping, pixel solid-angle/radiance corrections and resized detector calibration
are outside this model.

## Calibration and migration

The simulation API is `DISTORTION_API = OPTICAL_GEOMETRY_API = 3`.
Old `distortion.fits`, `distortion.sip` and `fpa.linear_wcs` inputs are rejected
with conversion guidance. Do not rename SIP coefficients into `dx/dy`: SIP
usually maps detector to ideal, the opposite direction.

`satsim-calibrate` imports supported TAN/TAN-SIP solves, handles image reflection
and sky-frame conversion, and fits the ideal-to-detector polynomial. Nominal
scale/roll stay in scene settings; constant and linear polynomial terms retain
the fitted reference and residual affine geometry such as skew. Default fitting
selects orders 2–7 to meet a separately sampled 0.001-pixel tolerance, reporting
maximum/RMS residuals and its domain. SIP inversion uses residual/Jacobian checks
and ignores potentially unreliable AP/BP hints. Unqualified fits fail explicitly.
Calibration files remain upstream analysis inputs, not simulation dependencies.

## Sky-coordinate output

`distortion_wcs.py` is an output-only adapter. It fits the inverse pixel mapping,
then converts it algebraically to a TPV FITS header for the frame's pointing and
exposure-start motion. The fitted inverse's linear part is put in CD, including
skew/parity, to keep TPV inversion well conditioned. Flips and crops are composed
once. It reopens the serialized header and checks sky-to-detector predictions on
an independent grid/random sample and detector-to-sky across the detector.
Both must meet 0.001 native pixel. Unsupported maps/motion raise an explicit
error instead of writing an unchecked header. Polynomial simulation itself has
no inverse fit or sky-coordinate dependency.

`WCSERR` records the maximum checked output error in pixels. Distortion metadata
includes parameters/hash, convention, inverse-fit order and error. Output fitting
error is additional to upstream calibration fitting error, and sampled bounds
are not global guarantees. Headers describe exposure start (`DATE-OBS`/`MJD-OBS`
in UTC; `WCSMJD` in TT), not integrated flux centroids. Downstream readers must
support TPV. Native catalog frames and celestial truth are preserved separately.

## Validation

`test_inline_distortion.py` checks polynomial formulas, validation, domain,
integer centers, boresight, normalization and oversampling. `test_distortion_projection.py`
checks projection/render/centroid/FITS agreement for quadratic and cubic models,
all flips, crop, RA wrap, multiple oversampling factors and FFT/ePSF.
`test_distortion.py` covers sampled motion, source entry, annotations, flux,
TensorFlow graph execution, catalog bounds and export failures.

Use `benchmarks/distortion/benchmark.py` for the direct-mapping measurement. See the
[benchmark index](../benchmarks/README.md) for recorded results and their scope.
