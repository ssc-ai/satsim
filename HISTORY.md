History
=======

0.27.2
---------------------

* Accelerate eager GPU deposition of moving native ePSF stars with a reusable TensorFlow graph and one in-flight stamp batch. Preserve temporal samples, interpolation, flux tiers and position callbacks; CPU, existing graph, single-sample and target rendering retain their previous paths. New graph shapes/settings have a first-use tracing cost, and GPU summation can still differ at rounding level. (#65)
* **BEHAVIORAL CHANGE:** Preserve the original padded grid size in real inverse FFT convolution. Fix column shifts and ringing in odd-width FFT renders and ePSF lookup-table generation. Previously cached ePSF LUTs are automatically rebuilt to remove the baked-in bias; affected seeded images can change. (#64)
* **BEHAVIORAL CHANGE:** Render native ePSF `streak` stars like `transform`: each star deposits its optical ePSF at every exposure-bin midpoint. This fixes clipped one-sided trails, field-rotation shape and phase carry. Stars whose ePSF wings reach the frame are now kept, so dense fields deposit slightly more stars. `epsf.crop` tier counts count the deposited stars, and frame metadata no longer includes `trail_det`. (#63)
* Document the accepted cost of correct per-source GPU streak sampling. Historical PR #63 qualification on an RTX 3090 measured a 6× dense moving renderer cost and a 7.4% increase in the complete-collect median versus the previous shared-trail approximation, with overlapping collect timing ranges. Equivalent `transform` rendering showed no unexplained slowdown. These measurements precede subsequent FFT fixes; see the [benchmark report](https://github.com/ssc-ai/satsim/blob/3b0ceabb089c4c6467b40f32a739a661ca0f2a57/benchmarks/epsf_gpu/report.md) for scope and limits. Further optimization is proposed in #67 for #65.
* Accelerate eager CPU ePSF star deposition with a compiled float64 kernel. It runs only when every render tensor, including `position_transform` output, is on the CPU; GPU, mixed-device and graph renders keep TensorFlow. Pixels can differ from TensorFlow at rounding level. (#63)
* Skip unused PSF generation for native ePSF `streak` frames with a warm LUT cache, and keep FFT temporal sampling and per-source star transforms when `fallback_to_fft_for_models` applies. (#63)
* **API CHANGE:** Deprecate the trailed ePSF path. `render_epsf` `phase_oversample` and `epsf_normalize`, and `resolve_epsf_tiers` `trail_det`, are ignored with a `DeprecationWarning`. `filter_epsf_sources_in_bounds` without `kernel_size` warns and applies the previous center-only rule. `build_trailed_epsf_lut`, `transform_and_add_trailed_epsf` and `add_epsf_counts_tiered` raise `NotImplementedError` naming their replacement. (#63)
* **BEHAVIORAL CHANGE:** Correct single-precision CPU FFT convolution on TensorFlow before 2.15 using double-precision internal arithmetic while preserving output dtype, padding and pixel registration. This prevents photometric error and spurious below-bias pixels in noiseless images. GPU and newer CPU FFT arithmetic retain their native precision. Rebuild older ePSF LUT caches on every device because tables generated on an affected CPU can later be reused elsewhere; affected seeded CPU images can change. Legacy CPU transforms require more FFT workspace and can take longer. Honor explicit `float64` real-FFT requests instead of silently reducing them to `float32`. (#68)

0.27.1
---------------------

* Support NumPy 2 and Astropy 6+ by replacing the removed `np.float_` alias with `np.float64`, removing the Astropy upper-version cap, and adding `asdf-astropy` for ASDF transform loading. (#55)
* Fix scalar great-circle propagation with NumPy 2.4+ by explicitly extracting the single longitude and latitude values returned by `pygc`. (#57)
* Accelerate star annotation generation by converting masked TensorFlow columns to NumPy once before iterating over stars, preserving annotation values, types, ordering and optional fields. (#62)
* **BEHAVIORAL CHANGE:** Standardize FITS `DATE-OBS` as the start of each simulated exposure. Previously, tracked output without distortion wrote the exposure midpoint, while distortion-WCS output wrote exposure start; untracked output could use file-generation time. Add `DATE-BEG`, arithmetic-midpoint `DATE-AVG`, `DATE-END`, and `TIMESYS='UTC'`; synchronize time-of-day, shutter, exposure-duration and UTC MJD fields, and serialize dates to six fractional digits without timezone suffixes. Astrometric midpoint quantities and the validated WCS reference epoch (`WCSMJD`, TT) are preserved. Consumers that treated affected legacy `DATE-OBS` as a midpoint should use `DATE-AVG` for new output or derive midpoint from exposure start and duration; do not add half an exposure to `DATE-AVG` again. Historical files are not modified; do not apply a blanket half-exposure correction because older output paths differ. Record the code revision in dataset provenance. (#58)
* Preserve zero-duration FITS writes at fractional-second epochs by canonicalizing validated start, midpoint and end timestamps to the same instant; apply the documented timing tolerance consistently. (#58)

0.27.0
---------------------

* **BEHAVIORAL CHANGE:** Preserve procedural optical texture inside saturated cloud coverage envelopes, floored haze and full-coverage layers by default. Existing cloud configurations remain valid, but seeded outputs and mean attenuation can change; reproduce historical datasets with their recorded code version.
* Add per-layer `transmission.min`, `transmission.max` and optional `transmission.gamma` controls. Interior transmission is mapped without clipping; edges blend toward clear sky at 1. Retain the same texture through moving crops and use the resulting transmission consistently for attenuation, glow and truth.
* Add scene-independent pixel polynomial distortion (orders 2–7), with integer pixel centers, sampled star/target motion, consistent annotations and validated exposure-start TPV output.
* Store distortion and affine corrections directly in portable JSON. FITS/WCS calibration and polynomial fitting stay upstream in satsim-calibrate; simulation distortion is independent of scene pointing.
* Accelerate CPU ePSF stamp deposition with segmented accumulation. Floating-point summation order changes can produce small pixel differences; GPU accumulation remains subject to rounding-level variation between repeated calls. (#51)
* Reduce SSTR7 star-zone cache memory by storing read-only NumPy arrays, vectorize catalog decoding, and add `satsim.geometry.sstr7.clear_star_cache()` to release cached zones. Catalog query values and ordering are preserved. (#51)
* **API CHANGE:** Remove `satsim.geometry.sstr7.load_stars_for_zone`; replace calls to its `cache_clear()` method with `clear_star_cache()`. `query_by_min_max` now returns fresh dictionaries with plain-float magnitudes; missing magnitudes are `32.0`, and all-missing `query_by_los` magnitude arrays are now float64.
* Include Markdown documentation and JSON schemas in the source distribution. Bound native thread pools by default in test/coverage Make targets while respecting explicit overrides, run coverage through the selected `python3` interpreter, and correctly hide GPUs in `make cpucoverage`.

0.26.2
---------------------

* Add reproducible POPPY atmospheric phase screens through `fpa.psf.turbulent_atmosphere.seed`, using distinct seeds for each zone.
* Bound target trajectory allocations with `sim.max_track_samples` (default 1,000,000) and chunk physical brightness and Earth-shadow evaluations with `sim.photometry_chunk_size` (default 256). Curve allocation checks use a conservative bound before rasterization. (#43)
* **BEHAVIORAL CHANGE:** Reject failed apparent target line-of-sight calculations instead of substituting geometric coordinates. Skip targets that fail propagation, brightness, shadow, or analytical line-of-sight evaluation without ending the collect, and omit their pixels, annotations, and cached orbits. (#43)
* Accelerate ephemeris interpolation by sharing Lagrange basis weights across coordinates while retaining the existing interpolation windows and segment boundaries. (#47)
* Add opt-in adaptive quadratic brightness sampling through `sim.brightness_sampling`, with interval limits, quarter-point accuracy checks, and exact fallbacks. Exact per-sample brightness remains the default, and Earth shadow remains evaluated at every original sample. (#48)


0.26.1
---------------------

* Accept state-vector and ephemeris sites


0.26.0
---------------------

* Add passive RF analytical TDOA/FDOA simulation using the SatSim analytical layout, multiple named `geometry.site` entries, RADAR-style per-receiver fields of view and timing, direct target RF properties, deterministic receiver-pair noise, uncertainty modeling, schemas, and standard per-frame output.
* Add optional `time.gap` scheduling to analytical RADAR and passive RF while preserving zero-gap frame timing by default.


0.25.2
---------------------

* Apply cloud transmission to point sources before PSF convolution.


0.25.1
---------------------

* Add cloud transmission truth planes, standalone cloud transmission TIFF output, and sampled cloud transmission annotations.
* Fix coordinate registration between oversampled source positions, rendered flux centroids, SatNet annotations, and cloud-transmission samples using one zero-based pixel-center convention.
* Fix target trajectories truncating sub-pixel coordinates. Targets now render at their exact positions relative to the star field; previously they were displaced by up to one oversampled pixel.


0.25.0
---------------------

* **BEHAVIORAL CHANGE:** Add sub-pixel bilinear point rendering for stars and targets, enabled by default. Rendered images differ slightly from v0.24.0; set `sim.point_rendering` to `floor` for legacy rendering.
* Add detector-space ePSF lookup-table renderer. Set `sim.mode` to `epsf` and configure `sim.epsf.*`.
* Add native ePSF trailed star rendering for `star_render_mode: streak` to speed dense rate-track star fields.
* Add tiered ePSF crop rendering for dim stars, defaulting to auto thresholds.
* Add ePSF transform renderer performance options: budget-scaled scatter batches, `point_rendering: phase_nearest`, and detector-pixel auto temporal sampling with bin-centered exposure samples.
* Add block-sum downsampling and normalization options for radial image models and PSF generation.


0.24.0
---------------------

* Add sky background models for artificial skyglow, moonlight, twilight, and daytime. Configure `background.*`.
* Add modular cloud layers with presets, optical depth, glow, source brightening, and wind motion. Configure `clouds`.
* Add deformable radial image models. Use `deformable_radial_poly2d` and `deformable_radial_falloff2d`.
* Add apparent RA/Dec FITS WCS metadata with GAPPT headers.


0.23.1
---------------------

* Change radar range rate to instantaneous measurement. See `radar.monostatic.range_rate`.


0.23.0
---------------------

* Add FITS compression support. Set `sim.fits_compression`.
* Add seed support for deterministic sampling in random functions. Use `seed` on `$sample`.
* Add simplex gradient noise and lognormal_mu_sigma random. Use `$sample: random.simplex` or `random.lognormal_mu_sigma`, with optional `center`.
* Add GCVS5 star variability catalog support. Configure `geometry.stars.gcvs5.*`.
* Add RA/Dec tracking support. Set `geometry.site.track.mode` to `radec`.
* Add compound operator normalization/evaluation and schema updates. Use `$compound` with `$operator`.
* Enhance radial_cos2d with falloff parameters. Set `falloff_height`, `falloff_width`, or `falloff_xy`.
* Implement PSF `none` mode to bypass convolution. Set `fpa.psf.mode: none`.
* Refactor FITS header generation to incorporate focal plane roll and fix CD/CRPIX.
* Add support for additional noise term in FPA and evaluation tests. Configure `fpa.noise.misc`.


0.22.2
---------------------

* Rename tracking pointing toggle from `light_transit_correction` to `track_apparent` in config and APIs.
* Add graceful fallback to geometric LOS when apparent LOS computation fails.


0.22.1
---------------------

* Add light transit correction toggle for pointing. Set `light_transit_correction` in `track`. Default is `true`.


0.22.0
---------------------

* Add analytical RADAR observations.
* Batched SGP4 propagation with FOV pre-filtering to speed large catalogs.
* Add profiling for image generation pipeline and object timing logs. Controlled by `enable_profiler`.


0.21.2
---------------------

* Fix stellar aberration being applied to target incorrectly.


0.21.1
---------------------

* Add field of view pre-filter to cull objects for performance. Controlled by `enable_fov_filter` and optional `fov_filter_radius` (degrees).


0.21.0
---------------------

* Add space-based observer.


0.20.4
---------------------

* Update analytical observation for specified observation frames. `analytical_obs_frame` can be `barycentric`, `geocentric`, or `observer`.


0.20.3
---------------------

* Fix star stellar aberration.
* Add simulation configuration options for deflection, light transit, and stellar aberration.


0.20.2
---------------------

* Refactor keras_tensor import logic to support multiple TensorFlow/Keras versions.
* Refactor great circle into separate module.
* Reorganize CLI tests to run first to prevent TensorFlow configuration errors.


0.20.1
---------------------

* Refactor two-body propagation to remove `poliastro` dependency.
* Add object identification fields to analytical observations.


0.20.0
---------------------

* Add analytical observations with noise modeling capability.
* Fix electronic noise casting issue in add_read_noise function.


0.19.4
---------------------

* Fix observer altitude units.
* Fix apparent position of stars to include stellar aberration.


0.19.3
---------------------

* Add object ID to FITS and annotations.


0.19.2
---------------------

* Add `rate-sidereal` track mode. This mode takes n - 1 rate track frames and the last frame in sidereal tracking mode.


0.19.1
---------------------

* Add `TRKMODE` to fits header.


0.19.0
---------------------

* Add lambertian sphere brightness model for objects. Set object key `model` `mode` to `lambertian_sphere` and set `albedo` and `size` (meters) parameters. For 2D simulations, `distance` (km) and `phase_angle` (degrees) parameters must also be specified.


0.18.0
---------------------

* Add turbulent atmosphere to POPPY PSF generation. Set with psf option `turbulent_atmosphere` and sim option `psf_sample_frequency`.
* Add star and object segmentation annotation. Set sim option `save_segmentation` to `true` and `star_annotation_threshold` to limit stars annotated.


0.17.1
---------------------

* Copy useful TensorFlow Addon (TFA) functions into `tfa` as the TFA project is no longer maintained.
* Copy tests.


0.17.0
---------------------

* Add JSON schema definitions. See `schema/v1/Document.json` for root schema file.
* Add observation object type.
* Add support for cropped sensors.
* Add altitude to site object. Set site `alt` parameter in km. Default is 0.


0.16.0
---------------------

* Add option to annotate stars. Set sim option `star_annotation_threshold` to the minimum star brightness magnitude or `false` to disable. Disabled by default.
* Add option to show annotated stars in annotated images. Set sim option `show_star_boxes` to `true` to enable.


0.15.1
---------------------

* Remove clipping of negative values in ground truth files by default.
* Fix missing dependencies for ground truth file generation.


0.15.0
---------------------

* Add support to save ground truth image data to the Annotations directory. Set sim option `save_ground_truth` to `true`.
* Add support for running on CPU with no GPU acceleration.
* Add CZML options for sensor visualization and object billboard image.


0.14.0
---------------------

* Add vector math library.
* Add CZML output for sensor visualization.
* Fix objects not updating properly when image renderer is off.


0.13.1
---------------------

* Add argument to set folder name in `gen_multi`.
* Add environment variable, `SATSIM_SKYFIELD_LOAD_DIR`, to specify location of Skyfield ephemeris files.
* Fix incorrect CZML output when image renderer is off.


0.13.0
---------------------

* Add ephemeris objects that are propagated with the Lagrange interpolator.
* Add Cesium CZML output. Set sim option `save_czml` to `false` to disable.
* Add CSV text file star catalog loader. This feature is useful for small catalogs such as Hipparcos and simulating wide FOV sensors.
* Add multiplier and clipping for radial cosine.
* Add option to skip image rendering. Set sim option `mode` to `none` to bypass image rendering.
* Update interfaces for newest version of Skyfield, Poliastro, POPPY, and AstroPy.
* Fix star renderer issue removing stars in field of view for non-square arrays.


0.12.0
---------------------

* Add augmentation of SatNet `tf.data.Dataset`. This feature allows injecting synthetic targets into real data during training.
* Add FFT convolution to `add_patch` sprite render and `scatter_shift` image augmenter for speed improvement.
* Add cache last PSF FFT to `fftconv2p` for speed improvement for static PSFs.
* Add two-body state vector as a trackable target.
* Add moon and sun model and misc methods to calculate phase angle and target brightness.


0.11.0
---------------------

* Add support to render star motion with FFT. Set sim option `star_render_mode` to `fft`.
* Add option to sample photon noise multiple times. Set sim option `num_shot_noise_samples` to integer number.
* Add support to render a satellite as a sprite. Set `model` option in obs.
* Add support to load and augment sprite model with `$pipeline` operator.
* Add cropped POPPY PSF generation.
* Fix GreatCircle propagator tracking offset.
* Fix runtime exception when site and track_mode are not defined.
* Add TensorFlow 2.6 and update TensorFlow 2.2 and 2.4 Docker build file.


0.10.0
---------------------

* Add support for piecewise rendering. Set sim option `render_size` to enable. For example, [256, 256].
* Add `fixed` tracking mode with mount azimuth and elevation.
* Add great circle propagator for targets.
* Add in-memory image generation. See generator function `image_generator`.
* Fix missing stars when FOV crosses zero degree RA.
* Add curved targets using bezier curve raster. Enabled by default. Set sim option `num_target_samples` to 2 to enable linear raster.
* Add LRU cache to star catalog reader.
* Add option to turn off SNR calculation. Set sim option `calculate_snr` to false will render targets and stars together.
* Handle unstable SGP4 TLEs.
* Add TensorFlow 2.4 Docker build file.
* Add debug output for pristine images of targets and stars.


0.9.1
---------------------

* Calculate POPPY input wavefront resolution to avoid PSF aliasing.
* Add support for additional FITS image data types (`int16`, `uint16`, `int32`, `uint32`, `float32`).
* Add batch processing to `transform_and_add_counts` to support batch processing of stars.
* Add `auto` option to calculate temporal oversample factor based on star velocities.
* Add option to turn off serializing config data to pickle file (`save_pickle`).
* Add option to turn off png movie output (`save_movie`).
* Add `crop_and_resize` and `flip` image augmentation.
* Set pixels with values beyond the pixel data type's capacity to the maximum value for that data type.
* Add `lognormal` function to generate a distribution with a true target mean.
* Fix issue with sidereal track.
* Fix issue with fragment velocity not being randomly sampled.


0.9.0
---------------------

* Add Physical Optics Propagation in Python (POPPY) PSF generation.
* Add PSF augmentation with `$pipeline` replacement key.
* Add `$function` and `$compound` replacement key.
* Add ability to generate stray light from a `$function` replacement key.
* Add built-in 2D polynomial image generator for stray light, `polygrid2d`.
* Add built-in cosine fourth image generator for irradiance falloff, `radial_cos2d`.
* Add built-in sine wave image generator for fix pattern noise, `sin2d`.
* Add built-in image generator from AstroPy model, `astropy_model2d`.
* Add built-in image augmentation, `scatter_shift` and `scatter_shift_polar`.
* Add `$cache` replacement key (caching works for PSF and `$function`).


0.8.3
---------------------

* Fix new Skyfield incompatibility.


0.8.2
---------------------

* Prefix replacement keys with `$` in SatSim configuration file.
* Add option to scale collision fragments by cosine of the exit angle.


0.8.1
---------------------

* Add astrometric metadata into FITS header
* Refactor WCS library
* Add option to flip images about x or y axis
* Add option to refresh stars for each frame
* Add RPO from TLE generator


0.8.0
---------------------

* Add two body propagator
* Add object `create`, `delete`, and `update` events
* Add collision generator
* Add breakup generator
* Add `ref` keyword to configuration
* Add `key` keyword to `import` configuration
* Refactor astrometric library


0.7.2
---------------------

* Add option to specify star and obs velocity in polar coordinates


0.7.1
---------------------

* Add option to turn off shot noise: `sim.enable_shot_noise: true`
* Add option to turn off annotation boxes in image: `sim.show_obs_boxes: true`
* Add option to specify velocity in arcseconds: `sim.velocity_units: arcsec`
* Fix PNG output threading issue


0.7.0
---------------------

* Add function pipelines to support variable target brightness


0.6.1
---------------------

* Fix built-in generators not included in distribution
* Add dockerfile


0.6.0
---------------------

* Add configuration import.
* Add configuration generator functions.
* Add built-in generator for breakups.
* Add built-in generator for CSOs.
* Add built-in generator for loading TLE files.


0.5.0
---------------------

* Runtime optimization.
* Add parallel processing and multi-gpu utilization.
* Add option to limit gpu memory usage.


0.4.0
---------------------

* Add signal to noise calculation for target pixels.


0.3.0
---------------------

* Add support for two line element set SGP4 satellite propagator.
* Add support for rate and sidereal track from topocentric site.


0.2.0
---------------------

* Add support for SSTR7 star catalog.


0.1.1
---------------------

* Add target position to annotation file.
* Updates to run GitLab CI.


0.1.0
---------------------

* First release.
