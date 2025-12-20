History
=======

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
