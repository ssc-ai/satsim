=====
Usage
=====

Overview
--------

SatSim can be used as a command line tool or as a Python module. A run starts
from a JSON or YAML configuration, transforms any dynamic configuration entries,
and writes a timestamped output directory containing imagery, labels, metadata,
and optional diagnostics.

Command Line Interface
----------------------

Show the global help, version, or run-command help:

.. code-block:: bash

    $ satsim --help
    $ satsim version
    $ satsim run --help

Run an EO simulation on GPU 0:

.. code-block:: bash

    $ satsim --debug INFO run --device 0 --mode eager --output_dir output/ input/config.json

The CLI accepts ``.json`` and ``.yml`` files. The only supported TensorFlow
backend mode is currently ``eager``.

Parallel Runs
~~~~~~~~~~~~~

Use ``--jobs`` to run multiple SatSim processes per selected GPU. The total
process count is ``jobs * number_of_devices``.

.. code-block:: bash

    # Three processes on GPU 0.
    $ satsim --debug INFO run --jobs 3 --device 0 --mode eager --output_dir output/ input/config.json

    # One process on each of GPUs 0 and 1.
    $ satsim --debug INFO run --jobs 1 --device 0,1 --mode eager --output_dir output/ input/config.json

    # Four processes per GPU, each limited to 7000 MB.
    $ satsim --debug INFO run --jobs 4 --memory 7000 --device 0,1,2,3 --mode eager --output_dir output/ input/config.json

Additional processes can improve throughput by increasing GPU utilization, but
each process needs its own memory allocation. If the allocation is too small or
the card is overcommitted, TensorFlow will fail with memory-related errors.

Radar Dispatch
~~~~~~~~~~~~~~

If the root configuration contains a ``radar`` block, ``satsim run``
automatically dispatches to the analytical radar simulator instead of the EO
image renderer:

.. code-block:: bash

    $ satsim --debug INFO run --output_dir output/ input/radar_config.json

Python Module
-------------

Generate image sets from Python:

.. code-block:: python

    from satsim import gen_multi, load_json

    ssp = load_json("input/config.json")
    ssp["sim"]["samples"] = 50
    ssp["geometry"]["obs"]["list"][0]["mv"] = 17.5

    gen_multi(ssp, eager=True, output_dir="output/")

For a single already-transformed configuration, call ``gen_images``. For a
streaming integration, use ``image_generator``.

.. code-block:: python

    import copy
    from satsim import gen_images, image_generator, load_json
    from satsim.config import transform

    ssp = load_json("input/config.json")
    transformed = transform(copy.deepcopy(ssp), "input")

    run_dir = gen_images(transformed, output_dir="output/")

    for frame in image_generator(transformed, with_meta=False, num_sets=1):
        # frame is a TensorFlow tensor containing one rendered frame.
        pass

Outputs
-------

Each CLI or ``gen_multi`` run creates a timestamped directory under
``output_dir``. Common EO products are:

* ``ImageFiles/*.fits``: rendered sensor frames.
* ``Annotations/*.json``: SatNet-style object annotations and frame metadata.
* ``AnnotatedImages/*.jpg``: optional annotated previews when
  ``sim.save_jpeg`` is true.
* ``AnnotatedImages/movie.png``: optional APNG movie when ``sim.save_movie`` is
  true.
* ``satsim.czml``: optional Cesium visualization when ``sim.save_czml`` is
  true.
* ``config.json``: the transformed configuration used for the run.
* ``config_pass_*.json``: intermediate transformation stages for debugging and
  reproducibility.
* ``Debug/*.pickle``: optional intermediate arrays when the CLI
  ``--output_intermediate`` flag is set.
* ``Annotations/*.tiff``: optional ground-truth planes when
  ``sim.save_ground_truth`` is true.
* ``Annotations/*_star_segmentation.tiff``,
  ``Annotations/*_object_segmentation.tiff``, and
  ``Annotations/*_cloud_segmentation.tiff``: optional segmentation masks when
  ``sim.save_segmentation`` is true.
* ``AnalyticalObservations/*.json``: optional analytical EO observations when
  ``sim.analytical_obs`` is true.

Radar runs write per-frame JSON observations through the analytical observation
writer.

Configuration Basics
--------------------

SatSim configurations are dictionaries serialized as JSON or YAML. The root
document includes:

* ``version``: configuration version.
* ``sim``: simulation, output, and renderer controls.
* ``fpa``: EO focal-plane sensor controls. Required for EO runs.
* ``radar``: analytical radar sensor controls. Required for radar runs.
* ``background``: sky background controls.
* ``clouds``: optional cloud layer stack.
* ``geometry``: time, observer, stars, targets, and tracking.

The v1 JSON schema in ``schema/v1`` is the most detailed option inventory. The
runtime also fills several optional defaults, but practical configurations
should still specify the sensor, background, and geometry values that define the
scenario.

Minimal EO Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

This is a compact complete EO configuration with procedural stars and one
image-plane target:

.. code-block:: json

    {
      "version": 1,
      "sim": {
        "mode": "fftconv2p",
        "spatial_osf": 5,
        "temporal_osf": "auto",
        "padding": 50,
        "samples": 1,
        "save_jpeg": true,
        "save_movie": false,
        "save_czml": false
      },
      "fpa": {
        "height": 512,
        "width": 512,
        "y_fov": 0.308312,
        "x_fov": 0.308312,
        "dark_current": 0.3,
        "gain": 1.0,
        "bias": 0.0,
        "zeropoint": 20.6663,
        "a2d": {
          "response": "linear",
          "fwc": 100000,
          "gain": 1.5,
          "bias": 1500
        },
        "noise": {
          "read": 9.0,
          "electronic": 0.0
        },
        "psf": {
          "mode": "gaussian",
          "eod": 0.15
        },
        "time": {
          "exposure": 5.0,
          "gap": 2.5
        },
        "num_frames": 6
      },
      "background": {
        "stray": { "mode": "none" },
        "galactic": 22.0,
        "moon": { "mode": "none" },
        "twilight": { "mode": "none" },
        "daytime": { "mode": "none" }
      },
      "geometry": {
        "time": [2015, 4, 24, 9, 7, 44.128],
        "stars": {
          "mode": "bins",
          "mv": {
            "bins": [10, 11, 12, 13, 14, 15, 16, 17, 18],
            "density": [5.0028, 10.269, 24.328, 35.192, 60.017, 110.06, 180.28, 285.53]
          },
          "motion": {
            "mode": "affine",
            "rotation": 0,
            "translation": [0.4, 7.0]
          }
        },
        "obs": {
          "mode": "list",
          "list": [
            {
              "mode": "line",
              "origin": [0.5, 0.5],
              "velocity": [0.0, 0.0],
              "mv": 17.5
            }
          ]
        }
      }
    }

Important ``sim`` Options
~~~~~~~~~~~~~~~~~~~~~~~~~

* ``mode``: ``fftconv2p`` renders images; ``none`` disables image rendering.
  ``none`` is useful with ``analytical_obs``.
* ``render_size``: ``"full"`` renders the full frame at once. A two-element
  list renders tiles and can reduce peak GPU memory for large images.
* ``spatial_osf``: spatial oversampling factor. ``spacial_osf`` is also accepted.
* ``temporal_osf``: temporal samples used to smear motion. ``"auto"`` derives a
  value from star motion.
* ``padding``: extra real pixels around the image for off-frame sources and
  convolution.
* ``samples``: number of independent transformed configurations to generate.
* ``star_render_mode``: ``transform`` renders per-star motion; ``fft`` uses a
  shared FFT-convolved star streak shape.
* ``star_catalog_query_mode``: ``frame`` refreshes catalog stars every frame;
  ``at_start`` queries once.
* ``apply_star_wrap_around``: repeats stars as they drift through the padded
  field.
* ``num_target_samples``: ``2`` draws straight target tracks; ``3`` draws a
  quadratic curve through start, midpoint, and end positions.
* ``enable_shot_noise`` and ``num_shot_noise_samples``: control Poisson shot
  noise and optional noise averaging.
* ``fits_compression``: ``none``, ``gzip``, ``gzip2``, ``rice``, ``hcompress``,
  or ``plio``.
* ``save_ground_truth``: writes target, star, background, noise, gain, and bias
  planes.
* ``save_segmentation`` and ``star_annotation_threshold``: write object, star,
  and cloud segmentation masks and optional star annotations.
* ``psf_sample_frequency``: regenerate a PSF ``once``, per ``collect``, or per
  ``frame``.
* ``enable_deflection``, ``enable_light_transit``, and
  ``enable_stellar_aberration``: control apparent-position corrections.
* ``enable_earth_shadow``: masks target brightness while targets are in Earth
  umbra.
* ``enable_fov_filter`` and ``fov_filter_radius``: skip propagation for objects
  well outside the padded field.
* ``analytical_obs``: write analytical EO observations in addition to, or
  instead of, rendered frames.
* ``analytical_obs_frame``: analytical observation reference frame:
  ``barycentric``, ``geocentric``, or ``observer``.
* ``enable_profiler``: log timing for object processing and image-generation
  stages.

FPA and PSF Options
~~~~~~~~~~~~~~~~~~~

The ``fpa`` block defines the detector, optics, timing, and measurement error
model. Scalars may be replaced by generated arrays where the runtime accepts
arrays, for example gain, bias, dark current, and background fields.

``fpa.psf`` supports:

* ``{"mode": "none"}``: no PSF convolution.
* ``{"mode": "gaussian", "eod": 0.15}``: Gaussian PSF using ensquared energy
  on detector.
* ``{"mode": "poppy", ...}``: POPPY physical-optics PSF with an
  ``optical_system``, wavelengths, weights, and optional
  ``turbulent_atmosphere``.

``fpa.detection`` controls synthetic detection uncertainty for analytical EO
observations:

.. code-block:: json

    {
      "detection": {
        "snr_threshold": 2.0,
        "pixel_error": 0.25,
        "false_alarm_rate": 0.0,
        "max_false": 10
      }
    }

Dynamic Configuration
---------------------

SatSim transforms configurations before rendering. Each transformed stage is
saved as ``config_pass_*.json`` in the output directory.

Random Sampling
~~~~~~~~~~~~~~~

Most numeric values can be replaced with ``$sample``. NumPy random
distributions are addressed as ``random.<distribution>``. SatSim also supports
``random.choice`` and ``random.list``.

.. code-block:: json

    {
      "mv": {
        "$sample": "random.uniform",
        "low": 12.0,
        "high": 18.0,
        "seed": 1234
      }
    }

When a ``seed`` is present, that sample is deterministic without globally
forcing every other sample to repeat.

Generate a random number of targets:

.. code-block:: json

    {
      "mode": "list",
      "list": {
        "$sample": "random.list",
        "length": { "$sample": "random.randint", "low": 0, "high": 4 },
        "value": {
          "mode": "line-polar",
          "origin": [
            { "$sample": "random.uniform", "low": 0.05, "high": 0.95 },
            { "$sample": "random.uniform", "low": 0.05, "high": 0.95 }
          ],
          "velocity": [
            { "$sample": "random.uniform", "low": 0, "high": 360 },
            { "$sample": "random.uniform", "low": 0.0, "high": 0.35 }
          ],
          "mv": { "$sample": "random.uniform", "low": 9.0, "high": 20.0 }
        }
      }
    }

References and Imports
~~~~~~~~~~~~~~~~~~~~~~

Use ``$ref`` to copy another value from the same configuration:

.. code-block:: json

    {
      "fpa": {
        "height": 512,
        "width": { "$ref": "fpa.height" }
      }
    }

Use ``$import`` to load reusable partial configurations. Imports are resolved
relative to the input configuration directory. ``key`` selects a nested object,
and ``override`` merges in local replacements.

.. code-block:: json

    {
      "fpa": {
        "$import": "../common/raven_fpa.json",
        "override": {
          "num_frames": 10
        }
      }
    }

Python Functions, Compounds, and Pipelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``$function`` calls a Python function and replaces the object with its return
value. This is useful for generated gain, bias, dark-current, or background
fields:

.. code-block:: json

    {
      "gain": {
        "$function": "radial_cos2d",
        "module": "satsim.image.model",
        "kwargs": {
          "height": { "$ref": "fpa.height" },
          "width": { "$ref": "fpa.width" },
          "y_scale": 0.08,
          "x_scale": 0.08,
          "mult": 1.0
        }
      }
    }

``$compound`` combines generated arrays. The default operator is addition; use
``"$operator": "multiply"`` for products.

.. code-block:: json

    {
      "bias": {
        "$compound": [
          100.0,
          {
            "$function": "sin2d",
            "module": "satsim.image.model",
            "kwargs": {
              "height": { "$ref": "fpa.height" },
              "width": { "$ref": "fpa.width" },
              "amplitude": 3.0,
              "frequency": 8
            }
          }
        ]
      }
    }

``$pipeline`` is evaluated into a Python callable by the renderer. Use it for
time-varying fields such as target brightness:

.. code-block:: json

    {
      "mv": {
        "$pipeline": [
          {
            "module": "satsim.pipeline",
            "function": "constant",
            "kwargs": { "value": 16.0 }
          },
          {
            "module": "satsim.pipeline",
            "function": "glint",
            "kwargs": { "period": 2.0, "magnitude": 10.0 }
          }
        ]
      }
    }

Generators
~~~~~~~~~~

``$generator`` calls a Python function that returns a configuration-compatible
object. Built-in observation generators cover simple geometry, TLE files, RPO
events, collision events, and breakup events.

Generate a cone of image-plane targets:

.. code-block:: json

    {
      "obs": {
        "$generator": {
          "module": "satsim.generator.obs.geometry",
          "function": "cone",
          "kwargs": {
            "n": { "$sample": "random.randint", "low": 500, "high": 1000 },
            "t": 0,
            "direction": [
              { "$sample": "random.uniform", "low": 0, "high": 360 },
              { "$sample": "random.uniform", "low": 10, "high": 30 }
            ],
            "velocity": [
              { "$sample": "random.uniform", "low": 1.0, "high": 5.0 },
              { "$sample": "random.uniform", "low": 3.0, "high": 5.0 }
            ],
            "origin": [
              { "$sample": "random.uniform", "low": 0.2, "high": 0.8 },
              { "$sample": "random.uniform", "low": 0.2, "high": 0.8 }
            ],
            "mv": [15.0, 17.0]
          }
        }
      }
    }

Geometry and Targets
--------------------

SatSim supports two broad geometry modes:

* 2D image-plane geometry for high-throughput object detection data. Targets
  are specified by normalized image origin, pixel velocity, and brightness.
* 3D astrometric geometry for physically propagated stars, observers, and
  targets. Targets are projected through the observer, tracker, gimbal, and WCS.

Stars
~~~~~

``geometry.stars.mode`` supports:

* ``bins``: procedurally sample stars from magnitude-density bins.
* ``sstr7``: query an SSTR7 catalog from ``path``.
* ``csv``: query a CSV star catalog from ``path``.
* ``none``: render no stars.

``geometry.stars.motion.mode`` supports ``affine``, ``affine-polar``, and
``none``. For catalog modes with a site/track, use ``motion.mode: none`` to let
SatSim compute the apparent star motion from the tracking geometry.

SSTR7 stars can be augmented with GCVS5 variable-star light curves:

.. code-block:: json

    {
      "stars": {
        "mode": "sstr7",
        "path": "/workspace/share/sstrc7",
        "gcvs5": {
          "enabled": true,
          "path": "internal/gcvs5.txt",
          "match_radius_arcsec": 5.0,
          "template": "auto",
          "mag_system": "V"
        },
        "motion": { "mode": "none" }
      }
    }

Targets
~~~~~~~

``geometry.obs`` accepts either ``$generator`` or ``{"mode": "list"}``. A list
may contain these target modes:

* ``line``: image-plane target with ``origin`` and row/column pixel velocity.
* ``line-polar``: image-plane target with ``origin`` and ``[theta, speed]``.
* ``tle``: SGP4 propagation from TLE lines.
* ``twobody``: two-body propagation from ECI position, velocity, and epoch.
* ``ephemeris``: Lagrange interpolation through ECI position/velocity samples.
* ``gc``: great-circle motion in local az/el or equatorial ra/dec coordinates.
* ``observation``: angles-only observation object.
* ``none``: disabled placeholder.

Astrometric Site and Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This example uses a topocentric site, rate tracking from a TLE, SSTR7 stars,
and one TLE target:

.. code-block:: json

    {
      "geometry": {
        "time": [2015, 4, 24, 9, 7, 44.128],
        "site": {
          "mode": "topo",
          "lat": "20.746111 N",
          "lon": "156.431667 W",
          "alt": 0.3,
          "gimbal": {
            "mode": "wcs",
            "rotation": 7
          },
          "track": {
            "mode": "rate",
            "tle": [
              "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992",
              "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
            ]
          }
        },
        "stars": {
          "mode": "sstr7",
          "path": "/workspace/share/sstrc7",
          "motion": { "mode": "none" }
        },
        "obs": {
          "mode": "list",
          "list": [
            {
              "mode": "tle",
              "tle": [
                "1 36411U 10008A   15115.45075343  .00000069  00000-0  00000+0 0  9992",
                "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"
              ],
              "mv": 12.0
            }
          ]
        }
      }
    }

Track modes include ``rate``, ``sidereal``, ``fixed``, ``radec``, and
``rate-sidereal``. A topocentric observer uses ``lat``, ``lon``, and ``alt``.
A space-based observer can be specified with site-level ``tle`` or
``tle1``/``tle2``.

RPO, Collision, and Breakup Generators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Generate an RPO pair near a tracked TLE:

.. code-block:: json

    {
      "obs": {
        "$generator": {
          "module": "satsim.generator.obs.rpo",
          "function": "rpo_from_tle",
          "kwargs": {
            "tle": { "$ref": "geometry.site.track.tle" },
            "epoch": { "$ref": "geometry.time" },
            "delta_distance": 10.0,
            "delta_velocity": 0.01,
            "target_mv": 12.0,
            "rpo_mv": 15.0
          }
        }
      }
    }

Generate collision fragments:

.. code-block:: json

    {
      "obs": {
        "$generator": {
          "module": "satsim.generator.obs.breakup",
          "function": "collision_from_tle",
          "kwargs": {
            "n": [407, 275],
            "tle": { "$ref": "geometry.site.track.tle" },
            "collision_time": { "$ref": "geometry.time" },
            "collision_time_offset": 17.5,
            "K": 0.13,
            "attack_angle": "random",
            "attack_velocity_scale": 0.55,
            "radius": 21.78,
            "target_mv": 11.17,
            "rpo_mv": 13.14
          }
        }
      }
    }

Use ``breakup_from_tle`` in the same module for single-object breakup events.

Background Configuration
------------------------

The ``background`` object defines sky background before dark current, sensor
noise, and analog-to-digital conversion. Existing configurations that only set
``background.galactic`` keep the legacy behavior: ``galactic`` is interpreted as
the total scalar sky background in visual magnitude per arcsec^2.

For artificial skyglow, add ``background.skyglow``:

.. code-block:: json

    {
      "background": {
        "galactic": 22.0,
        "skyglow": 21.5,
        "stray": { "mode": "none" },
        "moon": { "mode": "none" },
        "twilight": { "mode": "none" },
        "daytime": { "mode": "none" }
      }
    }

When ``skyglow`` is present, ``galactic`` is the natural moonless baseline and
``skyglow`` is the total moonless clear-sky site brightness, including
artificial light. ``skyglow`` may be a scalar or an FPA-shaped 2D field in
visual magnitude per arcsec^2. Magnitudes are logarithmic: SatSim converts both
values to linear photoelectrons and uses ``linear(skyglow) -
linear(galactic)`` as the artificial skyglow component. Numerically larger
``skyglow`` values are darker than ``galactic`` and are rejected because they
would imply negative artificial light.

SatSim currently treats configured background magnitudes as visual-equivalent
inputs. Sensor quantum efficiency and full spectral response are not modeled in
this background config yet.

The ``moon`` block enables lunar scattered sky brightness. ``mode: none``
disables it; ``mode: default`` currently aliases ``krisciunas-schaefer``. The
Krisciunas-Schaefer mode is a V-band model that uses lunar phase, Moon-sky
separation, Moon zenith distance, target zenith distance, and atmospheric
extinction. It requires a ground ``geometry.site`` with latitude and longitude.

The ``twilight`` block enables below-horizon solar twilight. ``mode: none``
disables it; ``mode: default`` currently aliases ``patat``. Patat twilight is a
V-band zenith model driven by Sun zenith distance. SatSim treats it as a solar
residual over ``background.galactic`` so the natural baseline is not counted
twice. It also requires a ground ``geometry.site`` with latitude and longitude.
The current Patat implementation clamps at its bright end between civil
twilight and sunrise; daytime mode then transitions from that endpoint to
Hosek-Wilkie across the first few degrees above the horizon.

The ``daytime`` block enables above-horizon daylight. ``mode: none`` disables
it; ``mode: default`` currently aliases ``hosek-wilkie``. The Hosek-Wilkie
daytime path uses a CIE Y-channel clear-sky sky-dome model with internal
turbidity and ground-albedo defaults, converts the resulting V-like luminance
to photoelectrons, and stores the result as a solar residual over
``background.galactic``. SatSim blends from the Patat twilight endpoint to
Hosek-Wilkie just above the horizon to avoid an artificial sunrise/sunset
brightness jump. The older ``mode: perez`` path remains available explicitly for
comparison; it uses a CIE/Perez relative luminance distribution scaled by
SatSim's internal fixed daytime zenith luminance default. Daytime modes require
a ground ``geometry.site`` with latitude and longitude.

Cloud Configuration
-------------------

Clouds are configured with an optional top-level ``clouds`` list. If the key is
missing or the list is empty, no cloud attenuation is applied. Each list item is
one cloud layer. Layers are generated independently and combined as an
atmospheric stack: transmission fields multiply, optical depth fields add, masks
are combined with logical OR, and density is combined as layered opacity.

Cloud transmission attenuates rendered stars, targets, and sky background before
sensor noise and analog-to-digital conversion. If a layer has ``brightness``,
SatSim adds configured cloud glow to the background in photoelectrons before
sensor noise. The glow uses the same visual magnitude per arcsec^2 convention as
``background.galactic`` and scales by ``1 - transmission`` for that layer.
Clouds can also receive source-driven glow from enabled artificial skyglow,
moonlight, twilight, and daytime background components. If
``sim.save_segmentation`` is enabled, SatSim writes an additional
``cloud_segmentation`` map.

Preset layers can be used with only a type and optional coverage:

.. code-block:: json

    {
      "clouds": [
        {
          "type": "patchy",
          "coverage": 0.2,
          "brightness": 17.5,
          "range": 3.0,
          "altitude": 1.2,
          "wind_speed": 8.0,
          "wind_direction": 90.0
        },
        {
          "type": "veil",
          "coverage": 0.5
        }
      ]
    }

Supported cloud types are ``patchy``, ``cellular``, ``veil``, ``sheet``,
``fog``, and ``custom``. The ``sheet`` preset approximates an optically thick
stratus deck; use ``veil`` for translucent layers. Preset layers start from
built-in defaults and then apply any user-provided fields. Custom layers start
from generic defaults and are intended for direct tuning:

.. code-block:: json

    {
      "clouds": [
        {
          "type": "custom",
          "name": "hand_tuned_haze",
          "coverage": 0.35,
          "texture": {
            "scales_m": [640, 1280, 2560, 5120],
            "edge_width": 0.25,
            "floor": 0.08,
            "contrast": 0.6,
            "locality_degree": 1
          },
          "optical": {
            "tau_min": 0.01,
            "tau_max": 0.8,
            "tau_gamma": 1.15
          },
          "geometry": {
            "range_km": 3.0,
            "altitude_km": 1.2
          },
          "motion": {
            "speed_m_per_s": 8.0,
            "direction_deg": 90.0
          },
          "illumination": {
            "brightness_mag_arcsec2": 17.5,
            "source_gains": {
              "artificial": 1.0,
              "lunar": 1.0,
              "solar": 1.0
            }
          }
        }
      ]
    }

The grouped ``texture``, ``optical``, ``geometry``, ``motion``, and
``illumination`` objects are aliases for the older flat fields. Do not set the
same logical field in both forms.

Each cloud layer supports these fields:

* ``type``: required cloud type name.
* ``name``: optional metadata label. Defaults to ``cloud_<index>``.
* ``enabled``: optional boolean. Defaults to ``true``.
* ``seed``: optional integer. Defaults to a deterministic seed derived from the
  simulation seed, layer index, and cloud type when a simulation seed is
  present. If no simulation seed is present, SatSim generates a random cloud
  seed for the run and records it in metadata.
* ``coverage``: optional approximate realized sky fraction of the cloud mask in
  the range ``0.0`` to ``1.0``. For floored presets such as ``veil``,
  ``sheet``, and ``fog``, coverage controls the above-floor cloud support while
  the mask remains full-frame because baseline attenuation is present
  everywhere.
* ``feature_scales_m`` or ``texture.scales_m``: optional list of positive
  physical texture scales in meters.
* ``density_edge_width`` or ``texture.edge_width``: optional nonnegative edge
  softness value.
* ``density_floor`` or ``texture.floor``: optional ``null`` or value in
  ``[0.0, 1.0]``.
* ``brightness`` or ``illumination.brightness``: optional cloud glow in visual
  magnitude per arcsec^2. Lower values are brighter.
* ``illumination.source_gains``: optional empirical brightening gains for
  source-driven cloud glow. Supported keys are ``artificial``, ``lunar``, and
  ``solar``.
* ``range`` or ``geometry.range``: optional cloud slant range in kilometers.
  It controls projected texture scale and converts wind speed to pixels per
  second. Defaults to ``3.0``.
* ``altitude`` or ``geometry.altitude``: optional cloud height in kilometers
  above the observer. It is used by source-driven cloud brightening gains. If
  omitted, SatSim uses ``range`` as the source-height proxy.
* ``wind_speed`` or ``motion.wind_speed``: optional cloud advection speed in
  meters per second on the cloud plane. Defaults to ``0.0``.
* ``wind_direction`` or ``motion.wind_direction``: optional cloud advection
  direction in degrees using SatSim's image-plane polar convention. ``0`` moves
  features toward positive column/right, and ``90`` moves features toward
  positive row/down.
* ``texture_contrast`` or ``texture.contrast``: optional value in
  ``[0.0, 1.0]``.
* ``locality_degree``: optional positive integer. Higher values make cloud
  support more localized and patchy.
* ``tau_min``, ``tau_max``, and ``tau_gamma`` or their ``optical`` aliases:
  optional optical depth controls.

Cloud motion uses a frame-midpoint approximation. Gimbal-induced cloud drift is
derived from the star-field translation, so clouds are treated as inertially
fixed rather than ground fixed; the error is the sidereal rate, which is small
relative to typical wind rates. Clouds are frozen at each frame midpoint, with
no intra-exposure cloud smear. For ``rate-sidereal`` tracking, the final frame
holds the cloud offset accumulated at frame start.

The v1 schema intentionally does not expose albedo, reflectance, blur,
``coverage_mode``, ``mask_threshold``, ``min_feature_scale_px``, or
``amplitude_decay`` fields. Unknown cloud layer fields raise a configuration
error.

Analytical EO Observations
--------------------------

Set ``sim.analytical_obs`` to write estimated observations from rendered object
tracks. Set ``sim.mode`` to ``none`` to skip image rendering and write only
annotations and analytical observations.

.. code-block:: json

    {
      "sim": {
        "mode": "none",
        "spatial_osf": 5,
        "temporal_osf": "auto",
        "padding": 50,
        "samples": 1,
        "analytical_obs": true,
        "analytical_obs_frame": "geocentric"
      }
    }

Analytical observations use ``fpa.detection`` for SNR thresholding, pixel error,
false alarms, and maximum false detections.

Radar Configuration
-------------------

Radar configurations use the same ``geometry`` target modes as EO where range
can be computed: ``tle``, ``twobody``/``statevector``, and ``ephemeris``. The
radar path is analytical; it does not render images.

.. code-block:: json

    {
      "version": 1,
      "sim": {
        "samples": 1
      },
      "radar": {
        "tx_power": 1000000.0,
        "tx_frequency": 3000000000.0,
        "antenna_diameter": 12.0,
        "efficiency": 0.55,
        "field_of_view": {
          "azimuth": [0.0, 360.0],
          "elevation": [10.0, 90.0]
        },
        "range_limits": [100.0, 50000.0],
        "detection": {
          "min_detectable_power": 1e-16,
          "snr_threshold": 1.0,
          "angle_error": 0.01,
          "range_error": 0.1,
          "range_rate_error": 0.001,
          "false_alarm_rate": 0.0
        },
        "time": {
          "dwell": 1.0
        },
        "num_frames": 10
      },
      "background": {
        "galactic": 22.0,
        "stray": { "mode": "none" },
        "moon": { "mode": "none" },
        "twilight": { "mode": "none" },
        "daytime": { "mode": "none" }
      },
      "geometry": {
        "time": [2024, 7, 12, 4, 34, 0.0],
        "site": {
          "lat": "20.746111 N",
          "lon": "156.431667 W",
          "alt": 0.3
        },
        "stars": {
          "mode": "none",
          "motion": { "mode": "none" }
        },
        "obs": {
          "mode": "list",
          "list": [
            {
              "mode": "tle",
              "name": "target",
              "rcs": 1.0,
              "tle": [
                "1 39533U 14008A   24194.19078549  .00000076  00000-0  00000-0 0  9992",
                "2 39533  53.6020 178.0486 0073391 221.0643 138.3710  2.00570029 75545"
              ]
            }
          ]
        }
      }
    }

Radar outputs contain azimuth, elevation, range, range rate, Doppler-equivalent
line-of-sight velocity, uncertainties, SNR, radar cross section, and optional
sensor/object identifiers. The current radar path applies detection thresholds
and Gaussian measurement noise; configured false alarms are not emitted yet.

TensorFlow Data Augmentation
----------------------------

SatSim can inject synthetic targets into TensorFlow datasets. The augmentation
path adds a generated SatSim frame to the input image and appends new synthetic
target boxes that pass the configured SNR threshold.

.. code-block:: python

    import tensorflow as tf
    from satsim import load_json
    from satsim.dataset.augment import augment_satnet_with_satsim

    satsim_params = load_json("input/augmentation_config.json")

    dataset = tf.data.TFRecordDataset("/path/to/tfrecords")
    dataset = augment_satnet_with_satsim(
        dataset,
        satsim_params,
        prob=0.5,
        rn=9.0,
        min_snr=2.0,
        box_pad=10,
    )

Common Notes
------------

* The CLI recognizes ``.json`` and ``.yml`` input files.
* Use ``spatial_osf`` for spatial oversampling. Legacy configurations using
  ``spacial_osf`` is also accepted.
* ``sim.mode: none`` disables image rendering. Use it with
  ``sim.analytical_obs`` for EO observation-only runs.
* ``sim.render_size`` can tile large images when full-frame rendering exceeds
  available memory.
* ``--jobs`` multiplies the number of processes per GPU; reduce
  ``spatial_osf``, ``padding``, frame size, or jobs if memory is exhausted.
* Use the schema in ``schema/v1`` to audit accepted keys. Newer blocks such as
  clouds intentionally reject unknown fields.
