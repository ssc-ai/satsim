=====
Usage
=====

Python Module
-------------

To use SatSim in a python project:

.. code-block:: python

    from satsim import gen_multi, load_json

    # load a template json file
    ssp = load_json('input/config.json')

    # edit some parameters
    ssp['sim']['samples'] = 50
    ssp['geometry']['obs']['list']['mv'] = 17.5

    # generate SatNet files to output/ directory
    gen_multi(ssp, eager=True, output_dir='output/')

Command Line Interface
----------------------

To run SatSim from the command line:

.. code-block:: bash

    # show help menu
    $ satsim --help

    # show run help menu
    $ satsim run --help

    # generate images on GPU 0 in eager mode
    $ satsim --debug INFO run --device 0 --mode eager --output_dir output/ input/config.json

To run multiple SatSim processes in parallel use the --jobs option:

.. code-block:: bash

    # single GPU with multiple processes (3 processes total)
    $ satsim --debug INFO run --jobs 3 --device 0 --mode eager --output_dir output/ input/config.json

    # multi GPU (2 processes total)
    $ satsim --debug INFO run --jobs 1 --device 0,1 --mode eager --output_dir output/ input/config.json

    # multi GPU and multiple processes per GPU (16 processes total)
    $ satsim --debug INFO run --jobs 4 --memory 7000 --device 0,1,2,3 --mode eager --output_dir output/ input/config.json

The last example will spawn 16 processes with 7000MB of maximum memory each 
(4 processes per GPU). If hardware memory is exceeded, TensorFlow will throw 
out-of-memory exceptions. If not enough memory is allocated per process, TensorFlow 
will throw misc exceptions.

Note on parallel processing: Additional processes on a single GPU may increase 
image generation throughput by increasing GPU utilization but requiring additional
memory allocation. Be sure there is enough available physical memory to support
the desired number of processes.

Parameters and Configuration
----------------------------

SatSim requires a dictionary or JSON for configuration. As of this version,
there is no parameter error checking or defaults. You must provide valid values
for all parameter. Note: some values have no effect and are placeholders for
future versions.

Most numeric parameters can be replaced with a `$sample` key, which SatSim
will use to randomly sample a value base on the desired distribution. Any NumPy
random distributions can be specified. For the list, see `NumPy`_.
You can also provide a `seed` inside a `$sample` dictionary to make that field
deterministic without affecting other samples.

.. _NumPy: https://docs.scipy.org/doc/numpy/reference/routines.random.html#distributions

Here is a complete SatSim parameter python dictionary example:

.. code-block:: python

    {
        "version": 1,
        "sim": {
            "mode": "fftconv2p",       # specifies convolution mode (placeholder)
            "spacial_osf": 15,         # number of pixels to upsample the image in each axis
            "temporal_osf": 100,       # number of transformations to apply to smear the background
            "padding": 100,            # number of real pixels to pad each side of the image
            "samples": 1               # number of sets to generate
        },
        "clouds": [
            {
                "type": "patchy",      # cloud type preset
                "coverage": 0.2,       # approximate cloud mask fraction
                "brightness": 17.5,    # optional cloud glow in mv/arcsec^2
                "wind_speed": 8.0,     # optional cloud advection speed in m/s
                "wind_direction": 90.0 # optional direction; 0 right, 90 down
            }
        ],
        "fpa": {
            "height": 512,             # height of image image in real pixels
            "width": 512,              # width of the image in real pixels
            "y_fov": 0.308312,         # vertical field of view in degrees
            "x_fov": 0.308312,         # horizontal field of view in degrees
            "dark_current": 0.3,       # dark current per pixel in pe/second (scalar or image)
            "gain": 1,                 # pixel response (scalar or image)
            "bias": 0,                 # pixel bias in pe (scalar or image)
            "zeropoint": 20.6663,      # zeropoint of the instrument
            "a2d": {
                "response": "linear",  # analog to digital converter response (placeholder)
                "fwc": 100000,         # full well capacity in pe
                "gain": 1.5,           # digital to analog multiplier
                "bias": 1500           # read out bias in digital counts
            },
            "noise": {
                "read": 9,             # RMS read noise in pe
                "electronic": 0        # RMS electronic noise in pe
            },
            "psf": {
                "mode": "gaussian",    # point spread function type
                "eod": 0.15            # energy on detector or ensquared energy from 0.0-1.0 of the gaussian
            },
            "time": {
                "exposure": 5.0,       # integration time in seconds
                "gap": 2.5             # read out time or gap between frames in seconds
            },
            "num_frames": 6            # number of frames per set of images
        },
        "background": {
            "stray": {                 # stray light mode
                "mode": "none"
            },
            "galactic": 22.0,          # natural sky background in mv/arcsec^2
            "skyglow": 21.5,           # optional total clear-sky site brightness
            "moon": { "mode": "none" },
            "twilight": { "mode": "none" },
            "daytime": { "mode": "none" }
        },
        "geometry": {
            "stars": {
                "mode": "bins",        # star generation type\
                "mv": {
                    "bins":            # visual magnitude bins
                        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    "density":         # density of stars in number/degrees^2 per bin
                        [0.019444,0,0.0055556,0.016667,0.036111,0.038889,0.097222,0.66944,2.4778,5.0028,10.269,24.328,35.192,60.017,110.06,180.28,285.53,446.14]
                },
                "motion": {
                    "mode": "affine",  # transform type (placeholder)
                    "rotation": 0,     # clockwise star rotation rate in radians/seconds
                    "translation":     # drift rate in pixels/second [row, column]
                        [0.4,7.0]
                }
            },
            "obs": {
                "mode": "list",            # obj generation type
                "list": {
                    "$sample": "random.list",
                    "length":              # number of targets to sample
                        { "$sample": "random.randint", "low": 0, "high": 15 },
                    "value": {
                        "mode": "line",    # draw type
                        "origin": [        # starting location of object in normalized array coordinates [row, col]
                            { "$sample": "random.uniform", "low": 0.1, "high": 0.9 },
                            { "$sample": "random.uniform", "low": 0.1, "high": 0.9 }
                        ],
                        "velocity": [      # velocity of the object in pixels/second [row, col]
                            { "$sample": "random.uniform", "low": -0.01, "high": 0.01 },
                            { "$sample": "random.uniform", "low": -0.01, "high": 0.01 }
                        ],
                        "mv":              # visual magnitude of the object
                            { "$sample": "random.uniform", "low": 5.0, "high": 22.0 }
                }
            }
        }
    }

Background Configuration
------------------------

The ``background`` object defines sky background before dark current, sensor
noise, and analog-to-digital conversion. Existing configurations that only set
``background.galactic`` keep the legacy behavior: ``galactic`` is interpreted as
the total scalar sky background in visual magnitude per arcsec^2.

For artificial skyglow, add ``background.skyglow``:

.. code-block:: python

    "background": {
        "galactic": 22.0,
        "skyglow": 21.5,
        "stray": { "mode": "none" },
        "moon": { "mode": "none" },
        "twilight": { "mode": "none" },
        "daytime": { "mode": "none" }
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

.. code-block:: python

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

Supported cloud types are ``patchy``, ``cellular``, ``veil``, ``sheet``,
``fog``, and ``custom``. The ``sheet`` preset approximates an optically thick
stratus deck; use ``veil`` for translucent layers. Preset layers start from
built-in defaults and then apply any user-provided fields. Custom layers start
from generic defaults and are intended for direct tuning:

.. code-block:: python

    "clouds": [
        {
            "type": "custom",
            "name": "hand_tuned_haze",
            "coverage": 0.35,
            "feature_scales_m": [640, 1280, 2560, 5120],
            "density_edge_width": 0.25,
            "density_floor": 0.08,
            "texture_contrast": 0.6,
            "locality_degree": 1,
            "tau_min": 0.01,
            "tau_max": 0.8,
            "tau_gamma": 1.15
        }
    ]

Each cloud layer supports these fields:

- ``type``: required cloud type name.
- ``name``: optional metadata label. Defaults to ``cloud_<index>``.
- ``enabled``: optional boolean. Defaults to ``true``.
- ``seed``: optional integer. Defaults to a deterministic seed derived from the
  simulation seed, layer index, and cloud type when a simulation seed is
  present. If no simulation seed is present, SatSim generates a random cloud
  seed for the run and records it in metadata.
- ``coverage``: optional approximate realized sky fraction of the cloud mask in
  the range ``0.0`` to ``1.0``. For floored presets such as ``veil``,
  ``sheet``, and ``fog``, coverage controls the above-floor cloud support while
  the mask remains full-frame because baseline attenuation is present
  everywhere.
- ``feature_scales_m``: optional list of positive physical texture scales in meters.
- ``density_edge_width``: optional nonnegative edge softness value.
- ``density_floor``: optional ``null`` or value in ``[0.0, 1.0]``.
- ``brightness``: optional cloud glow in visual magnitude per arcsec^2. Lower
  values are brighter. Photoelectrons are scaled by ``1 - transmission``.
- ``illumination.source_gains``: optional empirical brightening gains for
  source-driven cloud glow. Supported keys are ``artificial``, ``lunar``, and
  ``solar``. Each gain multiplies the in-band photoelectron residual from that
  source in the generic metadata-free model. For direct lunar and solar
  illumination metadata, SatSim applies the configured gain proportionally
  relative to the built-in default. Use measured clear/overcast sky brightness
  ratios to calibrate site-specific values. The default artificial gain is
  tuned as a moderate overcast amplification, while Berlin-class urban overcast
  sites may require gains near ``10`` to ``12``.
- ``range``: optional cloud slant range in kilometers. It controls projected
  texture scale and converts wind speed to pixels per second. Defaults to
  ``3.0``.
- ``altitude``: optional cloud height in kilometers above the observer. It is
  used by source-driven cloud brightening gains. If omitted, SatSim uses
  ``range`` as the source-height proxy.
- ``wind_speed``: optional cloud advection speed in meters per second on the
  cloud plane. Defaults to ``0.0``.
- ``wind_direction``: optional cloud advection direction in degrees using
  SatSim's image-plane polar convention. ``0`` moves features toward positive
  column/right, and ``90`` moves features toward positive row/down.
- ``texture_contrast``: optional value in ``[0.0, 1.0]``.
- ``locality_degree``: optional positive integer. Higher values make cloud
  support more localized and patchy.
- ``tau_min``, ``tau_max``, ``tau_gamma``: optional optical depth controls.

Cloud motion uses a frame-midpoint approximation. Gimbal-induced cloud drift is
derived from the star-field translation, so clouds are treated as inertially
fixed rather than ground fixed; the error is the sidereal rate, which is small
relative to typical wind rates. Clouds are frozen at each frame midpoint, with
no intra-exposure cloud smear. For ``rate-sidereal`` tracking, the final frame
holds the cloud offset accumulated at frame start.

The v1 schema intentionally does not expose albedo, reflectance, blur,
``coverage_mode``, ``mask_threshold``,
``min_feature_scale_px``, or ``amplitude_decay`` fields. Unknown cloud layer
fields raise a configuration error.

Here is a geometry example for a topocentric site and an SGP4 satellite track simulation:

.. code-block:: python

    "geometry": {
        "time": [2015, 4, 24, 9, 7, 44.128],    # start time of observation [year, month, day, hour, minute, seconds]
        "site": {                       # observing site configuration
            "mode": "topo",             # site type (topo for topocentric site)
            "lat": "20.746111 N",       # latitude for topocentric site
            "lon": "156.431667 W",      # longitude for topocentric site
            "alt": 0.3,                 # altitude for topocentric site in km
            "gimbal": {                 # gimbal type (placeholder, only wcs support)
                "mode": "wcs",          # placeholder
                "rotation": 7           # field rotation in degrees counter-clockwise
            },
            "track": {                  # track vector configuration
                "mode": "rate",         # rate or sidereal
                "tle1": "1 36411U 10008A   15115.45079343  .00000069  00000-0  00000+0 0  9992", # TLE line 1
                "tle2": "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866"  # TLE line 2
            }
        },
        "stars": {
            "mode": "sstr7",                    # use catalog type
            "path": "/workspace/share/sstrc7",  # path to sstr7 catalog
            "motion": { "mode": "none" }        # calculated using site and gimbal model when motion is set to none
        },
        "obs": {
            "mode": "list",             # obj generation type
            "list": [
                {
                    "mode": "tle",      # SGP4 two-line element set
                    "tle1": "1 36411U 10008A   15115.45075343  .00000069  00000-0  00000+0 0  9992", # TLE line 1
                    "tle2": "2 36411 000.0719 125.6855 0001927 217.7585 256.6121 01.00266852 18866", # TLE line 2
                    "mv": 12.0          # visual magnitude of the object 
                }
            ]
        }
    }

SatSim's `$generator` feature allows users to call built-in or custom defined python functions to dynamically generate configurations. 
The requirements are that the function is accessible from a python module and that the generator outputs configuration code that is
compatible with SatSim. Here is an example of using the built-in `cone` obs generator which will generate a list of targets that emanate
from a random point on the FPA:

.. code-block:: python

        "obs": {
            "generator": {
                "module": "satsim.generator.obs.geometry",  # the python module to import for which the function belongs to
                "function": "cone",                         # the generator function to call
                "kwargs": {                                 # keyword arguments to the generator function
                    "n": { "$sample": "random.randint", "low": 500, "high": 1000 }, # note that random sampling can be used
                    "t": 0,                                                         # static constants can also be used
                    "direction": [
                        { "$sample": "random.uniform", "low": 0, "high": 360 },
                        { "$sample": "random.uniform", "low": 10, "high": 30 }],
                    "velocity": [
                        { "$sample": "random.uniform", "low": 1.0, "high": 5.0}, 
                        { "$sample": "random.uniform", "low": 3.0, "high": 5.0 }],
                    "origin": [
                        { "$sample": "random.uniform", "low": 0.2, "high": 0.8 }, 
                        { "$sample": "random.uniform", "low": 0.2, "high": 0.8 }],
                    "mv": [15.0, 17.0]
                }
            }
        }

The SatSim configuration also supports importing partial configurations from external JSON files allowing the user to reuse common
configurations. For example, this configuration loads the `fpa` settings from an external file and overrides the `num_frames`
key with the value of 10:

.. code-block:: python

    "fpa": {
        "$import": "../common/random_raven_fpa.json",   # filename to import
        "override": {                                  # overrides values from the imported JSON
            "num_frames": 10
        }
    },
