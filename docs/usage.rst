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
            "galactic": 19.5           # background in mv/arcsec^2/sec
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
