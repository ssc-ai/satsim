"""Tests for `satsim.image.fpa` package."""
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="tensorflow")

import numpy as np
import tensorflow as tf

from satsim.dataset.augment import augment_satnet_with_satsim


def test_dataset_augment():

    def gen():
        yield tf.zeros((512,512, 1), dtype=tf.float32), tf.zeros((100,5), dtype=tf.float32), 'none', 'none'

    ds = tf.data.Dataset.from_generator(gen, output_types=(tf.float32, tf.float32, tf.string, tf.string))

    ssp = {
        "version": 1,
        "sim": {
            "mode": "fftconv2p",
            "spacial_osf": 3,
            "temporal_osf": 100,
            "padding": 100,
            "enable_shot_noise": False,
            "samples": 1
        },
        "fpa": {
            "height": 512,
            "width": 512,
            "y_fov": 0.3,
            "x_fov": 0.3,
            "dark_current": 0,
            "gain": 1,
            "bias": 0,
            "zeropoint": 20.6663,
            "a2d": {
                "response": "linear",
                "fwc": 100000,
                "gain": 1.5,
                "bias": 1000
            },
            "noise": {
                "read": 0,
                "electronic": 0
            },
            "psf": {
                "mode": "gaussian",
                "eod": 0.15,
            },
            "time": { "exposure": 1.0, "gap": 2.5},
            "num_frames": 1
        },
        "background": {
            "stray": {
                "mode": "none"
            },
            "galactic": 10000
        },
        "geometry": {
            "stars": {
                "mode": "none",
                "motion": { "mode": "none"}
            },
            "obs": {
                "mode": "list",
                "list": [{
                    "mode": "line-polar",
                    "origin": [0.5, 0.5],
                    "velocity": [90.0, 1.0],
                    "mv": 15
                }]
            }
        },
        "augment": {
            "image": {
                "post": None
            }
        }
    }

    data_satsim = augment_satnet_with_satsim(ds, ssp, prob=1.0)

    for i, b, f, a in data_satsim.skip(0).take(1):

        img = tf.squeeze(i).numpy()
        bb = b.numpy()

        # The noiseless pre-A/D target contains 184.7226 pe on both CPU and
        # GPU. After convolution, dividing by the 1.5 pe/DN gain and flooring
        # each pixel can differ by a couple of DN when backend roundoff moves
        # values across integer boundaries. Check the injected signal rather
        # than requiring a bit-identical reduction of the large bias plane.
        bias_dn = ssp['fpa']['a2d']['bias'] * img.size
        injected_dn = np.sum(img, dtype=np.float64) - bias_dn
        np.testing.assert_allclose(injected_dn, 108.0, rtol=0, atol=2.0)
        box_pad = 10.0 / ssp['fpa']['width']
        row_motion = 1.0 / ssp['fpa']['height']
        np.testing.assert_allclose(
            bb[0],
            [
                0.5 - box_pad,
                0.5 - box_pad,
                0.5 + row_motion + box_pad,
                0.5 + box_pad,
                1.0,
            ],
            rtol=0,
            atol=1e-7,
        )

    data_satsim = augment_satnet_with_satsim(ds, ssp, prob=0.0)

    for i, b, f, a in data_satsim.skip(0).take(1):

        img = tf.squeeze(i).numpy()
        bb = b.numpy()

        assert(int(np.sum(img)) == 0)
        np.testing.assert_array_equal(bb[0], [0, 0, 0, 0, 0])

        return

    # should not get here
    assert(False)
