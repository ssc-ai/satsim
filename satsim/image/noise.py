from __future__ import division, print_function, absolute_import

import tensorflow as tf


def add_photon_noise(fpa, samples=None):
    """Add photon noise. Photon noise results from the inherent statistical
    variation in the arrival rate of photons incident on the CCD. Photoelectrons
    generated within the semiconductor device constitute the signal, the
    magnitude of which fluctuates randomly with photon incidence at each
    measuring location (pixel) on the CCD. The interval between photon arrivals
    is governed by Poisson statistics, and therefore, the photon noise is
    equivalent to the square-root of the signal. In general, the term shot noise
    is applied to any noise component reflecting a similar statistical
    variation, or uncertainty, in measurements of the number of photons
    collected during a given time interval, and some references use that term in
    place of photon noise in discussions of CCD noise sources.

    Examples::

        fpa_photon_noise = add_photon_noise(fpa_no_noise+background+dark_current)

    Args:
        fpa: `Tensor`, input image as a 2D tensor in total photoelectrons per pixel
        samples: `int`, number of samples to generate then average. Typically used
            to estimate averaging of multiple images. default=None (or 1 sample)

    Returns:
        A `Tensor`, the 2D tensor with photon noise applied.
    """
    if samples is not None:
        fpa64 = tf.cast(fpa, tf.float64)
        fpa_tmp = tf.zeros_like(fpa64, dtype=tf.float64)
        for i in range(samples):
            fpa_tmp = fpa_tmp + tf.cast(tf.squeeze(tf.compat.v1.random.poisson(fpa64, [1])), tf.float64)
        return tf.cast(fpa_tmp / samples, fpa.dtype)
    else:
        return tf.squeeze(tf.compat.v1.random.poisson(fpa, [1]))


def add_read_noise(fpa, rn, en=0):
    """Add read noise. Read noise (RN) is a combination of noise from the pixel
    and from the analog to digital converter (ADC). The RN of the sensor is the
    equivalent noise level (in electrons RMS) at the output of the camera in the
    dark and at zero integration time. Note that the build up is different for a
    CMOS sensor and a CCD sensor. The ADC with CCD image sensors is done outside
    the sensor and the ADCs for a CMOS image sensor are in each pixel.

    Examples::

        fpa_with_read_noise = add_read_noise(fpa, 15)

    Args:
        fpa: `Tensor`, input image as a 2D tensor in real pixels (not oversampled).
        rn: `float`, electrons RMS value of the read noise.
        en: `float`, electrons RMS value of the electronic noise.

    Returns:
        A `Tensor`, the 2D tensor with read noise applied.
        A `Tensor`, the 2D tensor read noise.
    """
    rn = tf.cast(rn, tf.float32)
    en = tf.cast(en, tf.float32)
    noise = tf.random.normal(tf.shape(fpa)) * tf.math.sqrt(rn * rn + en * en)
    return fpa + noise, noise
