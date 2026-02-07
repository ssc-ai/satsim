from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import math


def signal_to_noise_ratio(signal, background, noise):
    """Calculates signal to noise ratio.

    Examples::

        snr = signal_to_noise_ratio(100, 20, 10)

    Args:
        signal: `array`, signal in total photoelectrons
        background: `array`, background in total photoelectrons (includes total
            dark current)
        noise: `float`, RMS noise

    Returns:
        An `array`, signal to noise ratio
    """

    return signal / tf.sqrt(signal + background + noise * noise)


def aperture_signal_to_noise_ratio(signal, background, noise, mask):
    """Calculates aperture signal to noise ratio for a mask.

    Args:
        signal: `array` or `float`, signal in total photoelectrons
        background: `array` or `float`, background in total photoelectrons
        noise: `array` or `float`, RMS noise
        mask: `array`, boolean mask for the aperture

    Returns:
        `float` SNR, or `None` if the mask is empty.
    """
    mask = np.asarray(mask, dtype=bool)
    n_pix = int(mask.sum())
    if n_pix == 0:
        return None

    def _masked_sum(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return float(value) * n_pix
        return float(value[mask].sum())

    def _masked_rn_sum(value):
        value = np.asarray(value)
        if value.ndim == 0:
            return float(value) * float(value) * n_pix
        return float(np.square(value[mask]).sum())

    signal_sum = _masked_sum(signal)
    background_sum = _masked_sum(background)
    rn_sum = _masked_rn_sum(noise)

    denom = math.sqrt(signal_sum + background_sum + rn_sum)
    if denom == 0:
        return 0.0
    return signal_sum / denom
