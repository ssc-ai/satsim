from __future__ import division, print_function, absolute_import

import tensorflow as tf


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
