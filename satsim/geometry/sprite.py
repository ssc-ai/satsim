from astropy.io import fits
import tensorflow as tf


def load_sprite_from_file(filename, normalize=True, dtype=tf.float32):
    """Load an image or sprite from file based on filename extension.
    Supported formats: FITS

    Args:
        filename: `str`, the image file name.
        normalize: `boolean`, normalize the sprite to 1. Default=True.

    Returns:
        A `Tensor`, the image, None if file type is not recognized
    """
    if filename.endswith('.fits'):

        hdul = fits.open(filename)
        img = tf.cast(hdul[0].data, dtype=dtype)

        if normalize:
            img = tf.cast(img / tf.math.reduce_sum(img), dtype)

        return img

    else:

        return None
