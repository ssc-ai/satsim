from __future__ import division, print_function, absolute_import

import numpy as np
from scipy import ndimage


def polygrid2d(height, width, c, low=-1, high=1):
    """ Generates a two dimensional image based on NumPy polygrid2d.

    Args:
        height: `int`, image height.
        width: `int`, image width.
        c: `array`, Array of coefficients ordered so that the coefficients for
            terms of degree i,j are contained in c[i,j]. If c has dimension
            greater than two the remaining indices enumerate multiple sets of
            coefficients.
        low: `float`, lowest number in the two dimensional series. Points are
            sampled linearly spaced between `low` and `high`. default=-1
        high: `float`, highest number in the two dimensional series. Points are
            sampled linearly spaced between `low` and `high`. default=1

    Returns:
        An `array`, a two dimensional image.
    """
    cc = np.linspace(low, high, width)
    rr = np.linspace(low, high, height)

    return np.polynomial.polynomial.polygrid2d(x=rr, y=cc, c=c)


def radial_cos2d(height, width, y_scale=0.1, x_scale=0.1, power=4, xy_scale=None, mult=1.0, clip=[0.0, 1.0]):
    """ Generates a cosine wave radially from the center of the image. Typically
    used to simulate optical vignette or irradiance falloff.

    Args:
        height: `int`, image height.
        width: `int`, image width.
        y_scale: `float`, the fraction of the cosine wave to generate across the
            rows from the center to the edge.
        x_scale: `float`, the fraction of the cosine wave to generate across the
            columns from the center to the edge.
        scale: `float`, if not None set y_scale and x_scale to this.
        power: `float`, the exponent of the cosine. Set to 4 to generate a
            "cosine fourth" irradiance falloff map. default=4
        mult: `float`, multiply cosine. default=1
        clip: `array`, clip returned value by minimum and maximum. default=[0.0, 1.0]

    Returns:
        An `array`, a two dimensional image.
    """
    if xy_scale is not None:
        y_scale = xy_scale
        x_scale = xy_scale

    x = np.linspace(-np.pi * x_scale, np.pi * x_scale, width)
    y = np.linspace(-np.pi * y_scale, np.pi * y_scale, height)
    xx, yy = np.meshgrid(x, y)
    z = mult * np.cos(np.sqrt(xx ** 2 + yy ** 2)) ** power

    if clip is not None:
        z = np.clip(z, clip[0], clip[1])

    return z


def sin2d(height, width, amplitude=1, frequency=5, bias=0, minimum=0, maximum=None, direction=0, damped=False):
    """ Generates a sine wave across the image in a specified direction.

    Args:
        height: `int`, image height.
        width: `int`, image width.
        amplitude: `float`, the amplitude of the sine wave. default=1
        frequency: `float`, number of full sine waves to generate over the width
            of the image. default=5
        bias: `float`, bias to add to the entire image. default=0
        minimum: `float`, set the minimum value in the image. default=0
        maximum: `float`, set the maximum value in the image. default=None
        direction: `float`, set the direction of the wave in degrees where 0 is
            horizontal from left to right. default=0
        damped: `boolean`, damp the sine wave by `e^-t`. default=False

    Returns:
        An `array`, a two dimensional image.
    """
    if frequency == 0:
        return np.ones((height, width))

    npix = width
    r = np.arange(0, npix, 1)
    t = r / npix * frequency

    if damped:
        s = amplitude * np.exp(-t) * np.sin(2 * np.pi * t) + bias
    else:
        s = amplitude * np.sin(2 * np.pi * t) + bias

    if direction == 0:
        img = np.tile(s, (height, 1))

    else:
        diag = int(np.sqrt(width * width + height * height))
        ox = int((diag - width) / 2)
        oy = int((diag - height) / 2)
        img = np.zeros((height, width))
        img = sin2d(diag, diag, amplitude, frequency, bias, minimum, maximum, 0, damped)
        img = ndimage.rotate(img, direction, reshape=False)
        img = img[oy:oy + height, ox:ox + width]

    if minimum is not None:
        img[img < minimum] = minimum

    if maximum is not None:
        img[img > maximum] = maximum

    return img


def astropy_model2d(height, width, filename, low=-1, high=1, clip=0):
    """ Loads an AstroPy Model. The model is evaluated between low and high
    using evenly spaced values (linspace).

    Args:
        height: `int`, image height.
        width: `int`, image width.
        filename: `string`, path to Advanced Scientific Data Format model file.
        low: `float`, lowest number in the two dimensional series. Points are
            sampled linearly spaced between `low` and `high`.
        high: `float`, highest number in the two dimensional series. Points are
            sampled linearly spaced between `low` and `high`.

    Returns:
        An `array`, a two dimensional image.
    """
    import asdf
    with asdf.open(filename) as f:
        model = f.tree['model']

    x, y = np.meshgrid(np.linspace(low, high, width), np.linspace(low, high, height))

    return np.nan_to_num(model(x, y)).clip(min=clip)
