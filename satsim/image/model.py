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


def radial_cos2d(height, width, y_scale=0.1, x_scale=0.1, power=4, xy_scale=None, normalize=False,
                 mult=1.0, clip=(0.0, 1.0),
                 falloff_height=None, falloff_width=None, falloff_xy=None):
    """ Generates a cosine wave radially from the center of the image. Typically
    used to simulate optical vignette or irradiance falloff.

    Args:
        height: `int`, image height.
        width: `int`, image width.
        y_scale: `float`, the fraction of the cosine wave to generate across the
            rows from the center to the edge of the falloff height.
        x_scale: `float`, the fraction of the cosine wave to generate across the
            columns from the center to the edge of the falloff width.
        power: `float`, the exponent of the cosine. Set to 4 to generate a
            "cosine fourth" irradiance falloff map. default=4
        xy_scale: `float`, if not None set y_scale and x_scale to this.
        normalize: `bool` or `str`, normalize the falloff before applying
            `mult`. True uses peak normalization; supported string modes are
            "peak", "median", and "mean". default=False
        mult: `float`, multiply cosine. default=1
        clip: `array`, clip returned value by minimum and maximum. default=(0.0, 1.0)
        falloff_height: `float`, effective height (in pixels) of the falloff
            grid. If larger than `height`, the vignette extends beyond the
            image; if smaller, the falloff is more aggressive. default=None
        falloff_width: `float`, effective width (in pixels) of the falloff
            grid. If larger than `width`, the vignette extends beyond the
            image; if smaller, the falloff is more aggressive. default=None
        falloff_xy: `float`, if not None set falloff_height and falloff_width to this.

    Returns:
        An `array`, a two dimensional image.
    """
    if xy_scale is not None:
        y_scale = xy_scale
        x_scale = xy_scale

    if falloff_xy is not None:
        falloff_height = falloff_xy
        falloff_width = falloff_xy

    if falloff_height is None:
        falloff_height = height
    if falloff_width is None:
        falloff_width = width

    def _axis_coords(size, falloff_size, scale):
        center = (size - 1) / 2.0
        if falloff_size is None or falloff_size <= 1:
            step = 0.0
        else:
            step = 2 * np.pi * scale / (falloff_size - 1)
        return (np.arange(size, dtype=np.float64) - center) * step

    x = _axis_coords(width, falloff_width, x_scale)
    y = _axis_coords(height, falloff_height, y_scale)
    xx, yy = np.meshgrid(x, y)
    z = np.cos(np.sqrt(xx ** 2 + yy ** 2)) ** power
    z = _normalize_values(z, normalize)
    z = mult * z

    if clip is not None:
        z = np.clip(z, clip[0], clip[1])

    return z


def _deformable_radius_grid(height, width, eta=1.0, center=None, center_x=None, center_y=None):
    if center is not None:
        center_x, center_y = center

    if center_x is None:
        center_x = (width - 1) / 2.0
    if center_y is None:
        center_y = (height - 1) / 2.0

    y = np.arange(height, dtype=np.float64)
    x = np.arange(width, dtype=np.float64)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt((xx - center_x) ** 2 + (eta * (yy - center_y)) ** 2)

    corners = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [0.0, height - 1.0],
        [width - 1.0, height - 1.0],
    ])
    r_max = np.max(np.sqrt((corners[:, 0] - center_x) ** 2 + (eta * (corners[:, 1] - center_y)) ** 2))
    return rr / r_max if r_max > 0 else rr


def _normalize_peak(z):
    z_max = np.max(z)
    return z / z_max if z_max != 0 else z


def _normalize_values(z, normalize):
    if normalize is None or normalize is False:
        return z

    if normalize is True:
        mode = 'peak'
    else:
        mode = str(normalize).strip().lower()

    if mode in {'peak', 'max'}:
        denom = np.max(z)
    elif mode in {'median', 'med'}:
        denom = np.median(z)
    elif mode == 'mean':
        denom = np.mean(z)
    else:
        raise ValueError("normalize must be false, true, 'peak', 'median', or 'mean'")

    return z / denom if denom != 0 else z


def deformable_radial_poly2d(height, width, coefficients, eta=1.0, center=None, center_x=None, center_y=None,
                             normalize=True, mult=1.0, clip=(0.0, 1.0)):
    """Generate a deformable radial polynomial vignette map.

    This follows the distance transform used by the deformable radial
    polynomial vignetting model:

        r_eta = sqrt((x - x_c)^2 + (eta * (y - y_c))^2)

    The polynomial is evaluated on r_eta normalized by the farthest image corner,
    so coefficients are stable across image sizes. ``center`` is in ``(x, y)``
    order; use ``center_x`` and ``center_y`` for explicit coordinates.
    """
    rr = _deformable_radius_grid(height, width, eta, center, center_x, center_y)
    z = np.polynomial.polynomial.polyval(rr, coefficients)

    z = _normalize_values(z, normalize)

    z = mult * z

    if clip is not None:
        z = np.clip(z, clip[0], clip[1])

    return z


def deformable_radial_falloff2d(height, width, amplitudes, eta=1.0, center=None, center_x=None, center_y=None,
                                base=1.0, normalize=False, mult=1.0, clip=(0.0, 1.0)):
    """Generate a monotonic deformable radial polynomial falloff map.

    This uses the same DRP distance transform as `deformable_radial_poly2d`,
    but constrains the radial function to decrease from the optical center:

        z = base - sum(amplitude_k * r_eta ** k)

    with k starting at 1 and r_eta normalized by the farthest image corner.
    ``center`` is in ``(x, y)`` order; use ``center_x`` and ``center_y`` for
    explicit coordinates.
    """
    rr = _deformable_radius_grid(height, width, eta, center, center_x, center_y)
    z = np.full((height, width), base, dtype=np.float64)
    for power, amplitude in enumerate(amplitudes, start=1):
        z = z - amplitude * rr ** power

    z = _normalize_values(z, normalize)

    z = mult * z

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
