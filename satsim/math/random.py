from __future__ import division, print_function, absolute_import

import numpy as np
import opensimplex


def gen_samples_from_bins(bins, density, mult=1.):
    """Generates random samples base on discrete bins. The number of samples of
    each bin is calculated from its density. A density multiplier can be
    specified in the case that `density` is rate based. Values are uniformly
    sampled between the min and max of each bin.

    Examples::

        s = gen_samples_from_bins([200,100,10], [30,50], 10.0)

    Args:
        bins: `list`, list of bins. Should be one length larger than density list
        density: `list`, number of occurrences per bin
        mult: `float`, density multiplier (e.g. density is rate based)

    Returns:
        A `list`, samples
    """
    d = np.floor(np.asarray(density) * mult)
    n = int(np.sum(d))

    s = np.zeros(n)

    j = 0
    ns = 0
    for i in range(len(density)):
        ns = int(density[i] * mult)
        s[j:j + ns] = np.random.uniform(bins[i], bins[i + 1], ns)
        j = j + ns

    return s


def gen_sample(type='uniform', seed=None, negate=0.0, **kwargs):
    """A wrapper function to generate a random number using the RNG function
    name. Useful for specifying random samples based on configuration files.

    Examples::
        x = gen_sample(**{"type":"normal", "loc":0.0, "scale":3.0})
        y = gen_sample(**{"type":"uniform", "low":-5.0, "high":5.0})

    Args:
        type: `string`, numpy.random RNG name or custom type
        seed: `int`, rng seed
        negate: `float`, probability between 0 and 1 of negating the sample
        kargs: `dict`, key value args for the RNG

    Returns:
        A `float`, the sample generated from the specified RNG
    """
    rng = np.random.RandomState(seed)

    invert_sign = False
    if negate > 0 and rng.uniform(low=0.0, high=1.0) < negate:
        invert_sign = True

    if type in _CUSTOM_TYPES:
        value = _CUSTOM_TYPES[type](seed=seed, **kwargs)
    else:
        f = getattr(rng, type)
        value = f(**kwargs)

    if invert_sign:
        return -value
    else:
        return value


def lognormal_mu_sigma(mu=1.0, sigma=1.0, size=1, mu_mode='median', seed=None):
    """ Draw samples from a log-normal distribution. Draw samples from a log-normal
    distribution with specified mean, standard deviation, and array shape. Note that
    the mean and standard deviation are the values for the distribution itself.

    Args:
        mu: `float`, Mean or median value of log normal distribution
        sigma: `float`, Standard deviation of the log normal distribution, valid for `mu_mode` == `mean`
        size: `int` or `tuple of ints`, Output shape
        mu_mode: `string`, `mean` or `median`
        seed: `int`, Seed for the random number generator

    Returns:
        A `ndarray` or `scalar`, the drawn samples from log-normal distribution
    """
    print("lognormal_mu_sigma called", mu, sigma, size, mu_mode, seed)
    if mu_mode == 'mean':
        lmu = np.log(sigma * sigma / np.sqrt(mu * mu + sigma * sigma))
    else:
        lmu = np.log(mu)

    rng = np.random.RandomState(seed)

    lsigma = np.sqrt(np.log(1 + (sigma * sigma) / (mu * mu)))

    return rng.lognormal(lmu, lsigma, size)


_SIMPLEX_SEED_MAX = 2 ** 32


def _simplex_shape(size=None, shape=None):
    if shape is None:
        shape = size

    if shape is None:
        raise ValueError('size must be (H, W)')

    if isinstance(shape, (int, np.integer)):
        shape = (int(shape), int(shape))

    if len(shape) != 2:
        raise ValueError('size must be (H, W)')

    height = int(shape[0])
    width = int(shape[1])

    if height <= 0 or width <= 0:
        raise ValueError('size must be positive')

    return height, width


def _simplex_seed(seed):
    if seed is None:
        rng = np.random.RandomState()
        seed = rng.randint(0, _SIMPLEX_SEED_MAX)

    return int(seed)


def simplex(size=(1, 1), sigma=1.0, scale=64.0, octaves=4, persistence=0.5,
            lacunarity=2.0, seed=None, x0=0.0, y0=0.0, normalize=True,
            dtype=np.float32, shape=None):
    """Generate a 2D OpenSimplex fractal Brownian motion (fBm) map.

    Args:
        size: `tuple`, output shape (H, W). Alias: `shape`
        sigma: `float`, desired output standard deviation
        scale: `float`, feature size in pixels
        octaves: `int`, number of fBm octaves
        persistence: `float`, amplitude multiplier per octave
        lacunarity: `float`, frequency multiplier per octave
        seed: `int`, OpenSimplex seed
        x0: `float`, x coordinate offset in pixels
        y0: `float`, y coordinate offset in pixels
        normalize: `bool`, zero-mean and scale to std = sigma
        dtype: numpy dtype

    Returns:
        A `ndarray`, the generated map
    """
    height, width = _simplex_shape(size=size, shape=shape)

    if scale <= 0:
        raise ValueError('scale must be > 0')
    if octaves < 1:
        raise ValueError('octaves must be >= 1')
    if sigma < 0:
        raise ValueError('sigma must be >= 0')

    opensimplex.seed(_simplex_seed(seed))

    x = (np.arange(width, dtype=np.float64) + float(x0)) / float(scale)
    y = (np.arange(height, dtype=np.float64) + float(y0)) / float(scale)

    out = np.zeros((height, width), dtype=np.float64)
    amp = 1.0
    freq = 1.0
    amp_sum = 0.0

    for _ in range(int(octaves)):
        out += amp * opensimplex.noise2array(x * freq, y * freq)
        amp_sum += amp
        amp *= float(persistence)
        freq *= float(lacunarity)

    if amp_sum > 0:
        out /= amp_sum

    if normalize:
        out -= out.mean()
        s = out.std()
        if s > 0 and sigma > 0:
            out = (out / s) * float(sigma)
        else:
            out.fill(0.0)
    else:
        out *= float(sigma)

    return out.astype(dtype, copy=False)


def simplex_stripe(size=(1, 1), axis='col', sigma=1.0, scale=256.0, octaves=3,
                   persistence=0.6, lacunarity=2.0, seed=None, coord_offset=0.0,
                   const_coord=0.0, normalize=True, dtype=np.float32, shape=None):
    """Generate column or row stripe maps using OpenSimplex fBm."""
    height, width = _simplex_shape(size=size, shape=shape)

    if scale <= 0:
        raise ValueError('scale must be > 0')
    if octaves < 1:
        raise ValueError('octaves must be >= 1')
    if sigma < 0:
        raise ValueError('sigma must be >= 0')

    axis = axis.lower().strip()
    if axis not in {'col', 'row'}:
        raise ValueError("axis must be 'col' or 'row'")

    opensimplex.seed(_simplex_seed(seed))

    amp = 1.0
    freq = 1.0
    amp_sum = 0.0

    if axis == 'col':
        x = (np.arange(width, dtype=np.float64) + float(coord_offset)) / float(scale)
        y = np.array([(float(const_coord) / float(scale))], dtype=np.float64)
        line = np.zeros((width,), dtype=np.float64)

        for _ in range(int(octaves)):
            n = opensimplex.noise2array(x * freq, y * freq)[0, :]
            line += amp * n
            amp_sum += amp
            amp *= float(persistence)
            freq *= float(lacunarity)

        if amp_sum > 0:
            line /= amp_sum

        if normalize:
            line -= line.mean()
            s = line.std()
            if s > 0 and sigma > 0:
                line = (line / s) * float(sigma)
            else:
                line.fill(0.0)
        else:
            line *= float(sigma)

        out = np.tile(line[None, :], (height, 1))

    else:
        y = (np.arange(height, dtype=np.float64) + float(coord_offset)) / float(scale)
        x = np.array([(float(const_coord) / float(scale))], dtype=np.float64)
        line = np.zeros((height,), dtype=np.float64)

        for _ in range(int(octaves)):
            n = opensimplex.noise2array(x * freq, y * freq)[:, 0]
            line += amp * n
            amp_sum += amp
            amp *= float(persistence)
            freq *= float(lacunarity)

        if amp_sum > 0:
            line /= amp_sum

        if normalize:
            line -= line.mean()
            s = line.std()
            if s > 0 and sigma > 0:
                line = (line / s) * float(sigma)
            else:
                line.fill(0.0)
        else:
            line *= float(sigma)

        out = np.tile(line[:, None], (1, width))

    return out.astype(dtype, copy=False)


_CUSTOM_TYPES = {
    'simplex': simplex,
    'simplex_stripe': simplex_stripe,
    'lognormal_mu_sigma': lognormal_mu_sigma
}
