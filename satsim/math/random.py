from __future__ import division, print_function, absolute_import

import numpy as np


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
    """A wrapper function to generate a numpy random number using the random
    number generator (RNG) function name. Useful for specifying random samples
    based on a configuration files.

    Examples::
        x = gen_sample(**{"type":"normal", "loc":0.0, "scale":3.0})
        y = gen_sample(**{"type":"uniform", "low":-5.0, "high":5.0})

    Args:
        type: `string`, numpy.random RNG name (see numpy.random docs)
        seed: `int`, rng seed
        negate: `float`, probability between 0 and 1 of negating the sample
        kargs: `dict`, key value args for the RNG

    Returns:
        A `float`, the sample generated from the specified RNG
    """
    rng = np.random.RandomState(seed)

    invert_sign = False
    if negate > 0 and np.random.uniform(low=0.0, high=1.0) < negate:
        invert_sign = True

    f = getattr(rng, type)

    if invert_sign:
        return -f(**kwargs)
    else:
        return f(**kwargs)


def lognormal(mu=1.0, sigma=1.0, size=1, mu_mode='median', seed=None):
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
    if mu_mode == 'mean':
        lmu = np.log(sigma * sigma / np.sqrt(mu * mu + sigma * sigma))
    else:
        lmu = np.log(mu)

    rng = np.random.RandomState(seed)

    lsigma = np.sqrt(np.log(1 + (sigma * sigma) / (mu * mu)))

    return rng.lognormal(lmu, lsigma, size)
