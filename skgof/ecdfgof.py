"""
Goodness-of-fit tests based on the empirical distribution function.

Interface is similar to `scipy.stats.kstest()`, but you may choose or provide
a statistic calculation routine and a matching distribution.

Three concrete tests are provided:
* `ks_test()` -- Kolmogorov-Smirnov supremum statistic; almost the same as
  `scipy.stats.kstest()` with `alternative='two-sided'`, but with (hopefully)
  somewhat more precise p-value calculation;
* `cvm_test()` -- Cramer-von Mises L2 statistic, with a rather crude estimation
  of the statistic distribution (but seemingly the best available);
* `ad_test()` -- Anderson-Darling statistic with a fair approximation of its
  distribution; unlike the "composite" `scipy.stats.anderson()` this one needs
  a fully specified hypothesized distribution.

Example::

    >>> from scipy.stats import uniform
    >>> from skgof import ks_test

    >>> ks_test((1, 2, 3), uniform(0, 4))
    GofResult(statistic=0.25, pvalue=0.97222...)

The result is accurate -- if we assume that the samples were drawn from the
specified distribution, then P(D <= 1/4) = 2/9 * (2 * 3/4 - 1)^3 = .97(2).
In general, for sample counts less than 150 you may expect good precision
with `ks_test()`, and a fair one above that.

Lectures 2 and 3 of http://www.win.tue.nl/~rmcastro/AppStat2013/ list formulas
for all three statistics. Their distributions are split into separate modules
as calculating each is a small research story -- see `ksdist`, `cvmdist`, and
`addist` for details and further references.
"""
from collections import namedtuple
from functools import partial

from numpy import arange, log, sort
from scipy._lib.six import string_types
from scipy.stats import distributions

from .addist import ad_unif
from .cvmdist import cvm_unif
from .ksdist import ks_unif

GofResult = namedtuple('GofResult', ('statistic', 'pvalue'))


def ks_stat(data):
    """
    Calculates the Kolmogorov-Smirnov statistic for sorted values from U(0, 1).
    """
    samples = len(data)
    uniform = arange(0, samples + 1) / samples
    d_plus = (uniform[1:] - data).max()
    d_minus = (data - uniform[:-1]).max()
    return max(d_plus, d_minus)


def cvm_stat(data):
    """
    Calculates the Cramer-von Mises statistic for sorted values from U(0, 1).
    """
    samples2 = 2 * len(data)
    minuends = arange(1, samples2, 2) / samples2
    return 1 / (6 * samples2) + ((minuends - data) ** 2).sum()


def ad_stat(data):
    """
    Calculates the Anderson-Darling statistic for sorted values from U(0, 1).

    The statistic is not defined if any of the values is exactly 0 or 1. You
    will get infinity as a result and a divide-by-zero warning for such values.
    The warning can be silenced or raised using numpy.errstate(divide=...).
    """
    samples = len(data)
    factors = arange(1, 2 * samples, 2)
    return -samples - (factors * log(data * (1 - data[::-1]))).sum() / samples


def simple_test(data, dist, args=(), stat=ad_stat, pdist=ad_unif,
                assume_sorted=False):
    """
    Tests goodness of fit of data to dist using a distribution-free statistic.
    """
    if isinstance(data, string_types):
        # Auto-generating samples from a named distribution is not supported.
        raise AttributeError("Data should be an array or list of values.")
    if isinstance(dist, string_types):
        dist = getattr(distributions, dist)(*args)
    elif args:
        dist = dist(args)
    if not assume_sorted:
        data = sort(data)
    statistic = stat(dist.cdf(data))
    pvalue = pdist(len(data)).sf(statistic)
    return GofResult(statistic, pvalue)

ks_test = partial(simple_test, stat=ks_stat, pdist=ks_unif)
cvm_test = partial(simple_test, stat=cvm_stat, pdist=cvm_unif)
ad_test = partial(simple_test, stat=ad_stat, pdist=ad_unif)
