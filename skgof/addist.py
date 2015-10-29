"""
Distributions of the Anderson-Darling statistic.

After doi:18637/jss.v009.i02.
"""
from numpy import exp, log, sqrt
from scipy.stats import rv_continuous

from .vect import vectorize


class ad_unif_gen(rv_continuous):
    """
    Approximate distribution of the uniform Anderson-Darling statistic
    (with the hypothesized distribution continuous and fully specified).
    """
    def _argcheck(self, samples):
        return samples > 0

    @vectorize(otypes=(float,))
    def _cdf(self, statistic, samples):
        if samples == 1:
            # Exact distribution for a single sample (a bit more precise than
            # the approximation). See doi:10.1214/aoms/1177704850 equation 8.
            if statistic <= log(4) - 1:
                return 0.
            else:
                return sqrt(1 - 4 * exp(-1 - statistic))
        pinf = ad_unif_inf(statistic)
        return pinf + ad_unif_fix(samples, pinf)

ad_unif = ad_unif_gen(a=0, name='ad-unif', shapes='samples')


def ad_unif_inf(statistic):
    """
    Approximates the limiting distribution to about 5 decimal digits.
    """
    z = statistic
    if z < 2:
        return (exp(-1.2337141 / z) / sqrt(z) *
                                (2.00012 + (.247105 - (.0649821 - (.0347962 -
                                (.011672 - .00168691 * z) * z) * z) * z) * z))
    else:
        return exp(-exp(1.0776 - (2.30695 - (.43424 - (.082433 -
                                (.008056 - .0003146 * z) * z) * z) * z) * z))


g1 = lambda x: sqrt(x) * (1 - x) * (49 * x - 102)
g2 = lambda x: (-.00022633 + (6.54034 - (14.6538 - (14.458 -
                                (8.259 - 1.91864 * x) * x) * x) * x) * x)
g3 = lambda x: (-130.2137 + (745.2337 - (1705.091 - (1950.646 -
                                (1116.36 - 255.7844 * x) * x) * x) * x) * x)


def ad_unif_fix(samples, pinf):
    """
    Corrects the limiting distribution for a finite sample size.
    """
    n = samples
    c = .01265 + .1757 / n
    if pinf < c:
        return (((.0037 / n + .00078) / n + .00006) / n) * g1(pinf / c)
    elif pinf < .8:
        return ((.01365 / n + .04213) / n) * g2((pinf - c) / (.8 - c))
    else:
        return g3(pinf) / n
