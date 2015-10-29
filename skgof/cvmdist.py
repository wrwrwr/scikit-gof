"""
Distributions of the Cramer-von Mises statistic.

After doi:10.2307/2346175.
"""
from numpy import arange, dot, exp, newaxis, pi, tensordot
from scipy.special import gamma, kv
from scipy.stats import rv_continuous

from .vect import varange, vectorize


class cvm_unif_gen(rv_continuous):
    """
    Approximate Cramer-von Mises statistic distribution for uniform data
    (with the hypothesized distribution continuous and fully specified).
    """
    def _argcheck(self, samples):
        return samples > 0

    @vectorize(otypes=(float,))
    def _cdf(self, statistic, samples):
        low = 1 / (12 * samples)
        # Basic bounds.
        if statistic <= low:
            return 0.
        if statistic >= samples / 3:
            return 1.
        # From the geometric approach of Csorgo and Faraway (equation 2.4).
        if statistic <= low + 1 / (4 * samples ** 2):
            return (gamma(samples + 1) / gamma(samples / 2 + 1) *
                                    (pi * (statistic - low)) ** (samples / 2))
        # Asymptotic distribution with a one-term correction (equation 1.8).
        return cvm_unif_inf(statistic) + cvm_unif_fix1(statistic) / samples

cvm_unif = cvm_unif_gen(a=0, name='cvm-unif', shapes='samples')


inf_ks41 = 4 * arange(11) + 1
inf_args = inf_ks41 ** 2 / 16
inf_cs = (inf_ks41 ** .5 * gamma(varange(.5, 11)) /
                                        (pi ** 1.5 * gamma(varange(1, 11))))


def cvm_unif_inf(statistic):
    """
    Calculates the limiting distribution of the Cramer-von Mises statistic.

    After the second line of equation 1.3 from the Csorgo and Faraway paper.
    """
    args = inf_args / statistic
    return (inf_cs * exp(-args) * kv(.25, args)).sum() / statistic ** .5


fix1_args = (4 * (varange((.5, 1., 1.5), 21)) - 1) ** 2 / 16
fix1_dens = 72 * pi ** 1.5 * gamma(varange(1, 21))
fix1_csa = fix1_args ** .75 * gamma(varange(1.5, 21)) / fix1_dens
fix1_csb = fix1_args ** 1.25 * gamma(varange((.5, 1.5, 2.5), 21)) / fix1_dens


def cvm_unif_fix1(statistic):
    """
    Approximates the first-term of the small sample count Gotze expansion.

    After equation 1.10 (with coefficients pulled out as csa / csb).
    """
    args = fix1_args / statistic
    kvs = kv((.25, .75, 1.25), args[:, :, newaxis])
    gs, hs = exp(-args) * tensordot(((1, 1, 0), (2, 3, -1)), kvs, axes=(1, 2))
    a = dot((7, 16, 7), fix1_csa * gs).sum() / statistic ** 1.5
    b = dot((1, 0, 24), fix1_csb * hs).sum() / statistic ** 2.5
    return cvm_unif_inf(statistic) / 12 - a - b
