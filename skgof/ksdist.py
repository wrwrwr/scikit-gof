"""
Distributions of the Kolmogorov-Smirnov supremum statistic.

After doi:10.18637/jss.v008.i18 and doi:10.18637/jss.v039.i11.
"""
from fractions import Fraction
from math import factorial, floor

from numpy import (arange, dot, exp, fmax, fromfunction, identity, log, modf,
                   pi, sqrt, tri)
from scipy.special import gamma, gammaln, smirnov
from scipy.stats import rv_continuous

from .vect import varange, vectorize


class ks_unif_gen(rv_continuous):
    """
    Approximate Kolmogorov-Smirnov two-sided, one-sample, distribution-free
    statistic (the hypothesized distribution continuous and fully specified).
    """
    def _argcheck(self, samples):
        return samples > 0

    @vectorize(otypes=(float,))
    def _cdf(self, statistic, samples):
        # Some simple, exact cases (more in Ruben & Gambino).
        if statistic <= 1 / (2 * samples):
            return 0.
        if statistic >= 1:
            return 1.
        if statistic <= 1 / samples:
            t = 2 * statistic - 1 / samples
            return exp(gammaln(samples + 1) + samples * log(t))
        if statistic >= 1 - 1 / samples:
            return 1 - 2 * (1 - statistic) ** samples

        # For small sample counts we may use an exact method when needed.
        if samples < 150:
            # With samples = 150 the matrix calculation takes about 100 ms
            # on a ~3 GFLOPS/core processor.
            if samples * statistic ** 2 < 7:
                # For a small threshold the Durbin matrix will be small.
                return ks_unif_durbin_matrix(samples, statistic)
            else:
                # Double the one-sided probability; accurate when close to one.
                return 1 - 2 * smirnov(samples, statistic)

        # Further we need to make a compromise between speed and accuracy.
        if samples < 100000 and samples * statistic ** 1.5 < 1.4:
            # The cost of the matrix calculation should still be acceptable.
            return ks_unif_durbin_matrix(samples, statistic)
        else:
            # No options left, but to use an asymptotic approximation.
            return ks_unif_pelz_good(samples, statistic)

    @vectorize(otypes=(float,))
    def _sf(self, statistic, samples):
        if statistic >= 1:
            # Statistic greater than 1 results in a NaN from Cephes smirnov().
            return 0.
        if statistic >= 1 - 1 / samples:
            # The _cdf code can suffer from some cancellation in this case.
            return min(1., 2 * (1 - statistic) ** samples)
        probability = 1 - self._cdf(statistic, samples)
        if probability > 1e-5:
            # Not too much precision got lost to cancellation.
            return probability
        else:
            # When the cdf float is very close to one it does not have bits
            # of small enough magnitude to express its 1-complement properly.
            # Hence, an approximate direct sf calculation may be more precise.
            return min(1., 2 * smirnov(samples, statistic))

ks_unif = ks_unif_gen(a=0, name='ks-unif', shapes='samples')


# Some arbitrary constants used for externalizing float exponents.
shift = 512
factor = float(2 ** shift)
factorr = float(2 ** -shift)


def ks_unif_durbin_matrix(samples, statistic):
    """
    Calculates the probability that the statistic is less than the given value,
    using a fairly accurate implementation of the Durbin's matrix formula.

    Not an exact transliteration of the Marsaglia code, but using the same
    ideas. Assumes samples > 0. See: doi:10.18637/jss.v008.i18.
    """
    # Construct the Durbin matrix.
    h, k = modf(samples * statistic)
    k = int(k)
    h = 1 - h
    m = 2 * k + 1
    A = tri(m, k=1)
    hs = h ** arange(1, m + 1)
    A[:, 0] -= hs
    A[-1] -= hs[::-1]
    if h > .5:
        A[-1, 0] += (2 * h - 1) ** m
    A /= fromfunction(lambda i, j: gamma(fmax(1, i - j + 2)), (m, m))
    # Calculate A ** n, expressed as P * 2 ** eP to avoid overflows.
    P = identity(m)
    s = samples
    eA, eP = 0, 0
    while s != 1:
        s, b = divmod(s, 2)
        if b == 1:
            P = dot(P, A)
            eP += eA
            if P[k, k] > factor:
                P /= factor
                eP += shift
        A = dot(A, A)
        eA *= 2
        if A[k, k] > factor:
            A /= factor
            eA += shift
    P = dot(P, A)
    eP += eA
    # Calculate n! / n ** n * P[k, k].
    x = P[k, k]
    for i in arange(1, samples + 1):
        x *= i / samples
        if x < factorr:
            x *= factor
            eP -= shift
    return x * 2 ** eP


def ks_unif_durbin_recurrence_rational(samples, statistic):
    """
    Calculates the probability that the statistic is less than the given value,
    using Durbin's recurrence and employing the standard fractions module.

    This is a (hopefully) exact reference implementation, likely too slow for
    practical usage. The statistic should be given as a Fraction instance and
    the result is also a Fraction. See: doi:10.18637/jss.v026.i02.
    """
    t = statistic * samples
    ft1 = floor(t) + 1
    fmt1 = floor(-t) + 1
    fdt1 = floor(2 * t) + 1
    qs = [Fraction(i ** i, factorial(i)) for i in range(ft1)]
    qs.extend(Fraction(i ** i, factorial(i)) - 2 * t *
                    sum((t + j) ** (j - 1) / factorial(j) *
                        (i - t - j) ** (i - j) / factorial(i - j)
                        for j in range(i + fmt1))
              for i in range(ft1, fdt1))
    qs.extend(-sum((-1) ** j * (2 * t - j) ** j / factorial(j) * qs[i - j]
                   for j in range(1, fdt1))
              for i in range(fdt1, samples + 1))
    return qs[samples] * factorial(samples) / samples ** samples


# Constants from the Pelz-Good approximation.
hs2 = varange(.5, 21) ** 2
ehs2 = exp(-hs2)
is2 = varange(1, 21) ** 2
pi2 = pi ** 2
pi4 = pi2 ** 2
pi6 = pi2 * pi4
hpi1d2 = sqrt(pi / 2)


def ks_unif_pelz_good(samples, statistic):
    """
    Approximates the statistic distribution by a transformed Li-Chien formula.

    This ought to be a bit more accurate than using the Kolmogorov limit, but
    should only be used with large squared sample count times statistic.
    See: doi:10.18637/jss.v039.i11 and http://www.jstor.org/stable/2985019.
    """
    x = 1 / statistic
    r2 = 1 / samples
    rx = sqrt(r2) * x
    r2x = r2 * x
    r2x2 = r2x * x
    r4x = r2x * r2
    r4x2 = r2x2 * r2
    r4x3 = r2x2 * r2x
    r5x3 = r4x2 * rx
    r5x4 = r4x3 * rx
    r6x3 = r4x2 * r2x
    r7x5 = r5x4 * r2x
    r9x6 = r7x5 * r2x
    r11x8 = r9x6 * r2x2
    a1 = rx * (-r6x3 / 108 + r4x2 / 18 - r4x / 36 - r2x / 3 + r2 / 6 + 2)
    a2 = pi2 / 3 * r5x3 * (r4x3 / 8 - r2x2 * 5 / 12 - r2x * 4 / 45 + x + 1 / 6)
    a3 = pi4 / 9 * r7x5 * (-r4x3 / 6 + r2x2 / 4 + r2x * 53 / 90 - 1 / 2)
    a4 = pi6 / 108 * r11x8 * (r2x2 / 6 - 1)
    a5 = pi2 / 18 * r5x3 * (r2x / 2 - 1)
    a6 = -pi4 * r9x6 / 108
    w = -pi2 / 2 * r2x2
    return hpi1d2 * ((a1 + (a2 + (a3 + a4 * hs2) * hs2) * hs2) * exp(w * hs2) +
                     (a5 + a6 * is2) * is2 * exp(w * is2)).sum()
