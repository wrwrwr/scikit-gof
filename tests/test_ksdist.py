from fractions import Fraction as F
from functools import partial
from math import sqrt

from numpy import allclose, isclose
import pytest

from ksref import (exact_values, almost_exact_values, marsaglia_values,
                   simard_values, simard_pelz_values, brown_values,
                   oconnor_values, oconnor_asymptotic_values)
from skgof.ksdist import (ks_unif, ks_unif_durbin_matrix, ks_unif_pelz_good,
                          ks_unif_durbin_recurrence_rational)

# Should we execute bigger cases (about 10 times slower than the "small" ones).
slow = pytest.config.getoption('--slow')

# We will specify the precision for each case separately.
allclose = partial(allclose, atol=0, rtol=0)
isclose = partial(isclose, atol=0, rtol=0)


class UnifTests:
    def test_cdf(self):
        # All cases should be quick and accurate to at least a few digits.
        cdf = lambda sp, st: ks_unif.cdf(st, sp)
        for sp, st, pr in oconnor_values:
            assert isclose(cdf(sp, st), pr, rtol=.5e-4)
        for sp, st, pr in brown_values:
            assert isclose(cdf(sp, st), pr, rtol=.5e-5)
        rtol = lambda sp: .5e-12 if sp < 100 else .5e-5
        for sp, st, pr in simard_values:
            assert isclose(cdf(sp, st), pr, rtol=rtol(sp))
        for sp, st, pr in marsaglia_values:
            assert isclose(cdf(sp, st), pr, rtol=rtol(sp))
        for sp, st, pr, prc in almost_exact_values:
            assert isclose(cdf(sp, st), pr, rtol=rtol(sp))
        for sp, st, pr in exact_values:
            assert isclose(cdf(sp, float(st)), float(pr), rtol=rtol(sp))

    def test_sf(self):
        # We do not have an exact method for the survival probability, so
        # some precision is lost in some cases due to subtracting from 1.
        sf = lambda sp, st: ks_unif.sf(st, sp)
        rtol = lambda sp: .5e-11 if sp < 100 else .5e-4
        for sp, st, pr, prc in almost_exact_values:
            assert isclose(sf(sp, st), prc, rtol=rtol(sp))
        for sp, st, pr in exact_values:
            assert isclose(sf(sp, float(st)), float(1 - pr), rtol=rtol(sp))

        # Some values that R used to get wrong, from doi:10.18637/jss.v039.i11.
        assert isclose(ks_unif(120).sf(.0874483967333), .30012, rtol=.5e-4)
        assert isclose(ks_unif(500).sf(.037527424), .47067, rtol=.5e-4)

        # Some values that MATLAB used to get wrong; after the same paper.
        assert isclose(ks_unif(20).sf(.8008915818), 2.5754e-14, rtol=.5e-4)
        assert isclose(ks_unif(20).sf(.9004583223), 1.8250e-20, rtol=.5e-4)

    def test_durbin_matrix(self):
        # Compare with the reference method for small sample counts.
        cdf = ks_unif_durbin_matrix
        small = lambda sp: (slow and sp < 50) or sp < 25
        for sp, st, pr, prc in almost_exact_values:
            if small(sp):
                assert isclose(cdf(sp, st), pr, atol=.5e-50, rtol=.5e-12)

        # And with the Marsaglia code for somewhat larger counts.
        for sp, st, pr in marsaglia_values[:81 if slow else 54]:
            assert isclose(cdf(sp, st), pr, rtol=.5e-14)

        # And also with the (approximate) Simard code for still larger counts.
        for sp, st, pr in simard_values[:45 if slow else 27]:
            assert isclose(cdf(sp, st), pr, rtol=.5e-5)

        # Only-11-digits of accuracy example from doi:10.18637/jss.v026.i02.
        assert isclose(cdf(16000, .0107438), .9506002390950460063, rtol=.5e-12)

        # NaNs in the Marsaglia code, after doi:10.18637/jss.v039.i11.
        assert isclose(cdf(11000, .0004135), .11746e-264, rtol=.5e-4)
        assert isclose(cdf(21001, .000480), .17917e-105, rtol=.5e-4)
        if slow:
            assert isclose(cdf(42001, .0002345), .12181e-222, rtol=.5e-4)
            assert isclose(cdf(62000, .004), .72640, rtol=.5e-4)

    def test_durbin_recurrence_rational(self):
        # Test the reference using external and hand-calculated values.
        cdf = ks_unif_durbin_recurrence_rational
        small = lambda sp: (slow and sp < 40) or sp < 20
        for sp, st, pr in oconnor_values:
            if small(sp):
                assert abs(cdf(sp, F.from_float(st)) - pr) / pr <= F('.5e-4')
        for sp, st, pr in brown_values:
            if small(sp):
                assert abs(cdf(sp, F.from_float(st)) - pr) / pr <= F('.5e-5')
        for sp, st, pr in simard_values:
            if small(sp):
                assert abs(cdf(sp, F.from_float(st)) - pr) / pr <= F('.5e-13')
        for sp, st, pr in marsaglia_values:
            if small(sp):
                assert abs(cdf(sp, F.from_float(st)) - pr) / pr <= F('.5e-13')
        for sp, st, pr in exact_values:
            if small(sp):
                # Reference is supposed to compute an exact result, with
                # inaccuracies resulting only from the limited precison
                # of the stored test values.
                assert cdf(sp, st) == pr

    def test_pelz_good(self):
        # Just about 4 digits of accuracy, and only in the lower left-tail.
        cdf = ks_unif_pelz_good
        for sp, st, pr in oconnor_asymptotic_values:
            assert isclose(cdf(sp, st), pr, rtol=.5e-2)
        for sp, st, pr in simard_pelz_values:
            assert isclose(cdf(sp, st), pr, rtol=.5e-12)
        for sp, st, pr in marsaglia_values:
            # The approximation is only supposed to work well in a tail.
            if sp * st ** 1.5 > 2:
                assert isclose(cdf(sp, st), pr, rtol=.5e-4)

        # A few values from the Pelz & Good 1976 paper.
        assert isclose(cdf(50000, .2 / sqrt(50000)), .63126e-12, rtol=.5e-3)
        assert isclose(cdf(50000, 1 / sqrt(50000)), .73080, rtol=.5e-3)
        assert isclose(cdf(50000, 2.2 / sqrt(50000)), .99988, rtol=.5e-3)
        assert isclose(cdf(100000, .2 / sqrt(100000)), .59181e-12, rtol=.5e-3)
        assert isclose(cdf(100000, 1 / sqrt(100000)), .73056, rtol=.5e-3)
        assert isclose(cdf(100000, 2.2 / sqrt(100000)), .99988, rtol=.5e-3)
