from functools import partial

from numpy import allclose, isclose
from numpy.testing import assert_array_equal

from skgof.cvmdist import cvm_unif, cvm_unif_inf, cvm_unif_fix1

# Reset default tolerance settings not to forget to specify some later.
allclose = partial(allclose, atol=0, rtol=0)
isclose = partial(isclose, atol=0, rtol=0)

# Values from table 1 in doi:10.2307/2346175 (samples, statistic, probability).
csorgo_values = (
    # "Exact calculation" (lower) entries.
    (2, .04326, .01), (2, .04565, .025), (2, .04962, .05), (2, .05758, .1),
    (2, .06554, .15), (2, .07350, .2), (2, .08146, .25), (2, .12659, .5),
    (2, .21522, .75), (2, .24743, .8), (2, .28854, .85), (2, .34343, .9),
    (2, .42480, .95), (2, .48901, .975), (2, .55058, .99), (2, .62858, .999),
    (5, .02869, .01), (5, .03422, .025), (5, .04036, .05), (5, .04969, .1),
    (5, .05800, .15), (5, .06610, .2), (5, .07427, .25), (5, .12250, .5),
    (5, .21164, .75), (5, .24237, .8), (5, .28305, .85), (5, .34238, .9),
    (5, .44697, .95), (5, .55056, .975), (5, .68352, .99), (5, .9873, .999),
    (10, .0265, .01), (10, .03212, .025), (10, .0384, .05), (10, .0478, .1),
    (10, .0561, .15), (10, .0641, .2), (10, .0721, .25), (10, .1207, .5),
    (10, .2104, .75), (10, .2417, .8), (10, .2836, .85), (10, .3450, .9),
    (10, .4542, .95), (10, .5659, .975), (10, .7147, .99), (10, 1.0822, .999),
    # Single-term linking approximation.
    (20, .02564, .01), (20, .03120, .025), (20, .03742, .05),
    (20, .04689, .1), (20, .05515, .15), (20, .06312, .2),
    (20, .07117, .25), (20, .11978, .5), (20, .20989, .75),
    (20, .24148, .8), (20, .28384, .85), (20, .34617, .9),
    (20, .45778, .95), (20, .57331, .975), (20, .72895, .99),
    (20, 1.11898, .999),
    (50, .02512, .01), (50, .03068, .025), (50, .03690, .05),
    (50, .04636, .1), (50, .05462, .15), (50, .06258, .2),
    (50, .07062, .25), (50, .11924, .5), (50, .20958, .75),
    (50, .24132, .8), (50, .28396, .85), (50, .34682, .9),
    (50, .45986, .95), (50, .57754, .975), (50, .73728, .99),
    (50, 1.14507, .999),
    (200, .02488, .01), (200, .03043, .025), (200, .03665, .05),
    (200, .04610, .1), (200, .05435, .15), (200, .06231, .2),
    (200, .07035, .25), (200, .11897, .5), (200, .20943, .75),
    (200, .24125, .8), (200, .28402, .85), (200, .34715, .9),
    (200, .46091, .95), (200, .57968, .975), (200, .74149, .99),
    (200, 1.15783, .999),
    (1000, .02481, .01), (1000, .03037, .025), (1000, .03658, .05),
    (1000, .04603, .1), (1000, .05428, .15), (1000, .06224, .2),
    (1000, .07027, .25), (1000, .11889, .5), (1000, .20938, .75),
    (1000, .24123, .8), (1000, .28403, .85), (1000, .34724, .9),
    (1000, .46119, .95), (1000, .58026, .975), (1000, .74262, .99),
    (1000, 1.16204, .999))


class UnifTests:
    def test_cdf(self):
        for sp, st, pr in csorgo_values:
            # The first batch of values comes from an extensive simulation,
            # the implemented approximation is less accurate.
            rtol = .05 if sp <= 10 else .001
            assert isclose(cvm_unif.cdf(st, sp), pr, rtol=rtol)

    def test_special(self):
        # Basic bounds.
        assert_array_equal(cvm_unif(5).cdf((-1, 0, .01, 1 / 60)), (0,) * 4)
        assert_array_equal(cvm_unif(9).cdf((3, 9, 12, 1000)), (1,) * 4)
        # Equation 2.4: (4! * pi ** 2) / 2! * (x - 1 / 48) ** 2.
        assert isclose(cvm_unif(4).cdf(1 / 47), .000023270343861, rtol=.5e-10)
        assert isclose(cvm_unif(4).cdf(7 / 197), .025591495453354, rtol=.5e-10)
        # Equation 2.4: (100! * pi ** 50) / 50! * (1 / (5 * 100 ** 2)) ** 50.
        assert isclose(cvm_unif(100).cdf(1 / 1200 + 1 / 50000),
                                            .248841150238417e-116, rtol=.5e-10)

    def test_inf(self):
        # A few values from the Csorgo and Faraway table 1.
        ps = [cvm_unif_inf(s) for s in (.0248, .06222, .11888, .24124, .74346)]
        assert allclose(ps, (.01, .2, .5, .8, .99), rtol=.5e-3)

        # Compared with R code from "goftest" package (the same approximation).
        ps = [cvm_unif_inf(s) for s in (.1, .5, 2)]
        assert allclose(ps, (.4151266, .9601668, .9999872), rtol=.5e-6)

    def test_fix1(self):
        # Again compared with the R code.
        cs = [cvm_unif_fix1(s) for s in (.1, .5, 2)]
        assert allclose(cs, (-.09126438, .02137429, .0001864369), rtol=.5e-6)
