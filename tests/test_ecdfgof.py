from collections import namedtuple
from functools import partial

from numpy import allclose, array, isclose, linspace
from scipy.stats import norm, uniform
from pytest import mark

from skgof.ecdfgof import (ad_stat, ad_test, cvm_stat, cvm_test,
                           ks_stat, ks_test, ecdfgof_test)

data1 = array((.125, .375, .625, .875))
data2 = array((.1, .2, .3, .4))
data3 = array((.6, .7, .8, .9))

# Hard-coded results precision.
allclose = partial(allclose, atol=0, rtol=.5e-5)
isclose = partial(isclose, atol=0, rtol=.5e-5)


class StatisticTests:
    def test_ks_stat(self):
        assert isclose(ks_stat(data1), .125)
        assert isclose(ks_stat(data2), .6)
        assert isclose(ks_stat(data3), .6)

    def test_cvm_stat(self):
        assert isclose(cvm_stat(data1), .0208333)
        assert isclose(cvm_stat(data2), .383333)
        assert isclose(cvm_stat(data3), .383333)

    def test_ad_stat(self):
        assert isclose(ad_stat(data1), .153334)
        assert isclose(ad_stat(data2), 1.749722)
        assert isclose(ad_stat(data3), 1.749722)


class TestTests:
    def test_basic(self):
        # Custom statistic calculation / distribution and object-like result.
        stat = lambda data: 3
        pdist = lambda samples: namedtuple('FD', 'sf')(sf=lambda statistic: .5)
        result = ecdfgof_test((1, 2, 3), norm, stat=stat, pdist=pdist)
        assert result.statistic == 3 and result.pvalue == .5

    def test_args(self):
        # It should be possible to specify a frozen distribution or a name and
        # parameters.
        assert ecdfgof_test((.2, .5, .8), uniform(1, 2)).pvalue == 0
        assert ecdfgof_test((.2, .5, .8), 'uniform', args=(1, 2)).pvalue == 0

    def test_ks_test(self):
        # Compared with R ks.test().
        result = ks_test((.1, .4, .7), uniform(0, 1))
        assert allclose(result, (.3, .886222))
        result = ks_test((.1, .4, .7), norm(0, 1))
        assert allclose(result, (.539828, .246999))

        # Some of the SciPy's kstest() cases ('norm' without arguments stands
        # for the normal distribution with mean 0 and standard deviation 1).
        result = ks_test(linspace(-1, 1, 9), 'norm')
        assert allclose(result, (.158655, .951641))
        result = ks_test(linspace(-15, 15, 9), 'norm')
        assert allclose(result, (.444356, .038850))
        data = norm.rvs(loc=.2, random_state=987654321, size=100)
        result = ks_test(data, 'norm')
        assert allclose(result, (.124643, .081973))
        result = ks_test((-.1, 1, 2), norm(1, 3))
        assert allclose(result, (.369441, .679447))

    def test_cvm_test(self):
        # Compared with cvm.test() from R goftest package.
        result = cvm_test((.1, .4, .7), uniform(0, 1))
        assert allclose(result, (.06, .851737))
        result = cvm_test((.1, .4, .7), norm(0, 1))
        assert allclose(result, (.196853, .281709))

    def test_ad_test(self):
        # Versus ad.test() from R goftest.
        result = ad_test((.1, .4, .7), uniform(0, 1))
        assert allclose(result, (.366028, .875957))
        result = ad_test((.1, .4, .7), norm(0, 1))
        assert allclose(result, (.921699, .390938))

        # Poles of the weight function.
        result = ad_test((0., .5), uniform(0, 1))
        assert allclose(result, (float('inf'), 0))
        result = ad_test((1., .5), uniform(0, 1))
        assert allclose(result, (float('inf'), 0))


class TestBenchmarks:
    @mark.benchmark(group='ks-test-small')
    def benchmark_ks_test_small(self, benchmark):
        result = benchmark(ks_test, linspace(0, 1, 1e3), uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='ks-test-small')
    def benchmark_ks_test_small_sorted(self, benchmark):
        result = benchmark(ks_test, linspace(0, 1, 1e3), uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='cvm-test-small')
    def benchmark_cvm_test_small(self, benchmark):
        result = benchmark(cvm_test, linspace(0, 1, 1e3), uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='cvm-test-small')
    def benchmark_cvm_test_small_sorted(self, benchmark):
        result = benchmark(cvm_test, linspace(0, 1, 1e3), uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='ad-test-small')
    def benchmark_ad_test_small(self, benchmark):
        result = benchmark(ad_test, linspace(0, 1, 1e3)[1:-1], uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='ad-test-small')
    def benchmark_ad_test_small_sorted(self, benchmark):
        result = benchmark(ad_test, linspace(0, 1, 1e3)[1:-1], uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-2)

    @mark.benchmark(group='ks-test-large')
    def benchmark_ks_test_large(self, benchmark):
        result = benchmark(ks_test, linspace(0, 1, 1e6), uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-4)

    @mark.benchmark(group='ks-test-large')
    def benchmark_ks_test_large_sorted(self, benchmark):
        result = benchmark(ks_test, linspace(0, 1, 1e6), uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-4)

    @mark.benchmark(group='cvm-test-large')
    def benchmark_cvm_test_large(self, benchmark):
        result = benchmark(cvm_test, linspace(0, 1, 1e6), uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-4)

    @mark.benchmark(group='cvm-test-large')
    def benchmark_cvm_test_large_sorted(self, benchmark):
        result = benchmark(cvm_test, linspace(0, 1, 1e6), uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-4)

    @mark.benchmark(group='ad-test-large')
    def benchmark_ad_test_large(self, benchmark):
        result = benchmark(ad_test, linspace(0, 1, 1e6)[1:-1], uniform(0, 1))
        assert allclose(result, (0., 1.), atol=.5e-4)

    @mark.benchmark(group='ad-test-large')
    def benchmark_ad_test_large_sorted(self, benchmark):
        result = benchmark(ad_test, linspace(0, 1, 1e6)[1:-1], uniform(0, 1),
                           assume_sorted=True)
        assert allclose(result, (0., 1.), atol=.5e-4)
