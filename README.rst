==========
scikit-gof
==========

Provides variants of Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling
goodness of fit tests for fully specified continuous distributions.

Example
=======

.. code:: python

    >>> from scipy.stats import norm, uniform
    >>> from skgof import ks_test, cvm_test, ad_test

    >>> ks_test((1, 2, 3), uniform(0, 4))
    GofResult(statistic=0.25, pvalue=0.97...)

    >>> cvm_test((1, 2, 3), uniform(0, 4))
    GofResult(statistic=0.04..., pvalue=0.95...)

    >>> data = norm(0, 1).rvs(random_state=1, size=100)
    >>> ad_test(data, norm(0, 1))
    GofResult(statistic=0.75..., pvalue=0.51...)
    >>> ad_test(data, norm(.3, 1))
    GofResult(statistic=3.52..., pvalue=0.01...)

Simple tests
============

Scikit-gof currently only offers three nonparametric tests that let you
compare a sample with a reference probability distribution. These are:

``ks_test()``
    Kolmogorov-Smirnov supremum statistic; almost the same as
    ``scipy.stats.kstest()`` with ``alternative='two-sided'`` but with
    (hopefully) somewhat more precise p-value calculation;

``cvm_test()``
    Cramer-von Mises L2 statistic, with a rather crude estimation of the
    statistic distribution (but seemingly the best available);

``ad_test()``
    Anderson-Darling statistic with a fair approximation of its distribution;
    unlike the composite ``scipy.stats.anderson()`` this one needs a fully
    specified hypothesized distribution.

Simple test functions use a common interface, taking as the first argument the
data (sample) to be compared and as the second argument a frozen ``scipy.stats``
distribution.
They return a named tuple with two fields: ``statistic`` and ``pvalue``.

For a simple example consider the hypothesis that the sample (.4, .1, .7) comes
from the uniform distribution on [0, 1]:

.. code:: python

    if ks_test((.4, .1, .7), unif(0, 1)).pvalue < .05:
        print("Hypothesis rejected with 5% significance.")

If your samples are very large and you have them sorted ahead of time, pass
``assume_sorted=True`` to save some time that would be wasted resorting.

Extending
=========

Simple tests are composed of two phases: calculating the test statistic and
determining how likely is the resulting value (under the hypothesis).
New tests may be defined by providing a new statistic calculation routine or an
alternative distribution for a statistic.

Functions calculating statistics are given evaluations of the reference
cumulative distribution function on sorted data and are expected to return
a single number.
For a simple test, if the sample indeed comes from the hypothesized (continuous)
distribution, the values passed to the function should be uniformly distributed
over [0, 1].

Here is a simplistic example of how a statistic function might look like:

.. code:: python

    def ex_stat(data):
        return abs(data.sum() - data.size / 2)

Statistic functions for the provided tests, ``ks_stat()``, ``cvm_stat()``,
and ``ad_stat()``, can be imported from ``skgof.ecdfgof``.

Statistic distributions should derive from ``rv_continuous`` and implement
at least one of the abstract ``_cdf()`` or ``_pdf()`` methods (you might
also consider directly coding ``_sf()`` for increased precision of results
close to 1). For example:

.. code:: python

    from numpy import sqrt
    from scipy.stats import norm, rv_continuous

    class ex_unif_gen(rv_continuous):
        def _cdf(self, statistic, samples):
            return 1 - 2 * norm.cdf(-statistic, scale=sqrt(samples / 12))

    ex_unif = ex_unif_gen(a=0, name='ex-unif', shapes='samples')

The provided distributions live in separate modules, respectively ``ksdist``,
``cvmdist``, and ``addist``.

Once you have a statistic calculation function and a statistic distribution the
two parts can be combined using ``simple_test``:

.. code:: python

    from functools import partial
    from skgof.ecdfgof import simple_test

    ex_test = partial(simple_test, stat=ex_stat, pdist=ex_unif)

**Exercise**: The example test has a fundamental flaw. Can you point it out?

..  The test is not consistent under all alternatives. For instance, if the
    hypothesis was that samples come from the uniform distribution on [0, 1],
    but they really were "drawn" from the degenerate distribution at .5, the
    test would never notice, even for arbitrarily large sample sizes.

    Moreover, the asymptotic distribution is not a good approximation of the
    actual statistic distribution for small sample sizes.

Installation
============

.. code:: bash

    pip install scikit-gof

Requires recent versions of Python (> 3), NumPy (>= 1.10) and SciPy.

Please fix or point out any errors, inaccuracies or typos you notice.
