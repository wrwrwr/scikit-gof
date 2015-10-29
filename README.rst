==========
scikit-gof
==========

Provides variants of Kolmogorov-Smirnov, Cramer-von Mises and Anderson-Darling
goodness of fit tests for fully specified continuous distributions.

The Kolmogorov-Smirnov statistic distribution is (hopefully) somewhat more
precise compared to what SciPy has to offer at the time of writing.

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

Installation
============

.. code:: bash

    pip install scikit-gof

Requires recent versions of Python (> 3), NumPy (>= 1.10) and SciPy.
