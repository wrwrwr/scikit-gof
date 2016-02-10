"""
A primitive statistic distribution simulation.
"""
from numpy import fromiter
from numpy.random import random_sample


def simulator(stat, samples, precision, rounds):
    """
    Simulates a distribution-free statistical test to estimate its p-values.

    The first argument should be a function that computes the simulated
    statistic for a vector of ordered, uniform(0, 1) samples.

    The second argument, samples, tells how many samples to generate and test
    with. To generate a typical p-values table you would run the function for
    a number of consecutive sample counts.

    The third argument, precision, decides how many critical values to return.
    For example with precision = 100 you will get 99 values; approximately, a
    statistic above value indexed 94 will have a probability lower than .05.

    The fourth argument, rounds, tells us how many times to repeat the data
    generation and statistic calculation. The more rounds the higher the
    quality of the results. Must be a (large) multiple of precision.

    Example::

        from skgof.testsim import simulator
        from skgof.ecdfgof import ks_stat

        # Repeat the simulation one million times for vectors of 10 samples.
        ks10 = simulator(ks_stat, 10, 100, 1e6)

        # Get the approximate 95% critical value (to about 2 decimal digits).
        ks10[94]  # 0.409...
    """
    rounds = int(rounds)
    data = random_sample(size=(rounds, samples))
    data.sort(axis=1)
    stats = fromiter((stat(d) for d in data), float, rounds)
    stats.sort()
    step = int(rounds / precision)
    return stats[step:rounds:step]
