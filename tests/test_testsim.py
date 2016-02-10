from numpy.testing import assert_array_equal

from skgof.testsim import simulator


class SimulatorTests:
    def test_signature(self):

        def stat(data):
            return 8

        assert_array_equal(simulator(stat, 3, 10, 10), [8] * 9)
        assert_array_equal(simulator(stat, 3, 10, 1e1), [8] * 9)

    def test_sorting(self):
        i = 9

        def stat(data):
            nonlocal i
            i -= 1
            return i

        assert_array_equal(simulator(stat, 3, 10, 10), range(9))
