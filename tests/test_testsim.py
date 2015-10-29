from numpy.testing import assert_array_equal

from skgof.testsim import simulator


class SimulatorTests:
    def test_signature(self):
        assert_array_equal(simulator(lambda data: 8, 3, 10, 10), [8] * 9)

    def test_sorting(self):
        i = 9

        def statistic(data):
            nonlocal i
            i -= 1
            return i
        assert_array_equal(simulator(statistic, 3, 10, 10), range(9))
