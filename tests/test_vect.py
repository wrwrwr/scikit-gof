from __future__ import division

from numpy import arange
from numpy.testing import assert_array_equal

from skgof.vect import varange, vectorize


class VectorizeTests:
    def test_function(self):
        @vectorize
        def f(x):
            return x + 1

        assert_array_equal(f((1, 2, 3)), (2, 3, 4))

    def test_method(self):
        class A:
            @vectorize
            def a(self, x):
                return x + 1

        assert_array_equal(A().a((1, 2, 3)), (2, 3, 4))

    def test_arguments(self):
        class A:
            @vectorize(excluded=('a',), otypes=(int,))
            def a(self, x, a):
                return x + a

        assert_array_equal(A().a((1, 2, 3), 2), (3, 4, 5))

    def test_doc(self):
        class A:
            @vectorize
            def a(self, x):
                """A docstring."""
                pass

        class B:
            @vectorize(doc="Another docstring.")
            def b(self, x):
                """A docstring."""
                pass

        assert A.a.__doc__ == """A docstring."""
        assert B.b.__doc__ == """Another docstring."""


class VarangeTests:
    def test_single(self):
        assert_array_equal(varange(.5, 4), arange(.5, 4))
        assert_array_equal(varange(1.5, 4), arange(1.5, 5))

    def test_sequence(self):
        a = varange((.5, 1., 1.5), 4)
        assert_array_equal(a, [arange(.5, 4), arange(1., 5), arange(1.5, 5)])
