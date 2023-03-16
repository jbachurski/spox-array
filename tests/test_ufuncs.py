import numpy as np

from spox_array.testing import assert_equiv_prop


def test_add_reduce():
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=False)
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=True)
