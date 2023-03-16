import numpy as np

from spox_array.testing import assert_equiv_prop


def test_concat():
    assert_equiv_prop(np.concatenate, [np.array([1]), np.array([2])])
