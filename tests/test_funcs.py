import numpy as np

from spox_array import const
from spox_array.testing import arr, assert_eq, val


def test_concat():
    assert_eq(val(np.concatenate([arr(const([1])), arr(const([2]))])), [1, 2])
