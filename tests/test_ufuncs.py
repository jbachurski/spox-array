import numpy as np

from spox_array import const
from spox_array.testing import arr, assert_eq, val


def test_add_reduce():
    assert_eq(val(np.add.reduce(arr(const([1, 2, 3])), keepdims=False)), 6)
    assert_eq(val(np.add.reduce(arr(const([1, 2, 3])), keepdims=True)), [6])
