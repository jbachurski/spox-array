import numpy as np

from spox_array import const
from spox_array.testing import arr, assert_eq, val


def col_loss(a: np.ndarray) -> np.ndarray:
    return np.mean((a[:, 0] - a[:, 1]) ** 2)


def test_col_loss():
    a = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 6.0]])
    assert_eq(val(col_loss(arr(const(a)))), col_loss(a))
