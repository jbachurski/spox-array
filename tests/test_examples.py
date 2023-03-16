import numpy as np

from spox_array import const
from spox_array.testing import arr, assert_eq, val


def col_loss(a: np.ndarray) -> np.ndarray:
    return np.mean((a[:, 0] - a[:, 1]) ** 2)


def test_col_loss():
    a = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 6.0]])
    assert_eq(val(col_loss(arr(const(a)))), col_loss(a))


def diff_loss(a: np.ndarray) -> np.ndarray:
    return (0.5 + (a[:-1] - a[1:]) ** 2).mean(dtype=np.float16)


def test_diff_loss():
    a = np.array([[1.0, 1.0], [3.0, 1.0], [5.0, 6.0]])
    assert_eq(val(diff_loss(arr(const(a)))), diff_loss(a))
