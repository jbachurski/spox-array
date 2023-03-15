from typing import cast

import numpy as np

from spox import Var
from spox_array import SpoxArray, const


def arr(var: Var) -> np.ndarray:
    return cast(np.ndarray, SpoxArray(var))  # type: ignore


def val(array: np.ndarray):
    if isinstance(array, SpoxArray):
        return array._var._get_value()  # noqa
    return np.asarray(array)


def test_add():
    assert val(arr(const(1.0)) + arr(const(2.0))) == 3
    assert val(arr(const(1.0)) + 2.0) == 3
    assert val(2.0 + arr(const(1.0))) == 3
    np.testing.assert_allclose(val(arr(const(1.0)) + np.array([2.0, -1.0])), [3, 0])


def test_add_reduce():
    assert val(np.add.reduce(arr(const([1, 2, 3])), keepdims=False)) == 6
    assert val(np.add.reduce(arr(const([1, 2, 3])), keepdims=True)) == [6]


def test_concat():
    np.testing.assert_allclose(
        val(np.concatenate([arr(const([1])), arr(const([2]))])), [1, 2]
    )
