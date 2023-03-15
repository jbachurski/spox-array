import operator
from typing import cast

import numpy as np
import pytest

from spox import Var
from spox_array import SpoxArray, const


def arr(var: Var) -> np.ndarray:
    return cast(np.ndarray, SpoxArray(var))  # type: ignore


def val(array: np.ndarray):
    if isinstance(array, SpoxArray):
        return array._var._get_value()  # noqa
    return np.asarray(array)


def assert_eq(got, expected):
    expected = np.array(expected)
    assert got.dtype == expected.dtype
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected)


@pytest.mark.parametrize(
    "bin_op",
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
)
@pytest.mark.parametrize("x", [1.5, 3, 5, [[2, 3]]])
@pytest.mark.parametrize("y", [0.5, 1, 2, [[1], [1.5]]])
def test_arithmetic(bin_op, x, y):
    x, y = np.array(x), np.array(y)
    assert_eq(val(bin_op(arr(const(x)), arr(const(y)))), bin_op(x, y))


def test_add_reduce():
    assert val(np.add.reduce(arr(const([1, 2, 3])), keepdims=False)) == 6
    assert val(np.add.reduce(arr(const([1, 2, 3])), keepdims=True)) == [6]


def test_concat():
    np.testing.assert_allclose(
        val(np.concatenate([arr(const([1])), arr(const([2]))])), [1, 2]
    )
