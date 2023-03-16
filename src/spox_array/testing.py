from typing import cast

import numpy as np

from spox import Var
from spox_array._array import SpoxArray, _nested_structure, const


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


def assert_equiv_prop(fun, *args, **kwargs):
    flat_args, restructure = _nested_structure(args)
    re_args = [arr(const(x)) if isinstance(x, np.ndarray) else x for x in flat_args]
    got = val(fun(*restructure(*re_args), **kwargs))
    expected = fun(*args, **kwargs)
    assert_eq(got, expected)
