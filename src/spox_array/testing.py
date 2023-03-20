from typing import cast

import numpy as np

from spox import Var
from spox_array._array import SpoxArray, _nested_structure


def arr(var: Var) -> np.ndarray:
    return cast(np.ndarray, SpoxArray(var))  # type: ignore


def val(array: np.ndarray):
    if isinstance(array, SpoxArray):
        return array._var._get_value()  # noqa
    return np.asarray(array)


def assert_eq(got, expected):
    expected = np.array(expected)
    assert got.dtype.type == expected.dtype.type
    assert got.shape == expected.shape
    if issubclass(expected.dtype.type, np.floating):
        np.testing.assert_allclose(got, expected, 1e-7, 1e-7)
    else:
        np.testing.assert_equal(got, expected)


def assert_equiv_prop(fun, *args, **kwargs):
    flat_args, restructure = _nested_structure(args)
    re_args = [arr(x) if isinstance(x, np.ndarray) else x for x in flat_args]
    got = val(fun(*restructure(*re_args), **kwargs))
    expected = val(fun(*args, **kwargs))
    assert_eq(got, expected)
