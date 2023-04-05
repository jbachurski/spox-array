import numpy as np

from spox_array._array import SpoxArray, _nested_structure, wrap


def toarray(value):
    if isinstance(value, SpoxArray):
        return value._var._get_value()  # noqa
    return np.array(value)


def assert_eq(got, expected):
    got, expected = toarray(got), toarray(expected)
    assert got.dtype.type == expected.dtype.type
    assert got.shape == expected.shape
    if issubclass(expected.dtype.type, np.floating):
        np.testing.assert_allclose(got, expected, 1e-7, 1e-7)
    else:
        np.testing.assert_equal(got, expected)


def assert_equiv_prop(fun, *args, **kwargs):
    flat_args, restructure = _nested_structure(args)
    re_args = [wrap(x) if isinstance(x, np.ndarray) else x for x in flat_args]
    got = toarray(fun(*restructure(*re_args), **kwargs))
    expected = toarray(fun(*args, **kwargs))
    assert_eq(got, expected)
