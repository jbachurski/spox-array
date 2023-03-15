import numpy as np
from spox_array import SpoxArray as Arr, const


def val(array: np.ndarray):
    if isinstance(array, Arr):
        return array._var._get_value()  # noqa
    return np.asarray(array)


def test_add():
    assert val(Arr(const(1.0)) + Arr(const(2.0))) == 3
    assert val(Arr(const(1.0)) + 2.0) == 3
    np.testing.assert_allclose(
        val(Arr(const(1.0)) + np.array([2.0, -1.0])),
        [3, 0]
    )


def test_concat():
    np.testing.assert_allclose(
        val(np.concatenate([Arr(const([1])), Arr(const([2]))])),
        [1, 2]
    )

