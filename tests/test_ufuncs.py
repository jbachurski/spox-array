import numpy as np
import pytest

from spox_array.testing import assert_equiv_prop

DTYPES = [
    np.uint32,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.float16,
    np.int32,
    np.int64,
]


def test_add_reduce():
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=False)
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=True)


@pytest.mark.parametrize("dtype1", DTYPES)
@pytest.mark.parametrize("dtype2", DTYPES)
def test_div(dtype1, dtype2):
    x, y = np.ones((3,), dtype1), np.ones((3,), dtype2)
    assert_equiv_prop(np.divide, x, y)


@pytest.mark.parametrize("dtype", DTYPES)
def test_sin(dtype):
    x = np.ones((3,), dtype)
    assert_equiv_prop(np.sin, x)
