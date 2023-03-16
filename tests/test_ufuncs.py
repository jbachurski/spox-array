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
COMMON_DTYPES = [np.int32, np.int64, np.float32, np.float64]

UNARY_UFUNCS = [
    "isnan",
    "absolute",
    "sign",
    "ceil",
    "floor",
    "sqrt",
    "log",
    "exp",
]

TRIG_UFUNCS = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "sinh",
    "cosh",
    "tanh",
    "arcsinh",
    "arccosh",
    "arctanh",
]

BINARY_UFUNCS = ["add", "subtract", "multiply", "divide", "floor_divide", "power"]


def test_add_reduce():
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=False)
    assert_equiv_prop(np.add.reduce, np.array([1, 2, 3]), keepdims=True)


@pytest.mark.parametrize("name", BINARY_UFUNCS)
@pytest.mark.parametrize("dtype1", COMMON_DTYPES)
@pytest.mark.parametrize("dtype2", COMMON_DTYPES)
def test_binary_ufunc(name, dtype1, dtype2):
    x = np.arange(1, 3, 0.5, dtype=dtype1)
    y = np.arange(3, 5, 0.5, dtype=dtype2)
    assert_equiv_prop(getattr(np, name), x, y)


@pytest.mark.parametrize("name", UNARY_UFUNCS)
@pytest.mark.parametrize("dtype", COMMON_DTYPES)
def test_unary_ufunc(name, dtype):
    x = np.arange(-2.1, 2.1, 0.66, dtype=dtype)
    assert_equiv_prop(getattr(np, name), x)


@pytest.mark.parametrize("name", TRIG_UFUNCS)
@pytest.mark.parametrize("dtype", [np.float32])
def test_trig_ufunc(name, dtype):
    x = np.arange(-2.1, 2.1, 0.66, dtype=dtype)
    assert_equiv_prop(getattr(np, name), x)


@pytest.mark.parametrize("name", TRIG_UFUNCS)
@pytest.mark.parametrize("dtype", [np.float32])
def test_recip(name, dtype):
    x = np.arange(0.5, 5, 0.5, dtype=dtype)
    assert_equiv_prop(np.reciprocal, x)
