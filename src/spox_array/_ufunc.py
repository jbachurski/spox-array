import functools

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import const, implements
from ._impl import prepare_call


def prepare_ufunc_call(obj=None, *, floating: int = 0):
    def wrapper(fun):
        @implements(method="__call__")
        @prepare_call(floating=floating)
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            return fun(*args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper


# Arithmetic


@prepare_ufunc_call
def add(x: Var, y: Var) -> Var:
    return op.add(x, y)


@prepare_ufunc_call
def subtract(x: Var, y: Var) -> Var:
    return op.sub(x, y)


@prepare_ufunc_call
def multiply(x: Var, y: Var) -> Var:
    return op.mul(x, y)


@prepare_ufunc_call(floating=2)
def divide(x: Var, y: Var) -> Var:
    return op.div(x, y)


@prepare_ufunc_call
def floor_divide(x: Var, y: Var) -> Var:
    if issubclass(x.unwrap_tensor().dtype.type, np.floating):
        return op.floor(op.div(x, y))
    return op.div(x, y)


@prepare_ufunc_call
def power(x: Var, y: Var) -> Var:
    return op.pow(x, y)


# Unary


@prepare_ufunc_call
def isnan(x: Var) -> Var:
    if not np.issubdtype(x.unwrap_tensor().dtype, np.floating):
        return op.expand(const(False), op.shape(x))
    return op.isnan(x)


# Unary - signs


@prepare_ufunc_call
def absolute(x: Var) -> Var:
    return op.abs(x)


@prepare_ufunc_call
def sign(x: Var) -> Var:
    return op.sign(x)


# Unary - rounding


@prepare_ufunc_call(floating=1)
def ceil(x: Var) -> Var:
    return op.ceil(x)


@prepare_ufunc_call(floating=1)
def floor(x: Var) -> Var:
    return op.floor(x)


# Unary - special-case exponents


@prepare_ufunc_call
def reciprocal(x: Var) -> Var:
    return op.reciprocal(x)


@prepare_ufunc_call(floating=1)
def sqrt(x: Var) -> Var:
    return op.sqrt(x)


# Unary - exponential


@prepare_ufunc_call(floating=1)
def log(x: Var) -> Var:
    return op.log(x)


@prepare_ufunc_call(floating=1)
def exp(x: Var) -> Var:
    return op.exp(x)


# Trigonometric


@prepare_ufunc_call(floating=1)
def sin(x: Var) -> Var:
    return op.sin(x)


@prepare_ufunc_call(floating=1)
def cos(x: Var) -> Var:
    return op.cos(x)


@prepare_ufunc_call(floating=1)
def tan(x: Var) -> Var:
    return op.tan(x)


# Inverse trigonometric


@prepare_ufunc_call(floating=1)
def arcsin(x: Var) -> Var:
    return op.asin(x)


@prepare_ufunc_call(floating=1)
def arccos(x: Var) -> Var:
    return op.acos(x)


@prepare_ufunc_call(floating=1)
def arctan(x: Var) -> Var:
    return op.atan(x)


# Hyperbolic


@prepare_ufunc_call(floating=1)
def sinh(x: Var) -> Var:
    return op.sinh(x)


@prepare_ufunc_call(floating=1)
def cosh(x: Var) -> Var:
    return op.cosh(x)


@prepare_ufunc_call(floating=1)
def tanh(x: Var) -> Var:
    return op.tanh(x)


# Inverse hyperbolic


@prepare_ufunc_call(floating=1)
def arcsinh(x: Var) -> Var:
    return op.asinh(x)


@prepare_ufunc_call(floating=1)
def arccosh(x: Var) -> Var:
    return op.acosh(x)


@prepare_ufunc_call(floating=1)
def arctanh(x: Var) -> Var:
    return op.atanh(x)


# Reduce


@implements(name="add", method="reduce")
@prepare_call
def add_reduce(x: Var, axis: int = 0, keepdims: bool = False):
    return op.reduce_sum(x, axes=const([axis]), keepdims=keepdims)
