import functools

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import const, implements, prepare_call


def binary_ufunc_call(obj=None, *, floating=False):
    def wrapper(fun):
        @implements(method="__call__")
        @prepare_call(floating=floating)
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            return fun(*args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper


@binary_ufunc_call
def add(x: Var, y: Var):
    return op.add(x, y)


@binary_ufunc_call
def subtract(x: Var, y: Var):
    return op.sub(x, y)


@binary_ufunc_call
def multiply(x: Var, y: Var):
    return op.mul(x, y)


@binary_ufunc_call(floating=True)
def divide(x: Var, y: Var):
    return op.div(x, y)


@binary_ufunc_call
def floor_divide(x: Var, y: Var):
    if issubclass(x.unwrap_tensor().dtype.type, np.floating):
        return op.floor(op.div(x, y))
    return op.div(x, y)


@binary_ufunc_call
def power(x: Var, y: Var):
    return op.pow(x, y)


@implements(name="add", method="reduce")
@prepare_call
def add_reduce(x: Var, axis: int = 0, keepdims: bool = False):
    return op.reduce_sum(x, axes=const([axis]), keepdims=keepdims)
