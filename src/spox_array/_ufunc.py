import functools

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import (
    const,
    handle_out,
    implements,
    promote_args,
    promote_args_floating,
    unwrap_vars,
    wrap_var,
)


def binary_ufunc_call(fun):
    @implements(method="__call__")
    @handle_out
    @wrap_var
    @promote_args
    @unwrap_vars
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        return fun(*args, **kwargs)

    return inner


def binary_ufunc_call_floating(fun):
    @implements(method="__call__")
    @handle_out
    @wrap_var
    @promote_args_floating
    @unwrap_vars
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        return fun(*args, **kwargs)

    return inner


@binary_ufunc_call
def add(x: Var, y: Var):
    return op.add(x, y)


@binary_ufunc_call
def subtract(x: Var, y: Var):
    return op.sub(x, y)


@binary_ufunc_call
def multiply(x: Var, y: Var):
    return op.mul(x, y)


@binary_ufunc_call_floating
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


@binary_ufunc_call
def mod(x: Var, y: Var):
    return op.pow(x, y)


@implements(name="add", method="reduce")
@handle_out
@wrap_var
@promote_args
@unwrap_vars
def add_reduce(x: Var, axis: int = 0, keepdims: bool = False):
    return op.reduce_sum(x, axes=const([axis]), keepdims=keepdims)
