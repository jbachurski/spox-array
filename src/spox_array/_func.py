import functools
from typing import Iterable, Sequence

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import const, implements, prepare_call


def wrap_axis_singleton(obj=None, var: bool = False):
    def wrapper(fun):
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            if (
                "axis" in kwargs
                and kwargs["axis"] is not None
                and not isinstance(kwargs["axis"], Iterable)
            ):
                kwargs["axis"] = (kwargs["axis"],)
            if (
                var
                and kwargs.get("axis") is not None
                and not isinstance(kwargs["axis"], Var)
            ):
                kwargs["axis"] = const(kwargs["axis"])
            return fun(*args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper


@implements
@prepare_call
def copy(var: Var) -> Var:
    return var


@implements
@prepare_call(array_args=1)
def reshape(var: Var, shape: Iterable[int]) -> Var:
    return op.reshape(var, const(list(shape)))


@implements
@prepare_call(array_args=1)
def transpose(var: Var, axes: Iterable[int] | None = None) -> Var:
    return op.transpose(var, perm=axes)


@implements
@prepare_call(array_args=1)
def concatenate(arrays: Sequence[Var], axis: int = 0) -> Var:
    return op.concat(arrays, axis=axis)


@implements(name="sum")
@wrap_axis_singleton(var=True)
@prepare_call(array_args=1)
def sum_(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_sum(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton
@prepare_call(array_args=1, floating=True)
def mean(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_mean(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton
@prepare_call(array_args=1)
def amin(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_min(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton
@prepare_call(array_args=1)
def amax(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_max(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton
@prepare_call(array_args=1)
def prod(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_prod(var, axes=axis, keepdims=keepdims)
