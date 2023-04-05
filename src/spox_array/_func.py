import functools
from typing import Any, Iterable, Sequence

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import SpoxArray, implements, promote, to_var
from ._impl import handle_out, prepare_call


def wrap_axis_singleton(obj=None, *, i: int | None = None, var: bool = False):
    def wrapper(fun):
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            axis: Any = None
            if "axis" in kwargs:
                axis = kwargs["axis"]
            elif i is not None and i < len(args):
                axis = args[i]
            if axis is not None and not isinstance(axis, Iterable):
                axis = (axis,)
            if var and axis is not None and not isinstance(axis, Var):
                axis = op.const(axis)  # type: ignore
            if axis is not None:
                if i is not None and i < len(args):
                    args = args[:i] + (axis,) + args[i + 1 :]
                    kwargs.pop("axis", None)
                else:
                    kwargs["axis"] = axis
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
    return op.reshape(var, op.const(list(shape)))


@implements
@prepare_call(array_args=1)
def transpose(var: Var, axes: Iterable[int] | None = None) -> Var:
    return op.transpose(var, perm=axes)


@implements
@prepare_call(array_args=1)
def concatenate(arrays: Sequence[Var], axis: int = 0) -> Var:
    return op.concat(arrays, axis=axis)


@implements(name="sum")
@wrap_axis_singleton(i=1, var=True)
@prepare_call(array_args=1)
def sum_(var: Var, axis: Var | None = None, keepdims: bool = False) -> Var:
    return op.reduce_sum(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton(i=1)
@prepare_call(array_args=1, floating=1)
def mean(var: Var, axis: Iterable[int] | None = None, keepdims: bool = False) -> Var:
    return op.reduce_mean(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton(i=1)
@prepare_call(array_args=1)
def amin(var: Var, axis: Iterable[int] | None = None, keepdims: bool = False) -> Var:
    return op.reduce_min(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton(i=1)
@prepare_call(array_args=1)
def amax(var: Var, axis: Iterable[int] | None = None, keepdims: bool = False) -> Var:
    return op.reduce_max(var, axes=axis, keepdims=keepdims)


@implements
@wrap_axis_singleton(i=1)
@prepare_call(array_args=1)
def prod(var: Var, axis: Iterable[int] | None = None, keepdims: bool = False) -> Var:
    return op.reduce_prod(var, axes=axis, keepdims=keepdims)


@implements
@prepare_call(array_args=1)
def argmin(var: Var, axis: int | None = None, keepdims: bool = False) -> Var:
    if axis is None:
        var = op.reshape(var, op.const([-1]))
        axis = 0
    return op.arg_min(var, axis=axis, keepdims=keepdims)


@implements
@prepare_call(array_args=1)
def argmax(var: Var, axis: int | None = None, keepdims: bool = False) -> Var:
    if axis is None:
        var = op.reshape(var, op.const([-1]))
        axis = 0
    return op.arg_max(var, axis=axis, keepdims=keepdims)


@implements
@prepare_call(floating=1)
def round_(x: Var) -> Var:
    return op.round(x)


@implements
@prepare_call(floating=1)
def around(x: Var) -> Var:
    return op.round(x)


@implements
@prepare_call(array_args=3)
def clip(x: Var, a: Var | None, b: Var | None) -> Var:
    return op.clip(x, a, b)


@implements
def where(x, a, b) -> SpoxArray:
    return SpoxArray(op.where(to_var(x), *promote(to_var(a), to_var(b))))


@implements
@wrap_axis_singleton(i=1, var=True)
@prepare_call(array_args=1)
def expand_dims(a: Var, axis: Var) -> Var:
    return op.unsqueeze(a, axis)


@implements
@wrap_axis_singleton(i=1, var=True)
@prepare_call(array_args=1)
def squeeze(a: Var, axis: Var | None = None) -> Var:
    return op.squeeze(a, axis)


@implements
@prepare_call(array_args=1)
def hstack(vs: Sequence[Var]) -> Var:
    arrays = [SpoxArray(v) for v in vs]
    rank = min(a.ndim for a in arrays)
    return to_var(np.concatenate(arrays, axis=1 if rank > 1 else 0))


@implements
@prepare_call(array_args=1)
def vstack(vs: Sequence[Var]) -> Var:
    arrays = [SpoxArray(v) for v in vs]
    rank = min(a.ndim for a in arrays)
    if rank == 1:
        arrays = [np.expand_dims(a, 0) for a in arrays]
    return to_var(np.concatenate(arrays, axis=0))


@implements
@handle_out
def compress(condition, a, axis: int | None = None) -> SpoxArray:
    return SpoxArray(op.compress(to_var(a), to_var(condition), axis=axis))


@implements
@prepare_call(array_args=1)
def cumsum(a: Var, axis: int | None = None) -> Var:
    if axis is None:
        a = op.reshape(a, op.const([-1]))
        axis = 0
    return op.cum_sum(a, op.const(axis))


@implements
def einsum(subscripts: str, *args, optimize: bool = False) -> SpoxArray:
    _ = optimize
    return SpoxArray(op.einsum([to_var(a) for a in args], equation=subscripts))
