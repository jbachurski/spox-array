from typing import Iterable, Sequence

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import const, implements, prepare_call


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


@implements
@prepare_call(array_args=1, floating=True)
def mean(
    var: Var, axis: int | Iterable[int] | None = None, keepdims: bool = False
) -> Var:
    if axis is not None and not isinstance(axis, Iterable):
        axis = (axis,)
    return op.reduce_mean(var, axes=axis, keepdims=keepdims)
