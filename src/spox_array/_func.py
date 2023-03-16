from typing import Iterable, Sequence

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import implements, prepare_call


@implements
@prepare_call
def concatenate(arrays: Sequence[Var], axis: int = 0) -> Var:
    return op.concat(arrays, axis=axis)


@implements
@prepare_call
def transpose(var: Var, axes: Iterable[int] | None = None) -> Var:
    return op.transpose(var, perm=axes)


@implements
@prepare_call(floating=True)
def mean(
    var: Var, axis: int | Iterable[int] | None = None, keepdims: bool = False
) -> Var:
    if axis is not None and not isinstance(axis, Iterable):
        axis = (axis,)
    return op.reduce_mean(var, axes=axis, keepdims=keepdims)
