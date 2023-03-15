from typing import Iterable, Sequence

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import handle_out, implements, promote_args, unwrap_vars, wrap_var


@implements
@handle_out
@wrap_var
@promote_args
@unwrap_vars
def concatenate(arrays: Sequence[Var], axis: int = 0):
    return op.concat(arrays, axis=axis)


@implements
@handle_out
@wrap_var
@promote_args
@unwrap_vars
def mean(var: Var, axis: int | tuple[int, ...] | None = None, keepdims: bool = False):
    if axis is not None and not isinstance(axis, Iterable):
        axis = (axis,)
    return op.reduce_mean(var, axes=axis, keepdims=keepdims)
