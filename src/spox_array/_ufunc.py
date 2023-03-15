import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import const, handle_out, implements, promote_args, unwrap_vars, wrap_var


@implements(method="__call__")
@handle_out
@wrap_var
@promote_args
@unwrap_vars
def add(x: Var, y: Var):
    return op.add(x, y)


@implements(name="add", method="reduce")
@handle_out
@wrap_var
@promote_args
@unwrap_vars
def add_reduce(x: Var, axis: int = 0, keepdims: bool = False):
    return op.reduce_sum(x, axes=const([axis]), keepdims=keepdims)
