import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import implements, promote_args, wrap_var


@implements(method="__call__")
@wrap_var
@promote_args
def add(x: Var, y: Var):
    return op.add(x, y)
