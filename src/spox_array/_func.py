from typing import Sequence

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._array import handle_out, implements, promote_args, wrap_var


@implements
@handle_out
@wrap_var
@promote_args
def concatenate(arrays: Sequence[Var], axis: int = 0):
    return op.concat(arrays, axis=axis)
