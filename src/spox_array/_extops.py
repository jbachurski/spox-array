import numpy as np
import numpy.typing as npt

import spox.opset.ai.onnx.v17 as op
from spox import Var


def const(value: npt.ArrayLike, dtype: npt.DTypeLike = None) -> Var:
    return op.constant(value=np.array(value, dtype))
