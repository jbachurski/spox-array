import operator

import numpy as np

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._extops import const

INDEX_MIN: int = np.iinfo(np.int64).min
INDEX_MAX: int = np.iinfo(np.int64).max


def getitem(var: Var, index) -> Var:
    index_ = index
    shape = var.unwrap_tensor().shape
    try:
        index = operator.index(index)
    except TypeError:
        pass
    else:
        pass
    if isinstance(index, slice):
        index = (index,) + (slice(None),) * (len(shape) - 1)
    if isinstance(index, tuple):
        axis_slices = {
            d: axis_slice
            for d, axis_slice in enumerate(index)
            if isinstance(axis_slice, slice) and axis_slice != slice(None)
        }
        axis_indices = {
            d: axis_index
            for d, axis_index in enumerate(index)
            if isinstance(axis_index, int)
        }
        starts: list[int] = [
            x.start if x.start is not None else 0 for x in axis_slices.values()
        ]
        ends: list[int] = [
            x.stop
            if x.stop is not None
            else (INDEX_MAX if x.step is None or x.step > 0 else INDEX_MIN)
            for x in axis_slices.values()
        ]
        steps: list[int] = [
            x.step if x.step is not None else 1 for x in axis_slices.values()
        ]
        indexed: Var = (
            op.slice(
                var,
                const(starts),
                const(ends),
                const(list(axis_slices.keys())),
                const(steps),
            )
            if axis_slices
            else var
        )
        for axis, axis_index in sorted(axis_indices.items(), reverse=True):
            indexed = op.gather(indexed, const(axis_index), axis=axis)
        return indexed
    raise TypeError(f"Cannot index SpoxArray with {index_!r}.")


def setitem(var: Var, index, value: Var) -> Var:
    return var
