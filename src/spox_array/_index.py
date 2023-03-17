import operator

import numpy as np
import numpy.typing as npt

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._extops import const

INDEX_MIN: int = np.iinfo(np.int64).min
INDEX_MAX: int = np.iinfo(np.int64).max


def _int_or_none(x) -> bool:
    return x is None or isinstance(x, int)


def normalize_index(
    index, rank: int
) -> Var | tuple[dict[tuple[int, int, int], slice], dict[int, int]]:
    index_ = index
    if isinstance(index, (list, np.ndarray)):
        index = const(index)
    if isinstance(index, Var):
        return index
    try:
        index = operator.index(index)
    except TypeError:
        pass
    else:
        pass
    if isinstance(index, (int, slice)):
        index = (index,) + (slice(None),) * (rank - 1)
    if isinstance(index, tuple):
        if not all(isinstance(d, (int, slice)) for d in index):
            raise TypeError(
                f"Bad inferred axis-index types in {index!r} from {index_!r}"
            )
        if not all(
            all(map(_int_or_none, (d.start, d.stop, d.step)))
            for d in index
            if isinstance(d, slice)
        ):
            raise TypeError(
                f"Bad inferred axis-slice types in {index!r} from {index_!r}"
            )
        if any(d.step == 0 for d in index if isinstance(d, slice)):
            raise TypeError(f"Zero-step slice in {index!r} from {index_!r}")
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
        return axis_slices, axis_indices
    raise TypeError(f"Cannot index with {index_!r} (transformed to {index!r}).")


def getitem(var: Var, index_) -> Var:
    index = normalize_index(index_, len(var.unwrap_tensor().shape))
    if isinstance(index, Var):
        index_dtype = index.unwrap_tensor().dtype
        if index_dtype == np.dtype(bool):
            return op.compress(var, index, axis=0)
        elif np.issubdtype(index_dtype, np.integer):
            return op.gather(var, index, axis=0)
        else:
            raise TypeError(
                f"Unsupported index array dtype {index_dtype} (from {index_!r})."
            )
    if isinstance(index, tuple):
        axis_slices, axis_indices = index
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
    raise TypeError(f"Cannot index with {index_!r}.")


def setitem(var: Var, index_, updates_: Var | npt.ArrayLike) -> Var:
    index = normalize_index(index_, len(var.unwrap_tensor().shape))
    updates = const(updates_) if not isinstance(updates_, Var) else updates_
    # updates_shape = updates.unwrap_tensor().shape
    shape = var.unwrap_tensor().shape
    shape_var = op.shape(var)
    dim_indices = [op.gather(shape_var, const(d)) for d in range(len(shape))]
    indices = None
    if isinstance(index, Var):
        index_dtype = index.unwrap_tensor().dtype
        index_shape = index.unwrap_tensor().shape
        if len(index_shape) != 1:
            raise TypeError("Update mask must be a vector.")
        if index_dtype == np.dtype(bool):
            initial_dim = dim_indices[0]
            indices = getitem(op.range(const(0), initial_dim, const(1)), index)
        elif np.issubdtype(index_dtype, np.integer):
            indices = index
        else:
            raise TypeError(
                f"Unsupported index array dtype {index_dtype} (from {index_!r})."
            )
    elif isinstance(index, tuple):
        indices = None
    return op.scatter_nd(var, indices, updates)
