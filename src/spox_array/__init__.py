import numpy as np
import numpy.typing as npt

from spox import Var

from . import _func, _ufunc, testing  # noqa: Register implementations
from ._array import SpoxArray, const


def wrap(var: Var) -> np.ndarray:
    """Wrap a Var in a SpoxArray and cast the type-hint to `numpy.ndarray`."""
    return SpoxArray(var)


def unwrap(arr: npt.ArrayLike | SpoxArray) -> Var:
    if isinstance(arr, SpoxArray):
        return arr.__var__()
    return const(arr)


__all__ = ["SpoxArray", "const", "testing", "wrap", "unwrap"]
