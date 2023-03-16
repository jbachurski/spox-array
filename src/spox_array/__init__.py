from typing import cast

import numpy as np
import numpy.typing as npt

from spox import Var

from . import _func, _ufunc, testing  # noqa: Register implementations
from ._array import SpoxArray, const


def wrap(value: npt.ArrayLike | Var | SpoxArray) -> np.ndarray:
    """Wrap a Var in a SpoxArray and cast the type-hint to `numpy.ndarray`."""
    return cast(np.ndarray, SpoxArray(value))


def unwrap(array: npt.ArrayLike | Var | SpoxArray) -> Var:
    if isinstance(array, SpoxArray):
        return array.__var__()
    elif isinstance(array, Var):
        return array
    return const(array)


__all__ = ["SpoxArray", "const", "testing", "wrap", "unwrap"]
