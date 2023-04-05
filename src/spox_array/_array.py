import functools
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import numpy.lib.mixins
import numpy.typing as npt

import spox.opset.ai.onnx.v17 as op
from spox import Var

from ._extops import const
from ._index import getitem, setitem
from ._numpy_dispatch import NumpyDispatchMixin


class SpoxArray(NumpyDispatchMixin):
    _var: Var

    def __init__(self, obj: npt.ArrayLike | Var | "SpoxArray"):
        if isinstance(obj, Var):
            var = obj
        elif isinstance(obj, SpoxArray):
            var = obj.__var__()
        else:
            var = op.const(obj)
        self.__var__(var)
        if self._var.unwrap_tensor().shape is None:
            raise TypeError("Rank of a SpoxArray must be known.")

    def __var__(self, var: Var | None = None) -> Var:
        if var is not None:
            self._var = var
        return self._var

    @property
    def dtype(self) -> np.dtype:
        return self._var.unwrap_tensor().dtype

    @property
    def shape(self) -> tuple[int | str | None, ...]:
        return self._var.unwrap_tensor().shape

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int | None:
        r = 1
        for x in self.shape:
            if not isinstance(x, int):
                return None
            r *= x
        return r

    def __repr__(self):
        return f"{self.__class__.__name__}({self._var})"

    def __getitem__(self, index) -> "SpoxArray":
        return SpoxArray(getitem(self.__var__(), index))

    def __setitem__(self, index, value) -> None:
        if isinstance(value, SpoxArray):
            value = value.__var__()
        var, value = promote(self.__var__(), value)
        self.__var__(setitem(var, index, value))

    @property
    def T(self) -> "SpoxArray":
        return SpoxArray(np.transpose(self))

    def copy(self) -> "SpoxArray":
        return SpoxArray(self)

    def astype(self, dtype: npt.DTypeLike) -> "SpoxArray":
        return type(self)(op.cast(self.__var__(), to=dtype))

    def reshape(self, shape, **kwargs) -> "SpoxArray":
        return SpoxArray(np.reshape(self, shape, **kwargs))

    def flatten(self) -> "SpoxArray":
        return self.reshape((-1,))

    def sum(self, **kwargs) -> "SpoxArray":
        return SpoxArray(np.sum(self, **kwargs))

    def mean(self, **kwargs) -> "SpoxArray":
        return SpoxArray(np.mean(self, **kwargs))

    def min(self, **kwargs) -> "SpoxArray":
        return SpoxArray(np.min(self, **kwargs))

    def max(self, **kwargs) -> "SpoxArray":
        return SpoxArray(np.max(self, **kwargs))

    def prod(self, **kwargs) -> "SpoxArray":
        return SpoxArray(np.prod(self, **kwargs))


implements = SpoxArray.implements_numpy


def to_var(
    x: Var | SpoxArray | npt.ArrayLike,
    dtype: npt.DTypeLike | None = None,
    *,
    casting: str | None = None,
) -> Var:
    if isinstance(x, SpoxArray):
        x = x.__var__()
    if isinstance(x, Var):
        x_dtype = x.unwrap_tensor().dtype
        if casting is not None and not np.can_cast(x_dtype, dtype, casting):
            raise TypeError(f"Cannot cast {x_dtype} to {dtype} with {casting=}.")
        return x if dtype is None or x_dtype == dtype else op.cast(x, to=dtype)
    if casting is not None and not np.can_cast(x, dtype, casting):
        raise TypeError(f"Cannot cast {x} to {dtype} with {casting=}.")
    return const(x, dtype)


def promote(
    *args: Var | SpoxArray | npt.ArrayLike | None,
    floating: int = 0,
    casting: str = "same_kind",
    dtype: Any = None,
) -> Sequence[SpoxArray | None]:
    if not args:
        return ()
    if dtype is None:
        on_args = (a for a in args if a is not None)
        if floating <= 0:
            target_type = result_type(*on_args)
        elif floating <= 1:
            target_type = result_type(*on_args, np.float16)
        elif floating <= 2:
            target_type = result_type(*on_args)
            target_type = np.common_type(np.array([], target_type))
        else:
            raise ValueError(f"Bad flag for floating: {floating}.")
    else:
        target_type = dtype

    def _promote_target(obj: Var | npt.ArrayLike | None) -> Optional[Var]:
        if obj is None:
            return None
        return to_var(obj, target_type, casting=casting)

    return tuple(var for var in map(_promote_target, args))


def _nested_structure(xs):
    if not isinstance(xs, Iterable) or isinstance(xs, (str, numpy.ndarray)):
        return [xs], lambda x: x
    sub = [_nested_structure(x) for x in xs]
    flat: list[Any] = sum((chunk for chunk, _ in sub), [])

    def restructure(*args):
        i = 0
        result = []
        for ch, re in sub:
            n = len(ch)
            result.append(re(*args[i : i + n]))
            i += n
        return result

    return flat, restructure


def promote_args(obj=None, *, array_args: int | None = None, floating: int = 0):
    def wrapper(fun):
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            pref = args[:array_args] if array_args is not None else args
            suff = args[array_args:] if array_args is not None else ()
            flat_args, restructure = _nested_structure(pref)
            promoted_args = promote(
                *flat_args,
                casting=kwargs.pop("casting", None),
                dtype=kwargs.pop("dtype", None),
                floating=floating,
            )
            re_args = tuple(restructure(*promoted_args)) + suff
            return fun(*re_args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper


@implements
def result_type(*args):
    targets: list[np.dtype | npt.ArrayLike] = [
        x.__var__().unwrap_tensor().dtype
        if isinstance(x, SpoxArray)
        else (x.unwrap_tensor().dtype if isinstance(x, Var) else x)
        for x in args
    ]
    return np.dtype(np.result_type(*targets))
