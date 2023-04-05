from typing import Any, ClassVar

import numpy as np


class NumpyDispatchMixin(np.lib.mixins.NDArrayOperatorsMixin):
    _ufunc_handlers: ClassVar[dict[str, dict[str, Any]]] = {}
    _function_handlers: ClassVar[dict[str, Any]] = {}

    @classmethod
    def implements_numpy(
        cls,
        target=None,
        *,
        name: str | None = None,
        method: str | None = None,
    ):
        def decorator(fun):
            nonlocal name
            if name is None:
                name = fun.__name__
            if method is not None:
                cls._ufunc_handlers.setdefault(name, {})[method] = fun
            else:
                cls._function_handlers[name] = fun
            return fun

        return decorator if target is None else decorator(target)

    def __array_ufunc__(self, ufunc: np.ufunc, method: str, *inputs, **kwargs):
        if (
            ufunc.__name__ in self._ufunc_handlers
            and method in self._ufunc_handlers[ufunc.__name__]
        ):
            return self._ufunc_handlers[ufunc.__name__][method](*inputs, **kwargs)
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if set(types) != {type(self)}:
            return NotImplemented
        if func.__name__ in self._function_handlers:
            return self._function_handlers[func.__name__](*args, **kwargs)
        return NotImplemented
