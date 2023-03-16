import functools

from ._array import SpoxArray, _nested_structure, promote_args


def handle_out(fun):
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        out = kwargs.pop("out", None)
        result: SpoxArray = fun(*args, **kwargs)
        if out is not None:
            if not isinstance(out, SpoxArray):
                raise TypeError(
                    f"Output for SpoxArrays must also be written to one, not {type(out).__name__}."
                )
            out.__var__(result.__var__())
            return out
        return result

    return inner


def var_wrapper(fun):
    @functools.wraps(fun)
    def inner(*args, **kwargs):
        flat_args, restructure = _nested_structure(args)
        re_args = restructure(
            *(arg.__var__() if isinstance(arg, SpoxArray) else arg for arg in flat_args)
        )
        return SpoxArray(fun(*re_args, **kwargs))

    return inner


def prepare_call(obj=None, *, array_args: int | None = None, floating: int = 0):
    def wrapper(fun):
        @handle_out
        @promote_args(array_args=array_args, floating=floating)
        @var_wrapper
        @functools.wraps(fun)
        def inner(*args, **kwargs):
            return fun(*args, **kwargs)

        return inner

    return wrapper(obj) if obj is not None else wrapper
