# spox-array

## Description

`spox-array` is a proof-of-concept implementation of an
[array API](https://data-apis.org/array-api/2022.12/index.html)
compatible with `numpy`.

It allows you to pass a special wrapper object, `SpoxArray`, into functions like these seamlessly:

```py
import numpy as np
from spox_array import wrap, unwrap
from spox import argument, Tensor, Var

# SpoxArray implements a np.ndarray (though it isn't a subclass)
# and can be passed to these
def col_loss(a: np.ndarray) -> np.ndarray:
    return np.mean((a[:, 0] - a[:, 1]) ** 2)


def diff_loss(a: np.ndarray) -> np.ndarray:
    return np.mean(0.5 + (a[:-1] - a[1:]) ** 2)

x: Var = argument(Tensor(float, ('N', 2)))
# We wrap the argument with the array interface, and then extract the Var.
# wrap(Var) -> np.ndarray  (SpoxArray under the hood)
# unwrap(SpoxArray) -> Var
# (the exact syntax for wrapping and unwrapping is preliminary)
f: Var = unwrap(col_loss(wrap(x)))

y: Var = argument(Tensor(float, ('M',)))
g: Var = unwrap(diff_loss(wrap(y)))

# Do whatever you would like in Spox with the Vars f and g!
```

## Features

Currently, only a few features are implemented:

- Basic arithmetic (ufunc calls)
- Index access (`__getitem__`)
- Numpy functions: `np.mean`, `np.concatenate`.

## Development

Install `requirements.txt` and use `pre-commit install` to install the pre commit hooks

- and `pre-commit run --all-files` to run them manually.
