# spox-array

## Description

`spox-array` is a proof-of-concept implementation of an
[array API](https://data-apis.org/array-api/2022.12/index.html)
compatible with `numpy`.

It allows you to pass a special wrapper object, `SpoxArray`,
into functions like [these](https://github.com/jbachurski/spox-array/blob/main/tests/test_examples.py) seamlessly:

```py
import numpy as np
from spox_array import wrap, unwrap
from spox import argument, build, Tensor, Var

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
# For example, built the above into a model:
model = build({'x': x, 'y': y}, {'f': f, 'g': g})
```

Since SpoxArray performs a normal Spox construction, propagated values
(eager execution) may be viewed, and all intermediate values are typed.

## Features

Only some `numpy` features are implemented:

- Numpy-compatible type promotion (in most cases)
- A good part of the ufuncs (arithmetic, trigonometry) with operator overloading – as defined in `_ufunc.py`
- Index access (`__getitem__`)
- Most common array methods (like `.shape`, `.T`, `.astype`, ...)
- Various numpy functions – as defined in `_func.py`

As SpoxArray tries to be compatible with the `numpy` dispatcher,
the API is kept closer to `numpy` than the Array API standard.

## Development

Install `requirements.txt` and use `pre-commit install` to install the pre commit hooks
and `pre-commit run --all-files` to run them manually.

`pip install -e .` and `pytest tests` should be enough to run the test suite.

When attempting to use Spox value propagation, it's recommended to
use ONNX Runtime for the backend (due to issues in `onnx.reference`):

```py
import logging
import spox._future

logging.getLogger().setLevel(logging.DEBUG)
spox._future.set_value_prop_backend(spox._future.ValuePropBackend.ONNXRUNTIME)
```
