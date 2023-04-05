import operator

import numpy as np
import pytest
from spox_array.testing import assert_equiv_prop


@pytest.mark.parametrize(
    "bin_op",
    [operator.add, operator.sub, operator.mul, operator.truediv, operator.floordiv],
)
@pytest.mark.parametrize("x", [1.5, 3, 5, [[2, 3]]])
@pytest.mark.parametrize("y", [0.5, 1, 2, [[1], [1.5]]])
def test_arithmetic(bin_op, x, y):
    assert_equiv_prop(bin_op, np.array(x), np.array(y))
