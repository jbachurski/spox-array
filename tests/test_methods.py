import numpy as np
import pytest

from spox_array.testing import assert_equiv_prop


def test_copy():
    assert_equiv_prop(lambda a: a.copy(), np.array([[1, 2, 3], [4, 5, 6]]))


def test_reshape():
    assert_equiv_prop(
        lambda a, *args: a.reshape(*args), np.array([[1, 2, 3], [4, 5, 6]]), (1, 2, 3)
    )


def test_flatten():
    assert_equiv_prop(lambda a: a.flatten(), np.array([[1, 2, 3], [4, 5, 6]]))


@pytest.mark.parametrize("reduce", ["sum", "mean", "min", "max", "prod"])
def test_reduces(reduce):
    def run(a, *args, **kwargs):
        return getattr(a, reduce)(*args, **kwargs)

    arr = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equiv_prop(run, arr)
    assert_equiv_prop(run, arr, axis=0)
    assert_equiv_prop(run, arr, axis=(0, 1))
