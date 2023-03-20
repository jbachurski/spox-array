import numpy as np
import pytest

from spox_array.testing import assert_equiv_prop

ARRAY = np.array([[1, 3], [2, 6], [3, 9]])

SLICES = [
    slice(None),
    1,
    slice(1, 2),
    slice(None, 3),
    slice(None, None, 2),
    slice(1, None, 2),
]


@pytest.mark.parametrize("m1", [False, True])
@pytest.mark.parametrize("m2", [False, True])
@pytest.mark.parametrize("m3", [False, True])
def test_get_boolean_mask(m1, m2, m3):
    assert_equiv_prop(lambda a: a[[m1, m2, m3]], ARRAY)


@pytest.mark.parametrize(
    "pi", [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
)
def test_get_integer_mask(pi):
    assert_equiv_prop(lambda a: a[pi], ARRAY)


@pytest.mark.parametrize("s", SLICES)
def test_get_slices_one(s):
    assert_equiv_prop(lambda a: a[s], ARRAY)


@pytest.mark.parametrize("s1", SLICES)
@pytest.mark.parametrize("s2", SLICES)
def test_get_slices_single(s1, s2):
    assert_equiv_prop(lambda a: a[s1, s2], ARRAY)


def test_get_slices_syntax():
    assert_equiv_prop(lambda a: a[::2, 0], ARRAY)


@pytest.mark.skip()
def test_set_boolean_mask():
    def run(a):
        a = a.copy()
        a[[False, True, True]] = a[:2]
        return a

    assert_equiv_prop(run, ARRAY[..., np.newaxis])


@pytest.mark.skip()
def test_set_integer_mask():
    def run(a):
        a = a.copy()
        a[[2, 1]] = a[:2]
        return a

    assert_equiv_prop(run, ARRAY[..., np.newaxis])


@pytest.mark.skip()
def test_set_slices():
    def run(a):
        a = a.copy()
        a[1:] = a[:2]
        return a

    assert_equiv_prop(run, ARRAY[..., np.newaxis])


@pytest.mark.skip()
def test_set_index():
    def run(a):
        a = a.copy()
        a[:, 0] = a[:, 1]
        return a

    assert_equiv_prop(run, ARRAY[..., np.newaxis])
