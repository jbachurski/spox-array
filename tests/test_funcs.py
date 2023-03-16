import numpy as np

from spox_array.testing import assert_equiv_prop


def test_concat():
    assert_equiv_prop(np.concatenate, [np.array([1]), np.array([2])])


def test_transpose():
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]))
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]), axis=(0, 1))
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]), axis=(1, 0))


def test_mean():
    assert_equiv_prop(np.mean, np.array([1, 3, 6, 8]))
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]), axis=(0, 1))
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]), axis=0)
    assert_equiv_prop(np.mean, np.array([[1, 3], [6, 8]]), axis=1)
