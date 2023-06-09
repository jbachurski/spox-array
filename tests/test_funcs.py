import numpy as np
import pytest
from spox_array.testing import assert_equiv_prop


def test_copy():
    assert_equiv_prop(np.copy, np.array([[1, 2, 3], [4, 5, 6]]))


def test_reshape():
    assert_equiv_prop(np.reshape, np.array([[1, 2, 3], [4, 5, 6]]), (1, 2, 3))


def test_transpose():
    assert_equiv_prop(np.transpose, np.array([[1, 2, 3], [4, 5, 6]]))
    assert_equiv_prop(np.transpose, np.array([[1, 2, 3], [4, 5, 6]]), axes=(0, 1))
    assert_equiv_prop(np.transpose, np.array([[1, 2, 3], [4, 5, 6]]), axes=(1, 0))


def test_concat():
    assert_equiv_prop(np.concatenate, [np.array([1]), np.array([2])])


@pytest.mark.parametrize("reduce", [np.sum, np.mean, np.min, np.max, np.prod])
def test_reduces(reduce):
    assert_equiv_prop(reduce, np.array([1, 3, 6, 8]))
    assert_equiv_prop(reduce, np.array([[1, 3], [6, 8]]), axis=(0, 1))
    assert_equiv_prop(reduce, np.array([[1, 3], [6, 8]]), axis=0)
    assert_equiv_prop(reduce, np.array([[1, 3], [6, 8]]), axis=1)


@pytest.mark.parametrize("reduce", [np.argmin, np.argmax])
def test_reduces_one_axis(reduce):
    a = np.array([[1, 3], [6, 8]], np.float32)  # f32 for ORT support
    assert_equiv_prop(reduce, a)
    assert_equiv_prop(reduce, a, axis=0)
    assert_equiv_prop(reduce, a, axis=1)


def test_round():
    a = np.array([-1, -0.7, 0.3, 0.1, 0.7, 1])
    assert_equiv_prop(np.around, a)
    assert_equiv_prop(np.round_, a)


def test_clip():
    assert_equiv_prop(np.clip, np.array([1, 2, 3, 4]), 2, 3)
    assert_equiv_prop(np.clip, np.array([1, 2, 3, 4]), 3, None)
    assert_equiv_prop(np.clip, np.array([1, 2, 3, 4]), None, 2)


def test_where():
    assert_equiv_prop(
        np.where,
        np.array([True, False, True, False]),
        [-1, -2, -3, -4],
        [1, 2, 3, 4],
    )


def test_expand_dims():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equiv_prop(np.expand_dims, a, 0)
    assert_equiv_prop(np.expand_dims, a, (1, 2))


def test_squeeze():
    a = np.array([[[1, 2, 3], [4, 5, 6]]])
    assert_equiv_prop(np.squeeze, a)
    assert_equiv_prop(np.squeeze, a, 0)


@pytest.mark.parametrize("stack", [np.hstack, np.vstack])
def test_stacks(stack):
    assert_equiv_prop(stack, [np.array([1, 2, 3]), np.array([4, 5, 6])])
    assert_equiv_prop(
        stack,
        [np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])],
    )


def test_compress():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equiv_prop(np.compress, [False, True, True], a, axis=1)
    assert_equiv_prop(np.compress, [False, True], a, axis=0)


def test_cumsum():
    a = np.array([[1, 2, 3], [4, 5, 6]])
    assert_equiv_prop(np.cumsum, a)
    assert_equiv_prop(np.cumsum, a, 0)
    assert_equiv_prop(np.cumsum, a, axis=1)


def test_einsum():
    a = np.array([[1, 2]])
    b = np.array([[3, 4, 5], [6, 7, 8]])
    c = np.array([[1, 2], [3, 4]])
    assert_equiv_prop(np.einsum, "ii", c)
    assert_equiv_prop(np.einsum, "ii->i", c)
    assert_equiv_prop(np.einsum, "ij->ji", b)
    assert_equiv_prop(np.einsum, "ij,jk", a, b)
