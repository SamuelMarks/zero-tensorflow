import pytest
from zero_tensorflow import (
    Variable,
    function,
    GradientTape,
    data,
    math,
    Tensor,
)
import numpy as np


def test_variable():
    v = Variable(1.0)
    assert v.value == 1.0

    # Test initialization with Tensor
    v2 = Variable(Tensor(2.0))
    assert v2.value == 2.0

    # Test Tensor initialization with Tensor
    t_inner = Tensor(1.0)
    t_outer = Tensor(t_inner)
    assert t_outer.numpy() == 1.0

    # Test assign methods
    v.assign(3.0)
    assert v.value == 3.0
    v.assign(Tensor(4.0))
    assert v.value == 4.0

    v.assign_add(1.0)
    assert v.value == 5.0

    v.assign_sub(2.0)
    assert v.value == 3.0


def test_function():
    @function
    def f(x, msg):
        return x + 1, msg

    # f(1) traces and returns the result in tracing context
    result, msg = f(1, "hello")

    # Check if the result acts as a tensor
    assert isinstance(result, Tensor)
    assert msg == "hello"

    # Try calling without tracing
    assert math.add(1, 1).numpy() == 2


def test_gradient_tape():
    v = Variable(1.0)
    with GradientTape() as tape:
        tape.watch(v)
        y = v.value + 1.0
    grads = tape.gradient(y, [v])
    assert len(grads) == 1

    grads_single = tape.gradient(y, v)
    assert len(grads_single) == 1


def test_dataset():
    ds = data.Dataset.from_tensor_slices([1, 2, 3])
    ds = ds.batch(2).map(lambda x: x).shuffle(10)
    assert isinstance(ds, data.Dataset)


def test_math():
    assert math.add(1, 2).numpy() == 3
    assert math.subtract(2, 1).numpy() == 1
    assert math.multiply(2, 3).numpy() == 6
    assert math.divide(6, 2).numpy() == 3.0

    assert np.allclose(math.exp(0).numpy(), 1.0)
    assert np.allclose(math.log(np.e).numpy(), 1.0)
    assert math.pow(2, 3).numpy() == 8
    assert math.sqrt(4).numpy() == 2

    x = np.array([[1, 2], [3, 4]])
    assert math.reduce_sum(x).numpy() == 10
    assert math.reduce_mean(x).numpy() == 2.5
    assert math.reduce_max(x).numpy() == 4
    assert math.reduce_min(x).numpy() == 1

    a = np.array([[1, 0], [0, 1]])
    b = np.array([[1, 2], [3, 4]])
    assert np.array_equal(math.matmul(a, b).numpy(), b)
    assert np.array_equal(math.tensordot(a, b, axes=1).numpy(), b)


def test_tensor_magic_methods_eager():
    t1 = Tensor(2)
    t2 = Tensor(3)
    assert (t1 + t2).numpy() == 5
    assert (t1 - t2).numpy() == -1
    assert (t1 * t2).numpy() == 6
    assert (t1 / t2).numpy() == 2 / 3
    assert (1 + t1).numpy() == 3
    assert (3 - t1).numpy() == 1
    assert (3 * t1).numpy() == 6
    assert (6 / t1).numpy() == 3

    assert bool(t1 == 2)
    assert bool(t1 != 3)
    assert bool(t1 < 3)
    assert bool(t1 <= 2)
    assert bool(t1 > 1)
    assert bool(t1 >= 2)

    assert bool(Tensor(True))
    assert bool(Tensor(1))
    assert not bool(Tensor(0))
    assert Tensor(1).__nonzero__()


def test_tensor_magic_methods_traced():
    @function
    def f(x, y):
        a = x + y
        b = x - y
        c = x * y
        d = x / y
        e = 1 + x
        f_ = 3 - x
        g = 3 * x
        h = 6 / x
        return a, b, c, d, e, f_, g, h

    res = f(Tensor(2), Tensor(3))
    assert all(isinstance(r, Tensor) for r in res)

    @function
    def cmp_f(x, y):
        a = x == y
        b = x != y
        c = x < y
        d = x <= y
        e = x > y
        f_ = x >= y
        return a, b, c, d, e, f_

    cmp_res = cmp_f(Tensor(2), Tensor(3))
    assert all(isinstance(r, Tensor) for r in cmp_res)

    with pytest.raises(TypeError):

        @function
        def bool_f(x):
            return bool(x)

        bool_f(Tensor(1))

    with pytest.raises(ValueError):

        @function
        def numpy_f(x):
            return x.numpy()

        numpy_f(Tensor(1))


def test_math_traced():
    @function
    def math_f(x, y, a, b):
        return (
            math.exp(x),
            math.log(x),
            math.pow(x, y),
            math.sqrt(x),
            math.reduce_sum(x),
            math.reduce_mean(x),
            math.reduce_max(x),
            math.reduce_min(x),
            math.matmul(a, b),
            math.tensordot(a, b, axes=1),
        )

    x = Tensor(2.0)
    y = Tensor(3.0)
    a = Tensor(np.eye(2))
    b = Tensor(np.eye(2))
    res = math_f(x, y, a, b)
    assert all(isinstance(r, Tensor) for r in res)
