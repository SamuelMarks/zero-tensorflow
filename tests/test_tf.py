"""Unit tests for `test_tf`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import GradientTape, Tensor, Variable, data, function, math

import numpy as np


def test_variable() -> None:
    """Execute `test_variable`."""
    v = Variable(1.0)
    assert v.value == 1.0
    v2 = Variable(Tensor(2.0))
    assert v2.value == 2.0
    t_inner = Tensor(1.0)
    t_outer = Tensor(t_inner)
    assert t_outer.numpy() == 1.0
    v.assign(3.0)
    assert v.value == 3.0
    v.assign(Tensor(4.0))
    assert v.value == 4.0
    v.assign_add(1.0)
    assert v.value == 5.0
    v.assign_sub(2.0)
    assert v.value == 3.0


def test_function() -> None:
    """Execute `test_function`."""

    @function
    def f(x: Tensor | int, msg: str) -> tuple[Tensor, str]:
        """Execute `f`.

        Args:
            x: Parameter `x`.
            msg: Parameter `msg`.

        Returns:
            tuple[Tensor, str]: Computed result.
        """
        return (Tensor(x) + 1, msg)

    result, msg = f(1, "hello")
    assert isinstance(result, Tensor)
    assert msg == "hello"
    assert math.add(1, 1).numpy() == 2


def test_gradient_tape() -> None:
    """Execute `test_gradient_tape`."""
    v = Variable(1.0)
    with GradientTape(persistent=True) as tape:
        tape.watch(v)
        y = v.value + 1.0
    grads = tape.gradient(y, [v])
    assert isinstance(grads, list) and len(grads) == 1 and grads[0] is not None
    assert grads[0].numpy() == 1.0
    grads_single = tape.gradient(y, v)
    assert isinstance(grads_single, Tensor)
    assert grads_single.numpy() == 1.0


def test_dataset() -> None:
    """Execute `test_dataset`."""
    ds = data.Dataset.from_tensor_slices([1, 2, 3])
    ds = ds.batch(2).map(lambda x: x).shuffle(10)
    assert isinstance(ds, data.Dataset)


def test_math() -> None:
    """Execute `test_math`."""
    assert math.add(1, 2).numpy() == 3
    assert math.subtract(2, 1).numpy() == 1
    assert math.multiply(2, 3).numpy() == 6
    assert math.divide(6, 2).numpy() == 3.0
    exp_val = math.exp(0).numpy()
    log_val = math.log(np.e).numpy()
    assert isinstance(exp_val, (float, np.ndarray)) and np.allclose(exp_val, 1.0)
    assert isinstance(log_val, (float, np.ndarray)) and np.allclose(log_val, 1.0)
    assert math.pow(2, 3).numpy() == 8
    assert math.sqrt(4).numpy() == 2
    x = np.array([[1, 2], [3, 4]])
    assert math.reduce_sum(x).numpy() == 10
    assert math.reduce_mean(x).numpy() == 2.5
    assert math.reduce_max(x).numpy() == 4
    assert math.reduce_min(x).numpy() == 1
    a = np.array([[1, 0], [0, 1]])
    b = np.array([[1, 2], [3, 4]])
    mm_val = tf.linalg.matmul(a, b).numpy()
    td_val = tf.linalg.tensordot(a, b, axes=1).numpy()
    np.testing.assert_array_equal(mm_val, b)
    np.testing.assert_array_equal(td_val, b)


def test_tensor_magic_methods_eager() -> None:
    """Execute `test_tensor_magic_methods_eager`."""
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


def test_tensor_magic_methods_traced() -> None:
    """Execute `test_tensor_magic_methods_traced`."""

    @function
    def f(x: Tensor, y: Tensor) -> tuple[Tensor, ...]:
        """Execute `f`.

        Args:
            x: Parameter `x`.
            y: Parameter `y`.

        Returns:
            tuple[Tensor, ...]: Computed result.
        """
        a = x + y
        b = x - y
        c = x * y
        d = x / y
        e = 1 + x
        f_ = 3 - x
        g = 3 * x
        h = 6 / x
        return (a, b, c, d, e, f_, g, h)

    res = f(Tensor(2), Tensor(3))
    assert all(isinstance(r, Tensor) for r in res)

    @function
    def cmp_f(x: Tensor, y: Tensor) -> tuple[Tensor | bool, ...]:
        """Execute `cmp_f`.

        Args:
            x: Parameter `x`.
            y: Parameter `y`.

        Returns:
            tuple[Tensor | bool, ...]: Computed result.
        """
        a = x == y
        b = x != y
        c = x < y
        d = x <= y
        e = x > y
        f_ = x >= y
        return (a, b, c, d, e, f_)

    cmp_res = cmp_f(Tensor(2), Tensor(3))
    assert all(isinstance(r, Tensor) for r in cmp_res)


def test_math_traced() -> None:
    """Execute `test_math_traced`."""

    @function
    def math_f(x: Tensor, y: Tensor, a: Tensor, b: Tensor) -> tuple[Tensor, ...]:
        """Execute `math_f`.

        Args:
            x: Parameter `x`.
            y: Parameter `y`.
            a: Parameter `a`.
            b: Parameter `b`.

        Returns:
            tuple[Tensor, ...]: Computed result.
        """
        return (
            math.exp(x),
            math.log(x),
            math.pow(x, y),
            math.sqrt(x),
            math.reduce_sum(x),
            math.reduce_mean(x),
            math.reduce_max(x),
            math.reduce_min(x),
            tf.linalg.matmul(a, b),
            tf.linalg.tensordot(a, b, axes=1),
        )

    x = Tensor(2.0)
    y = Tensor(3.0)
    a = Tensor(np.eye(2))
    b = Tensor(np.eye(2))
    res = math_f(x, y, a, b)
    assert all(isinstance(r, Tensor) for r in res)
