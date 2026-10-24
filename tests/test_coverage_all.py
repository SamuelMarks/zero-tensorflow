"""Unit tests for `test_coverage_all`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import Tensor

import numpy as np


def test_math_coverage() -> None:
    """Verify `tf.math` operations produce accurate numerical `Tensor` outputs."""
    x = Tensor([1.0, 4.0])
    y = Tensor([2.0, 2.0])

    np.testing.assert_allclose(tf.math.add(x, y).numpy(), [3.0, 6.0])
    np.testing.assert_allclose(tf.math.subtract(x, y).numpy(), [-1.0, 2.0])
    np.testing.assert_allclose(tf.math.multiply(x, y).numpy(), [2.0, 8.0])
    np.testing.assert_allclose(tf.math.divide(x, y).numpy(), [0.5, 2.0])
    np.testing.assert_allclose(tf.math.sqrt(x).numpy(), [1.0, 2.0])
    np.testing.assert_allclose(tf.math.square(x).numpy(), [1.0, 16.0])
    np.testing.assert_allclose(tf.math.abs(Tensor([-3.0, 4.0])).numpy(), [3.0, 4.0])
    np.testing.assert_allclose(tf.math.reduce_sum(x).numpy(), 5.0)
    np.testing.assert_allclose(tf.math.reduce_mean(x).numpy(), 2.5)
    np.testing.assert_allclose(tf.math.reduce_max(x).numpy(), 4.0)
    np.testing.assert_allclose(tf.math.reduce_min(x).numpy(), 1.0)
    np.testing.assert_allclose(tf.math.reduce_prod(x).numpy(), 4.0)
    np.testing.assert_allclose(tf.math.maximum(x, y).numpy(), [2.0, 4.0])
    np.testing.assert_allclose(tf.math.minimum(x, y).numpy(), [1.0, 2.0])
    np.testing.assert_allclose(tf.math.pow(x, y).numpy(), [1.0, 16.0])


def test_nn_coverage() -> None:
    """Verify `tf.nn` activation and normalization operations produce accurate outputs."""
    x = Tensor([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(tf.nn.relu(x).numpy(), [0.0, 0.0, 2.0])
    np.testing.assert_allclose(tf.nn.leaky_relu(Tensor([0.0, 2.0])).numpy(), [0.0, 2.0])
    np.testing.assert_allclose(tf.nn.sigmoid(Tensor([0.0])).numpy(), [0.5])
    np.testing.assert_allclose(tf.nn.tanh(Tensor([0.0])).numpy(), [0.0])
    np.testing.assert_allclose(tf.nn.softmax(Tensor([[1.0, 1.0]])).numpy(), [[0.5, 0.5]])


def test_linalg_coverage() -> None:
    """Verify `tf.linalg` matrix operations produce accurate `Tensor` outputs."""
    a = Tensor([[1.0, 2.0], [3.0, 4.0]])
    b = Tensor([[2.0, 0.0], [0.0, 2.0]])
    np.testing.assert_allclose(tf.linalg.matmul(a, b).numpy(), [[2.0, 4.0], [6.0, 8.0]])
    np.testing.assert_allclose(tf.linalg.det(b).numpy(), 4.0)
    np.testing.assert_allclose(
        tf.linalg.cholesky(b).numpy(), [[np.sqrt(2.0), 0.0], [0.0, np.sqrt(2.0)]]
    )
    np.testing.assert_allclose(tf.linalg.inv(b).numpy(), [[0.5, 0.0], [0.0, 0.5]])
    np.testing.assert_allclose(tf.linalg.diag(Tensor([2.0, 3.0])).numpy(), [[2.0, 0.0], [0.0, 3.0]])


def test_bitwise_coverage() -> None:
    """Verify `tf.bitwise` integer bitwise operations produce accurate outputs."""
    a = Tensor([5, 3], dtype=tf.int32)
    b = Tensor([3, 6], dtype=tf.int32)
    np.testing.assert_array_equal(tf.bitwise.bitwise_and(a, b).numpy(), [1, 2])
    np.testing.assert_array_equal(tf.bitwise.bitwise_or(a, b).numpy(), [7, 7])
    np.testing.assert_array_equal(tf.bitwise.bitwise_xor(a, b).numpy(), [6, 5])
    np.testing.assert_array_equal(
        tf.bitwise.left_shift(
            Tensor([1, 2], dtype=tf.int32), Tensor([2, 1], dtype=tf.int32)
        ).numpy(),
        [4, 4],
    )
    np.testing.assert_array_equal(
        tf.bitwise.right_shift(
            Tensor([8, 4], dtype=tf.int32), Tensor([2, 1], dtype=tf.int32)
        ).numpy(),
        [2, 2],
    )


def test_toplevel_coverage() -> None:
    """Verify top-level TensorFlow operations (`zeros`, `ones`, `eye`, `fill`, `range`, `linspace`, `where`)."""
    np.testing.assert_allclose(tf.zeros((2, 2)).numpy(), [[0.0, 0.0], [0.0, 0.0]])
    np.testing.assert_allclose(tf.ones((2, 2)).numpy(), [[1.0, 1.0], [1.0, 1.0]])
    np.testing.assert_allclose(tf.eye(2).numpy(), [[1.0, 0.0], [0.0, 1.0]])
    np.testing.assert_allclose(tf.fill((2, 2), 7.0).numpy(), [[7.0, 7.0], [7.0, 7.0]])
    np.testing.assert_allclose(tf.range(0, 4, 1).numpy(), [0, 1, 2, 3])
    np.testing.assert_allclose(tf.linspace(0.0, 1.0, 3).numpy(), [0.0, 0.5, 1.0])
    cond = Tensor([True, False])
    np.testing.assert_allclose(
        tf.where(cond, Tensor([10.0, 20.0]), Tensor([30.0, 40.0])).numpy(),
        [10.0, 40.0],
    )


def test_gradient_elementary_functions() -> None:
    """Verify `GradientTape` differentiates polynomials, linear expressions, and transcendental functions."""
    x = tf.Variable(3.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * x
    grad_sq = tape.gradient(y, x)
    assert isinstance(grad_sq, Tensor)
    np.testing.assert_allclose(grad_sq.numpy(), 6.0)

    with tf.GradientTape() as tape2:
        tape2.watch(x)
        y2 = x * 3.0 + tf.Tensor(2.0)
    grad_lin = tape2.gradient(y2, x)
    assert isinstance(grad_lin, Tensor)
    np.testing.assert_allclose(grad_lin.numpy(), 3.0)

    with tf.GradientTape() as tape3:
        tape3.watch(x)
        y3 = tf.math.sin(x)
    grad_sin = tape3.gradient(y3, x)
    assert isinstance(grad_sin, Tensor)
    np.testing.assert_allclose(grad_sin.numpy(), np.cos(3.0))


def test_gradient_val_2() -> None:
    """Verify `GradientTape` differentiates `x * 2.0` to `2.0`."""
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x * 2.0
    grad = tape.gradient(y, x)
    assert isinstance(grad, Tensor) and grad.numpy() == 2.0


def test_gradient_val_1() -> None:
    """Verify `GradientTape` differentiates `x + 1.0` to `1.0`."""
    x = tf.Variable(2.0)
    with tf.GradientTape() as tape:
        tape.watch(x)
        y = x + 1.0
    grad = tape.gradient(y, x)
    assert isinstance(grad, Tensor) and grad.numpy() == 1.0
