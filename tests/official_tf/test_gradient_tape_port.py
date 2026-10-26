"""Ported `GradientTape` backpropagation unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase


class BackpropTest(TensorFlowTestCase):
    """Ported `GradientTape` test case."""

    def testGradientTape(self) -> None:
        """Verify nested `GradientTape` differentiation."""
        with tf.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
            with tf.GradientTape() as gg:
                gg.watch(y)
                z = 2 * y
            inner_grads = gg.gradient(z, [y])
            assert isinstance(inner_grads, list) and inner_grads[0] is not None
            inner_grad = inner_grads[0]
            self.assertEqual(self.evaluate(inner_grad), 2.0)
            y = y + inner_grad
        grads = g.gradient(y, [x])
        assert isinstance(grads, list) and grads[0] is not None
        self.assertEqual(self.evaluate(grads[0]), 6.0)

    def testGradientTapeBasicStub(self) -> None:
        """Verify single-level `GradientTape` differentiation of `x * x`."""
        with tf.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
        grads = g.gradient(y, [x])
        assert isinstance(grads, list) and grads[0] is not None
        self.assertEqual(self.evaluate(grads[0]), 6.0)

    def testHigherOrderAndJacobians(self) -> None:
        """Verify nested `GradientTape`, `jacobian`, and `batch_jacobian`."""
        x = tf.Tensor(3.0)
        with tf.GradientTape() as outer:
            outer.watch(x)
            with tf.GradientTape() as inner:
                inner.watch(x)
                y = x * x * x
            dy_dx = inner.gradient(y, x)
            assert isinstance(dy_dx, tf.Tensor)
        d2y_dx2 = outer.gradient(dy_dx, x)
        assert isinstance(d2y_dx2, tf.Tensor)
        self.assertAllClose(d2y_dx2, 18.0)

        vec = tf.Tensor([2.0, 3.0])
        with tf.GradientTape() as jtape:
            jtape.watch(vec)
            vec_sq = vec * vec
        jac = jtape.jacobian(vec_sq, vec)
        assert isinstance(jac, tf.Tensor)
        self.assertAllClose(jac, [[4.0, 0.0], [0.0, 6.0]])

        b_vec = tf.Tensor([[1.0, 2.0], [3.0, 4.0]])
        with tf.GradientTape() as bjtape:
            bjtape.watch(b_vec)
            b_sq = b_vec * b_vec
        bjac = bjtape.batch_jacobian(b_sq, b_vec)
        assert isinstance(bjac, tf.Tensor)
        self.assertEqual(bjac.shape, (2, 2, 2))
