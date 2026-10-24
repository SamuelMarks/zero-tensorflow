"""Ported `GradientTape` backpropagation unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import FakeBackprop, TensorFlowTestCase


class BackpropTest(TensorFlowTestCase):
    """Ported `GradientTape` test case."""

    def testGradientTape(self) -> None:
        """Verify nested `GradientTape` differentiation."""
        with FakeBackprop.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
            with FakeBackprop.GradientTape() as gg:
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
        with FakeBackprop.GradientTape() as g:
            x = tf.Tensor(3.0)
            g.watch(x)
            y = x * x
        grads = g.gradient(y, [x])
        assert isinstance(grads, list) and grads[0] is not None
        self.assertEqual(self.evaluate(grads[0]), 6.0)
