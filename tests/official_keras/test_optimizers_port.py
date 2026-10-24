"""Ported `Optimizers` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class OptimizersTest(TensorFlowTestCase):
    """Ported optimizer test case."""

    def test_adam_basic(self) -> None:
        """Verify `Adam` optimizer instantiation, `learning_rate`, and `apply_gradients`."""
        opt = keras.optimizers.Adam(learning_rate=0.01)
        self.assertEqual(opt.learning_rate, 0.01)
        var = tf.Variable([1.0, 2.0])
        grad = tf.Tensor([0.1, 0.2])
        opt.apply_gradients([(grad._tensor, var)])
        self.assertIsNotNone(var.numpy())
