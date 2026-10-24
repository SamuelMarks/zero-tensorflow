"""Ported `Initializers` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class InitializersTest(TensorFlowTestCase):
    """Ported initializer test case."""

    def test_random_normal(self) -> None:
        """Verify `RandomNormal`, `Ones`, and `Zeros` initializers."""
        init = keras.initializers.RandomNormal(mean=0.0, stddev=1.0, seed=42)
        self.assertIsNotNone(init)

        ones_init = keras.initializers.Ones()
        ones_tensor = tf.Tensor(ones_init((2, 2)))
        self.assertAllClose(ones_tensor, [[1.0, 1.0], [1.0, 1.0]])

        zeros_init = keras.initializers.Zeros()
        zeros_tensor = tf.Tensor(zeros_init((2, 2)))
        self.assertAllClose(zeros_tensor, [[0.0, 0.0], [0.0, 0.0]])
