"""Ported `Sequential` model unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class SequentialTest(TensorFlowTestCase):
    """Ported `Sequential` model test case."""

    def test_sequential_basic(self) -> None:
        """Verify `Sequential` layer stacking, `build`, and forward call behavior."""
        model = keras.models.Sequential(
            [keras.layers.Dense(2, activation="relu"), keras.layers.Dense(1)]
        )
        model.build((None, 2))
        self.assertEqual(len(model.layers), 2)
        self.assertTrue(model.built)
        out = tf.Tensor(model(tf.Tensor([[1.0, 2.0]])._tensor))
        self.assertEqual(out.shape, (1, 1))
