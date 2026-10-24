"""Ported `Dense` layer correctness and signature tests."""

from __future__ import annotations

from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras

import numpy as np


class DenseTest(TensorFlowTestCase):
    """Ported `Dense` layer test case."""

    def test_dense_correctness(self) -> None:
        """Verify forward pass of `Dense` with weights, bias, and ReLU activation."""
        layer = keras.layers.Dense(
            units=2,
            activation="relu",
            use_bias=True,
        )
        layer.build((1, 2))
        layer.set_weights(
            [
                np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([0.5, -0.5]),
            ]
        )
        inputs = np.array([[1.0, 1.0]])
        expected_output = np.array([[4.5, 5.5]])

        outputs = layer(inputs)
        self.assertAllClose(outputs, expected_output)

    def test_dense_api_signature(self) -> None:
        """Verify `Dense` build state and `units` property."""
        layer = keras.layers.Dense(units=2, activation="relu", use_bias=True)
        layer.build((1, 2))
        self.assertTrue(layer.built)
        self.assertEqual(layer.units, 2)
