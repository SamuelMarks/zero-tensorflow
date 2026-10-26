"""Ported `Losses` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class LossesTest(TensorFlowTestCase):
    """Ported loss test case."""

    def test_mse_basic(self) -> None:
        """Verify `MeanSquaredError` instantiation and numerical evaluation."""
        mse = keras.losses.MeanSquaredError()
        self.assertIsNotNone(mse)
        y_true = tf.Tensor([1.0, 2.0])
        y_pred = tf.Tensor([3.0, 4.0])
        loss_val = tf.Tensor(mse(y_true, y_pred))
        self.assertAllClose(loss_val, 4.0)
        self.assertDifferentialParity(
            lambda backend: backend.keras.losses.MeanSquaredError()(y_true, y_pred)
        )
