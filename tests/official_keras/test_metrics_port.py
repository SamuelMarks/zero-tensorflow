"""Ported `Metrics` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class MetricsTest(TensorFlowTestCase):
    """Ported metric test case."""

    def test_accuracy_basic(self) -> None:
        """Verify `Accuracy` metric instantiation, `update_state`, and `result`."""
        acc = keras.metrics.Accuracy()
        self.assertIsNotNone(acc)
        acc.update_state(tf.Tensor([1, 0, 1, 1]), tf.Tensor([1, 0, 0, 1]))
        res = tf.Tensor(acc.result())
        self.assertAllClose(res, 0.75)
