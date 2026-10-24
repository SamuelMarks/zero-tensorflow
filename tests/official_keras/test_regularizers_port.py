"""Ported `Regularizers` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import keras


class RegularizersTest(TensorFlowTestCase):
    """Ported regularizer test case."""

    def test_l2_basic(self) -> None:
        """Verify `L2` regularizer instantiation and penalty computation."""
        reg = keras.regularizers.L2(0.01)
        self.assertIsNotNone(reg)
        weights = tf.Tensor([2.0, 3.0])
        penalty = tf.Tensor(reg(weights._tensor))
        self.assertAllClose(penalty, 0.01 * (4.0 + 9.0))
