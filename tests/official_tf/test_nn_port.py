"""Ported `nn_ops` (`relu`, `softmax`) unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase
from zero_tensorflow import nn as nn_ops


class NNOpsTest(TensorFlowTestCase):
    """Ported neural network operations test case."""

    def testRelu(self) -> None:
        """Verify `nn_ops.relu` on a 2D array with negative and positive values."""
        x = tf.Tensor([[-1.0, 2.0], [3.0, -4.0]])
        res = nn_ops.relu(x)
        self.assertAllClose(res, [[0.0, 2.0], [3.0, 0.0]])
        self.assertDifferentialParity(lambda backend: backend.nn.relu(x))

    def testSoftmax(self) -> None:
        """Verify `nn_ops.softmax` on uniform 2-class logits."""
        x = tf.Tensor([[1.0, 1.0]])
        res = nn_ops.softmax(x)
        self.assertAllClose(res, [[0.5, 0.5]])
        self.assertDifferentialParity(lambda backend: backend.nn.softmax(x))
