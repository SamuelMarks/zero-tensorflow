"""Ported `tf.math.reduce_sum` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import FakeTestUtil, TensorFlowTestCase


@FakeTestUtil.run_all_in_graph_and_eager_modes
class ReduceTest(TensorFlowTestCase):
    """Ported math reduction test case."""

    def testReduceSum(self) -> None:
        """Verify `tf.math.reduce_sum` on a 2D integer array."""
        x = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.int32)
        assert x is not None
        with FakeTestUtil.device(use_gpu=True):
            y_tf = self.evaluate(tf.math.reduce_sum(x))
            self.assertEqual(y_tf, 21)
            self.assertDifferentialParity(lambda backend: backend.math.reduce_sum(x), ref_module=tf)
            self.assertDifferentialParity(lambda backend: backend.math.reduce_sum(x))
