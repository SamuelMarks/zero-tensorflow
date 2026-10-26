"""Ported `constant_op` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase


class ConstantOpTest(TensorFlowTestCase):
    """Ported constant operation test case."""

    def test_np_array_memory_not_shared(self) -> None:
        """Verify mutating an input sequence does not mutate a created constant tensor."""
        for _ in range(100):
            x = list(range(10))
            xt = tf.constant(x)
            x[3] = 42
            assert xt is not None
            self.assertEqual(self.evaluate(xt[3]), 3)
