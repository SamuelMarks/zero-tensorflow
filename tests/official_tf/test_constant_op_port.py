"""Ported `constant_op` unit tests."""

from __future__ import annotations

from tests.conftest import FakeConstantOp, TensorFlowTestCase

import numpy as np


class ConstantOpTest(TensorFlowTestCase):
    """Ported constant operation test case."""

    def test_np_array_memory_not_shared(self) -> None:
        """Verify mutating an input NumPy array does not mutate a created constant tensor."""
        for _ in range(100):
            x = np.arange(10)
            xt = FakeConstantOp.constant(x)
            x[3] = 42
            arr = np.asarray(xt.numpy())
            self.assertEqual(int(arr[3]), 3)
