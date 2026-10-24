"""Ported `tf.data.Dataset` unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase


class DatasetTest(TensorFlowTestCase):
    """Ported `Dataset` pipeline test case."""

    def testFromTensorSlices(self) -> None:
        """Verify chaining `batch`, `map`, and `shuffle` on `from_tensor_slices`."""
        ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
        ds = ds.batch(2).map(lambda x: x).shuffle(10, seed=42)
        self.assertIsInstance(ds, tf.data.Dataset)
        batches = list(ds)
        self.assertEqual(len(batches), 2)
        self.assertIn([1, 2], batches)
        self.assertIn([3], batches)
