"""Ported `tf.function` tracing unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import FakeDefFunction, TensorFlowTestCase
from zero_tensorflow._typing import TensorOperand


class FunctionTest(TensorFlowTestCase):
    """Ported `tf.function` compilation test case."""

    def testBasic(self) -> None:
        """Verify `@tf.function` traces a binary tensor addition and preserves numerical value."""

        @FakeDefFunction.function
        def add(*args: TensorOperand, **kwargs: TensorOperand) -> tf.Tensor:
            """Add two `Tensor` positional arguments.

            Args:
                *args: Two `Tensor` operands `(a, b)`.
                **kwargs: Unused keyword arguments.

            Returns:
                tf.Tensor: Sum `a + b`.
            """
            del kwargs
            a = args[0]
            b = args[1]
            if isinstance(a, tf.Tensor) and isinstance(b, tf.Tensor):
                return a + b
            return tf.Tensor(0.0)

        res = add(tf.Tensor(1.0), tf.Tensor(2.0))
        self.assertIsInstance(res, tf.Tensor)
        self.assertAllClose(res, 3.0)
