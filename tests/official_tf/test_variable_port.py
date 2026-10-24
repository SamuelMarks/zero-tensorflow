"""Ported `tf.Variable` state mutation unit tests."""

from __future__ import annotations

import zero_tensorflow as tf
from tests.conftest import TensorFlowTestCase


class VariableTest(TensorFlowTestCase):
    """Ported `Variable` test case."""

    def testBasic(self) -> None:
        """Verify initial value of a `Variable`."""
        v = tf.Variable(1.0)
        self.assertEqual(v.value.numpy(), 1.0)

    def testAssign(self) -> None:
        """Verify `Variable.assign` updates the stored value."""
        v = tf.Variable(1.0)
        v.assign(3.0)
        self.assertEqual(v.value.numpy(), 3.0)

    def testAssignAdd(self) -> None:
        """Verify `Variable.assign_add` increments the stored value."""
        v = tf.Variable(1.0)
        v.assign_add(2.0)
        self.assertEqual(v.value.numpy(), 3.0)

    def testAssignSub(self) -> None:
        """Verify `Variable.assign_sub` decrements the stored value."""
        v = tf.Variable(3.0)
        v.assign_sub(1.0)
        self.assertEqual(v.value.numpy(), 2.0)
