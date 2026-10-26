"""Root pytest configuration and shared test utilities for `zero_tensorflow`."""

from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from collections.abc import Callable, Iterator
from typing import TypeVar

import pytest

sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ml-switcheroo-compiler/src")),
)

import zero_tensorflow as tf
from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    NestedTensorData,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.debugging import assert_near

_TargetT = TypeVar("_TargetT", bound="type | CallableOperandProtocol")


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Provide a default autouse pytest fixture for compiler configuration isolation.

    Yields:
        None: Execution control to the active test.
    """
    yield


@pytest.mark.reference
class TensorFlowTestCase(unittest.TestCase):
    """Base `unittest.TestCase` emulating `tensorflow.python.framework.test_util.TensorFlowTestCase`."""

    def evaluate(
        self,
        tensor: TensorOperand | tf.Tensor,
    ) -> TensorOperand | NestedTensorData:
        """Evaluate a tensor or nested structure to its materialized value.

        Args:
            tensor: Input tensor or scalar value to evaluate.

        Returns:
            TensorOperand | NestedTensorData: Materialized value.
        """
        if isinstance(tensor, (tf.Tensor, NumpyConvertibleProtocol)):
            return as_operand(tensor.numpy())
        return tensor

    def assertAllClose(
        self,
        a: TensorOperand | tf.Tensor,
        b: TensorOperand | tf.Tensor,
    ) -> None:
        """Assert that two tensors or array-like values are numerically close.

        Args:
            a: First tensor or array-like operand.
            b: Second tensor or array-like operand.
        """
        assert_near(a, b)

    def assertDifferentialParity(
        self,
        op_fn: Callable[[types.ModuleType], TensorOperand | tf.Tensor],
        ref_module: types.ModuleType | None = None,
    ) -> None:
        """Execute `op_fn` against `zero_tensorflow` and the reference TensorFlow backend and assert parity.

        Args:
            op_fn: Callable receiving a TensorFlow-compatible module and returning a tensor/array.
            ref_module: Optional reference backend module (`default: zero_tensorflow`).
        """
        ztf_val = self.evaluate(op_fn(tf))
        target_ref = ref_module if ref_module is not None else tf
        ref_val = self.evaluate(op_fn(target_ref))
        self.assertAllClose(as_operand(ztf_val), as_operand(ref_val))


class FakeTestUtil:
    """Compatibility helper emulating `tensorflow.python.framework.test_util`."""

    TensorFlowTestCase = TensorFlowTestCase

    @staticmethod
    def run_all_in_graph_and_eager_modes(
        cls_or_fn: _TargetT | None = None,
    ) -> _TargetT | Callable[[_TargetT], _TargetT]:
        """Return class or function decorator running tests in eager/graph modes.

        Args:
            cls_or_fn: Target test class or method.

        Returns:
            _TargetT | Callable[[_TargetT], _TargetT]: Unmodified target or identity decorator.
        """
        if cls_or_fn is None:

            def _identity(target: _TargetT) -> _TargetT:
                """Return `target` unchanged.

                Args:
                    target: Test class or callable.

                Returns:
                    _TargetT: Unmodified test class or callable.
                """
                return target

            return _identity
        return cls_or_fn

    @staticmethod
    @contextlib.contextmanager
    def device(use_gpu: bool = False) -> Iterator[None]:
        """Context manager emulating device placement in ported tests.

        Args:
            use_gpu: Whether GPU placement is requested.

        Yields:
            None: Device context scope.
        """
        del use_gpu
        yield


class FakeConstantOp:
    """Compatibility namespace emulating `tensorflow.python.framework.constant_op`."""

    constant = tf.Tensor


class FakeBackprop:
    """Compatibility namespace emulating `tensorflow.python.eager.backprop`."""

    GradientTape = tf.GradientTape


class FakeDefFunction:
    """Compatibility namespace emulating `tensorflow.python.eager.def_function`."""

    function = staticmethod(tf.function)


class FakePlatformTest:
    """Compatibility namespace emulating `tensorflow.python.platform.test`."""

    main = staticmethod(unittest.main)


__all__ = [
    "ArrayLikeProtocol",
    "FakeBackprop",
    "FakeConstantOp",
    "FakeDefFunction",
    "FakePlatformTest",
    "FakeTestUtil",
    "TensorFlowTestCase",
    "switcheroo_config",
]
