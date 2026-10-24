"""Root pytest configuration and shared test utilities for `zero_tensorflow`."""

from __future__ import annotations

import contextlib
import os
import sys
import types
import unittest
from collections.abc import Callable, Iterator
from typing import TypeVar

import numpy as np
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

import zero_keras as keras  # isort: skip

_TargetT = TypeVar("_TargetT", bound="type | CallableOperandProtocol")


@pytest.fixture(autouse=True)
def switcheroo_config() -> Iterator[None]:
    """Provide a default autouse pytest fixture for compiler configuration isolation.

    Yields:
        None: Execution control to the active test.
    """
    yield


def _to_float_array(
    val: TensorOperand | tf.Tensor,
) -> np.ndarray[tuple[int, ...], np.dtype[np.float64]]:
    """Convert a tensor or operand into a NumPy float64 array for comparison.

    Args:
        val: Input tensor, array-like object, or scalar.

    Returns:
        np.ndarray: NumPy float64 array.
    """
    raw = val.numpy() if isinstance(val, (tf.Tensor, NumpyConvertibleProtocol)) else val
    if isinstance(raw, ArrayLikeProtocol):
        return np.asarray(raw.tolist(), dtype=np.float64)
    if isinstance(raw, (int, float, bool)):
        return np.asarray(float(raw), dtype=np.float64)
    if isinstance(raw, (list, tuple)):
        return np.asarray(raw, dtype=np.float64)
    return np.asarray(0.0, dtype=np.float64)


class TensorFlowTestCase(unittest.TestCase):
    """Base `unittest.TestCase` emulating `tensorflow.python.framework.test_util.TensorFlowTestCase`."""

    def evaluate(
        self,
        tensor: TensorOperand | tf.Tensor,
    ) -> TensorOperand | NestedTensorData:
        """Evaluate a tensor or nested structure to its NumPy/Python value.

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
        arr_a = _to_float_array(a)
        arr_b = _to_float_array(b)
        np.testing.assert_allclose(arr_a, arr_b)


class FakeTestUtil:
    """Mock implementation of `tensorflow.python.framework.test_util`."""

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
        """Context manager mocking device placement in ported tests.

        Args:
            use_gpu: Whether GPU placement is requested.

        Yields:
            None: Device context scope.
        """
        del use_gpu
        yield


class FakeDTypes:
    """Mock `tensorflow.dtypes` namespace exposing NumPy dtype equivalents."""

    float16 = np.float16
    float32 = np.float32
    float64 = np.float64
    int32 = np.int32
    int64 = np.int64
    complex64 = np.complex64
    complex128 = np.complex128
    bool = np.bool_
    uint8 = np.uint8
    int8 = np.int8
    int16 = np.int16
    uint16 = np.uint16
    uint32 = np.uint32
    uint64 = np.uint64
    bfloat16 = np.float32


class FakeConstantOp:
    """Mock `tensorflow.python.framework.constant_op` namespace."""

    constant = tf.Tensor


class FakeBackprop:
    """Mock `tensorflow.python.eager.backprop` namespace."""

    GradientTape = tf.GradientTape


class FakeDefFunction:
    """Mock `tensorflow.python.eager.def_function` namespace."""

    function = staticmethod(tf.function)


class FakePlatformTest:
    """Mock `tensorflow.python.platform.test` namespace."""

    main = staticmethod(unittest.main)


class _FakeTensorFlowModule(types.ModuleType):
    """Typed `types.ModuleType` subclass representing the fake `tensorflow` module."""

    def __init__(self) -> None:
        """Initialize `_FakeTensorFlowModule` with `zero_tensorflow` attributes."""
        super().__init__("tensorflow")
        self.Tensor = tf.Tensor
        self.Variable = tf.Variable
        self.GradientTape = tf.GradientTape
        self.function = tf.function
        self.data = tf.data
        self.math = tf.math
        self.linalg = tf.linalg
        self.bitwise = tf.bitwise
        self.nn = tf.nn
        self.keras = keras
        self.test_util = FakeTestUtil()
        self.dtypes = FakeDTypes()
        self.math_ops = tf.math
        self.constant_op = FakeConstantOp()
        self.backprop = FakeBackprop()
        self.def_function = FakeDefFunction()
        self.nn_ops = tf.nn
        self.test = FakePlatformTest()
        self.TensorFlowTestCase = TensorFlowTestCase
        self.run_all_in_graph_and_eager_modes = FakeTestUtil.run_all_in_graph_and_eager_modes
        self.device = FakeTestUtil.device


def install_fake_tensorflow() -> None:
    """Populate `sys.modules['tensorflow']` and internal `tensorflow.python.*` aliases."""
    tf_mod = _FakeTensorFlowModule()

    sys.modules["tensorflow"] = tf_mod
    for sub in (
        "tensorflow.python",
        "tensorflow.python.framework",
        "tensorflow.python.framework.test_util",
        "tensorflow.python.eager",
        "tensorflow.python.ops",
        "tensorflow.python.ops.ragged",
        "tensorflow.python.platform",
        "tensorflow.core",
        "tensorflow.core.framework",
    ):
        sys.modules[sub] = tf_mod

    sys.modules["tensorflow.keras"] = keras


__all__ = [
    "ArrayLikeProtocol",
    "FakeBackprop",
    "FakeConstantOp",
    "FakeDTypes",
    "FakeDefFunction",
    "FakePlatformTest",
    "FakeTestUtil",
    "TensorFlowTestCase",
    "install_fake_tensorflow",
    "switcheroo_config",
]
