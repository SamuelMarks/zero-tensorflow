"""TensorFlow testing utilities module."""

from __future__ import annotations

import contextlib
import tempfile
import unittest
from collections.abc import Iterator, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, Variable
from zero_tensorflow.debugging import (
    assert_equal,
    assert_greater,
    assert_less,
    assert_near,
)

import ml_switcheroo_compiler.core as msc_core

__all__ = [
    "TestCase",
    "experimental",
    "gpu_device_name",
    "is_built_with_cuda",
    "is_built_with_gpu_support",
    "is_built_with_rocm",
    "is_built_with_xla",
    "main",
]


def _evaluate_nested(
    tensors: Tensor | Variable | TensorOperand,
) -> TensorOperand:
    """Recursively evaluate `Tensor` and `Variable` objects into NumPy/Python values.

    Args:
        tensors: Tensor, Variable, or nested tuple/list/dict of tensors.

    Returns:
        TensorOperand: Evaluated NumPy array, scalar, or nested structure.
    """
    if isinstance(tensors, (Tensor, Variable, NumpyConvertibleProtocol)):
        return as_operand(tensors.numpy())
    if isinstance(tensors, tuple):
        return tuple(_evaluate_nested(t) for t in tensors)
    if isinstance(tensors, list):
        return [_evaluate_nested(t) for t in tensors]
    return tensors


def _shape_of(val: Tensor | Variable | TensorOperand) -> tuple[int, ...]:
    """Return the shape tuple of `val`.

    Args:
        val: Tensor, Variable, or array-like object.

    Returns:
        tuple[int, ...]: Shape tuple.
    """
    if isinstance(val, (Tensor, Variable, msc_core.Tensor, ArrayLikeProtocol)):
        return tuple(val.shape)
    if isinstance(val, NumpyConvertibleProtocol):
        return _shape_of(as_operand(val.numpy()))
    dims: list[int] = []
    cur = val
    while isinstance(cur, (list, tuple)):
        dims.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(dims)


class TestCase(unittest.TestCase):
    """Base class for TensorFlow unit tests with numerical and shape assertions."""

    def evaluate(
        self,
        tensors: Tensor | Variable | TensorOperand,
    ) -> TensorOperand:
        """Evaluate a tensor or nested structure of tensors in eager/session mode.

        Args:
            tensors: Tensor, Variable, or nested collection of tensors.

        Returns:
            TensorOperand: Evaluated NumPy array or Python scalar structure.
        """
        return _evaluate_nested(tensors)

    def assertAllClose(
        self,
        a: Tensor | Variable | TensorOperand,
        b: Tensor | Variable | TensorOperand,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        msg: str | None = None,
    ) -> None:
        """Assert that `a` and `b` are element-wise close within `rtol` and `atol`.

        Args:
            a: First tensor or array-like value.
            b: Second tensor or array-like value.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            msg: Optional failure message.

        Raises:
            AssertionError: If `a` and `b` are not close.
        """
        try:
            assert_near(a, b, rtol=rtol, atol=atol, message=msg)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc

    def assertNotAllClose(
        self,
        a: Tensor | Variable | TensorOperand,
        b: Tensor | Variable | TensorOperand,
        rtol: float = 1e-6,
        atol: float = 1e-6,
        msg: str | None = None,
    ) -> None:
        """Assert that `a` and `b` are not element-wise close within `rtol` and `atol`.

        Args:
            a: First tensor or array-like value.
            b: Second tensor or array-like value.
            rtol: Relative tolerance.
            atol: Absolute tolerance.
            msg: Optional failure message.

        Raises:
            AssertionError: If `a` and `b` are close.
        """
        try:
            assert_near(a, b, rtol=rtol, atol=atol, message=msg)
        except ValueError:
            return
        raise AssertionError(msg or "Expected arrays not to be all close.")

    def assertAllEqual(
        self,
        a: Tensor | Variable | TensorOperand,
        b: Tensor | Variable | TensorOperand,
        msg: str | None = None,
    ) -> None:
        """Assert that `a` and `b` are element-wise equal.

        Args:
            a: First tensor or array-like value.
            b: Second tensor or array-like value.
            msg: Optional failure message.

        Raises:
            AssertionError: If `a` and `b` are not equal.
        """
        try:
            assert_equal(a, b, message=msg)
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc

    def assertNotAllEqual(
        self,
        a: Tensor | Variable | TensorOperand,
        b: Tensor | Variable | TensorOperand,
        msg: str | None = None,
    ) -> None:
        """Assert that `a` and `b` differ in at least one element.

        Args:
            a: First tensor or array-like value.
            b: Second tensor or array-like value.
            msg: Optional failure message.

        Raises:
            AssertionError: If `a` and `b` are completely equal.
        """
        try:
            assert_equal(a, b, message=msg)
        except ValueError:
            return
        raise AssertionError(msg or "Expected arrays not to be all equal.")

    def assertAllGreater(
        self,
        a: Tensor | Variable | TensorOperand,
        comparison_target: float | int,
    ) -> None:
        """Assert that all elements of `a` are strictly greater than `comparison_target`.

        Args:
            a: Input tensor or array-like value.
            comparison_target: Scalar lower bound.

        Raises:
            AssertionError: If any element of `a <= comparison_target`.
        """
        shape = _shape_of(a)
        n_elems = 1
        for d in shape:
            n_elems *= d
        try:
            assert_greater(a, [comparison_target] * max(1, n_elems))
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc

    def assertAllLess(
        self,
        a: Tensor | Variable | TensorOperand,
        comparison_target: float | int,
    ) -> None:
        """Assert that all elements of `a` are strictly less than `comparison_target`.

        Args:
            a: Input tensor or array-like value.
            comparison_target: Scalar upper bound.

        Raises:
            AssertionError: If any element of `a >= comparison_target`.
        """
        shape = _shape_of(a)
        n_elems = 1
        for d in shape:
            n_elems *= d
        try:
            assert_less(a, [comparison_target] * max(1, n_elems))
        except ValueError as exc:
            raise AssertionError(str(exc)) from exc

    def assertShapeEqual(
        self,
        np_array: Tensor | Variable | TensorOperand,
        tf_tensor: Tensor | Variable | TensorOperand,
        msg: str | None = None,
    ) -> None:
        """Assert that `np_array` and `tf_tensor` have identical shapes.

        Args:
            np_array: Reference array or tensor.
            tf_tensor: Target tensor.
            msg: Optional failure message.

        Raises:
            AssertionError: If shapes differ.
        """
        sh_a = _shape_of(np_array)
        sh_b = _shape_of(tf_tensor)
        if sh_a != sh_b:
            raise AssertionError(msg or f"Shape mismatch: {sh_a} != {sh_b}")

    @contextlib.contextmanager
    def cached_session(self) -> Iterator[TestCase]:
        """Return a context manager yielding `self` as an eager-compatible test session.

        Yields:
            TestCase: Current test case instance.
        """
        yield self

    @contextlib.contextmanager
    def session(self) -> Iterator[TestCase]:
        """Return a context manager yielding `self` as a test session.

        Yields:
            TestCase: Current test case instance.
        """
        yield self

    def get_temp_dir(self) -> str:
        """Return a temporary directory path for use during the test.

        Returns:
            str: Path to a temporary directory.
        """
        return tempfile.mkdtemp()


def is_built_with_cuda() -> bool:
    """Return whether TensorFlow was built with CUDA support.

    Returns:
        bool: Always `False` in the pure-Python backend.
    """
    return False


def is_built_with_gpu_support() -> bool:
    """Return whether TensorFlow was built with GPU (CUDA or ROCm) support.

    Returns:
        bool: Always `False` in the pure-Python backend.
    """
    return False


def is_built_with_rocm() -> bool:
    """Return whether TensorFlow was built with ROCm support.

    Returns:
        bool: Always `False` in the pure-Python backend.
    """
    return False


def is_built_with_xla() -> bool:
    """Return whether TensorFlow was built with XLA support.

    Returns:
        bool: Always `False` in the pure-Python backend.
    """
    return False


def gpu_device_name() -> str:
    """Return the name of the first available GPU device, or `""` if none.

    Returns:
        str: Empty string `""` when no GPU is present.
    """
    return ""


def main(argv: Sequence[str] | None = None) -> None:
    """Run all `unittest.TestCase` suites in the calling module.

    Args:
        argv: Optional command-line argument list passed to `unittest.main`.
    """
    unittest.main(argv=list(argv) if argv is not None else ["test"], exit=False)


class experimental:
    """Experimental testing utilities."""

    @staticmethod
    def sync_devices(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Synchronize all active compute devices.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        return None
