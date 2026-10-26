"""TensorFlow debugging utilities and runtime assertions module."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    DTypeLike,
    NestedTensorData,
    NumpyConvertibleProtocol,
    TensorOperand,
)
from zero_tensorflow.core_tensor import Tensor, Variable, _TensorRef, _wrap_tensor, to_dtype

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.core.dtype as msc_dtype
from ml_switcheroo_compiler.core.shape import TensorOrScalar

__all__ = [
    "assert_equal",
    "assert_greater",
    "assert_less",
    "assert_near",
    "assert_non_negative",
    "assert_positive",
    "assert_proper_iterable",
    "assert_rank",
    "assert_shapes",
    "assert_type",
    "check_numerics",
    "experimental",
    "is_numeric_tensor",
]

_DEBUG_DUMP_ROOT: str | None = None


class experimental:
    """Experimental debugging APIs."""

    @staticmethod
    def disable_dump_debug_info(*args: TensorOperand, **kwargs: TensorOperand) -> None:
        """Disable dumping of execution debug information.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        global _DEBUG_DUMP_ROOT
        _DEBUG_DUMP_ROOT = None

    @staticmethod
    def enable_dump_debug_info(
        dump_root: str | TensorOperand = None,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Enable dumping of execution debug information to disk.

        Args:
            dump_root: Optional directory path for dumping debug info.
            *args: Positional arguments.
            **kwargs: Keyword arguments.
        """
        del args, kwargs
        global _DEBUG_DUMP_ROOT
        _DEBUG_DUMP_ROOT = dump_root if isinstance(dump_root, str) else None
        return None


def _flatten_floats(val: Tensor | TensorOperand | TensorOrScalar) -> list[float]:
    """Flatten a tensor or nested sequence into a 1D list of floats.

    Args:
        val: Input tensor or array-like value.

    Returns:
        list[float]: Flattened numeric values.
    """
    if isinstance(val, NumpyConvertibleProtocol):
        return _flatten_floats(val.numpy())
    if isinstance(val, ArrayLikeProtocol):
        return _flatten_floats(val.tolist())
    if isinstance(val, (list, tuple)):
        out: list[float] = []
        for item in val:
            out.extend(_flatten_floats(item))
        return out
    if isinstance(val, (int, float, bool)):
        return [float(val)]
    return []


def _infer_shape(val: Tensor | TensorOperand | TensorOrScalar) -> tuple[int, ...]:
    """Infer the shape tuple of `val`.

    Args:
        val: Input tensor or nested sequence.

    Returns:
        tuple[int, ...]: Inferred shape tuple.
    """
    if isinstance(val, (Tensor, Variable, msc_core.Tensor, ArrayLikeProtocol)):
        return tuple(val.shape)
    if isinstance(val, NumpyConvertibleProtocol):
        return _infer_shape(val.numpy())
    dims: list[int] = []
    cur: NestedTensorData | TensorOperand | TensorOrScalar = val
    while isinstance(cur, (list, tuple)):
        dims.append(len(cur))
        if not cur:
            break
        cur = cur[0]
    return tuple(dims)


def assert_equal(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x == y` element-wise.

    Args:
        x: First tensor or value.
        y: Second tensor or value.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If `x` and `y` are not element-wise equal.
    """
    del summarize, name
    if x is None and y is None:
        return None
    fx = _flatten_floats(x)
    fy = _flatten_floats(y)
    if len(fx) != len(fy) or any(a != b for a, b in zip(fx, fy)):
        raise ValueError(message or f"assert_equal failed: {fx} != {fy}")


def assert_near(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    rtol: float = 1e-5,
    atol: float = 1e-5,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x` and `y` are element-wise close within `atol + rtol * |y|`.

    Args:
        x: First tensor or value.
        y: Second tensor or value.
        rtol: Relative tolerance.
        atol: Absolute tolerance.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If any elements of `x` and `y` differ by more than tolerance.
    """
    del summarize, name
    if x is None and y is None:
        return None
    fx = _flatten_floats(x)
    fy = _flatten_floats(y)
    if len(fx) != len(fy) or any(abs(a - b) > atol + rtol * abs(b) for a, b in zip(fx, fy)):
        raise ValueError(message or f"assert_near failed: {fx} not close to {fy}")


def assert_less(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x < y` element-wise.

    Args:
        x: First tensor or value.
        y: Second tensor or value.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If any element of `x >= y`.
    """
    del summarize, name
    if x is None and y is None:
        return None
    fx = _flatten_floats(x)
    fy = _flatten_floats(y)
    if len(fx) != len(fy) or any(a >= b for a, b in zip(fx, fy)):
        raise ValueError(message or f"assert_less failed: {fx} not < {fy}")


def assert_greater(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x > y` element-wise.

    Args:
        x: First tensor or value.
        y: Second tensor or value.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If any element of `x <= y`.
    """
    del summarize, name
    if x is None and y is None:
        return None
    fx = _flatten_floats(x)
    fy = _flatten_floats(y)
    if len(fx) != len(fy) or any(a <= b for a, b in zip(fx, fy)):
        raise ValueError(message or f"assert_greater failed: {fx} not > {fy}")


def assert_non_negative(
    x: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x >= 0` element-wise.

    Args:
        x: Input tensor or value.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If any element of `x < 0`.
    """
    del summarize, name
    if x is None:
        return None
    fx = _flatten_floats(x)
    if any(a < 0.0 for a in fx):
        raise ValueError(message or f"assert_non_negative failed on {fx}")


def assert_positive(
    x: Tensor | TensorOperand = None,
    message: str | None = None,
    summarize: int | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x > 0` element-wise.

    Args:
        x: Input tensor or value.
        message: Optional error message prefix.
        summarize: Optional number of elements to print on failure.
        name: Optional operation name.

    Raises:
        ValueError: If any element of `x <= 0`.
    """
    del summarize, name
    if x is None:
        return None
    fx = _flatten_floats(x)
    if any(a <= 0.0 for a in fx):
        raise ValueError(message or f"assert_positive failed on {fx}")


def assert_proper_iterable(
    values: Tensor | Variable | Iterable[TensorOperand] | TensorOperand,
) -> None:
    """Assert that `values` is a proper Python iterable and not a single Tensor, Variable, or string.

    Args:
        values: Candidate iterable object.

    Raises:
        TypeError: If `values` is a `Tensor`, `Variable`, `str`, `bytes`, or non-iterable.
    """
    if isinstance(values, (Tensor, Variable, msc_core.Tensor, str, bytes)) or not isinstance(
        values, Iterable
    ):
        raise TypeError("Expected a proper iterable (not a Tensor, Variable, str, or bytes).")


def assert_rank(
    x: Tensor | TensorOperand = None,
    rank: int | None = None,
    message: str | None = None,
    name: str | None = None,
) -> None:
    """Assert that `x` has rank equal to `rank`.

    Args:
        x: Input tensor or array-like value.
        rank: Expected integer rank.
        message: Optional error message.
        name: Optional operation name.

    Raises:
        ValueError: If the rank of `x` does not equal `rank`.
    """
    del name
    if x is None or rank is None:
        return None
    actual_rank = len(_infer_shape(x))
    if actual_rank != int(rank):
        raise ValueError(message or f"Expected rank {rank}, but got rank {actual_rank}.")


def assert_shapes(
    shapes: Mapping[Tensor | _TensorRef, Sequence[int | str | None]]
    | Sequence[tuple[Tensor | _TensorRef | TensorOperand, Sequence[int | str | None]]],
    data: Sequence[TensorOperand] | None = None,
    summarize: int | None = None,
    message: str | None = None,
    name: str | None = None,
) -> None:
    """Assert that multiple tensors match their expected shape specifications.

    Args:
        shapes: Mapping or sequence of `(tensor, shape_spec)` pairs.
        data: Optional tensors to print on failure.
        summarize: Optional number of elements to print.
        message: Optional error message.
        name: Optional operation name.

    Raises:
        ValueError: If any tensor shape or named dimension symbol mismatches.
    """
    del data, summarize, name
    pairs: list[tuple[Tensor | _TensorRef | TensorOperand, Sequence[int | str | None]]] = (
        list(shapes.items()) if isinstance(shapes, Mapping) else list(shapes)
    )
    symbol_bindings: dict[str, int] = {}
    for raw_tensor, spec in pairs:
        tensor = raw_tensor.deref() if isinstance(raw_tensor, _TensorRef) else raw_tensor
        actual = _infer_shape(tensor)
        if len(actual) != len(spec):
            raise ValueError(
                message or f"Rank mismatch: expected {len(spec)} dims, got {len(actual)}."
            )
        for act_dim, exp_dim in zip(actual, spec):
            if exp_dim is not None and exp_dim != ".":
                if isinstance(exp_dim, int):
                    if act_dim != exp_dim:
                        raise ValueError(
                            message or f"Dimension mismatch: expected {exp_dim}, got {act_dim}."
                        )
                else:
                    if exp_dim in symbol_bindings and symbol_bindings[exp_dim] != act_dim:
                        raise ValueError(
                            message
                            or f"Symbolic dimension '{exp_dim}' mismatch: {symbol_bindings[exp_dim]} != {act_dim}."
                        )
                    symbol_bindings[exp_dim] = act_dim


def assert_type(
    tensor: Tensor | Variable | msc_core.Tensor,
    tf_type: DTypeLike,
    message: str | None = None,
    name: str | None = None,
) -> None:
    """Assert that `tensor` is of data type `tf_type`.

    Args:
        tensor: Input `Tensor` or `Variable`.
        tf_type: Expected `DType` or type specifier.
        message: Optional error message.
        name: Optional operation name.

    Raises:
        TypeError: If `tensor.dtype` does not match `tf_type`.
    """
    del name
    expected = to_dtype(tf_type)
    actual = to_dtype(tensor.dtype)
    if actual != expected:
        raise TypeError(message or f"Expected dtype {expected}, got {actual}.")


def check_numerics(
    tensor: Tensor | TensorOperand,
    message: str,
    name: str | None = None,
) -> Tensor:
    """Check a floating-point tensor for `NaN` and `Inf` values.

    Args:
        tensor: Input floating-point tensor.
        message: Prefix message included in the exception if `NaN` or `Inf` is found.
        name: Optional operation name.

    Returns:
        Tensor: The validated input `Tensor`.

    Raises:
        ValueError: If `tensor` contains any `NaN` or `Inf` values.
    """
    del name
    vals = _flatten_floats(tensor)
    for v in vals:
        if math.isnan(v) or math.isinf(v):
            raise ValueError(f"{message}: Tensor had NaN or Inf values.")
    return _wrap_tensor(tensor)


def is_numeric_tensor(tensor: Tensor | Variable | TensorOperand) -> bool:
    """Return `True` if `tensor` is a tensor with a numeric (float, int, uint, or complex) dtype.

    Args:
        tensor: Candidate object to inspect.

    Returns:
        bool: `True` if `tensor` is a numeric tensor, else `False`.
    """
    if not isinstance(tensor, (Tensor, Variable, msc_core.Tensor)):
        return False
    dt = tensor.dtype
    dt_str = str(dt.value if isinstance(dt, msc_dtype.DType) else dt).lower()
    return any(k in dt_str for k in ("float", "int", "uint", "complex", "bfloat"))
