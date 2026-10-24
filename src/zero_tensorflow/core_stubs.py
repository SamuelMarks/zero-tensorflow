"""Top-level core TensorFlow helper and stub functions."""

from __future__ import annotations

import math
from types import TracebackType
from typing import Protocol, runtime_checkable

from zero_tensorflow._typing import DTypeLike, TensorOperand
from zero_tensorflow.core_tensor import Tensor, Variable, _to_tensor, _wrap_tensor
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.sparse import SparseTensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as msc_ops


@runtime_checkable
class _SymbolicFlagProtocol(Protocol):
    """Protocol for objects declaring an `is_symbolic` boolean attribute."""

    @property
    def is_symbolic(self) -> bool:
        """Return whether the tensor is symbolic.

        Returns:
            bool: True if symbolic.
        """


@runtime_checkable
class _ShapedProtocol(Protocol):
    """Protocol for objects declaring a `.shape` sequence attribute."""

    @property
    def shape(self) -> tuple[int, ...]:
        """Return tensor shape tuple.

        Returns:
            tuple[int, ...]: Shape dimensions.
        """


def double(
    *args: Tensor | msc_core.Tensor | float | int | bool,
    **kwargs: TensorOperand,
) -> Tensor:
    """Cast the first positional argument to `float64`.

    Args:
        *args: Positional arguments where `args[0]` is the input tensor.
        **kwargs: Unused keyword arguments.

    Returns:
        Tensor: Tensor cast to `float64`.
    """
    del kwargs
    first = args[0] if args else 0.0
    return _wrap_tensor(msc_ops.cast(_to_tensor(first), msc_core.DType.Float64))


def import_graph_def(*args: TensorOperand, **kwargs: TensorOperand) -> None:
    """Import a serialized `GraphDef` into the current graph.

    Args:
        *args: Positional graph arguments.
        **kwargs: Keyword graph arguments.
    """
    del args, kwargs
    return None


class init_scope:
    """Context manager lifting operations out of function-building graphs."""

    def __enter__(self) -> init_scope:
        """Enter the `init_scope` context.

        Returns:
            init_scope: Active context manager instance.
        """
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        """Exit the `init_scope` context.

        Args:
            exc_type: Exception type if raised, else None.
            exc_val: Exception instance if raised, else None.
            exc_tb: Traceback if raised, else None.
        """
        del exc_type, exc_val, exc_tb


def is_symbolic_tensor(tensor: Tensor | TensorOperand | _SymbolicFlagProtocol) -> bool:
    """Check whether `tensor` is a symbolic graph tensor.

    Args:
        tensor: Candidate object to inspect.

    Returns:
        bool: True if `tensor` has `is_symbolic` set to True.
    """
    if isinstance(tensor, _SymbolicFlagProtocol):
        return bool(tensor.is_symbolic)
    return False


def is_tensor(
    tensor: Tensor | Variable | SparseTensor | RaggedTensor | TensorOperand,
) -> bool:
    """Check whether `tensor` is a recognized compiler or framework tensor type.

    Args:
        tensor: Candidate object to check.

    Returns:
        bool: True if `tensor` is a dense, variable, sparse, or ragged tensor.
    """
    return isinstance(
        tensor,
        (
            Tensor,
            Variable,
            SparseTensor,
            RaggedTensor,
            msc_core.Tensor,
            msc_core.SparseTensor,
            msc_core.RaggedTensor,
        ),
    )


def ragged_fill_empty_rows(
    *args: Tensor | TensorOperand, **kwargs: TensorOperand
) -> tuple[Tensor | TensorOperand, ...]:
    """Fill empty rows in a 2D ragged or sparse tensor.

    Args:
        *args: Input tensor arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        tuple[Tensor | TensorOperand, ...]: Tuple of positional arguments.
    """
    del kwargs
    return args


def ragged_fill_empty_rows_grad(
    *args: Tensor | TensorOperand, **kwargs: TensorOperand
) -> tuple[Tensor | TensorOperand, ...]:
    """Compute gradient for `ragged_fill_empty_rows`.

    Args:
        *args: Input gradient arguments.
        **kwargs: Optional keyword arguments.

    Returns:
        tuple[Tensor | TensorOperand, ...]: Tuple of positional arguments.
    """
    del kwargs
    return args


def size(
    input: Tensor | TensorOperand,
    out_type: DTypeLike = None,
    name: str | None = None,
) -> int:
    """Return the total number of elements in `input`.

    Args:
        input: Input tensor or shaped object.
        out_type: Optional output integer data type.
        name: Optional operation name.

    Returns:
        int: Total number of elements in `input`, or `0` if unshaped.
    """
    del out_type, name
    if isinstance(input, _ShapedProtocol) and input.shape:
        return int(math.prod(input.shape))
    return 0


def space_to_batch(
    *args: Tensor | TensorOperand, **kwargs: TensorOperand
) -> Tensor | TensorOperand | None:
    """Zero-pad and rearrange blocks of spatial data into batch dimension.

    Args:
        *args: Positional arguments where `args[0]` is the input tensor.
        **kwargs: Keyword arguments.

    Returns:
        Tensor | TensorOperand | None: First positional argument if provided, else None.
    """
    del kwargs
    return args[0] if args else None


def space_to_batch_nd(
    *args: Tensor | TensorOperand, **kwargs: TensorOperand
) -> Tensor | TensorOperand | None:
    """Divide N-D spatial dimensions into a batch of blocks.

    Args:
        *args: Positional arguments where `args[0]` is the input tensor.
        **kwargs: Keyword arguments.

    Returns:
        Tensor | TensorOperand | None: First positional argument if provided, else None.
    """
    del kwargs
    return args[0] if args else None


def string(*args: Tensor | TensorOperand, **kwargs: TensorOperand) -> Tensor | TensorOperand | None:
    """Convert or pass through a string tensor operand.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor | TensorOperand | None: First positional argument if provided, else None.
    """
    del kwargs
    return args[0] if args else None
