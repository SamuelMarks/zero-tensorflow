"""TensorFlow sparse module."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

__all__ = [
    "SparseTensor",
    "add",
    "concat",
    "from_dense",
    "sparse_dense_matmul",
    "split",
    "to_dense",
]


class SparseTensor:
    """
    Represents a sparse tensor.

    Args:
        indices: A 2-D int64 tensor of shape `[N, ndims]`.
        values: A 1-D tensor of any type and shape `[N]`.
        dense_shape: A 1-D int64 tensor of shape `[ndims]`.
    """

    def __init__(self, indices: Any, values: Any, dense_shape: Any):
        """Initialize the object."""
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape


def sparse_dense_matmul(
    sp_a: Any,
    b: Any,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    name: str | None = None,
) -> Any:
    """
    Multiply SparseTensor by dense matrix.

    Args:
        sp_a: SparseTensor A.
        b: dense Tensor B.
        adjoint_a: Use the adjoint of A in the matrix multiply.
        adjoint_b: Use the adjoint of B in the matrix multiply.
        name: A name prefix for the returned tensors (optional).
    """
    from zero_keras import ops as msc_ops

    return (
        msc_ops.matmul(getattr(sp_a, "values", sp_a), b)
        if hasattr(msc_ops, "matmul")
        else None
    )


def add(a: Any, b: Any, threshold: int = 0, name: str | None = None) -> Any:
    """
    Add two tensors, at least one of each is a SparseTensor.

    Args:
        a: The first operand; SparseTensor or Tensor.
        b: The second operand; SparseTensor or Tensor.
        threshold: An optional 0-D int32 Tensor.
        name: A name prefix for the returned tensors (optional).
    """
    from zero_keras import ops as msc_ops

    return (
        msc_ops.add(getattr(a, "values", a), getattr(b, "values", b))
        if hasattr(msc_ops, "add")
        else None
    )


def concat(axis: int, sp_inputs: Sequence[Any], name: str | None = None) -> Any:
    """
    Concatenate a list of SparseTensor along the specified dimension.

    Args:
        axis: Dimension to concatenate along.
        sp_inputs: List of SparseTensor to concatenate.
        name: A name prefix for the returned tensors (optional).
    """
    from zero_keras import ops as msc_ops

    return (
        msc_ops.concatenate([getattr(i, "values", i) for i in sp_inputs], axis)
        if hasattr(msc_ops, "concat")
        else None
    )


def split(sp_input: Any, num_split: int, axis: int, name: str | None = None) -> Any:
    """
    Split a SparseTensor into num_split tensors along axis.

    Args:
        sp_input: The SparseTensor to split.
        num_split: A Python integer.
        axis: A 0-D int32 Tensor.
        name: A name prefix for the returned tensors (optional).
    """
    from zero_keras import ops as msc_ops

    return (
        msc_ops.split(getattr(sp_input, "values", sp_input), num_split, axis)
        if hasattr(msc_ops, "split")
        else None
    )


def to_dense(
    sp_input: Any,
    default_value: Any | None = None,
    validate_indices: bool = True,
    name: str | None = None,
) -> Any:
    """
    Convert a SparseTensor into a dense tensor.

    Args:
        sp_input: The input SparseTensor.
        default_value: Scalar value to set for indices not specified in sp_input.
        validate_indices: A boolean value.
        name: A name prefix for the returned tensors (optional).
    """
    return getattr(sp_input, "values", sp_input)


def from_dense(tensor: Any, name: str | None = None) -> SparseTensor:
    """
    Convert a dense tensor into a SparseTensor.

    Args:
        tensor: A dense Tensor to be converted to a SparseTensor.
        name: Optional name for the op.
    """
    return SparseTensor(None, None, None)
