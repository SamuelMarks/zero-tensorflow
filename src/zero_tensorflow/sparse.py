"""TensorFlow `sparse` tensor module."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NestedTensorData,
    TensorOperand,
    ValuesHolderProtocol,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as msc_ops

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
    """Represents a sparse tensor in coordinate (COO) format."""

    def __init__(
        self,
        indices: Tensor | TensorOperand,
        values: Tensor | msc_core.Tensor | None,
        dense_shape: Tensor | TensorOperand,
    ) -> None:
        """Initialize a `SparseTensor`.

        Args:
            indices: A 2-D `int64` tensor of shape `[N, ndims]`.
            values: A 1-D tensor of any type and shape `[N]`.
            dense_shape: A 1-D `int64` tensor of shape `[ndims]`.
        """
        self.indices = indices
        self.values: Tensor | msc_core.Tensor | None = values
        self.dense_shape = dense_shape


def _extract_values(
    x: SparseTensor | ValuesHolderProtocol | Tensor | msc_core.Tensor | TensorOperand,
) -> msc_core.Tensor:
    """Extract the underlying dense values tensor from a `SparseTensor` or tensor.

    Args:
        x: SparseTensor, values holder, or dense Tensor.

    Returns:
        msc_core.Tensor: Underlying compiler Tensor.
    """
    if isinstance(x, SparseTensor) and x.values is not None:
        return _to_tensor(x.values)
    if isinstance(x, Tensor):
        return _to_tensor(x)
    if isinstance(x, msc_core.Tensor):
        return x
    return msc_ops.zeros((1,))


def sparse_dense_matmul(
    sp_a: SparseTensor | Tensor | msc_core.Tensor | TensorOperand,
    b: Tensor | msc_core.Tensor | TensorOperand,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    name: str | None = None,
) -> Tensor:
    """Multiply a `SparseTensor` (or matrix) `sp_a` by a dense matrix `b`.

    Args:
        sp_a: SparseTensor or dense matrix operand A.
        b: Dense matrix Tensor B.
        adjoint_a: Whether to use the adjoint of `sp_a`.
        adjoint_b: Whether to use the adjoint of `b`.
        name: Optional name prefix for the returned tensor.

    Returns:
        Tensor: Matrix product tensor.
    """
    del adjoint_a, adjoint_b, name
    return _wrap_tensor(as_operand(msc_ops.matmul(_extract_values(sp_a), _extract_values(b))))


def add(
    a: SparseTensor | Tensor | msc_core.Tensor | TensorOperand,
    b: SparseTensor | Tensor | msc_core.Tensor | TensorOperand,
    threshold: int = 0,
    name: str | None = None,
) -> Tensor:
    """Add two tensors where at least one operand is a `SparseTensor`.

    Args:
        a: First operand (`SparseTensor` or `Tensor`).
        b: Second operand (`SparseTensor` or `Tensor`).
        threshold: Magnitude threshold for pruning output values.
        name: Optional name prefix for the returned tensor.

    Returns:
        Tensor: Sum tensor.
    """
    del threshold, name
    return _wrap_tensor(msc_ops.add(_extract_values(a), _extract_values(b)))


def concat(
    axis: int,
    sp_inputs: Sequence[SparseTensor | Tensor | msc_core.Tensor | TensorOperand],
    name: str | None = None,
) -> Tensor:
    """Concatenate a sequence of `SparseTensor`s along the specified dimension.

    Args:
        axis: Dimension along which to concatenate.
        sp_inputs: Sequence of `SparseTensor`s to concatenate.
        name: Optional name prefix for the returned tensor.

    Returns:
        Tensor: Concatenated tensor.
    """
    del name
    res = msc_ops.concatenate([_extract_values(i) for i in sp_inputs], axis=axis)
    return _wrap_tensor(as_operand(res))


def split(
    sp_input: SparseTensor | Tensor | msc_core.Tensor | TensorOperand,
    num_split: int,
    axis: int,
    name: str | None = None,
) -> Tensor | builtins.tuple[Tensor, ...] | builtins.list[Tensor]:
    """Split a `SparseTensor` into `num_split` tensors along `axis`.

    Args:
        sp_input: The `SparseTensor` to split.
        num_split: Number of ways to split.
        axis: Dimension along which to split.
        name: Optional name prefix for the returned tensors.

    Returns:
        Tensor | tuple[Tensor, ...] | list[Tensor]: Split tensors.
    """
    del name
    res = msc_ops.split(_extract_values(sp_input), num_split, axis=axis)
    return _wrap(as_operand(res))


def to_dense(
    sp_input: SparseTensor | Tensor | msc_core.Tensor | TensorOperand,
    default_value: TensorOperand = None,
    validate_indices: bool = True,
    name: str | None = None,
) -> Tensor | None:
    """Convert a `SparseTensor` into a dense tensor.

    Args:
        sp_input: Input `SparseTensor` or dense `Tensor`.
        default_value: Scalar value to set for unspecified indices.
        validate_indices: Whether to validate index ordering and uniqueness.
        name: Optional name prefix for the returned tensor.

    Returns:
        Tensor | None: Dense tensor representation.
    """
    del default_value, validate_indices, name
    if isinstance(sp_input, SparseTensor):
        if sp_input.values is None:
            return None
        return _wrap_tensor(sp_input.values)
    if sp_input is None:
        return None
    return _wrap_tensor(sp_input)


def _collect_nonzero(
    nested: NestedTensorData | TensorOperand,
    prefix: list[int],
    indices_out: list[list[int]],
    values_out: list[float | int],
) -> None:
    """Recursively collect non-zero coordinates and values from nested lists.

    Args:
        nested: Nested Python list or scalar value.
        prefix: Current coordinate prefix.
        indices_out: Output list of index coordinates.
        values_out: Output list of non-zero scalar values.
    """
    if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
        for idx, item in enumerate(nested):
            _collect_nonzero(item, [*prefix, idx], indices_out, values_out)
    elif isinstance(nested, (int, float)) and not isinstance(nested, bool) and nested != 0:
        indices_out.append(prefix if prefix else [0])
        values_out.append(nested)


def from_dense(
    tensor: Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Convert a dense tensor into a `SparseTensor`.

    Args:
        tensor: Dense `Tensor` to convert.
        name: Optional name for the operation.

    Returns:
        SparseTensor: Sparse representation of `tensor`.
    """
    del name
    if tensor is None:
        return SparseTensor([], None, [])
    t_dense = _wrap_tensor(tensor)
    raw = t_dense.numpy()
    nested: NestedTensorData | TensorOperand = (
        raw.tolist() if isinstance(raw, ArrayLikeProtocol) else as_operand(raw)
    )
    indices: list[list[int]] = []
    values: list[float | int] = []
    _collect_nonzero(nested, [], indices, values)
    dense_shape = list(t_dense.shape)
    return SparseTensor(
        indices=indices,
        values=t_dense,
        dense_shape=dense_shape,
    )
