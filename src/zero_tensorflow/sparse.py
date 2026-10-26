"""TensorFlow `sparse` tensor module."""

from __future__ import annotations

import builtins
import math
from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    NestedTensorData,
    NumpyConvertibleProtocol,
    TensorOperand,
    ValuesHolderProtocol,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops

from zero_keras import ops as _k_ops  # isort: skip

__all__ = [
    "SparseTensor",
    "add",
    "concat",
    "from_dense",
    "sparse_dense_matmul",
    "split",
    "to_dense",
]


def _extract_values(x: Tensor | TensorOperand) -> msc_core.Tensor:
    """Extract or convert the underlying compiler `Tensor` from `x`.

    Args:
        x: Input `SparseTensor`, `Tensor`, or array-like operand.

    Returns:
        msc_core.Tensor: Underlying compiler tensor.
    """
    if isinstance(x, SparseTensor):
        dense = to_dense(x)
        if dense is not None:
            return _to_tensor(dense)
        return _to_tensor([0.0])
    if isinstance(x, ValuesHolderProtocol):
        return _to_tensor(x.values)
    return _to_tensor(x)


def _to_int_list(raw: TensorOperand) -> list[int]:
    """Convert a shape or coordinate sequence into a flat `list[int]`.

    Args:
        raw: Input shape or coordinate sequence.

    Returns:
        list[int]: Flat list of integers.
    """
    if isinstance(raw, NumpyConvertibleProtocol):
        raw = as_operand(raw.numpy())
    if isinstance(raw, ArrayLikeProtocol):
        raw = raw.tolist()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [int(v) for v in raw if isinstance(v, (int, float))]
    return []


def _to_coord_list(raw: TensorOperand) -> list[list[int]]:
    """Convert an `indices` operand into a 2-D `list[list[int]]`.

    Args:
        raw: Input indices tensor or sequence.

    Returns:
        list[list[int]]: 2-D coordinate list.
    """
    if isinstance(raw, NumpyConvertibleProtocol):
        raw = as_operand(raw.numpy())
    if isinstance(raw, ArrayLikeProtocol):
        raw = raw.tolist()
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        coords: list[list[int]] = []
        for row in raw:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                coords.append([int(c) for c in row if isinstance(c, (int, float))])
            else:
                coords.append([int(row) if isinstance(row, (int, float)) else 0])
        return coords
    return []


def _to_scalar_list(raw: TensorOperand) -> list[float]:
    """Convert a 1-D values operand into a flat `list[float]`.

    Args:
        raw: Input values tensor or sequence.

    Returns:
        list[float]: Flat list of numeric values.
    """
    if isinstance(raw, NumpyConvertibleProtocol):
        raw = as_operand(raw.numpy())
    if isinstance(raw, ArrayLikeProtocol):
        raw = raw.tolist()
    if isinstance(raw, (int, float)):
        return [float(raw)]
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [float(item) for item in raw if isinstance(item, (int, float))]
    return []


class SparseTensor:
    """Represents a sparse tensor via `(indices, values, dense_shape)`."""

    def __init__(
        self,
        indices: Tensor | TensorOperand,
        values: Tensor | TensorOperand,
        dense_shape: Tensor | TensorOperand,
    ) -> None:
        """Initialize a `SparseTensor`.

        Args:
            indices: 2-D int64 tensor of shape `[N, ndims]`.
            values: 1-D tensor of length `N` holding non-zero values.
            dense_shape: 1-D int64 tensor of length `ndims` specifying dense dimensions.
        """
        self.indices = indices
        self.values = values
        self.dense_shape = dense_shape

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the dense shape tuple of the sparse tensor.

        Returns:
            tuple[int, ...]: Dense shape tuple.
        """
        return tuple(_to_int_list(self.dense_shape))


def sparse_dense_matmul(
    sp_a: SparseTensor | Tensor | TensorOperand,
    b: Tensor | TensorOperand,
    adjoint_a: bool = False,
    adjoint_b: bool = False,
    name: str | None = None,
) -> Tensor:
    """Multiply a `SparseTensor` (or dense tensor) `sp_a` by a dense matrix `b`.

    Args:
        sp_a: Left-hand sparse or dense matrix.
        b: Right-hand dense matrix.
        adjoint_a: If `True`, transpose/adjoint `sp_a` before multiplication.
        adjoint_b: If `True`, transpose/adjoint `b` before multiplication.
        name: Optional operation name.

    Returns:
        Tensor: Dense matrix product.
    """
    del name
    val_a = _extract_values(sp_a)
    val_b = _to_tensor(b)
    if adjoint_a and len(val_a.shape) >= 2:
        val_a = _k_ops.transpose(val_a)
    if adjoint_b and len(val_b.shape) >= 2:
        val_b = _k_ops.transpose(val_b)
    return _wrap_tensor(as_operand(_ops.matmul(val_a, val_b)))


def add(
    a: SparseTensor | Tensor | TensorOperand,
    b: SparseTensor | Tensor | TensorOperand,
    threshold: float = 0,
    name: str | None = None,
) -> SparseTensor | Tensor:
    """Add two tensors, at least one of which is a `SparseTensor`.

    Args:
        a: First `SparseTensor` or `Tensor`.
        b: Second `SparseTensor` or `Tensor`.
        threshold: Magnitude threshold when both inputs are `SparseTensor`.
        name: Optional operation name.

    Returns:
        SparseTensor | Tensor: `SparseTensor` if both inputs are `SparseTensor`, else `Tensor`.
    """
    del threshold, name
    res = as_operand(_ops.add(_extract_values(a), _extract_values(b)))
    if isinstance(a, SparseTensor) and isinstance(b, SparseTensor):
        return SparseTensor(a.indices, _wrap_tensor(res), a.dense_shape)
    return _wrap_tensor(res)


def concat(
    axis: int,
    sp_inputs: Sequence[SparseTensor | Tensor | TensorOperand],
    name: str | None = None,
    expand_nonconcat_dims: bool = False,
    axis_arg: int | None = None,
) -> SparseTensor:
    """Concatenate a list of `SparseTensor`s along the specified `axis`.

    Args:
        axis: Dimension along which to concatenate.
        sp_inputs: Sequence of `SparseTensor`s to concatenate.
        name: Optional operation name.
        expand_nonconcat_dims: Whether to expand non-concatenated dimensions.
        axis_arg: Optional explicit axis override.

    Returns:
        SparseTensor: Concatenated `SparseTensor`.
    """
    del name, expand_nonconcat_dims
    actual_axis = axis_arg if axis_arg is not None else axis
    vals = [_extract_values(x) for x in sp_inputs]
    res = _k_ops.concatenate(vals, axis=actual_axis)
    first = sp_inputs[0] if sp_inputs else None
    indices = first.indices if isinstance(first, SparseTensor) else None
    return SparseTensor(indices, _wrap_tensor(res), list(res.shape))


def split(
    sp_input: SparseTensor | Tensor | TensorOperand = None,
    num_split: int = 1,
    axis: int = 0,
    name: str | None = None,
) -> list[SparseTensor]:
    """Split a `SparseTensor` into `num_split` tensors along `axis`.

    Args:
        sp_input: Input `SparseTensor` to split.
        num_split: Number of ways to split.
        axis: Dimension along which to split.
        name: Optional operation name.

    Returns:
        list[SparseTensor]: List of split `SparseTensor`s.
    """
    del name
    val = _extract_values(sp_input)
    res = _wrap(as_operand(_ops.split(val, num_split, axis=axis)))
    parts: Sequence[Tensor] = res if isinstance(res, (builtins.list, builtins.tuple)) else [res]
    indices = sp_input.indices if isinstance(sp_input, SparseTensor) else None
    return [SparseTensor(indices, r, list(r.shape)) for r in parts]


def to_dense(
    sp_input: SparseTensor | Tensor | TensorOperand,
    default_value: float | int = 0,
    validate_indices: bool = True,
    name: str | None = None,
) -> Tensor | None:
    """Convert a `SparseTensor` into a dense `Tensor`.

    Args:
        sp_input: Input `SparseTensor` or `Tensor`.
        default_value: Scalar value to set for indices not specified in `sp_input`.
        validate_indices: If `True`, validate ordering and uniqueness of indices.
        name: Optional operation name.

    Returns:
        Tensor | None: Dense `Tensor` representation, or `None` if `sp_input` is empty.
    """
    del validate_indices, name
    if sp_input is None:
        return None
    if isinstance(sp_input, SparseTensor):
        if sp_input.values is None:
            return None
        shape_list = _to_int_list(sp_input.dense_shape)
        coords = _to_coord_list(sp_input.indices)
        vals = _to_scalar_list(sp_input.values)
        if shape_list and coords and len(coords) == len(vals) and len(coords[0]) == len(shape_list):
            total = math.prod(shape_list)
            flat: list[float] = [float(default_value)] * total
            strides: list[int] = [1] * len(shape_list)
            for d in range(len(shape_list) - 2, -1, -1):
                strides[d] = strides[d + 1] * shape_list[d + 1]
            for coord, v in zip(coords, vals):
                flat_idx = sum(c * s for c, s in zip(coord, strides))
                flat[flat_idx] = v
            return _wrap_tensor(_k_ops.reshape(_to_tensor(flat), tuple(shape_list)))
        return _wrap_tensor(_to_tensor(sp_input.values))
    return _wrap_tensor(_to_tensor(sp_input))


def _collect_nonzero(
    arr: NestedTensorData,
    prefix: list[int],
    indices: list[list[int]],
    values: list[float | int],
) -> None:
    """Recursively collect non-zero coordinates and values from a nested sequence.

    Args:
        arr: Nested sequence or scalar value.
        prefix: Current coordinate prefix.
        indices: Output list of non-zero coordinate lists.
        values: Output list of non-zero scalar values.
    """
    if isinstance(arr, Sequence) and not isinstance(arr, (str, bytes)):
        for idx, sub in enumerate(arr):
            _collect_nonzero(sub, [*prefix, idx], indices, values)
    elif isinstance(arr, (int, float)) and arr != 0:
        indices.append(prefix)
        values.append(arr)


def from_dense(
    tensor: Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Convert a dense `Tensor` into a `SparseTensor`.

    Args:
        tensor: Dense input `Tensor` or array-like operand.
        name: Optional operation name.

    Returns:
        SparseTensor: Sparse representation containing only non-zero entries.
    """
    del name
    if tensor is None:
        return SparseTensor(indices=[], values=[], dense_shape=[])
    t_dense = _wrap_tensor(_to_tensor(tensor))
    raw = t_dense.numpy()
    nested: NestedTensorData = raw.tolist() if isinstance(raw, ArrayLikeProtocol) else raw
    indices: list[list[int]] = []
    values: list[float | int] = []
    _collect_nonzero(nested, [], indices, values)
    dense_shape = list(t_dense.shape)
    return SparseTensor(
        indices=indices,
        values=_wrap_tensor(values),
        dense_shape=dense_shape,
    )
