"""TensorFlow `sparse` tensor module."""

from __future__ import annotations

import builtins
import itertools
import math
from collections.abc import Callable, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    DTypeLike,
    NestedTensorData,
    NumpyConvertibleProtocol,
    TensorOperand,
    ValuesHolderProtocol,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap, _wrap_tensor, to_dtype
from zero_tensorflow.dtypes import DType

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops

from zero_keras import ops as _k_ops  # isort: skip

__all__ = [
    "SparseTensor",
    "add",
    "bincount",
    "concat",
    "cross",
    "cross_hashed",
    "expand_dims",
    "eye",
    "fill_empty_rows",
    "from_dense",
    "map_values",
    "mask",
    "maximum",
    "minimum",
    "reduce_max",
    "reduce_sum",
    "reorder",
    "reset_shape",
    "reshape",
    "retain",
    "segment_mean",
    "segment_sqrt_n",
    "segment_sum",
    "slice",
    "softmax",
    "sparse_dense_matmul",
    "split",
    "to_dense",
    "to_indicator",
    "transpose",
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

    @property
    def dtype(self) -> DType:
        """Return the element `DType` of `self.values`.

        Returns:
            DType: Element data type.
        """
        if isinstance(self.values, Tensor) and self.values.dtype is not None:
            return self.values.dtype
        return DType.Float32


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


def bincount(
    values: SparseTensor | Tensor | TensorOperand = None,
    weights: SparseTensor | Tensor | TensorOperand = None,
    axis: int = 0,
    minlength: int | None = None,
    maxlength: int | None = None,
    binary_output: bool = False,
    name: str | None = None,
) -> SparseTensor:
    """Count the number of occurrences of each value in an integer `SparseTensor` or `Tensor`.

    Args:
        values: Integer `SparseTensor` or `Tensor`.
        weights: Optional weight tensor.
        axis: Reduction axis (`0` or `-1`).
        minlength: Optional minimum bin count.
        maxlength: Optional maximum bin count.
        binary_output: Whether to output binary indicators (`1` if present).
        name: Optional operation name.

    Returns:
        SparseTensor: Sparse bincount representation.
    """
    del weights, axis, name
    raw_v = values.values if isinstance(values, SparseTensor) else values
    ints = [v for v in _to_int_list(raw_v) if v >= 0 and (maxlength is None or v < maxlength)]
    max_b = max(ints) + 1 if ints else 0
    nbins = max(max_b, int(minlength) if minlength is not None else 0)
    counts = [0] * nbins
    for v in ints:
        counts[v] = 1 if binary_output else counts[v] + 1
    return from_dense(counts)


def cross(
    inputs: Sequence[SparseTensor | Tensor | TensorOperand] = (),
    name: str | None = None,
    separator: str = "_X_",
) -> SparseTensor:
    """Generate sparse feature cross from a list of sparse and dense tensors.

    Args:
        inputs: Sequence of 2-D `SparseTensor` or `Tensor` instances.
        name: Optional operation name.
        separator: Separator string joining crossed tokens.

    Returns:
        SparseTensor: 2-D `SparseTensor` of crossed string values.
    """
    del name
    if not inputs:
        return SparseTensor(indices=[], values=Tensor([], dtype=DType.String), dense_shape=[0, 0])
    dense_rows = [Tensor(_extract_values(inp)).tolist() for inp in inputs]
    nrows = min((len(r) if isinstance(r, builtins.list) else 0 for r in dense_rows), default=0)
    indices: list[list[int]] = []
    vals: list[str] = []
    max_cols = 0
    for b in range(nrows):
        row_tokens = [
            [str(x) for x in (r[b] if isinstance(r[b], builtins.list) else [r[b]])]
            for r in dense_rows
            if isinstance(r, builtins.list)
        ]
        crossed = [separator.join(p) for p in itertools.product(*row_tokens)]
        max_cols = max(max_cols, len(crossed))
        for c, tok in enumerate(crossed):
            indices.append([b, c])
            vals.append(tok)
    return SparseTensor(
        indices=indices,
        values=Tensor(vals, dtype=DType.String),
        dense_shape=[nrows, max_cols],
    )


def cross_hashed(
    inputs: Sequence[SparseTensor | Tensor | TensorOperand] = (),
    num_buckets: int = 0,
    hash_key: int | None = None,
    name: str | None = None,
) -> SparseTensor:
    """Generate hashed sparse feature cross from a list of sparse and dense tensors.

    Args:
        inputs: Sequence of 2-D `SparseTensor` or `Tensor` instances.
        num_buckets: Number of hash buckets (`0` uses `2**31`).
        hash_key: Optional custom hash seed.
        name: Optional operation name.

    Returns:
        SparseTensor: 2-D `SparseTensor` of `int64` hashed feature crosses.
    """
    del hash_key
    sp_c = cross(inputs=inputs, name=name)
    mod = max(1, int(num_buckets)) if num_buckets > 0 else (1 << 31)
    raw_v = sp_c.values.tolist() if isinstance(sp_c.values, Tensor) else []
    hashed = [
        abs(hash(str(v))) % mod for v in (raw_v if isinstance(raw_v, builtins.list) else [raw_v])
    ]
    return SparseTensor(
        indices=sp_c.indices,
        values=Tensor(hashed, dtype=DType.Int64),
        dense_shape=sp_c.dense_shape,
    )


def expand_dims(
    sp_input: SparseTensor | Tensor | TensorOperand = None,
    axis: int = 0,
    name: str | None = None,
) -> SparseTensor:
    """Insert a dimension of size `1` into `sp_input`'s shape at `axis`.

    Args:
        sp_input: Input `SparseTensor` or `Tensor`.
        axis: Dimension index at which to insert the singleton axis.
        name: Optional operation name.

    Returns:
        SparseTensor: Expanded `SparseTensor`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    old_shape = _to_int_list(sp.dense_shape)
    norm_ax = axis if axis >= 0 else len(old_shape) + 1 + axis
    new_shape = [*old_shape[:norm_ax], 1, *old_shape[norm_ax:]]
    new_coords = [[*c[:norm_ax], 0, *c[norm_ax:]] for c in _to_coord_list(sp.indices)]
    return SparseTensor(indices=new_coords, values=sp.values, dense_shape=new_shape)


def eye(
    num_rows: int = 1,
    num_columns: int | None = None,
    dtype: DTypeLike = "float32",
    name: str | None = None,
) -> SparseTensor:
    """Create a 2-D `SparseTensor` with ones along the diagonal and zeros elsewhere.

    Args:
        num_rows: Non-negative number of rows.
        num_columns: Optional number of columns (defaults to `num_rows`).
        dtype: Element `DType`.
        name: Optional operation name.

    Returns:
        SparseTensor: Sparse identity matrix.
    """
    del name
    r = max(0, int(num_rows))
    c = r if num_columns is None else max(0, int(num_columns))
    diag_n = min(r, c)
    indices = [[i, i] for i in range(diag_n)]
    vals = Tensor([1.0] * diag_n, dtype=to_dtype(dtype))
    return SparseTensor(indices=indices, values=vals, dense_shape=[r, c])


def fill_empty_rows(
    sp_input: SparseTensor | Tensor | TensorOperand = None,
    default_value: float | int = 0,
    name: str | None = None,
) -> tuple[SparseTensor, Tensor]:
    """Fill empty rows in a 2-D `SparseTensor` with `default_value`.

    Args:
        sp_input: 2-D `SparseTensor`.
        default_value: Scalar value to insert into empty rows at column `0`.
        name: Optional operation name.

    Returns:
        tuple[SparseTensor, Tensor]: `(filled_sparse_tensor, empty_row_indicator)`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    shape_l = _to_int_list(sp.dense_shape)
    nrows = shape_l[0] if shape_l else 1
    ncols = shape_l[1] if len(shape_l) >= 2 else 1
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    present_rows = {c[0] for c in coords if c}
    empty_ind: list[bool] = []
    paired = list(zip(coords, vals))
    for r in range(nrows):
        is_emp = r not in present_rows
        empty_ind.append(is_emp)
        if is_emp:
            paired.append(([r, 0], float(default_value)))
    paired.sort(key=lambda item: item[0])
    out_coords = [p[0] for p in paired]
    out_vals = Tensor([p[1] for p in paired], dtype=sp.dtype)
    return (
        SparseTensor(indices=out_coords, values=out_vals, dense_shape=[nrows, ncols]),
        Tensor(empty_ind, dtype=DType.Bool),
    )


def map_values(
    op: Callable[[TensorOperand], TensorOperand] | CallableOperandProtocol,
    *args: SparseTensor | Tensor | TensorOperand,
    **kwargs: TensorOperand,
) -> SparseTensor:
    """Apply `op` to the `.values` of one or more `SparseTensor`s sharing the same indices.

    Args:
        op: Callable applied to `.values`.
        *args: Input `SparseTensor`s or `Tensor`s.
        **kwargs: Keyword arguments forwarded to `op`.

    Returns:
        SparseTensor: `SparseTensor` with transformed `.values`.
    """
    first_sp = next((a for a in args if isinstance(a, SparseTensor)), None)
    unwrapped = [a.values if isinstance(a, SparseTensor) else a for a in args]
    mapped = _wrap_tensor(op(*unwrapped, **kwargs))
    if first_sp is not None:
        return SparseTensor(
            indices=first_sp.indices,
            values=mapped,
            dense_shape=first_sp.dense_shape,
        )
    return from_dense(mapped)


def mask(
    a: SparseTensor | Tensor | TensorOperand = None,
    mask_indices: Sequence[int] | Tensor = (),
    name: str | None = None,
) -> SparseTensor:
    """Mask elements of `SparseTensor` or `IndexedSlices` `a` matching `mask_indices`.

    Args:
        a: Input `SparseTensor` or `Tensor`.
        mask_indices: 1-D sequence of row indices to remove.
        name: Optional operation name.

    Returns:
        SparseTensor: Masked `SparseTensor`.
    """
    del name
    sp = a if isinstance(a, SparseTensor) else from_dense(a)
    masked_set = set(_to_int_list(mask_indices))
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    kept_c: list[list[int]] = []
    kept_v: list[float] = []
    for c, v in zip(coords, vals):
        if not c or c[0] not in masked_set:
            kept_c.append(c)
            kept_v.append(v)
    return SparseTensor(
        indices=kept_c,
        values=Tensor(kept_v, dtype=sp.dtype),
        dense_shape=sp.dense_shape,
    )


def maximum(
    sp_a: SparseTensor | Tensor | TensorOperand,
    sp_b: SparseTensor | Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Return the element-wise maximum of two `SparseTensor`s.

    Args:
        sp_a: First `SparseTensor`.
        sp_b: Second `SparseTensor`.
        name: Optional operation name.

    Returns:
        SparseTensor: Element-wise maximum `SparseTensor`.
    """
    del name
    return from_dense(_wrap_tensor(_k_ops.maximum(_extract_values(sp_a), _extract_values(sp_b))))


def minimum(
    sp_a: SparseTensor | Tensor | TensorOperand,
    sp_b: SparseTensor | Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Return the element-wise minimum of two `SparseTensor`s.

    Args:
        sp_a: First `SparseTensor`.
        sp_b: Second `SparseTensor`.
        name: Optional operation name.

    Returns:
        SparseTensor: Element-wise minimum `SparseTensor`.
    """
    del name
    return from_dense(_wrap_tensor(_k_ops.minimum(_extract_values(sp_a), _extract_values(sp_b))))


def reduce_max(
    sp_input: SparseTensor | Tensor | TensorOperand,
    axis: int | Sequence[int] | None = None,
    keepdims: bool | None = None,
    output_is_sparse: bool = False,
    name: str | None = None,
) -> SparseTensor | Tensor:
    """Compute `max` of elements across dimensions of a `SparseTensor`.

    Args:
        sp_input: Input `SparseTensor`.
        axis: Dimension(s) to reduce.
        keepdims: If `True`, retain reduced dimensions with length `1`.
        output_is_sparse: If `True`, return a `SparseTensor` instead of a dense `Tensor`.
        name: Optional operation name.

    Returns:
        SparseTensor | Tensor: Reduced maximum tensor.
    """
    del name
    ax_val: int | tuple[int, ...] | None = (
        axis
        if isinstance(axis, int)
        else (tuple(int(x) for x in axis) if isinstance(axis, Sequence) else None)
    )
    res = _wrap_tensor(_k_ops.max(_extract_values(sp_input), axis=ax_val, keepdims=bool(keepdims)))
    return from_dense(res) if output_is_sparse else res


def reduce_sum(
    sp_input: SparseTensor | Tensor | TensorOperand,
    axis: int | Sequence[int] | None = None,
    keepdims: bool | None = None,
    output_is_sparse: bool = False,
    name: str | None = None,
) -> SparseTensor | Tensor:
    """Compute `sum` of elements across dimensions of a `SparseTensor`.

    Args:
        sp_input: Input `SparseTensor`.
        axis: Dimension(s) to reduce.
        keepdims: If `True`, retain reduced dimensions with length `1`.
        output_is_sparse: If `True`, return a `SparseTensor` instead of a dense `Tensor`.
        name: Optional operation name.

    Returns:
        SparseTensor | Tensor: Reduced sum tensor.
    """
    del name
    ax_val: int | tuple[int, ...] | None = (
        axis
        if isinstance(axis, int)
        else (tuple(int(x) for x in axis) if isinstance(axis, Sequence) else None)
    )
    res = _wrap_tensor(_k_ops.sum(_extract_values(sp_input), axis=ax_val, keepdims=bool(keepdims)))
    return from_dense(res) if output_is_sparse else res


def reorder(
    sp_input: SparseTensor | Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Reorder a `SparseTensor` into canonical lexicographical row-major index order.

    Args:
        sp_input: Input `SparseTensor`.
        name: Optional operation name.

    Returns:
        SparseTensor: Canonically ordered `SparseTensor`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    paired = sorted(zip(coords, vals), key=lambda item: item[0])
    return SparseTensor(
        indices=[p[0] for p in paired],
        values=Tensor([p[1] for p in paired], dtype=sp.dtype),
        dense_shape=sp.dense_shape,
    )


def reset_shape(
    sp_input: SparseTensor | Tensor | TensorOperand,
    new_shape: Sequence[int] | Tensor | None = None,
    name: str | None = None,
) -> SparseTensor:
    """Reset the `dense_shape` of `sp_input` to `new_shape` (or tight bounding box if `None`).

    Args:
        sp_input: Input `SparseTensor`.
        new_shape: Optional new `dense_shape`.
        name: Optional operation name.

    Returns:
        SparseTensor: `SparseTensor` with updated `dense_shape`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    if new_shape is not None:
        target_s = _to_int_list(new_shape)
    else:
        coords = _to_coord_list(sp.indices)
        old_s = _to_int_list(sp.dense_shape)
        target_s = [
            max((c[d] + 1 for c in coords if d < len(c)), default=0) for d in range(len(old_s))
        ]
    return SparseTensor(indices=sp.indices, values=sp.values, dense_shape=target_s)


def reshape(
    sp_input: SparseTensor | Tensor | TensorOperand,
    shape: Sequence[int] | Tensor,
    name: str | None = None,
) -> SparseTensor:
    """Reshape a `SparseTensor` to represent values in a new dense `shape`.

    Args:
        sp_input: Input `SparseTensor`.
        shape: Target 1-D dense shape (may contain `-1` for one inferred dimension).
        name: Optional operation name.

    Returns:
        SparseTensor: Reshaped `SparseTensor`.
    """
    del name
    dense_t = _extract_values(sp_input)
    new_shape = tuple(_to_int_list(shape))
    reshaped = _wrap_tensor(_k_ops.reshape(dense_t, new_shape))
    return from_dense(reshaped)


def retain(
    sp_input: SparseTensor | Tensor | TensorOperand,
    to_retain: Sequence[bool] | Tensor | TensorOperand,
) -> SparseTensor:
    """Retain specified non-empty values within a `SparseTensor`.

    Args:
        sp_input: Input `SparseTensor`.
        to_retain: Boolean mask of length `len(sp_input.values)`.

    Returns:
        SparseTensor: Filtered `SparseTensor`.
    """
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    raw_mask = to_retain.tolist() if isinstance(to_retain, Tensor) else to_retain
    mask_bools = (
        [bool(b) for b in raw_mask]
        if isinstance(raw_mask, Sequence) and not isinstance(raw_mask, (str, bytes))
        else []
    )
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    out_c = [c for c, keep in zip(coords, mask_bools) if keep]
    out_v = [v for v, keep in zip(vals, mask_bools) if keep]
    return SparseTensor(
        indices=out_c,
        values=Tensor(out_v, dtype=sp.dtype),
        dense_shape=sp.dense_shape,
    )


def _sparse_segment_reduce(
    data: Tensor | TensorOperand,
    indices: Sequence[int] | Tensor,
    segment_ids: Sequence[int] | Tensor,
    mode: str,
) -> Tensor:
    """Gather rows of `data` by `indices` and reduce by `segment_ids`.

    Args:
        data: Dense input tensor.
        indices: 1-D row indices into `data`.
        segment_ids: 1-D segment IDs corresponding to `indices`.
        mode: `'sum'`, `'mean'`, or `'sqrt_n'`.

    Returns:
        Tensor: Segment-reduced dense tensor.
    """
    t_data = _to_tensor(data)
    idx_list = _to_int_list(indices)
    seg_list = _to_int_list(segment_ids)
    num_segs = max(seg_list) + 1 if seg_list else 0
    gathered = _k_ops.take(t_data, _to_tensor(idx_list, dtype=msc_core.DType.Int32), axis=0)
    rows: list[msc_core.Tensor] = []
    for s_id in range(num_segs):
        pos = [i for i, seg in enumerate(seg_list) if seg == s_id]
        if not pos:
            rows.append(_ops.zeros((1, *t_data.shape[1:]), dtype=t_data.dtype))
            continue
        seg_slice = _k_ops.take(gathered, _to_tensor(pos, dtype=msc_core.DType.Int32), axis=0)
        summed = _k_ops.sum(seg_slice, axis=0, keepdims=True)
        if mode == "mean":
            summed = _ops.divide(summed, _to_tensor(float(len(pos)), dtype=t_data.dtype))
        elif mode == "sqrt_n":
            summed = _ops.divide(summed, _to_tensor(math.sqrt(float(len(pos))), dtype=t_data.dtype))
        rows.append(summed)
    if not rows:
        return _wrap_tensor(_ops.zeros((0, *t_data.shape[1:]), dtype=t_data.dtype))
    return _wrap_tensor(_k_ops.concatenate(rows, axis=0))


def segment_mean(
    data: Tensor | TensorOperand,
    indices: Sequence[int] | Tensor,
    segment_ids: Sequence[int] | Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute the mean along sparse segments of a tensor.

    Args:
        data: Dense tensor with data to segment.
        indices: 1-D tensor of indices into `data`.
        segment_ids: 1-D tensor of segment IDs.
        num_segments: Optional total number of segments.
        name: Optional operation name.

    Returns:
        Tensor: Segment mean tensor.
    """
    del num_segments, name
    return _sparse_segment_reduce(data, indices, segment_ids, "mean")


def segment_sqrt_n(
    data: Tensor | TensorOperand,
    indices: Sequence[int] | Tensor,
    segment_ids: Sequence[int] | Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute the sum divided by `sqrt(N)` along sparse segments of a tensor.

    Args:
        data: Dense tensor with data to segment.
        indices: 1-D tensor of indices into `data`.
        segment_ids: 1-D tensor of segment IDs.
        num_segments: Optional total number of segments.
        name: Optional operation name.

    Returns:
        Tensor: Segment `sqrt_n`-normalized sum tensor.
    """
    del num_segments, name
    return _sparse_segment_reduce(data, indices, segment_ids, "sqrt_n")


def segment_sum(
    data: Tensor | TensorOperand,
    indices: Sequence[int] | Tensor,
    segment_ids: Sequence[int] | Tensor,
    num_segments: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute the sum along sparse segments of a tensor.

    Args:
        data: Dense tensor with data to segment.
        indices: 1-D tensor of indices into `data`.
        segment_ids: 1-D tensor of segment IDs.
        num_segments: Optional total number of segments.
        name: Optional operation name.

    Returns:
        Tensor: Segment sum tensor.
    """
    del num_segments, name
    return _sparse_segment_reduce(data, indices, segment_ids, "sum")


def slice(
    sp_input: SparseTensor | Tensor | TensorOperand,
    start: Sequence[int] | Tensor,
    size: Sequence[int] | Tensor,
    name: str | None = None,
) -> SparseTensor:
    """Slice a `SparseTensor` based on `start` offset and `size` extent.

    Args:
        sp_input: Input `SparseTensor`.
        start: 1-D integer sequence specifying slice start per dimension.
        size: 1-D integer sequence specifying slice size per dimension.
        name: Optional operation name.

    Returns:
        SparseTensor: Sliced `SparseTensor` with shifted coordinates and updated `dense_shape`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    st_list = _to_int_list(start)
    sz_list = _to_int_list(size)
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    out_c: list[list[int]] = []
    out_v: list[float] = []
    for c, v in zip(coords, vals):
        if all(st_list[d] <= c[d] < st_list[d] + sz_list[d] for d in range(len(st_list))):
            out_c.append([c[d] - st_list[d] for d in range(len(st_list))])
            out_v.append(v)
    return SparseTensor(
        indices=out_c,
        values=Tensor(out_v, dtype=sp.dtype),
        dense_shape=sz_list,
    )


def softmax(
    sp_input: SparseTensor | Tensor | TensorOperand,
    name: str | None = None,
) -> SparseTensor:
    """Apply softmax to a batched N-D `SparseTensor` across the innermost dimension.

    Args:
        sp_input: Input `SparseTensor` of rank `>= 2`.
        name: Optional operation name.

    Returns:
        SparseTensor: `SparseTensor` with row-wise softmax probabilities on non-empty entries.
    """
    del name
    sp = reorder(sp_input)
    coords = _to_coord_list(sp.indices)
    vals = _to_scalar_list(sp.values)
    groups: dict[tuple[int, ...], list[int]] = {}
    for idx, c in enumerate(coords):
        prefix = tuple(c[:-1])
        groups.setdefault(prefix, []).append(idx)
    out_vals = list(vals)
    for member_idxs in groups.values():
        max_v = max(vals[i] for i in member_idxs)
        exps = [math.exp(vals[i] - max_v) for i in member_idxs]
        sum_e = sum(exps)
        for i, e_val in zip(member_idxs, exps):
            out_vals[i] = e_val / sum_e
    return SparseTensor(
        indices=coords,
        values=Tensor(out_vals, dtype=sp.dtype),
        dense_shape=sp.dense_shape,
    )


def to_indicator(
    sp_input: SparseTensor | Tensor | TensorOperand,
    vocab_size: int = 1,
    name: str | None = None,
) -> Tensor:
    """Convert a `SparseTensor` of integer IDs into a dense boolean indicator tensor.

    Args:
        sp_input: Input `SparseTensor` with integer `.values` in `[0, vocab_size)`.
        vocab_size: Size of the innermost vocabulary indicator dimension.
        name: Optional operation name.

    Returns:
        Tensor: Dense boolean indicator `Tensor` of shape `[*sp_input.dense_shape[:-1], vocab_size]`.
    """
    del name
    sp = sp_input if isinstance(sp_input, SparseTensor) else from_dense(sp_input)
    shape_l = _to_int_list(sp.dense_shape)
    nrows = shape_l[0] if shape_l else 1
    v_size = max(1, int(vocab_size))
    grid = [[False] * v_size for _ in range(nrows)]
    for c, v in zip(_to_coord_list(sp.indices), _to_scalar_list(sp.values)):
        r_idx = c[0] % nrows
        col_id = int(v) % v_size
        grid[r_idx][col_id] = True
    return Tensor(grid, dtype=DType.Bool)


def transpose(
    a: SparseTensor | Tensor | TensorOperand,
    perm: Sequence[int] | None = None,
    name: str | None = None,
) -> SparseTensor:
    """Transpose a `SparseTensor` according to dimension permutation `perm`.

    Args:
        a: Input `SparseTensor`.
        perm: Optional dimension permutation (defaults to reversing all dimensions).
        name: Optional operation name.

    Returns:
        SparseTensor: Transposed and canonically reordered `SparseTensor`.
    """
    del name
    sp = a if isinstance(a, SparseTensor) else from_dense(a)
    old_shape = _to_int_list(sp.dense_shape)
    ndim = len(old_shape)
    p = [int(x) for x in perm] if perm is not None else list(reversed(range(ndim)))
    new_shape = [old_shape[i] for i in p]
    new_coords = [[c[i] for i in p] for c in _to_coord_list(sp.indices)]
    return reorder(SparseTensor(indices=new_coords, values=sp.values, dense_shape=new_shape))
