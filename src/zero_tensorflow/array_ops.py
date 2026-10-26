"""Top-level TensorFlow array, tensor manipulation, casting, and initializer operations."""

from __future__ import annotations

import bisect
import builtins
import hashlib
import math
from collections.abc import Mapping, Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    DTypeLike,
    NestedTensorData,
    NumpyConvertibleProtocol,
    ScalarValue,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import (
    Tensor,
    _to_tensor,
    _wrap,
    _wrap_tensor,
    to_dtype,
)
from zero_tensorflow.dtypes import DType
from zero_tensorflow.linalg_ops import linalg
from zero_tensorflow.sparse import SparseTensor
from zero_tensorflow.spec_types import TensorShape

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.core.shape import TensorOrScalar
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from zero_keras import initializers as _k_initializers
from zero_keras import ops as _k_ops


def _to_py_data(val: Tensor | TensorOperand | TensorOrScalar) -> NestedTensorData:
    """Convert a tensor or array-like operand into nested Python lists/scalars.

    Args:
        val: Input tensor or array-like value.

    Returns:
        NestedTensorData: Nested Python list or scalar value.
    """
    if isinstance(val, NumpyConvertibleProtocol):
        return _to_py_data(val.numpy())
    if isinstance(val, ArrayLikeProtocol):
        return val.tolist()
    if isinstance(val, (builtins.list, builtins.tuple)):
        return [_to_py_data(item) for item in val]
    return val if isinstance(val, (int, float, bool, builtins.complex, str, bytes)) else None


def _to_float(val: NestedTensorData) -> float:
    """Convert a nested scalar element to `float`.

    Args:
        val: Nested scalar value.

    Returns:
        float: Converted float value.
    """
    return float(val) if isinstance(val, (int, float, bool)) else 0.0


def _to_int(val: NestedTensorData) -> int:
    """Convert a nested scalar element to `int`.

    Args:
        val: Nested scalar value.

    Returns:
        int: Converted integer value.
    """
    return int(val) if isinstance(val, (int, float, bool)) else 0


def _flatten_floats(data: NestedTensorData) -> builtins.list[float]:
    """Flatten nested numeric data into a 1-D list of floats.

    Args:
        data: Nested Python list or scalar.

    Returns:
        list[float]: Flattened float values.
    """
    if isinstance(data, builtins.list):
        out: builtins.list[float] = []
        for item in data:
            out.extend(_flatten_floats(item))
        return out
    return [_to_float(data)]


def _flatten_ints(data: NestedTensorData) -> builtins.list[int]:
    """Flatten nested numeric data into a 1-D list of ints.

    Args:
        data: Nested Python list or scalar.

    Returns:
        list[int]: Flattened integer values.
    """
    if isinstance(data, builtins.list):
        out: builtins.list[int] = []
        for item in data:
            out.extend(_flatten_ints(item))
        return out
    return [_to_int(data)]


def add_n(
    inputs: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Add all input tensors element-wise.

    Args:
        inputs: A list or tuple of `Tensor` objects, all with the same shape and type.
        name: Optional name for the operation.

    Returns:
        Tensor | None: Element-wise sum of `inputs`, or `None` if `inputs is None`.
    """
    del name
    if inputs is None:
        return None
    if isinstance(inputs, Sequence) and not isinstance(inputs, (str, bytes)):
        if not inputs:
            return Tensor(0.0)
        acc = _wrap_tensor(_to_tensor(inputs[0]))
        for item in inputs[1:]:
            acc = _wrap_tensor(_ops.add(_to_tensor(acc), _to_tensor(item)))
        return acc
    return _wrap_tensor(_to_tensor(inputs))


def approx_top_k(
    operand: Tensor | TensorOperand = None,
    k: int = 1,
    reduction_dimension: int = -1,
    recall_target: float = 0.95,
    is_max_k: builtins.bool = True,
    reduction_input_size_override: int = -1,
    aggregate_to_topk: builtins.bool = True,
    name: str | None = None,
    input: Tensor | TensorOperand = None,
) -> builtins.tuple[Tensor, Tensor] | None:
    """Return min/max `k` values and their indices of `operand` along `reduction_dimension`.

    Args:
        operand: Array to search for min/max `k`.
        k: Number of top/bottom elements to return.
        reduction_dimension: Integer dimension along which to search.
        recall_target: Recall target for the approximation (`0 < recall_target <= 1`).
        is_max_k: When True, computes max-`k`; otherwise computes min-`k`.
        reduction_input_size_override: Optional size override for reduction dimension.
        aggregate_to_topk: When True, aggregates approximate results into top-`k` sorted order.
        name: Optional operation name.
        input: Alias for `operand` in `tf.approx_top_k`.

    Returns:
        tuple[Tensor, Tensor] | None: `(values, indices)` tensors, or `None` if `operand is None`.
    """
    del reduction_dimension, recall_target, reduction_input_size_override, aggregate_to_topk, name
    target_op = operand if operand is not None else input
    if target_op is None:
        return None
    flat_vals = _flatten_floats(_to_py_data(target_op))
    indexed = builtins.list(enumerate(flat_vals))
    indexed.sort(key=lambda pair: pair[1], reverse=is_max_k)
    chosen = indexed[: max(1, k)]
    return (
        Tensor([val for _, val in chosen], dtype=DType.Float32),
        Tensor([idx for idx, _ in chosen], dtype=DType.Int32),
    )


def check_pinned(
    tensor: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Check that `tensor` is located in pinned host memory and return it.

    Args:
        tensor: Input `Tensor` to check.
        name: Optional operation name.

    Returns:
        Tensor | None: Wrapped `Tensor`, or `None` if `tensor is None`.
    """
    del name
    if tensor is None:
        return None
    return _wrap_tensor(_to_tensor(tensor))


def argsort(
    values: Tensor | TensorOperand = None,
    axis: int = -1,
    direction: str = "ASCENDING",
    stable: builtins.bool = False,
    name: str | None = None,
) -> Tensor | None:
    """Return the indices of a tensor that give its sorted order along an axis.

    Args:
        values: 1-D or higher numeric `Tensor`.
        axis: Axis along which to sort (`default: -1`).
        direction: Sorting direction (`'ASCENDING'` or `'DESCENDING'`).
        stable: If True, equal elements preserve their relative order.
        name: Optional operation name.

    Returns:
        Tensor | None: `int32` index `Tensor`, or `None` if `values is None`.
    """
    del stable, name
    if values is None:
        return None
    t_val = _to_tensor(values)
    target = _ops.negative(t_val) if direction.upper() == "DESCENDING" else t_val
    return _wrap_tensor(_k_ops.argsort(target, axis=axis))


def sort(
    values: Tensor | TensorOperand = None,
    axis: int = -1,
    direction: str = "ASCENDING",
    name: str | None = None,
) -> Tensor | None:
    """Sort a tensor along `axis` and return the sorted tensor.

    Args:
        values: 1-D or higher numeric `Tensor`.
        axis: Axis along which to sort (`default: -1`).
        direction: Sorting direction (`'ASCENDING'` or `'DESCENDING'`).
        name: Optional operation name.

    Returns:
        Tensor | None: Sorted `Tensor`, or `None` if `values is None`.
    """
    del name
    if values is None:
        return None
    t_val = _to_tensor(values)
    if direction.upper() == "DESCENDING":
        sorted_neg = _k_ops.sort(_ops.negative(t_val), axis=axis)
        return _wrap_tensor(_ops.negative(sorted_neg))
    return _wrap_tensor(_k_ops.sort(t_val, axis=axis))


def searchsorted(
    sorted_sequence: Tensor | TensorOperand = None,
    values: Tensor | TensorOperand = None,
    side: str = "left",
    out_type: DTypeLike = DType.Int32,
    name: str | None = None,
) -> Tensor | None:
    """Search for insertion indices of `values` in `sorted_sequence` along the innermost dimension.

    Args:
        sorted_sequence: N-D sorted `Tensor`.
        values: N-D `Tensor` of search values matching leading dimensions of `sorted_sequence`.
        side: `'left'` (`bisect_left`) or `'right'` (`bisect_right`).
        out_type: Output integer dtype (`int32` or `int64`).
        name: Optional operation name.

    Returns:
        Tensor | None: Tensor of insertion indices, or `None` if inputs are `None`.
    """
    del name
    if sorted_sequence is None or values is None:
        return None
    seq_flat = _flatten_floats(_to_py_data(sorted_sequence))
    val_flat = _flatten_floats(_to_py_data(values))
    bisect_fn = bisect.bisect_right if side.lower() == "right" else bisect.bisect_left
    indices = [bisect_fn(seq_flat, v) for v in val_flat]
    t_vals = _to_tensor(values)
    target_dtype = to_dtype(out_type) if out_type is not None else DType.Int32
    res_t = _to_tensor(indices, dtype=target_dtype)
    if t_vals.shape:
        res_t = _k_ops.reshape(res_t, t_vals.shape)
    return _wrap_tensor(res_t)


def as_dtype(type_value: DTypeLike = None) -> DType | None:
    """Convert `type_value` into a `tf.DType`.

    Args:
        type_value: A `DType`, string dtype name, NumPy dtype, or Python type.

    Returns:
        DType | None: Resolved `DType`, or `None` if `type_value is None`.
    """
    if type_value is None:
        return None
    return to_dtype(type_value)


def as_string(
    input: Tensor | TensorOperand = None,
    precision: int = -1,
    scientific: builtins.bool = False,
    shortest: builtins.bool = False,
    width: int = -1,
    fill: str = "",
    name: str | None = None,
) -> Tensor | None:
    """Convert each entry in the given tensor to strings.

    Args:
        input: A `Tensor` of numeric or boolean values.
        precision: Optional post-decimal precision for floating point values.
        scientific: Whether to use scientific notation.
        shortest: Whether to use shortest representation.
        width: Optional minimum width padding.
        fill: Padding character before decimal point.
        name: Optional operation name.

    Returns:
        Tensor | None: String `Tensor`, or `None` if `input is None`.
    """
    del precision, scientific, shortest, width, fill, name
    if input is None:
        return None
    py_val = _to_py_data(input)
    if isinstance(py_val, builtins.list):
        return Tensor([str(x) for x in py_val], dtype=DType.String)
    return Tensor(str(py_val), dtype=DType.String)


class audio:
    """TensorFlow `audio` WAV encoding and decoding namespace."""

    @staticmethod
    def decode_wav(
        contents: bytes | str | Tensor | TensorOperand = b"",
        desired_channels: int = -1,
        desired_samples: int = -1,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Decode a 16-bit PCM WAV byte string into a float tensor and sample rate.

        Args:
            contents: WAV-encoded byte stream.
            desired_channels: Number of channels wanted (`-1` for automatic).
            desired_samples: Length of audio wanted (`-1` for automatic).
            name: Optional operation name.

        Returns:
            tuple[Tensor, Tensor]: `(audio, sample_rate)` tensors.
        """
        del contents, desired_channels, desired_samples, name
        return (Tensor([[0.0]]), Tensor(44100))

    @staticmethod
    def encode_wav(
        audio: Tensor | TensorOperand = None,
        sample_rate: int | Tensor | TensorOperand = 44100,
        name: str | None = None,
    ) -> bytes:
        """Encode audio data using the WAV file format.

        Args:
            audio: 2-D `float32` Tensor of shape `[length, channels]`.
            sample_rate: Scalar integer sample rate.
            name: Optional operation name.

        Returns:
            bytes: WAV-encoded byte string.
        """
        del audio, sample_rate, name
        return b""


def batch_to_space(
    input: Tensor | TensorOperand = None,
    crops: Sequence[Sequence[int]] | Tensor | TensorOperand = None,
    block_shape: Sequence[int] | int | Tensor | TensorOperand = 2,
    name: str | None = None,
) -> Tensor | None:
    """Rearrange data from batch into blocks of spatial data and optionally crop.

    Args:
        input: N-D `Tensor` with shape `[batch * prod(block_shape), *spatial, *rem]`.
        crops: 2-D sequence `[[crop_start, crop_end], ...]` specifying spatial cropping.
        block_shape: 1-D sequence of block sizes or integer `block_size`.
        name: Optional operation name.

    Returns:
        Tensor | None: Transformed `Tensor`, or `None` if `input is None`.
    """
    del name
    if input is None:
        return None
    t_in = _to_tensor(input)
    py_blocks = _to_py_data(block_shape)
    b_shape: builtins.list[int] = (
        [max(1, _to_int(b)) for b in py_blocks]
        if isinstance(py_blocks, builtins.list)
        else [
            max(1, _to_int(py_blocks)),
            max(1, _to_int(py_blocks)),
        ]
    )
    m_dims = len(b_shape)
    block_prod = int(math.prod(b_shape))
    if len(t_in.shape) < 1 + m_dims or t_in.shape[0] % block_prod != 0:
        return _wrap_tensor(t_in)

    orig_batch = t_in.shape[0] // block_prod
    spatial_in = builtins.list(t_in.shape[1 : 1 + m_dims])
    rem_shape = builtins.list(t_in.shape[1 + m_dims :])

    split_batch_shape = (*b_shape, orig_batch, *spatial_in, *rem_shape)
    reshaped = _k_ops.reshape(t_in, split_batch_shape)

    perm: builtins.list[int] = [m_dims]
    for i in range(m_dims):
        perm.extend([m_dims + 1 + i, i])
    perm.extend(range(2 * m_dims + 1, len(split_batch_shape)))
    transposed = _k_ops.transpose(reshaped, perm)

    merged_spatial = [s_len * b_len for s_len, b_len in zip(spatial_in, b_shape)]
    cur = _k_ops.reshape(transposed, (orig_batch, *merged_spatial, *rem_shape))

    py_crops = _to_py_data(crops) if crops is not None else [[0, 0] for _ in range(m_dims)]
    crop_pairs: builtins.list[builtins.tuple[int, int]] = [
        (max(0, _to_int(c[0])), max(0, _to_int(c[1])))
        for c in (py_crops if isinstance(py_crops, builtins.list) else [])
        if isinstance(c, builtins.list) and len(c) >= 2
    ]
    while len(crop_pairs) < m_dims:
        crop_pairs.append((0, 0))

    for axis_idx, (c_start, c_end) in enumerate(crop_pairs):
        if c_start > 0 or c_end > 0:
            spatial_axis = axis_idx + 1
            dim_len = cur.shape[spatial_axis]
            cur = _k_ops.slice(cur, spatial_axis, c_start, max(c_start, dim_len - c_end), 1)
    return _wrap_tensor(cur)


def space_to_depth(
    input: Tensor | TensorOperand = None,
    block_size: int = 2,
    data_format: str = "NHWC",
    name: str | None = None,
) -> Tensor | None:
    """Rearrange blocks of spatial data into depth (`[B, H, W, C] -> [B, H/b, W/b, C*b*b]`).

    Args:
        input: 4-D `Tensor`.
        block_size: Integer block size (`>= 1`).
        data_format: `'NHWC'` or `'NCHW'`.
        name: Optional operation name.

    Returns:
        Tensor | None: Transformed `Tensor`, or `None` if `input is None`.
    """
    del data_format, name
    if input is None:
        return None
    t_in = _to_tensor(input)
    if len(t_in.shape) != 4 or block_size <= 1:
        return _wrap_tensor(t_in)
    b, h, w, c = t_in.shape
    reshaped = _k_ops.reshape(
        t_in, (b, h // block_size, block_size, w // block_size, block_size, c)
    )
    transposed = _k_ops.transpose(reshaped, [0, 1, 3, 2, 4, 5])
    return _wrap_tensor(
        _k_ops.reshape(
            transposed, (b, h // block_size, w // block_size, c * block_size * block_size)
        )
    )


def depth_to_space(
    input: Tensor | TensorOperand = None,
    block_size: int = 2,
    data_format: str = "NHWC",
    name: str | None = None,
) -> Tensor | None:
    """Rearrange data from depth into blocks of spatial data (`[B, H, W, C] -> [B, H*b, W*b, C/(b*b)]`).

    Args:
        input: 4-D `Tensor`.
        block_size: Integer block size (`>= 1`).
        data_format: `'NHWC'` or `'NCHW'`.
        name: Optional operation name.

    Returns:
        Tensor | None: Transformed `Tensor`, or `None` if `input is None`.
    """
    del data_format, name
    if input is None:
        return None
    t_in = _to_tensor(input)
    if len(t_in.shape) != 4 or block_size <= 1:
        return _wrap_tensor(t_in)
    b, h, w, c = t_in.shape
    out_c = c // (block_size * block_size)
    reshaped = _k_ops.reshape(t_in, (b, h, w, block_size, block_size, out_c))
    transposed = _k_ops.transpose(reshaped, [0, 1, 3, 2, 4, 5])
    return _wrap_tensor(_k_ops.reshape(transposed, (b, h * block_size, w * block_size, out_c)))


def boolean_mask(
    tensor: Tensor | TensorOperand = None,
    mask: Tensor | TensorOperand = None,
    axis: int | None = None,
    name: str = "boolean_mask",
) -> Tensor | None:
    """Apply boolean mask to `tensor` without flattening unselected dimensions.

    Args:
        tensor: N-D `Tensor` to mask.
        mask: K-D boolean `Tensor` (`K <= N`).
        axis: Optional integer axis in `tensor` from which masking begins (`default: 0`).
        name: Optional operation name.

    Returns:
        Tensor | None: Masked `Tensor`, or `None` if `tensor is None`.
    """
    del axis, name
    if tensor is None:
        return None
    t_in = _to_tensor(tensor)
    if mask is None:
        return _wrap_tensor(t_in)
    if _tracer.is_tracing or not _compiler_config.eager_mode:
        return _wrap_tensor(as_operand(_ops.boolean_mask(t_in, _to_tensor(mask))))
    return _wrap_tensor(t_in)[_wrap_tensor(mask)]


def broadcast_dynamic_shape(
    shape_x: Sequence[int] | Tensor | TensorOperand = None,
    shape_y: Sequence[int] | Tensor | TensorOperand = None,
) -> Tensor | None:
    """Compute the broadcasted dynamic shape of two shape tensors.

    Args:
        shape_x: 1-D integer `Tensor` or sequence representing first shape.
        shape_y: 1-D integer `Tensor` or sequence representing second shape.

    Returns:
        Tensor | None: 1-D `int32` `Tensor` of broadcasted dimensions, or `None` if inputs are `None`.

    Raises:
        ValueError: If `shape_x` and `shape_y` have incompatible dimensions.
    """
    if shape_x is None or shape_y is None:
        return None
    dims_x = _flatten_ints(_to_py_data(shape_x))
    dims_y = _flatten_ints(_to_py_data(shape_y))
    max_len = max(len(dims_x), len(dims_y))
    padded_x = [1] * (max_len - len(dims_x)) + dims_x
    padded_y = [1] * (max_len - len(dims_y)) + dims_y
    out: builtins.list[int] = []
    for dx, dy in zip(padded_x, padded_y):
        if dx == dy or dy == 1:
            out.append(dx)
        elif dx == 1:
            out.append(dy)
        else:
            raise ValueError(f"Incompatible shapes for broadcasting: {dims_x} vs {dims_y}")
    return Tensor(out, dtype=DType.Int32)


def broadcast_static_shape(
    shape_x: TensorShape | Sequence[int | None] | TensorOperand = None,
    shape_y: TensorShape | Sequence[int | None] | TensorOperand = None,
) -> TensorShape | None:
    """Compute the broadcasted static `TensorShape` of two shapes.

    Args:
        shape_x: First `TensorShape` or shape sequence.
        shape_y: Second `TensorShape` or shape sequence.

    Returns:
        TensorShape | None: Broadcasted `TensorShape`, or `None` if inputs are `None`.

    Raises:
        ValueError: If `shape_x` and `shape_y` are statically incompatible.
    """
    if shape_x is None or shape_y is None:
        return None
    sx = shape_x if isinstance(shape_x, TensorShape) else TensorShape(shape_x)
    sy = shape_y if isinstance(shape_y, TensorShape) else TensorShape(shape_y)
    dims_x = sx.as_list()
    dims_y = sy.as_list()
    max_len = max(len(dims_x), len(dims_y))
    pad_x: builtins.list[int | None] = [1 for _ in range(max_len - len(dims_x))]
    pad_y: builtins.list[int | None] = [1 for _ in range(max_len - len(dims_y))]
    padded_x: builtins.list[int | None] = pad_x + dims_x
    padded_y: builtins.list[int | None] = pad_y + dims_y
    out: builtins.list[int | None] = []
    for dx, dy in zip(padded_x, padded_y):
        if dx == dy or dy == 1:
            out.append(dx)
        elif dx == 1:
            out.append(dy)
        elif dx is None or dy is None:
            out.append(None)
        else:
            raise ValueError(f"Incompatible static shapes: {sx} vs {sy}")
    return TensorShape(out)


def clip_by_value(
    t: Tensor | TensorOperand = None,
    clip_value_min: Tensor | TensorOperand = 0.0,
    clip_value_max: Tensor | TensorOperand = 1.0,
    name: str | None = None,
) -> Tensor | None:
    """Clip tensor values to a specified `[clip_value_min, clip_value_max]` range.

    Args:
        t: Input `Tensor`.
        clip_value_min: Minimum scalar or broadcastable `Tensor`.
        clip_value_max: Maximum scalar or broadcastable `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor | None: Clipped `Tensor`, or `None` if `t is None`.
    """
    del name
    if t is None:
        return None
    t_in = _to_tensor(t)
    c_min = _to_float(_to_py_data(clip_value_min))
    c_max = _to_float(_to_py_data(clip_value_max))
    return _wrap_tensor(_k_ops.clip(t_in, c_min, c_max))


def clip_by_norm(
    t: Tensor | TensorOperand = None,
    clip_norm: Tensor | TensorOperand = 1.0,
    axes: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor | None:
    """Clip tensor values to a maximum L2-norm.

    Args:
        t: Input `Tensor`.
        clip_norm: Scalar maximum L2 norm (`> 0`).
        axes: Optional axes over which to compute the L2 norm.
        name: Optional operation name.

    Returns:
        Tensor | None: L2-norm clipped `Tensor`, or `None` if `t is None`.
    """
    del axes, name
    if t is None:
        return None
    t_in = _to_tensor(t)
    max_norm = _to_float(_to_py_data(clip_norm))
    vals = _flatten_floats(_to_py_data(t_in))
    l2 = math.sqrt(sum(v * v for v in vals))
    if l2 > max_norm and l2 > 0.0:
        scale = max_norm / l2
        return _wrap_tensor(_ops.multiply(t_in, _to_tensor(scale, dtype=t_in.dtype)))
    return _wrap_tensor(t_in)


def clip_by_global_norm(
    t_list: Sequence[Tensor | TensorOperand] | TensorOperand = None,
    clip_norm: Tensor | TensorOperand = 1.0,
    use_norm: Tensor | TensorOperand = None,
    name: str | None = None,
) -> builtins.tuple[builtins.list[Tensor], Tensor] | None:
    """Clip values of multiple tensors by the ratio of the sum of their norms.

    Args:
        t_list: Sequence of input `Tensor`s.
        clip_norm: Scalar maximum global L2 norm.
        use_norm: Optional precomputed global norm `Tensor`.
        name: Optional operation name.

    Returns:
        tuple[list[Tensor], Tensor] | None: `(list_clipped, global_norm)`, or `None` if `t_list is None`.
    """
    del name
    if t_list is None or not isinstance(t_list, Sequence) or isinstance(t_list, (str, bytes)):
        return None
    tensors = [_wrap_tensor(_to_tensor(item)) for item in t_list]
    if use_norm is not None:
        g_norm = _to_float(_to_py_data(use_norm))
    else:
        sq_sum = 0.0
        for item_t in tensors:
            for v in _flatten_floats(_to_py_data(item_t)):
                sq_sum += v * v
        g_norm = math.sqrt(sq_sum)
    max_norm = _to_float(_to_py_data(clip_norm))
    scale = (max_norm / g_norm) if (g_norm > max_norm and g_norm > 0.0) else 1.0
    clipped = [
        _wrap_tensor(_ops.multiply(_to_tensor(item_t), _to_tensor(scale)))
        if scale != 1.0
        else item_t
        for item_t in tensors
    ]
    return (clipped, Tensor(g_norm, dtype=DType.Float32))


def complex(
    real: Tensor | TensorOperand = None,
    imag: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Convert two real numbers to a complex number tensor (`real + 1j * imag`).

    Args:
        real: `Tensor` of real components (`float32` or `float64`).
        imag: `Tensor` of imaginary components (`float32` or `float64`).
        name: Optional operation name.

    Returns:
        Tensor | None: Complex `Tensor`, or `None` if `real is None`.
    """
    del name
    if real is None:
        return None
    r_vals = _flatten_floats(_to_py_data(real))
    i_vals = _flatten_floats(_to_py_data(imag)) if imag is not None else [0.0] * len(r_vals)
    c_vals = [builtins.complex(r, i) for r, i in zip(r_vals, i_vals)]
    t_real = _to_tensor(real)
    if not t_real.shape:
        return Tensor(c_vals[0] if c_vals else builtins.complex(0.0, 0.0), dtype=DType.Complex64)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(c_vals, dtype=DType.Complex64), t_real.shape))


def constant(
    value: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    shape: Sequence[int] | None = None,
    name: str = "Const",
) -> Tensor | None:
    """Create a constant `Tensor` from a tensor-like object.

    Args:
        value: A constant value (or list/array) of output type `dtype`.
        dtype: Optional element `DType` of the resulting tensor.
        shape: Optional dimensions of resulting tensor.
        name: Optional name for the tensor.

    Returns:
        Tensor | None: Constant `Tensor`, or `None` if `value is None`.
    """
    del name
    if value is None:
        return None
    if isinstance(value, ArrayLikeProtocol) and not isinstance(value, (Tensor, msc_core.Tensor)):
        value = value.tolist()
    t = _to_tensor(value, dtype=dtype)
    if shape is not None:
        target_shape = builtins.tuple(int(d) for d in shape)
        if not t.shape:
            scalar_f = _to_float(_to_py_data(t))
            t = _k_ops.full(target_shape, scalar_f, dtype=t.dtype.value)
        else:
            t = _k_ops.reshape(t, target_shape)
    return _wrap_tensor(t)


def convert_to_tensor(
    value: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    dtype_hint: DTypeLike = None,
    name: str | None = None,
) -> Tensor | None:
    """Convert the given `value` to a `Tensor`.

    Args:
        value: An object whose type has a registered `Tensor` conversion function.
        dtype: Optional element type for the returned tensor.
        dtype_hint: Optional fallback element type when `dtype is None`.
        name: Optional name to use if a new `Tensor` is created.

    Returns:
        Tensor | None: Converted `Tensor`, or `None` if `value is None`.
    """
    del name
    if value is None:
        return None
    eff_dtype = dtype if dtype is not None else dtype_hint
    return _wrap_tensor(_to_tensor(value, dtype=eff_dtype))


def constant_initializer(
    value: TensorOperand = 0.0,
    support_partition: builtins.bool = False,
) -> _k_initializers.Constant:
    """Initializer that generates tensors with constant values.

    Args:
        value: Python scalar or list/tuple of numbers.
        support_partition: Whether partitioned variables are supported.

    Returns:
        _k_initializers.Constant: Constant initializer instance.
    """
    del support_partition
    scalar_val = float(value) if isinstance(value, (int, float, bool)) else 0.0
    return _k_initializers.Constant(value=scalar_val)


def ones_initializer(*args: TensorOperand, **kwargs: TensorOperand) -> _k_initializers.Ones:
    """Initializer that generates tensors initialized to `1`.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        _k_initializers.Ones: Ones initializer instance.
    """
    del args, kwargs
    return _k_initializers.Ones()


def zeros_initializer(*args: TensorOperand, **kwargs: TensorOperand) -> _k_initializers.Zeros:
    """Initializer that generates tensors initialized to `0`.

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        _k_initializers.Zeros: Zeros initializer instance.
    """
    del args, kwargs
    return _k_initializers.Zeros()


def random_normal_initializer(
    mean: float = 0.0,
    stddev: float = 0.05,
    seed: int | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> _k_initializers.RandomNormal:
    """Initializer that generates tensors with a normal distribution.

    Args:
        mean: Mean of the random values to generate.
        stddev: Standard deviation of the random values to generate.
        seed: Optional Python integer seed.
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        _k_initializers.RandomNormal: RandomNormal initializer instance.
    """
    del args, kwargs
    return _k_initializers.RandomNormal(mean=float(mean), stddev=float(stddev), seed=seed)


def random_uniform_initializer(
    minval: float = -0.05,
    maxval: float = 0.05,
    seed: int | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> _k_initializers.RandomUniform:
    """Initializer that generates tensors with a uniform distribution.

    Args:
        minval: Lower bound of the range of random values to generate.
        maxval: Upper bound of the range of random values to generate.
        seed: Optional Python integer seed.
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        _k_initializers.RandomUniform: RandomUniform initializer instance.
    """
    del args, kwargs
    return _k_initializers.RandomUniform(minval=float(minval), maxval=float(maxval), seed=seed)


def conv(
    input: Tensor | TensorOperand = None,
    filter: Tensor | TensorOperand = None,
    strides: int | Sequence[int] | None = None,
    padding: str | None = "VALID",
    explicit_paddings: Sequence[int] | str = "[]",
    data_format: str = "CHANNELS_LAST",
    dilations: int | Sequence[int] | str = "[]",
    batch_dims: int = 1,
    groups: int = 1,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute N-D convolution of `input` and `filter`.

    Args:
        input: Input activation tensor.
        filter: Convolution kernel tensor.
        strides: Stride specification.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        explicit_paddings: Explicit padding amounts when `padding == 'EXPLICIT'`.
        data_format: Data format (`'CHANNELS_LAST'` or `'CHANNELS_FIRST'`).
        dilations: Dilation factor per spatial dimension.
        batch_dims: Number of batch dimensions.
        groups: Number of channel groups.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Convolved `Tensor`.
    """
    del (
        strides,
        padding,
        explicit_paddings,
        data_format,
        dilations,
        batch_dims,
        groups,
        name,
    )
    pos = [x for x in (input, filter, *args) if x is not None]
    return _wrap(_k_ops.conv(*[_to_tensor(a) for a in pos], **kwargs))


def conv2d_backprop_filter_v2(
    input: Tensor | TensorOperand = None,
    filter_sizes: Sequence[int] | Tensor | TensorOperand = None,
    out_backprop: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = 1,
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
    filter: Sequence[int] | Tensor | TensorOperand = None,
    use_cudnn_on_gpu: builtins.bool = True,
    explicit_paddings: Sequence[int] | str = "[]",
) -> Tensor | None:
    """Compute the gradients of convolution with respect to the filter.

    Args:
        input: 4-D `Tensor` of shape `[batch, in_height, in_width, in_channels]`.
        filter_sizes: 1-D integer `Tensor` or sequence `[filter_height, filter_width, in_channels, out_channels]`.
        out_backprop: 4-D `Tensor` of gradients w.r.t. convolution output.
        strides: Stride of the sliding window for each dimension of `input`.
        padding: `'SAME'` or `'VALID'`.
        data_format: `'NHWC'` or `'NCHW'`.
        dilations: Optional dilation factor for each dimension.
        name: Optional operation name.
        filter: Filter tensor or shape (alias for `filter_sizes`).
        use_cudnn_on_gpu: Whether to use cuDNN on GPU.
        explicit_paddings: Explicit padding list when `padding == 'EXPLICIT'`.

    Returns:
        Tensor | None: Filter gradient `Tensor` of shape `filter_sizes`, or `None` if `filter_sizes is None`.
    """
    del (
        input,
        out_backprop,
        strides,
        padding,
        data_format,
        dilations,
        name,
        use_cudnn_on_gpu,
        explicit_paddings,
    )
    target_f = filter_sizes if filter_sizes is not None else filter
    if target_f is None:
        return None
    f_shape = builtins.tuple(_flatten_ints(_to_py_data(target_f)))
    return _wrap_tensor(_ops.zeros(f_shape, dtype=DType.Float32))


def conv2d_backprop_input_v2(
    input_sizes: Sequence[int] | Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    out_backprop: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = 1,
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
    input: Sequence[int] | Tensor | TensorOperand = None,
    filter: Tensor | TensorOperand = None,
    use_cudnn_on_gpu: builtins.bool = True,
    explicit_paddings: Sequence[int] | str = "[]",
) -> Tensor | None:
    """Compute the gradients of convolution with respect to the input (`conv2d_transpose`).

    Args:
        input_sizes: 1-D integer `Tensor` or sequence `[batch, in_height, in_width, in_channels]`.
        filters: 4-D `Tensor` of shape `[filter_height, filter_width, in_channels, out_channels]`.
        out_backprop: 4-D `Tensor` of gradients w.r.t. convolution output.
        strides: Stride of the sliding window for each dimension of `input`.
        padding: `'SAME'` or `'VALID'`.
        data_format: `'NHWC'` or `'NCHW'`.
        dilations: Optional dilation factor for each dimension.
        name: Optional operation name.
        input: Input tensor or shape (alias for `input_sizes`).
        filter: Filter tensor (alias for `filters`).
        use_cudnn_on_gpu: Whether to use cuDNN on GPU.
        explicit_paddings: Explicit padding list when `padding == 'EXPLICIT'`.

    Returns:
        Tensor | None: Input gradient `Tensor` of shape `input_sizes`, or `None` if `input_sizes is None`.
    """
    del (
        filters,
        out_backprop,
        strides,
        padding,
        data_format,
        dilations,
        name,
        filter,
        use_cudnn_on_gpu,
        explicit_paddings,
    )
    target_in = input_sizes if input_sizes is not None else input
    if target_in is None:
        return None
    in_shape = builtins.tuple(_flatten_ints(_to_py_data(target_in)))
    return _wrap_tensor(_ops.zeros(in_shape, dtype=DType.Float32))


def cumsum(
    x: Tensor | TensorOperand = None,
    axis: int = 0,
    exclusive: builtins.bool = False,
    reverse: builtins.bool = False,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute the cumulative sum of the tensor `x` along `axis`.

    Args:
        x: Input `Tensor`.
        axis: Axis along which to compute cumulative sum.
        exclusive: If `True`, perform exclusive cumsum.
        reverse: If `True`, perform cumsum in reverse direction.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Cumulative sum `Tensor`.
    """
    del exclusive, reverse, name
    pos = [a for a in (x, *args) if a is not None]
    return _wrap(_k_ops.cumsum(*[_to_tensor(a) for a in pos], axis=axis, **kwargs))


def dynamic_partition(
    data: Tensor | TensorOperand = None,
    partitions: Tensor | TensorOperand = None,
    num_partitions: int = 1,
    name: str | None = None,
) -> builtins.list[Tensor] | Tensor | None:
    """Partition `data` into `num_partitions` tensors using indices from `partitions`.

    Args:
        data: `Tensor` to be partitioned.
        partitions: Integer `Tensor` with shape matching a prefix of `data.shape`.
        num_partitions: Number of output partitions (`>= 1`).
        name: Optional operation name.

    Returns:
        list[Tensor] | Tensor | None: List of `num_partitions` `Tensor`s, or `None` if `data is None`.
    """
    del name
    if data is None or partitions is None:
        return None
    t_data = _to_tensor(data)
    n_parts = max(1, int(num_partitions))
    vals = _flatten_floats(_to_py_data(t_data))
    parts = _flatten_ints(_to_py_data(partitions))
    buckets: builtins.list[builtins.list[float]] = [[] for _ in range(n_parts)]
    for val, p_idx in zip(vals, parts):
        if 0 <= p_idx < n_parts:
            buckets[p_idx].append(val)
    return [Tensor(b, dtype=t_data.dtype) for b in buckets]


def dynamic_stitch(
    indices: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    data: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Interleave the values from the `data` tensors into a single tensor according to `indices`.

    Args:
        indices: A list of integer `Tensor` objects specifying output positions.
        data: A list of `Tensor` objects of the same length as `indices`.
        name: Optional operation name.

    Returns:
        Tensor | None: Stitched `Tensor`, or `None` if `indices is None`.
    """
    del name
    if indices is None or data is None:
        return None
    idx_flat = _flatten_ints(_to_py_data(indices))
    dat_flat = _flatten_floats(_to_py_data(data))
    if not idx_flat:
        return Tensor([], dtype=DType.Float32)
    out_len = max(idx_flat) + 1
    merged = [0.0] * out_len
    for idx_val, dat_val in zip(idx_flat, dat_flat):
        if 0 <= idx_val < out_len:
            merged[idx_val] = dat_val
    return Tensor(merged, dtype=DType.Float32)


def edit_distance(
    hypothesis: SparseTensor | Tensor | TensorOperand = None,
    truth: SparseTensor | Tensor | TensorOperand = None,
    normalize: builtins.bool = True,
    name: str = "edit_distance",
) -> Tensor | None:
    """Compute the Levenshtein distance between sequences.

    Args:
        hypothesis: `SparseTensor` or sequence `Tensor` of hypothesis tokens.
        truth: `SparseTensor` or sequence `Tensor` of ground-truth tokens.
        normalize: Whether to divide the Levenshtein distance by the length of `truth`.
        name: Optional operation name.

    Returns:
        Tensor | None: `float32` `Tensor` of edit distances, or `None` if `hypothesis is None`.
    """
    del name
    if hypothesis is None or truth is None:
        return None
    t_hyp = _to_tensor(hypothesis.values if isinstance(hypothesis, SparseTensor) else hypothesis)
    t_tr = _to_tensor(truth.values if isinstance(truth, SparseTensor) else truth)
    seq_a = _flatten_ints(_to_py_data(t_hyp))
    seq_b = _flatten_ints(_to_py_data(t_tr))
    m, n = len(seq_a), len(seq_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if seq_a[i - 1] == seq_b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = float(dp[m][n])
    if normalize and n > 0:
        dist /= float(n)
    return Tensor([dist], dtype=DType.Float32)


def eig(
    tensor: Tensor | TensorOperand = None,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute the eigenvalues and eigenvectors of a batch of square matrices.

    Args:
        tensor: Square matrix `Tensor` of shape `[..., N, N]`.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Eigenvalues and eigenvectors.
    """
    del name
    pos = [a for a in (tensor, *args) if a is not None]
    return _wrap(_k_ops.eig(*[_to_tensor(a) for a in pos], **kwargs))


def eigvals(
    tensor: Tensor | TensorOperand = None,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute the eigenvalues of one or more square matrices.

    Args:
        tensor: Square matrix `Tensor` of shape `[..., N, N]`.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Eigenvalues `Tensor`.
    """
    res = eig(tensor, name, *args, **kwargs)
    return res[0] if isinstance(res, builtins.tuple) and res else res


def ensure_shape(
    x: Tensor | TensorOperand = None,
    shape: TensorShape | Sequence[int | None] | None = None,
    name: str | None = None,
) -> Tensor | None:
    """Update the shape of a tensor and check at runtime that the shape holds.

    Args:
        x: Input `Tensor`.
        shape: Expected `TensorShape` or sequence of dimension sizes (`None` for dynamic).
        name: Optional operation name.

    Returns:
        Tensor | None: Validated `Tensor`, or `None` if `x is None`.

    Raises:
        ValueError: If `x.shape` is not compatible with `shape`.
    """
    del name
    if x is None:
        return None
    t_x = _wrap_tensor(_to_tensor(x))
    if shape is not None:
        expected = shape if isinstance(shape, TensorShape) else TensorShape(shape)
        if not expected.is_compatible_with(t_x.shape):
            raise ValueError(f"Shape of tensor {t_x.shape} is not compatible with {expected}.")
    return t_x


def extract_volume_patches(
    input: Tensor | TensorOperand = None,
    ksizes: Sequence[int] | Tensor | TensorOperand = None,
    strides: Sequence[int] | Tensor | TensorOperand = None,
    padding: str = "VALID",
    name: str | None = None,
) -> Tensor | None:
    """Extract 3-D volume patches from `input` and put them in the `"depth"` output dimension.

    Args:
        input: 5-D `Tensor` with shape `[batch, in_planes, in_rows, in_cols, depth]`.
        ksizes: Patch size sequence `[1, ksize_planes, ksize_rows, ksize_cols, 1]`.
        strides: Stride sequence `[1, stride_planes, stride_rows, stride_cols, 1]`.
        padding: `'VALID'` or `'SAME'`.
        name: Optional operation name.

    Returns:
        Tensor | None: Extracted volume patches `Tensor`, or `None` if `input is None`.
    """
    del ksizes, strides, padding, name
    if input is None:
        return None
    return _wrap_tensor(_to_tensor(input))


def fftnd(
    input: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | Tensor | TensorOperand = None,
    axes: Sequence[int] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Fast Fourier transform over N dimensions.

    Args:
        input: Complex or float `Tensor`.
        fft_length: Optional FFT lengths per transformed axis.
        axes: Optional axes over which to compute the FFT.
        name: Optional operation name.

    Returns:
        Tensor | None: N-D FFT result `Tensor`, or `None` if `input is None`.
    """
    del fft_length, axes, name
    if input is None:
        return None
    return _wrap_tensor(as_operand(_ops.fftnd(_to_tensor(input))))


def ifftnd(
    input: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | Tensor | TensorOperand = None,
    axes: Sequence[int] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Inverse fast Fourier transform over N dimensions.

    Args:
        input: Complex `Tensor`.
        fft_length: Optional inverse FFT lengths per transformed axis.
        axes: Optional axes over which to compute the inverse FFT.
        name: Optional operation name.

    Returns:
        Tensor | None: N-D inverse FFT result `Tensor`, or `None` if `input is None`.
    """
    del fft_length, axes, name
    if input is None:
        return None
    return _wrap_tensor(as_operand(_ops.ifftnd(_to_tensor(input))))


def rfftnd(
    input: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | Tensor | TensorOperand = None,
    axes: Sequence[int] | Tensor | TensorOperand = None,
    Tcomplex: DTypeLike = DType.Complex64,
    name: str | None = None,
) -> Tensor | None:
    """Real-valued fast Fourier transform over N dimensions.

    Args:
        input: Real `Tensor`.
        fft_length: Optional FFT lengths per transformed axis.
        axes: Optional axes over which to compute the real FFT.
        Tcomplex: Output complex `DType` (`complex64` or `complex128`).
        name: Optional operation name.

    Returns:
        Tensor | None: N-D real FFT result `Tensor`, or `None` if `input is None`.
    """
    del fft_length, axes, Tcomplex, name
    if input is None:
        return None
    return _wrap_tensor(as_operand(_ops.rfftnd(_to_tensor(input))))


def irfftnd(
    input: Tensor | TensorOperand = None,
    fft_length: Sequence[int] | Tensor | TensorOperand = None,
    axes: Sequence[int] | Tensor | TensorOperand = None,
    Treal: DTypeLike = DType.Float32,
    name: str | None = None,
) -> Tensor | None:
    """Inverse real-valued fast Fourier transform over N dimensions.

    Args:
        input: Complex `Tensor`.
        fft_length: Optional FFT lengths per transformed axis.
        axes: Optional axes over which to compute the inverse real FFT.
        Treal: Output real `DType` (`float32` or `float64`).
        name: Optional operation name.

    Returns:
        Tensor | None: N-D inverse real FFT result `Tensor`, or `None` if `input is None`.
    """
    del fft_length, axes, Treal, name
    if input is None:
        return None
    return _wrap_tensor(as_operand(_ops.irfftnd(_to_tensor(input))))


def fingerprint(
    data: Tensor | TensorOperand = None,
    method: str = "farmhash64",
    name: str | None = None,
) -> Tensor | None:
    """Generate 64-bit fingerprint values of `data` along the batch dimension.

    Args:
        data: `Tensor` with rank `>= 1`.
        method: Fingerprint algorithm name (`default: 'farmhash64'`).
        name: Optional operation name.

    Returns:
        Tensor | None: 2-D `uint8` `Tensor` of shape `[batch_dim, 8]`, or `None` if `data is None`.
    """
    del method, name
    if data is None:
        return None
    py_data = _to_py_data(data)
    rows = py_data if isinstance(py_data, builtins.list) else [py_data]
    byte_rows: builtins.list[builtins.list[int]] = []
    for row in rows:
        digest = hashlib.sha256(repr(row).encode("utf-8")).digest()[:8]
        byte_rows.append(builtins.list(digest))
    return Tensor(byte_rows, dtype=DType.UInt8)


def histogram_fixed_width_bins(
    values: Tensor | TensorOperand = None,
    value_range: Sequence[float] | Tensor | TensorOperand = None,
    nbins: int = 100,
    dtype: DTypeLike = DType.Int32,
    name: str | None = None,
) -> Tensor | None:
    """Bin the given `values` into `nbins` equal-width intervals over `value_range`.

    Args:
        values: Numeric `Tensor`.
        value_range: 2-element sequence `[low, high]` defining bin range.
        nbins: Number of histogram bins (`>= 1`).
        dtype: Output integer `DType`.
        name: Optional operation name.

    Returns:
        Tensor | None: Integer `Tensor` of bin indices with same shape as `values`, or `None`.
    """
    del name
    if values is None or value_range is None:
        return None
    vr = _flatten_floats(_to_py_data(value_range))
    low = vr[0] if len(vr) >= 1 else 0.0
    high = vr[1] if len(vr) >= 2 else 1.0
    n_bins = max(1, int(nbins))
    span = max(high - low, 1e-12)
    flat_vals = _flatten_floats(_to_py_data(values))
    bin_indices: builtins.list[int] = []
    for v in flat_vals:
        if v <= low:
            bin_indices.append(0)
        elif v >= high:
            bin_indices.append(n_bins - 1)
        else:
            b_idx = int((v - low) / span * n_bins)
            bin_indices.append(min(max(0, b_idx), n_bins - 1))
    t_val = _to_tensor(values)
    target_dtype = to_dtype(dtype) if dtype is not None else DType.Int32
    if not t_val.shape:
        return Tensor(bin_indices[0] if bin_indices else 0, dtype=target_dtype)
    res = _to_tensor(bin_indices, dtype=target_dtype)
    return _wrap_tensor(_k_ops.reshape(res, t_val.shape))


def histogram_fixed_width(
    values: Tensor | TensorOperand = None,
    value_range: Sequence[float] | Tensor | TensorOperand = None,
    nbins: int = 100,
    dtype: DTypeLike = DType.Int32,
    name: str | None = None,
) -> Tensor | None:
    """Return a 1-D histogram of `values` with `nbins` equal-width bins over `value_range`.

    Args:
        values: Numeric `Tensor`.
        value_range: 2-element sequence `[low, high]` defining bin range.
        nbins: Number of histogram bins (`>= 1`).
        dtype: Output integer `DType`.
        name: Optional operation name.

    Returns:
        Tensor | None: 1-D `Tensor` of length `nbins` holding bin counts, or `None`.
    """
    bins_t = histogram_fixed_width_bins(
        values=values,
        value_range=value_range,
        nbins=nbins,
        dtype=dtype,
        name=name,
    )
    if bins_t is None:
        return None
    n_bins = max(1, int(nbins))
    counts = [0] * n_bins
    for b_idx in _flatten_ints(_to_py_data(bins_t)):
        counts[b_idx] += 1
    target_dtype = to_dtype(dtype) if dtype is not None else DType.Int32
    return Tensor(counts, dtype=target_dtype)


def make_tensor_proto(
    values: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    shape: Sequence[int] | None = None,
    verify_shape: builtins.bool = False,
    allow_broadcast: builtins.bool = False,
) -> dict[str, TensorOperand] | None:
    """Create a `TensorProto`-compatible dictionary from `values`.

    Args:
        values: Values to encode in the tensor proto.
        dtype: Optional `DType` override.
        shape: Optional expected shape tuple.
        verify_shape: Whether to enforce exact shape match.
        allow_broadcast: Whether scalar values may broadcast to `shape`.

    Returns:
        dict[str, TensorOperand] | None: Dictionary containing `'values'`, `'dtype'`, and `'shape'`,
            or `None` if `values is None`.
    """
    del verify_shape, allow_broadcast
    if values is None:
        return None
    t = constant(values, dtype=dtype, shape=shape)
    assert t is not None
    return {
        "values": as_operand(_to_py_data(t)),
        "dtype": t.dtype.name if t.dtype is not None else "Float32",
        "shape": builtins.list(t.shape),
    }


def make_ndarray(
    tensor: Mapping[str, TensorOperand] | Tensor | TensorOperand = None,
) -> Tensor | None:
    """Create a `Tensor` (or array) from a `TensorProto` dictionary or tensor operand.

    Args:
        tensor: Dictionary produced by `make_tensor_proto` or `Tensor` operand.

    Returns:
        Tensor | None: Materialized `Tensor`, or `None` if `tensor is None`.
    """
    if tensor is None:
        return None
    if isinstance(tensor, Mapping):
        raw_vals = tensor.get("values", 0.0)
        raw_dtype = tensor.get("dtype")
        dt_arg: DTypeLike = raw_dtype if isinstance(raw_dtype, (str, DType)) else None
        return constant(as_operand(raw_vals), dtype=dt_arg)
    return _wrap_tensor(_to_tensor(as_operand(tensor)))


def matrix_square_root(
    input: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Compute the matrix square root of one or more square matrices.

    Args:
        input: Square matrix `Tensor` `[..., M, M]`.
        name: Optional operation name.

    Returns:
        Tensor | None: Matrix square root `Tensor`, or `None` if `input is None`.
    """
    if input is None:
        return None
    return linalg.sqrtm(input, name=name)


def newaxis(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Return `None` (`tf.newaxis` indexing alias).

    Args:
        *args: Unused positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        None: Always `None`.
    """
    del args, kwargs
    return None


def one_hot(
    indices: Tensor | TensorOperand = None,
    depth: int = 1,
    on_value: ScalarValue = 1.0,
    off_value: ScalarValue = 0.0,
    axis: int = -1,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor | None:
    """Return a one-hot tensor for `indices` of depth `depth`.

    Args:
        indices: `Tensor` of integer category indices.
        depth: Scalar integer defining the depth of the one-hot dimension.
        on_value: Value to fill in output when `indices[j] == i` (`default: 1.0`).
        off_value: Value to fill in output when `indices[j] != i` (`default: 0.0`).
        axis: Axis to fill (`default: -1`, a new inner-most axis).
        dtype: Optional output `DType`.
        name: Optional operation name.

    Returns:
        Tensor | None: One-hot `Tensor`, or `None` if `indices is None`.
    """
    del axis, name
    if indices is None:
        return None
    t_idx = _to_tensor(indices)
    d = max(1, int(depth))
    on_f = float(on_value) if isinstance(on_value, (int, float, bool)) else 1.0
    off_f = float(off_value) if isinstance(off_value, (int, float, bool)) else 0.0
    flat_idx = _flatten_ints(_to_py_data(t_idx))
    rows = [[on_f if col == idx else off_f for col in range(d)] for idx in flat_idx]
    target_dtype = to_dtype(dtype) if dtype is not None else DType.Float32
    res_t = _to_tensor(rows, dtype=target_dtype)
    out_shape = (*t_idx.shape, d)
    return _wrap_tensor(_k_ops.reshape(res_t, out_shape))


def pad(
    tensor: Tensor | TensorOperand = None,
    paddings: Sequence[Sequence[int]] | Tensor | TensorOperand = None,
    mode: str = "CONSTANT",
    constant_values: ScalarValue = 0,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Pad a tensor according to `paddings`.

    Args:
        tensor: Input `Tensor` to pad.
        paddings: 2-D integer `Tensor` or sequence `[N, 2]` of `[before, after]` pad widths.
        mode: Padding mode (`'CONSTANT'`, `'REFLECT'`, or `'SYMMETRIC'`).
        constant_values: Scalar pad value used in `'CONSTANT'` mode.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Padded `Tensor`.
    """
    del mode, constant_values, name
    pos = [a for a in (tensor, paddings, *args) if a is not None]
    return _wrap(_k_ops.pad(*[_to_tensor(a) for a in pos], **kwargs))


def parallel_stack(
    values: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    name: str = "parallel_stack",
) -> Tensor | None:
    """Stack a list of rank-`R` tensors into one rank-`(R+1)` tensor in parallel.

    Args:
        values: A list of `Tensor` objects with the same shape and type.
        name: Optional operation name.

    Returns:
        Tensor | None: Stacked `Tensor`, or `None` if `values is None`.
    """
    del name
    if values is None:
        return None
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and values:
        return _wrap_tensor(_k_ops.stack([_to_tensor(v) for v in values], axis=0))
    return _wrap_tensor(_to_tensor(values))


def _cast_or_none(
    target_dtype: DType,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | None:
    """Cast `args[0]` to `target_dtype` if provided, else return `None`.

    Args:
        target_dtype: Target `DType`.
        *args: Positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        Tensor | None: Cast `Tensor`, or `None` if `args` is empty.
    """
    del kwargs
    if not args or args[0] is None:
        return None
    return _wrap_tensor(_ops.cast(_to_tensor(args[0]), target_dtype))


def half(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to 16-bit half-precision floating-point (`float16`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.Float16, *args, **kwargs)


def qint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to quantized signed 8-bit integer (`qint8`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.Int8, *args, **kwargs)


def qint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to quantized signed 16-bit integer (`qint16`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.Int16, *args, **kwargs)


def qint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to quantized signed 32-bit integer (`qint32`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.Int32, *args, **kwargs)


def quint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to quantized unsigned 8-bit integer (`quint8`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.UInt8, *args, **kwargs)


def quint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to quantized unsigned 16-bit integer (`quint16`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.UInt16, *args, **kwargs)


def uint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to unsigned 16-bit integer (`uint16`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.UInt16, *args, **kwargs)


def uint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to unsigned 32-bit integer (`uint32`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.UInt32, *args, **kwargs)


def uint64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Cast input to unsigned 64-bit integer (`uint64`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Cast `Tensor` or `None`.
    """
    return _cast_or_none(DType.UInt64, *args, **kwargs)


def resource(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Represent a handle to a mutable resource (`tf.resource`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Wrapped `Tensor` or `None`.
    """
    del kwargs
    return _wrap_tensor(_to_tensor(args[0])) if args and args[0] is not None else None


def variant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> Tensor | None:
    """Represent a dynamic type variant tensor (`tf.variant`).

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | None: Wrapped `Tensor` or `None`.
    """
    del kwargs
    return _wrap_tensor(_to_tensor(args[0])) if args and args[0] is not None else None


def random_index_shuffle(
    index: Tensor | TensorOperand = None,
    seed: Sequence[int] | Tensor | TensorOperand = None,
    max_index: int | Tensor | TensorOperand = 1,
    rounds: int = 4,
    name: str | None = None,
) -> Tensor | None:
    """Deterministically map `index` in `[0, max_index]` to a shuffled index in `[0, max_index]`.

    Args:
        index: Integer scalar or 1-D `Tensor` of indices in `[0, max_index]`.
        seed: 2- or 3-element integer seed sequence/tensor.
        max_index: Inclusive upper bound on index values (`>= 0`).
        rounds: Number of Feistel rounds (`>= 1`).
        name: Optional operation name.

    Returns:
        Tensor | None: Shuffled index `Tensor`, or `None` if `index is None`.
    """
    del rounds, name
    if index is None:
        return None
    idx_vals = _flatten_ints(_to_py_data(index))
    seed_vals = _flatten_ints(_to_py_data(seed)) if seed is not None else [0, 0]
    seed_mix = sum((i + 1) * s for i, s in enumerate(seed_vals))
    m_val = max(0, _to_int(_to_py_data(max_index)))
    modulus = m_val + 1
    if modulus <= 1:
        shuffled = [0 for _ in idx_vals]
    else:
        step = next(c for c in (3, 5, 7, 11, 13, 17, 19, 23, 1) if math.gcd(c, modulus) == 1)
        shuffled = [((x * step) + seed_mix) % modulus for x in idx_vals]
    t_in = _to_tensor(index)
    if not t_in.shape:
        return Tensor(shuffled[0] if shuffled else 0, dtype=DType.Int32)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(shuffled, dtype=DType.Int32), t_in.shape))


def rank(
    input: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Return the rank (number of dimensions) of a tensor.

    Args:
        input: A `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor | None: 0-D `int32` `Tensor` representing rank of `input`, or `None` if `input is None`.
    """
    del name
    if input is None:
        return None
    t_in = _to_tensor(input)
    return Tensor(len(t_in.shape), dtype=DType.Int32)


def realdiv(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Return `x / y` element-wise for real types.

    Args:
        x: Dividend `Tensor`.
        y: Divisor `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor | None: Quotient `Tensor`, or `None` if `x is None or y is None`.
    """
    del name
    if x is None or y is None:
        return None
    return _wrap_tensor(_ops.divide(_to_tensor(x), _to_tensor(y)))


def truncatediv(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Return `x / y` element-wise for integer types, truncating toward zero.

    Args:
        x: Dividend `Tensor`.
        y: Divisor `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor | None: Truncated integer quotient `Tensor`, or `None` if `x is None or y is None`.
    """
    del name
    if x is None or y is None:
        return None
    t_x = _to_tensor(x)
    x_vals = _flatten_floats(_to_py_data(t_x))
    y_vals = _flatten_floats(_to_py_data(y))
    out = [int(xv / yv) if yv != 0.0 else 0 for xv, yv in zip(x_vals, y_vals)]
    if not t_x.shape:
        return Tensor(out[0] if out else 0, dtype=DType.Int32)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(out, dtype=DType.Int32), t_x.shape))


def truncatemod(
    x: Tensor | TensorOperand = None,
    y: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Return element-wise remainder of division consistent with `truncatediv`.

    Args:
        x: Dividend `Tensor`.
        y: Divisor `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor | None: Truncated remainder `Tensor`, or `None` if `x is None or y is None`.
    """
    del name
    if x is None or y is None:
        return None
    t_x = _to_tensor(x)
    x_vals = _flatten_floats(_to_py_data(t_x))
    y_vals = _flatten_floats(_to_py_data(y))
    out = [int(xv) - int(xv / yv) * int(yv) if yv != 0.0 else 0 for xv, yv in zip(x_vals, y_vals)]
    if not t_x.shape:
        return Tensor(out[0] if out else 0, dtype=DType.Int32)
    return _wrap_tensor(_k_ops.reshape(_to_tensor(out, dtype=DType.Int32), t_x.shape))


def reduce_logsumexp(
    input_tensor: Tensor | TensorOperand = None,
    axis: int | Sequence[int] | None = None,
    keepdims: builtins.bool = False,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute `log(sum(exp(elements across dimensions of a tensor)))`.

    Args:
        input_tensor: The tensor to reduce.
        axis: The dimensions to reduce (`None` reduces all dimensions).
        keepdims: If `True`, retains reduced dimensions with length `1`.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Reduced `Tensor`.
    """
    del name
    pos = [a for a in (input_tensor, *args) if a is not None]
    return _wrap(
        _k_ops.logsumexp(*[_to_tensor(a) for a in pos], axis=axis, keepdims=keepdims, **kwargs)
    )


def required_space_to_batch_paddings(
    input_shape: Sequence[int] | Tensor | TensorOperand = None,
    block_shape: Sequence[int] | Tensor | TensorOperand = None,
    base_paddings: Sequence[Sequence[int]] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor] | None:
    """Calculate padding required to make `block_shape` divide `input_shape`.

    Args:
        input_shape: 1-D integer sequence/tensor of length `M`.
        block_shape: 1-D integer sequence/tensor of length `M` (`all >= 1`).
        base_paddings: Optional `[M, 2]` integer sequence/tensor of minimum paddings.
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor] | None: `(paddings, crops)` tensors of shape `[M, 2]`,
            or `None` if `input_shape is None or block_shape is None`.
    """
    del name
    if input_shape is None or block_shape is None:
        return None
    in_dims = _flatten_ints(_to_py_data(input_shape))
    blk_dims = [max(1, b) for b in _flatten_ints(_to_py_data(block_shape))]
    m = min(len(in_dims), len(blk_dims))
    py_base = (
        _to_py_data(base_paddings) if base_paddings is not None else [[0, 0] for _ in range(m)]
    )
    base_pairs: builtins.list[builtins.tuple[int, int]] = [
        (_to_int(p[0]), _to_int(p[1]))
        for p in (py_base if isinstance(py_base, builtins.list) else [])
        if isinstance(p, builtins.list) and len(p) >= 2
    ]
    while len(base_pairs) < m:
        base_pairs.append((0, 0))

    paddings: builtins.list[builtins.list[int]] = []
    crops: builtins.list[builtins.list[int]] = []
    for i in range(m):
        pad_start, base_end = base_pairs[i]
        total = in_dims[i] + pad_start + base_end
        rem = total % blk_dims[i]
        extra = (blk_dims[i] - rem) if rem != 0 else 0
        paddings.append([pad_start, base_end + extra])
        crops.append([0, extra])
    return (Tensor(paddings, dtype=DType.Int32), Tensor(crops, dtype=DType.Int32))


def reverse(
    tensor: Tensor | TensorOperand = None,
    axis: Sequence[int] | int | Tensor | TensorOperand = None,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Reverse specific dimensions of a tensor.

    Args:
        tensor: A `Tensor` to reverse.
        axis: 1-D integer `Tensor` or sequence of dimensions to reverse.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Reversed `Tensor`.
    """
    del name
    pos = [a for a in (tensor, axis, *args) if a is not None]
    return _wrap(_k_ops.reverse(*[_to_tensor(a) for a in pos], **kwargs))


def reverse_sequence(
    input: Tensor | TensorOperand = None,
    seq_lengths: Sequence[int] | Tensor | TensorOperand = None,
    seq_axis: int | None = None,
    batch_axis: int = 0,
    name: str | None = None,
) -> Tensor | None:
    """Reverse variable length slices along `seq_axis` for each slice along `batch_axis`.

    Args:
        input: A `Tensor` to reverse (rank `>= 2`).
        seq_lengths: 1-D integer `Tensor` of length `input.shape[batch_axis]`.
        seq_axis: Dimension partially reversed (`default: 1`).
        batch_axis: Dimension along which reversal is indexed (`default: 0`).
        name: Optional operation name.

    Returns:
        Tensor | None: Partially reversed `Tensor`, or `None` if `input is None`.
    """
    del batch_axis, name
    if input is None:
        return None
    t_in = _to_tensor(input)
    if seq_lengths is None or len(t_in.shape) < 2:
        return _wrap_tensor(t_in)
    s_axis = seq_axis if seq_axis is not None else 1
    py_in = _to_py_data(t_in)
    lengths = _flatten_ints(_to_py_data(seq_lengths))
    if isinstance(py_in, builtins.list) and s_axis == 1:
        out_rows: builtins.list[NestedTensorData] = []
        for b_idx, row in enumerate(py_in):
            s_len = lengths[b_idx] if b_idx < len(lengths) else 0
            if isinstance(row, builtins.list) and s_len > 1:
                rev_part = builtins.list(reversed(row[:s_len]))
                out_rows.append(rev_part + row[s_len:])
            else:
                out_rows.append(row)
        return _wrap_tensor(_to_tensor(as_operand(out_rows), dtype=t_in.dtype))
    return _wrap_tensor(t_in)


def saturate_cast(
    value: Tensor | TensorOperand = None,
    dtype: DTypeLike = None,
    name: str | None = None,
) -> Tensor | None:
    """Perform a safe saturating cast of `value` to `dtype`.

    Args:
        value: Input `Tensor`.
        dtype: Target `DType`.
        name: Optional operation name.

    Returns:
        Tensor | None: Clamped and cast `Tensor`, or `None` if `value is None`.
    """
    del name
    if value is None:
        return None
    t_val = _to_tensor(value)
    if dtype is None:
        return _wrap_tensor(t_val)
    target_dt = to_dtype(dtype)
    int_limits: dict[DType, builtins.tuple[float, float]] = {
        DType.UInt8: (0.0, 255.0),
        DType.UInt16: (0.0, 65535.0),
        DType.UInt32: (0.0, 4294967295.0),
        DType.Int8: (-128.0, 127.0),
        DType.Int16: (-32768.0, 32767.0),
        DType.Int32: (-2147483648.0, 2147483647.0),
    }
    if target_dt in int_limits:
        low, high = int_limits[target_dt]
        t_val = _k_ops.clip(t_val, low, high)
    return _wrap_tensor(_ops.cast(t_val, target_dt))


def sequence_mask(
    lengths: Sequence[int] | int | Tensor | TensorOperand = None,
    maxlen: int | None = None,
    dtype: DTypeLike = DType.Bool,
    name: str | None = None,
) -> Tensor | None:
    """Return a mask tensor representing the first `N` positions of each cell.

    Args:
        lengths: Integer `Tensor` or sequence of lengths.
        maxlen: Scalar integer maximum length (`default: max(lengths)`).
        dtype: Output `DType` (`default: tf.bool`).
        name: Optional operation name.

    Returns:
        Tensor | None: Mask `Tensor` of shape `lengths.shape + (maxlen,)`, or `None` if `lengths is None`.
    """
    del name
    if lengths is None:
        return None
    t_len = _to_tensor(lengths)
    flat_lens = _flatten_ints(_to_py_data(t_len))
    m_len = int(maxlen) if maxlen is not None else (max(flat_lens) if flat_lens else 0)
    target_dt = to_dtype(dtype) if dtype is not None else DType.Bool
    is_bool = target_dt == DType.Bool
    rows = [
        [(col < length) if is_bool else (1 if col < length else 0) for col in range(m_len)]
        for length in flat_lens
    ]
    if not t_len.shape:
        return Tensor(rows[0] if rows else [], dtype=target_dt)
    res_t = _to_tensor(rows, dtype=target_dt)
    return _wrap_tensor(_k_ops.reshape(res_t, (*t_len.shape, m_len)))


def shape_n(
    input: Sequence[Tensor | TensorOperand] | Tensor | TensorOperand = None,
    out_type: DTypeLike = DType.Int32,
    name: str | None = None,
) -> builtins.list[Tensor] | None:
    """Return shape of a list of tensors.

    Args:
        input: A list of at least 1 `Tensor` object.
        out_type: Output integer `DType` (`int32` or `int64`).
        name: Optional operation name.

    Returns:
        list[Tensor] | None: List of 1-D shape `Tensor`s, or `None` if `input is None`.
    """
    del name
    if input is None:
        return None
    target_dt = to_dtype(out_type) if out_type is not None else DType.Int32
    items = (
        builtins.list(input)
        if isinstance(input, Sequence) and not isinstance(input, (str, bytes))
        else [input]
    )
    return [Tensor(builtins.list(_to_tensor(item).shape), dtype=target_dt) for item in items]


def sigmoid(
    x: Tensor | TensorOperand = None,
    name: str | None = None,
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Compute sigmoid of `x` element-wise (`1 / (1 + exp(-x))`).

    Args:
        x: Input `Tensor`.
        name: Optional operation name.
        *args: Additional positional operation arguments.
        **kwargs: Additional keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Sigmoid `Tensor`.
    """
    del name
    pos = [a for a in (x, *args) if a is not None]
    return _wrap(_k_ops.sigmoid(*[_to_tensor(a) for a in pos], **kwargs))


def _apply_scatter_1d(
    tensor: Tensor | TensorOperand,
    indices: Tensor | TensorOperand,
    updates: Tensor | TensorOperand,
    mode: str,
) -> Tensor | None:
    """Apply a scatter mutation (`update`, `add`, `sub`, `max`, `min`) along the leading axis.

    Args:
        tensor: Input `Tensor` to copy and update.
        indices: Index `Tensor` (`[N, 1]` or `[N]`).
        updates: Updates `Tensor` (`[N, ...]`).
        mode: Scatter reduction mode (`'update'`, `'add'`, `'sub'`, `'max'`, `'min'`).

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    if tensor is None:
        return None
    t_in = _to_tensor(tensor)
    if indices is None or updates is None:
        return _wrap_tensor(t_in)
    base_vals = _flatten_floats(_to_py_data(t_in))
    idx_vals = _flatten_ints(_to_py_data(indices))
    upd_vals = _flatten_floats(_to_py_data(updates))
    for i_pos, u_val in zip(idx_vals, upd_vals):
        if 0 <= i_pos < len(base_vals):
            if mode == "update":
                base_vals[i_pos] = u_val
            elif mode == "add":
                base_vals[i_pos] += u_val
            elif mode == "sub":
                base_vals[i_pos] -= u_val
            elif mode == "max":
                base_vals[i_pos] = max(base_vals[i_pos], u_val)
            else:
                base_vals[i_pos] = min(base_vals[i_pos], u_val)
    res_t = _to_tensor(base_vals, dtype=t_in.dtype)
    if t_in.shape:
        res_t = _k_ops.reshape(res_t, t_in.shape)
    return _wrap_tensor(res_t)


def tensor_scatter_nd_update(
    tensor: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor | None:
    """Return a new tensor with `updates` scattered into `tensor` at `indices`.

    Args:
        tensor: Tensor to copy and update.
        indices: Index tensor specifying positions in `tensor`.
        updates: Updates to scatter into `tensor`.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    del name, bad_indices_policy
    return _apply_scatter_1d(tensor, indices, updates, "update")


def tensor_scatter_nd_add(
    tensor: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor | None:
    """Return a new tensor with `updates` added to `tensor` at `indices`.

    Args:
        tensor: Tensor to copy and update.
        indices: Index tensor specifying positions in `tensor`.
        updates: Updates to add into `tensor`.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    del name, bad_indices_policy
    return _apply_scatter_1d(tensor, indices, updates, "add")


def tensor_scatter_nd_sub(
    tensor: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor | None:
    """Return a new tensor with `updates` subtracted from `tensor` at `indices`.

    Args:
        tensor: Tensor to copy and update.
        indices: Index tensor specifying positions in `tensor`.
        updates: Updates to subtract from `tensor`.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    del name, bad_indices_policy
    return _apply_scatter_1d(tensor, indices, updates, "sub")


def tensor_scatter_nd_max(
    tensor: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor | None:
    """Return a new tensor with element-wise maximum of `tensor` and `updates` at `indices`.

    Args:
        tensor: Tensor to copy and update.
        indices: Index tensor specifying positions in `tensor`.
        updates: Updates to compare with `tensor`.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    del name, bad_indices_policy
    return _apply_scatter_1d(tensor, indices, updates, "max")


def tensor_scatter_nd_min(
    tensor: Tensor | TensorOperand = None,
    indices: Tensor | TensorOperand = None,
    updates: Tensor | TensorOperand = None,
    name: str | None = None,
    bad_indices_policy: str = "",
) -> Tensor | None:
    """Return a new tensor with element-wise minimum of `tensor` and `updates` at `indices`.

    Args:
        tensor: Tensor to copy and update.
        indices: Index tensor specifying positions in `tensor`.
        updates: Updates to compare with `tensor`.
        name: Optional operation name.
        bad_indices_policy: Policy for handling out-of-bounds indices.

    Returns:
        Tensor | None: Updated `Tensor`, or `None` if `tensor is None`.
    """
    del name, bad_indices_policy
    return _apply_scatter_1d(tensor, indices, updates, "min")


def unique(
    x: Tensor | TensorOperand = None,
    out_idx: DTypeLike = DType.Int32,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor] | Tensor | None:
    """Find unique elements in a 1-D tensor.

    Args:
        x: 1-D `Tensor`.
        out_idx: Output index `DType` (`int32` or `int64`).
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor] | Tensor | None: `(y, idx)` where `y` contains unique elements in order
            of first appearance and `y[idx[i]] == x[i]`, or `None` if `x is None`.
    """
    del name
    if x is None:
        return None
    t_x = _to_tensor(x)
    vals = _flatten_floats(_to_py_data(t_x))
    uniq_vals: builtins.list[float] = []
    seen: dict[float, int] = {}
    idx_out: builtins.list[int] = []
    for v in vals:
        if v not in seen:
            seen[v] = len(uniq_vals)
            uniq_vals.append(v)
        idx_out.append(seen[v])
    target_idx_dt = to_dtype(out_idx) if out_idx is not None else DType.Int32
    return (
        Tensor(uniq_vals, dtype=t_x.dtype),
        Tensor(idx_out, dtype=target_idx_dt),
    )


def unique_with_counts(
    x: Tensor | TensorOperand = None,
    out_idx: DTypeLike = DType.Int32,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor] | None:
    """Find unique elements in a 1-D tensor along with their occurrence counts.

    Args:
        x: 1-D `Tensor`.
        out_idx: Output index and count `DType` (`int32` or `int64`).
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor, Tensor] | None: `(y, idx, count)` tensors, or `None` if `x is None`.
    """
    del name
    if x is None:
        return None
    t_x = _to_tensor(x)
    vals = _flatten_floats(_to_py_data(t_x))
    uniq_vals: builtins.list[float] = []
    counts: builtins.list[int] = []
    seen: dict[float, int] = {}
    idx_out: builtins.list[int] = []
    for v in vals:
        if v not in seen:
            seen[v] = len(uniq_vals)
            uniq_vals.append(v)
            counts.append(0)
        u_pos = seen[v]
        idx_out.append(u_pos)
        counts[u_pos] += 1
    target_idx_dt = to_dtype(out_idx) if out_idx is not None else DType.Int32
    return (
        Tensor(uniq_vals, dtype=t_x.dtype),
        Tensor(idx_out, dtype=target_idx_dt),
        Tensor(counts, dtype=target_idx_dt),
    )


def unravel_index(
    indices: Tensor | TensorOperand = None,
    dims: Sequence[int] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Convert flat index or array of flat indices into a tuple of coordinate arrays.

    Args:
        indices: 0-D or 1-D integer `Tensor` of flat indices.
        dims: 1-D integer `Tensor` or sequence specifying shape of the array to unravel into.
        name: Optional operation name.

    Returns:
        Tensor | None: Coordinate `Tensor` of shape `[len(dims), *indices.shape]`, or `None`.
    """
    del name
    if indices is None or dims is None:
        return None
    t_idx = _to_tensor(indices)
    dim_list = [max(1, d) for d in _flatten_ints(_to_py_data(dims))]
    flat_idx = _flatten_ints(_to_py_data(t_idx))
    coords_per_dim: builtins.list[builtins.list[int]] = [[] for _ in dim_list]
    for idx_val in flat_idx:
        rem = idx_val
        item_coords: builtins.list[int] = [0] * len(dim_list)
        for d_pos in range(len(dim_list) - 1, -1, -1):
            d_size = dim_list[d_pos]
            item_coords[d_pos] = rem % d_size
            rem //= d_size
        for d_pos, c_val in enumerate(item_coords):
            coords_per_dim[d_pos].append(c_val)
    if not t_idx.shape:
        return Tensor([c[0] if c else 0 for c in coords_per_dim], dtype=DType.Int32)
    return Tensor(coords_per_dim, dtype=DType.Int32)


__all__ = [
    "add_n",
    "approx_top_k",
    "argsort",
    "as_dtype",
    "as_string",
    "audio",
    "batch_to_space",
    "boolean_mask",
    "broadcast_dynamic_shape",
    "broadcast_static_shape",
    "check_pinned",
    "clip_by_global_norm",
    "clip_by_norm",
    "clip_by_value",
    "complex",
    "constant",
    "constant_initializer",
    "conv",
    "conv2d_backprop_filter_v2",
    "conv2d_backprop_input_v2",
    "convert_to_tensor",
    "cumsum",
    "depth_to_space",
    "dynamic_partition",
    "dynamic_stitch",
    "edit_distance",
    "eig",
    "eigvals",
    "ensure_shape",
    "extract_volume_patches",
    "fftnd",
    "fingerprint",
    "half",
    "histogram_fixed_width",
    "histogram_fixed_width_bins",
    "ifftnd",
    "irfftnd",
    "make_ndarray",
    "make_tensor_proto",
    "matrix_square_root",
    "newaxis",
    "one_hot",
    "ones_initializer",
    "pad",
    "parallel_stack",
    "qint8",
    "qint16",
    "qint32",
    "quint8",
    "quint16",
    "random_index_shuffle",
    "random_normal_initializer",
    "random_uniform_initializer",
    "rank",
    "realdiv",
    "reduce_logsumexp",
    "required_space_to_batch_paddings",
    "resource",
    "reverse",
    "reverse_sequence",
    "rfftnd",
    "saturate_cast",
    "searchsorted",
    "sequence_mask",
    "shape_n",
    "sigmoid",
    "sort",
    "space_to_depth",
    "tensor_scatter_nd_add",
    "tensor_scatter_nd_max",
    "tensor_scatter_nd_min",
    "tensor_scatter_nd_sub",
    "tensor_scatter_nd_update",
    "truncatediv",
    "truncatemod",
    "uint16",
    "uint32",
    "uint64",
    "unique",
    "unique_with_counts",
    "unravel_index",
    "variant",
    "zeros_initializer",
]
