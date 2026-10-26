"""Top-level core TensorFlow helper and stub functions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from types import TracebackType
from typing import Protocol, runtime_checkable

from zero_tensorflow._typing import (
    DTypeLike,
    ScalarValue,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.array_ops import _to_float, _to_int, _to_py_data
from zero_tensorflow.core_tensor import Tensor, Variable, _to_tensor, _wrap_tensor, to_dtype
from zero_tensorflow.dtypes import DType
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.sparse import SparseTensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as msc_ops
from ml_switcheroo_compiler.core.config import config as _compiler_config
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import LogicalEdge, LogicalGraph, LogicalNode
from zero_keras import ops as _k_ops


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


def import_graph_def(
    graph_def: LogicalGraph
    | Mapping[str, Sequence[LogicalNode]]
    | Sequence[LogicalNode]
    | TensorOperand = None,
    input_map: Mapping[str, Tensor | LogicalNode] | None = None,
    return_elements: Sequence[str] | None = None,
    name: str | None = None,
    op_dict: Mapping[str, str] | None = None,
    producer_op_list: Sequence[str] | None = None,
) -> list[LogicalNode] | LogicalGraph | None:
    """Import a `LogicalGraph` or node collection into the active graph.

    Args:
        graph_def: `LogicalGraph`, node mapping, or sequence of `LogicalNode` objects to import.
        input_map: Optional mapping from input names to replacement nodes/tensors.
        return_elements: Optional list of node names to return after import.
        name: Optional prefix scope name prepended to imported node IDs (`default: "import"`).
        op_dict: Optional dictionary of allowed operation names.
        producer_op_list: Optional list of producer operation names.

    Returns:
        list[LogicalNode] | LogicalGraph | None: Requested `return_elements` list, the target
            `LogicalGraph` when `graph_def` is provided without `return_elements`, or `None`.

    Raises:
        ValueError: If a requested `return_elements` entry is not found in the imported graph
            or an imported node fails validation (`op_dict` mismatch).
    """
    del producer_op_list
    if graph_def is None:
        return None

    prefix = f"{name}/" if name else ""
    target_graph = (
        _tracer.active_graph
        if _tracer.active_graph is not None
        else LogicalGraph(name=name or "imported_graph")
    )

    nodes_to_import: list[LogicalNode] = []
    edges_to_import: list[LogicalEdge] = []
    if isinstance(graph_def, LogicalGraph):
        nodes_to_import = list(graph_def.nodes.values())
        edges_to_import = list(graph_def.edges)
    elif isinstance(graph_def, Mapping):
        raw_nodes = graph_def.get("nodes")
        if isinstance(raw_nodes, Sequence):
            nodes_to_import = [item for item in raw_nodes if isinstance(item, LogicalNode)]
    elif isinstance(graph_def, Sequence) and not isinstance(graph_def, (str, bytes)):
        nodes_to_import = [item for item in graph_def if isinstance(item, LogicalNode)]

    imported_map: dict[str, LogicalNode] = {}
    for node in nodes_to_import:
        if op_dict is not None and node.op_type not in op_dict:
            raise ValueError(f"Unrecognized op_type '{node.op_type}' not in op_dict.")
        new_id = f"{prefix}{node.id}" if prefix else node.id
        if input_map is not None and node.id in input_map:
            mapped_val = input_map[node.id]
            if isinstance(mapped_val, LogicalNode):
                imported_node = LogicalNode(
                    id=new_id,
                    op_type=mapped_val.op_type,
                    attributes=dict(mapped_val.attributes),
                    shape_metadata=mapped_val.shape_metadata,
                )
            else:
                imported_node = LogicalNode(
                    id=new_id,
                    op_type="Identity",
                    attributes=dict(node.attributes),
                    shape_metadata=node.shape_metadata,
                )
        else:
            imported_node = LogicalNode(
                id=new_id,
                op_type=node.op_type,
                attributes=dict(node.attributes),
                shape_metadata=node.shape_metadata,
            )
        target_graph.nodes[new_id] = imported_node
        imported_map[node.id] = imported_node
        imported_map[new_id] = imported_node

    for edge in edges_to_import:
        src = f"{prefix}{edge.source}" if prefix else edge.source
        tgt = f"{prefix}{edge.target}" if prefix else edge.target
        target_graph.add_edge(src, tgt)

    if return_elements is not None:
        returned: list[LogicalNode] = []
        for elem in return_elements:
            clean_elem = elem.split(":")[0]
            if clean_elem in imported_map:
                returned.append(imported_map[clean_elem])
            else:
                raise ValueError(f"Requested return_element '{elem}' not found in graph_def.")
        return returned

    return target_graph


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
    x: Tensor | Variable | SparseTensor | RaggedTensor | TensorOperand = None,
) -> bool:
    """Check whether `x` is a recognized compiler or framework tensor type.

    Args:
        x: Candidate object to check.

    Returns:
        bool: True if `x` is a dense, variable, sparse, or ragged tensor.
    """
    return isinstance(
        x,
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
    indices: Tensor | TensorOperand = None,
    values: Tensor | TensorOperand = None,
    dense_shape: Tensor | TensorOperand = None,
    default_value: Tensor | TensorOperand = 0,
    name: str | None = None,
    value_rowids: Tensor | TensorOperand = None,
    nrows: Tensor | TensorOperand = None,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Fill empty rows in a 2D ragged or sparse tensor with a default value.

    Args:
        indices: 2-D `int64` tensor of shape `[N, 2]` with `(row, col)` coordinates.
        values: 1-D tensor of shape `[N]` containing values at `indices`.
        dense_shape: 1-D tensor/sequence `[num_rows, num_cols]`.
        default_value: Scalar value inserted into empty rows at column `0`.
        name: Optional operation name.
        value_rowids: Optional 1-D `int64` row IDs tensor (alias for `indices`).
        nrows: Optional scalar `int64` row count tensor (alias for `dense_shape`).

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]: Tuple of
            `(output_indices, output_values, empty_row_indicator, reverse_index_map)`.
    """
    del name, value_rowids, nrows
    py_indices = _to_py_data(indices) if indices is not None else []
    py_values = _to_py_data(values) if values is not None else []
    py_shape = _to_py_data(dense_shape) if dense_shape is not None else None
    py_default = _to_py_data(default_value)
    default_scalar: ScalarValue = (
        py_default if isinstance(py_default, (int, float, bool, complex, str, bytes)) else 0
    )

    idx_pairs: list[tuple[int, int, int]] = [
        (_to_int(row_entry[0]), _to_int(row_entry[1]), orig_i)
        for orig_i, row_entry in enumerate(py_indices if isinstance(py_indices, list) else [])
        if isinstance(row_entry, list) and len(row_entry) >= 2
    ]

    val_list: list[ScalarValue] = [
        v
        for v in (py_values if isinstance(py_values, list) else [])
        if isinstance(v, (int, float, bool, complex, str, bytes))
    ]

    num_rows = (
        _to_int(py_shape[0])
        if isinstance(py_shape, list) and len(py_shape) >= 1
        else (max((r for r, _, _ in idx_pairs), default=-1) + 1)
    )

    rows_bucket: dict[int, list[tuple[int, int]]] = {r: [] for r in range(num_rows)}
    for r, c, orig_i in idx_pairs:
        if r not in rows_bucket:
            rows_bucket[r] = []
        rows_bucket[r].append((c, orig_i))

    output_indices: list[list[int]] = []
    output_values: list[ScalarValue] = []
    empty_row_indicator: list[bool] = []
    reverse_index_map: list[int] = [0] * len(idx_pairs)

    for r in range(num_rows):
        entries = rows_bucket.get(r, [])
        if not entries:
            empty_row_indicator.append(True)
            output_indices.append([r, 0])
            output_values.append(default_scalar)
        else:
            empty_row_indicator.append(False)
            for c, orig_i in entries:
                out_pos = len(output_indices)
                output_indices.append([r, c])
                val_item = val_list[orig_i] if orig_i < len(val_list) else default_scalar
                output_values.append(val_item)
                reverse_index_map[orig_i] = out_pos

    return (
        Tensor(output_indices, dtype=msc_core.DType.Int64),
        Tensor(output_values),
        Tensor(empty_row_indicator, dtype=msc_core.DType.Bool),
        Tensor(reverse_index_map, dtype=msc_core.DType.Int64),
    )


def ragged_fill_empty_rows_grad(
    reverse_index_map: Tensor | TensorOperand = None,
    grad_values: Tensor | TensorOperand = None,
    name: str | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute backward gradients for `ragged_fill_empty_rows`.

    Args:
        reverse_index_map: 1-D `int64` tensor mapping input indices to `output_values` positions.
        grad_values: 1-D tensor of gradients with respect to `output_values`.
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor]: `(d_values, d_default_value)` gradient tensors.
    """
    del name
    py_map = _to_py_data(reverse_index_map) if reverse_index_map is not None else []
    py_grads = _to_py_data(grad_values) if grad_values is not None else []

    map_list: list[int] = (
        [_to_int(m) for m in py_map]
        if isinstance(py_map, list)
        else ([_to_int(py_map)] if isinstance(py_map, (int, float)) else [])
    )
    grad_list: list[float] = (
        [_to_float(g) for g in py_grads]
        if isinstance(py_grads, list)
        else ([_to_float(py_grads)] if isinstance(py_grads, (int, float)) else [])
    )

    visited: set[int] = set()
    d_values: list[float] = []
    for out_idx in map_list:
        visited.add(out_idx)
        d_values.append(grad_list[out_idx] if 0 <= out_idx < len(grad_list) else 0.0)

    d_default = sum(g for idx, g in enumerate(grad_list) if idx not in visited)
    return (Tensor(d_values), Tensor(d_default))


def size(
    input: Tensor | TensorOperand,
    out_type: DTypeLike = None,
    name: str | None = None,
) -> Tensor:
    """Return a 0-D scalar `Tensor` representing the number of elements in `input`.

    Args:
        input: Input tensor or shaped object.
        out_type: Optional output integer data type (defaults to `tf.int32`).
        name: Optional operation name.

    Returns:
        Tensor: 0-D `Tensor` of type `out_type` containing the total element count.
    """
    target_dt = to_dtype(out_type) if out_type is not None else DType.Int32
    count = int(math.prod(input.shape)) if isinstance(input, _ShapedProtocol) and input.shape else 0
    if _tracer.is_tracing and _tracer.active_graph is not None:
        op_id = f"{name or 'Size'}_{len(_tracer.active_graph.nodes)}"
        _tracer.active_graph.nodes[op_id] = LogicalNode(
            id=op_id,
            op_type="Size",
            attributes={"out_type": target_dt.name, "value": count},
            shape_metadata=(),
        )
    prev_tracing = _tracer.is_tracing
    prev_eager = _compiler_config.eager_mode
    _tracer.is_tracing = False
    _compiler_config.eager_mode = True
    try:
        return Tensor(count, dtype=target_dt)
    finally:
        _tracer.is_tracing = prev_tracing
        _compiler_config.eager_mode = prev_eager


def space_to_batch_nd(
    input: Tensor | TensorOperand = None,
    block_shape: Sequence[int] | Tensor | TensorOperand = None,
    paddings: Sequence[Sequence[int]] | Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor | None:
    """Divide N-D spatial dimensions of `input` into a grid of blocks interleaved into batch.

    Args:
        input: N-D Tensor with shape `[batch, *spatial_shape, *remaining_shape]`.
        block_shape: 1-D sequence of length `M` specifying block sizes `[B_1, ..., B_M]`.
        paddings: 2-D sequence of shape `[M, 2]` specifying `[pad_start, pad_end]` per spatial axis.
        name: Optional operation name.

    Returns:
        Tensor | None: Space-to-batch transformed `Tensor`, or `None` if `input is None`.
    """
    del name
    if input is None:
        return None
    t_in = _to_tensor(input)
    if block_shape is None:
        return _wrap_tensor(t_in)

    if _tracer.is_tracing or not _compiler_config.eager_mode:
        pads_arg = paddings if paddings is not None else [[0, 0] for _ in range(1)]
        return _wrap_tensor(
            as_operand(
                msc_ops.space_to_batch_nd(t_in, _to_tensor(block_shape), _to_tensor(pads_arg))
            )
        )

    py_blocks = _to_py_data(block_shape)
    b_shape: list[int] = (
        [max(1, _to_int(b)) for b in py_blocks] if isinstance(py_blocks, list) else [1]
    )
    m_dims = len(b_shape)

    py_pads = _to_py_data(paddings) if paddings is not None else [[0, 0] for _ in range(m_dims)]
    pad_pairs: list[tuple[int, int]] = [
        (max(0, _to_int(p[0])), max(0, _to_int(p[1])))
        for p in (py_pads if isinstance(py_pads, list) else [])
        if isinstance(p, list) and len(p) >= 2
    ]
    while len(pad_pairs) < m_dims:
        pad_pairs.append((0, 0))

    if len(t_in.shape) < 1 + m_dims:
        return _wrap_tensor(t_in)

    cur = t_in
    for axis_idx in range(m_dims):
        spatial_axis = axis_idx + 1
        pad_before, pad_after = pad_pairs[axis_idx]
        if pad_before > 0 or pad_after > 0:
            parts: list[msc_core.Tensor] = []
            if pad_before > 0:
                b_sh = list(cur.shape)
                b_sh[spatial_axis] = pad_before
                parts.append(msc_ops.zeros(tuple(b_sh), dtype=cur.dtype))
            parts.append(cur)
            if pad_after > 0:
                a_sh = list(cur.shape)
                a_sh[spatial_axis] = pad_after
                parts.append(msc_ops.zeros(tuple(a_sh), dtype=cur.dtype))
            cur = _k_ops.concatenate(parts, axis=spatial_axis)

    batch_size = cur.shape[0]
    padded_spatial = list(cur.shape[1 : 1 + m_dims])
    rem_shape = list(cur.shape[1 + m_dims :])

    split_shape: list[int] = [batch_size]
    for p_len, b_len in zip(padded_spatial, b_shape):
        split_shape.extend([p_len // b_len, b_len])
    split_shape.extend(rem_shape)

    reshaped = _k_ops.reshape(cur, tuple(split_shape))
    block_axes = [2 * i + 2 for i in range(m_dims)]
    grid_axes = [2 * i + 1 for i in range(m_dims)]
    rem_axes = list(range(1 + 2 * m_dims, len(split_shape)))
    perm = [*block_axes, 0, *grid_axes, *rem_axes]
    transposed = _k_ops.transpose(reshaped, perm)

    final_batch = batch_size * int(math.prod(b_shape))
    final_spatial = [p_len // b_len for p_len, b_len in zip(padded_spatial, b_shape)]
    final_shape = (final_batch, *final_spatial, *rem_shape)
    return _wrap_tensor(_k_ops.reshape(transposed, final_shape))


def space_to_batch(
    input: Tensor | TensorOperand = None,
    paddings: Sequence[Sequence[int]] | Tensor | TensorOperand = None,
    block_size: int = 2,
    name: str | None = None,
    block_shape: Sequence[int] | int | Tensor | TensorOperand = None,
) -> Tensor | None:
    """Zero-pad and rearrange 2-D blocks of spatial data into the batch dimension.

    Args:
        input: 4-D Tensor with shape `[batch, height, width, depth]`.
        paddings: 2-D sequence of shape `[2, 2]` specifying spatial padding.
        block_size: Integer block size (`>= 1`).
        name: Optional operation name.
        block_shape: Optional 1-D block shape sequence or integer overriding `block_size`.

    Returns:
        Tensor | None: Space-to-batch transformed `Tensor`, or `None` if `input is None`.
    """
    del block_shape
    if input is None:
        return None
    if paddings is None:
        return _wrap_tensor(input)
    return space_to_batch_nd(
        input,
        block_shape=[block_size, block_size],
        paddings=paddings,
        name=name,
    )


string: DType = DType.String
