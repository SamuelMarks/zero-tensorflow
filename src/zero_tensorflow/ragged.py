"""TensorFlow `ragged` tensor module."""

from __future__ import annotations

import builtins
import itertools
from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, to_dtype
from zero_tensorflow.dtypes import DType
from zero_tensorflow.sparse import SparseTensor

__all__ = [
    "boolean_mask",
    "constant",
    "cross",
    "cross_hashed",
    "map_flat_values",
    "range",
    "row_splits_to_segment_ids",
    "segment_ids_to_row_splits",
    "stack",
    "stack_dynamic_partitions",
]


def _flatten(val: TensorOperand, out: list[TensorOperand]) -> None:
    """Recursively flatten nested sequences into `out`.

    Args:
        val: Input scalar or nested sequence.
        out: Output accumulator list.
    """
    if isinstance(val, RaggedTensor):
        out.extend(val.flat_values)
    elif isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        for item in val:
            _flatten(item, out)
    elif val is not None:
        out.append(val)


def _compute_row_splits(val: TensorOperand) -> tuple[int, ...]:
    """Compute top-level `row_splits` for a nested sequence `val`.

    Args:
        val: Input nested sequence or scalar.

    Returns:
        tuple[int, ...]: Cumulative row split offsets starting at `0`.
    """
    if isinstance(val, RaggedTensor):
        return val.row_splits
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        splits: list[int] = [0]
        acc = 0
        for row in val:
            if isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                acc += len(row)
            else:
                acc += 1
            splits.append(acc)
        return tuple(splits)
    return (0, 1) if val is not None else (0,)


class RaggedTensor:
    """Represents a tensor with ragged (variable-length) dimensions."""

    def __init__(
        self,
        values: TensorOperand = None,
        row_splits: Sequence[int] | None = None,
        ragged_rank: int = 1,
        nested_row_splits: Sequence[Sequence[int]] | None = None,
    ) -> None:
        """Initialize a `RaggedTensor` from nested values and optional `row_splits`.

        Args:
            values: Nested sequence, `RaggedTensor`, or flat values.
            row_splits: Optional explicit cumulative row split offsets.
            ragged_rank: Number of ragged dimensions.
            nested_row_splits: Optional sequence of row_splits for multi-level raggedness.
        """
        flat: list[TensorOperand] = []
        _flatten(values, flat)
        self.flat_values: list[TensorOperand] = flat
        self._has_nested_rows: bool = (
            row_splits is not None
            or nested_row_splits is not None
            or isinstance(values, RaggedTensor)
            or (
                isinstance(values, Sequence)
                and not isinstance(values, (str, bytes))
                and bool(values)
                and isinstance(values[0], Sequence)
                and not isinstance(values[0], (str, bytes))
            )
        )
        if nested_row_splits is not None and len(nested_row_splits) > 0:
            self._nested_row_splits: tuple[tuple[int, ...], ...] = tuple(
                tuple(int(s) for s in sp) for sp in nested_row_splits
            )
            self._row_splits: tuple[int, ...] = self._nested_row_splits[0]
            self._ragged_rank: int = len(self._nested_row_splits)
        else:
            self._row_splits = (
                tuple(int(s) for s in row_splits)
                if row_splits is not None
                else _compute_row_splits(values)
            )
            self._nested_row_splits = (self._row_splits,)
            self._ragged_rank = ragged_rank

    @property
    def values(self) -> list[TensorOperand]:
        """Return the flattened values of this ragged tensor.

        Returns:
            list[TensorOperand]: Flattened values list.
        """
        return self.flat_values

    @property
    def row_splits(self) -> tuple[int, ...]:
        """Return the row-split offset tuple of this ragged tensor.

        Returns:
            tuple[int, ...]: Cumulative row splits starting with `0`.
        """
        return self._row_splits

    @property
    def nested_row_splits(self) -> tuple[tuple[int, ...], ...]:
        """Return the tuple of row-split tuples for all ragged dimensions.

        Returns:
            tuple[tuple[int, ...], ...]: Tuple of row-split tuples.
        """
        return self._nested_row_splits

    @property
    def ragged_rank(self) -> int:
        """Return the number of ragged dimensions in this `RaggedTensor`.

        Returns:
            int: Ragged rank.
        """
        return self._ragged_rank

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return the static shape tuple `(nrows, None)` of this `RaggedTensor`.

        Returns:
            tuple[int | None, ...]: Shape tuple with `None` for ragged dimensions.
        """
        nrows = max(len(self._row_splits) - 1, 0)
        return (nrows, *([None] * self._ragged_rank))

    @property
    def dtype(self) -> DType:
        """Return the `DType` of the `RaggedTensor`'s flat values.

        Returns:
            DType: Inferred element data type.
        """
        if self.flat_values:
            first = self.flat_values[0]
            if isinstance(first, (str, bytes)):
                return DType.String
            return (
                to_dtype(int)
                if isinstance(first, int) and not isinstance(first, bool)
                else to_dtype(float)
            )
        return to_dtype(float)

    def __getitem__(
        self,
        key: int
        | slice[int | None, int | None, int | None]
        | tuple[int | slice[int | None, int | None, int | None], ...],
    ) -> RaggedTensor | Tensor:
        """Index or slice rows (and inner dimensions) of this `RaggedTensor`.

        Args:
            key: Integer row index, slice, or multi-axis index tuple.

        Returns:
            RaggedTensor | Tensor: Sliced `RaggedTensor` or row `Tensor`.
        """
        rows = self.to_list()
        if isinstance(key, int):
            return Tensor(rows[key])
        if isinstance(key, slice):
            return RaggedTensor(rows[key], ragged_rank=self._ragged_rank)
        r_key, c_key = key[0], key[1]
        if isinstance(r_key, int):
            row_val = rows[r_key]
            if isinstance(row_val, list):
                return Tensor(row_val[c_key])
            return Tensor(row_val)
        sliced_rows = [r[c_key] if isinstance(r, list) else r for r in rows[r_key]]
        return RaggedTensor(sliced_rows, ragged_rank=self._ragged_rank)

    @classmethod
    def from_row_splits(
        cls,
        values: Sequence[TensorOperand] | RaggedTensor,
        row_splits: Sequence[int],
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `values` and `row_splits`.

        Args:
            values: Flat values sequence or inner `RaggedTensor`.
            row_splits: Cumulative row-split offsets starting with `0`.
            name: Optional operation name.
            validate: If `True`, validate `row_splits` monotonicity.

        Returns:
            RaggedTensor: Constructed ragged tensor.

        Raises:
            ValueError: If `validate` is `True` and `row_splits` is invalid.
        """
        del name
        if isinstance(values, RaggedTensor):
            return cls.from_nested_row_splits(
                values.flat_values,
                (tuple(int(s) for s in row_splits), *values.nested_row_splits),
                validate=validate,
            )
        if validate and (not row_splits or row_splits[0] != 0 or row_splits[-1] != len(values)):
            raise ValueError("Invalid row_splits for values.")
        return cls(list(values), row_splits=row_splits)

    @classmethod
    def from_nested_row_splits(
        cls,
        flat_values: Sequence[TensorOperand],
        nested_row_splits: Sequence[Sequence[int]],
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `flat_values` and `nested_row_splits`.

        Args:
            flat_values: Innermost flat values sequence.
            nested_row_splits: Sequence of row-split sequences from outermost to innermost.
            name: Optional operation name.
            validate: If `True`, validate split boundaries.

        Returns:
            RaggedTensor: Constructed multi-level `RaggedTensor`.

        Raises:
            ValueError: If `validate=True` and `nested_row_splits` are inconsistent.
        """
        del name
        splits_tuple = tuple(tuple(int(s) for s in sp) for sp in nested_row_splits)
        if validate and (
            not splits_tuple
            or not splits_tuple[-1]
            or splits_tuple[-1][0] != 0
            or splits_tuple[-1][-1] != len(flat_values)
        ):
            raise ValueError("Invalid nested_row_splits for flat_values.")
        return cls(
            list(flat_values),
            ragged_rank=len(splits_tuple),
            nested_row_splits=splits_tuple,
        )

    @classmethod
    def from_row_lengths(
        cls,
        values: Sequence[TensorOperand],
        row_lengths: Sequence[int],
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `values` and `row_lengths`.

        Args:
            values: Flat values sequence.
            row_lengths: Sequence of non-negative row lengths.
            name: Optional operation name.
            validate: If `True`, validate that sum of `row_lengths` equals `len(values)`.

        Returns:
            RaggedTensor: Constructed ragged tensor.
        """
        splits: list[int] = [0]
        for length in row_lengths:
            splits.append(splits[-1] + int(length))
        return cls.from_row_splits(values, splits, name=name, validate=validate)

    @classmethod
    def from_nested_row_lengths(
        cls,
        flat_values: Sequence[TensorOperand],
        nested_row_lengths: Sequence[Sequence[int]],
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `flat_values` and `nested_row_lengths`.

        Args:
            flat_values: Innermost flat values sequence.
            nested_row_lengths: Sequence of row-length sequences from outermost to innermost.
            name: Optional operation name.
            validate: If `True`, validate row length sums.

        Returns:
            RaggedTensor: Constructed multi-level `RaggedTensor`.
        """
        nested_splits: list[list[int]] = []
        for lengths in nested_row_lengths:
            sp = [0]
            for ln in lengths:
                sp.append(sp[-1] + int(ln))
            nested_splits.append(sp)
        return cls.from_nested_row_splits(
            flat_values,
            nested_splits,
            name=name,
            validate=validate,
        )

    @classmethod
    def from_value_rowids(
        cls,
        values: Sequence[TensorOperand],
        value_rowids: Sequence[int],
        nrows: int | None = None,
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `values` and `value_rowids`.

        Args:
            values: Flat values sequence.
            value_rowids: Row index for each element in `values`.
            nrows: Optional total number of rows.
            name: Optional operation name.
            validate: If `True`, validate `value_rowids`.

        Returns:
            RaggedTensor: Constructed ragged tensor.
        """
        total_rows = nrows if nrows is not None else (max(value_rowids) + 1 if value_rowids else 0)
        counts = [0] * total_rows
        for rid in value_rowids:
            counts[rid] += 1
        return cls.from_row_lengths(values, counts, name=name, validate=validate)

    @classmethod
    def from_nested_value_rowids(
        cls,
        flat_values: Sequence[TensorOperand],
        nested_value_rowids: Sequence[Sequence[int]],
        nested_nrows: Sequence[int | None] | None = None,
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `flat_values` and `nested_value_rowids`.

        Args:
            flat_values: Innermost flat values sequence.
            nested_value_rowids: Sequence of value-rowid sequences.
            nested_nrows: Optional sequence of row counts per level.
            name: Optional operation name.
            validate: If `True`, validate value rowids.

        Returns:
            RaggedTensor: Constructed multi-level `RaggedTensor`.
        """
        nrows_seq = (
            list(nested_nrows) if nested_nrows is not None else [None] * len(nested_value_rowids)
        )
        nested_lengths: list[list[int]] = []
        for rowids, nr in zip(nested_value_rowids, nrows_seq):
            total_r = nr if nr is not None else (max(rowids) + 1 if rowids else 0)
            cnts = [0] * total_r
            for rid in rowids:
                cnts[rid] += 1
            nested_lengths.append(cnts)
        return cls.from_nested_row_lengths(
            flat_values,
            nested_lengths,
            name=name,
            validate=validate,
        )

    @classmethod
    def from_tensor(
        cls,
        tensor: Tensor | TensorOperand,
        lengths: Sequence[int] | None = None,
        padding: TensorOperand = None,
        ragged_rank: int = 1,
        name: str | None = None,
    ) -> RaggedTensor:
        """Convert a dense `Tensor` or nested sequence into a `RaggedTensor`.

        Args:
            tensor: Dense `Tensor` or nested sequence.
            lengths: Optional row lengths per row.
            padding: Optional padding value to strip.
            ragged_rank: Target ragged rank.
            name: Optional operation name.

        Returns:
            RaggedTensor: Converted ragged tensor.
        """
        del lengths, padding, name
        raw: TensorOperand = (
            as_operand(tensor.numpy()) if isinstance(tensor, NumpyConvertibleProtocol) else tensor
        )
        py_val: TensorOperand = raw.tolist() if isinstance(raw, ArrayLikeProtocol) else raw
        return cls(py_val, ragged_rank=ragged_rank)

    def to_list(self) -> list[TensorOperand]:
        """Reconstruct a Python list using `nested_row_splits` and `flat_values`.

        Returns:
            list[TensorOperand]: Nested or flat Python list.
        """
        if not self._has_nested_rows:
            return list(self.flat_values)
        current: list[TensorOperand] = list(self.flat_values)
        for splits in reversed(self._nested_row_splits):
            next_level: list[TensorOperand] = []
            for i in builtins.range(len(splits) - 1):
                next_level.append(list(current[splits[i] : splits[i + 1]]))
            current = next_level
        return current

    def to_tensor(
        self,
        default_value: TensorOperand = 0,
        name: str | None = None,
        shape: Sequence[int | None] | None = None,
    ) -> Tensor:
        """Convert this `RaggedTensor` into a padded dense `Tensor`.

        Args:
            default_value: Padding value for short rows.
            name: Optional operation name.
            shape: Optional target dense shape.

        Returns:
            Tensor: Padded dense `Tensor`.
        """
        del name
        if not self.flat_values:
            return Tensor([])
        rows: list[list[TensorOperand]] = []
        for i in builtins.range(len(self._row_splits) - 1):
            start = self._row_splits[i]
            end = self._row_splits[i + 1]
            rows.append(list(self.flat_values[start:end]))
        max_cols = (
            int(shape[1])
            if shape is not None and len(shape) >= 2 and shape[1] is not None
            else max((len(r) for r in rows), default=0)
        )
        target_rows = (
            int(shape[0])
            if shape is not None and len(shape) >= 1 and shape[0] is not None
            else len(rows)
        )
        padded: list[list[TensorOperand]] = []
        for r_idx in builtins.range(target_rows):
            row = rows[r_idx][:max_cols] if r_idx < len(rows) else []
            padded.append(row + [default_value] * (max_cols - len(row)))
        return Tensor(padded)

    def to_sparse(self, name: str | None = None) -> SparseTensor:
        """Convert this `RaggedTensor` into a `SparseTensor`.

        Args:
            name: Optional operation name.

        Returns:
            SparseTensor: Sparse representation of the ragged elements.
        """
        del name
        indices: list[list[int]] = []
        vals: list[TensorOperand] = []
        nrows = max(len(self._row_splits) - 1, 0)
        max_cols = 0
        for r_idx in builtins.range(nrows):
            start = self._row_splits[r_idx]
            end = self._row_splits[r_idx + 1]
            row = self.flat_values[start:end]
            max_cols = max(max_cols, len(row))
            for c_idx, val in enumerate(row):
                indices.append([r_idx, c_idx])
                vals.append(val)
        return SparseTensor(
            indices=indices,
            values=Tensor(vals),
            dense_shape=[nrows, max_cols],
        )


def constant(
    pylist: TensorOperand,
    dtype: TensorOperand = None,
    ragged_rank: int | None = None,
    inner_shape: Sequence[int] | None = None,
    name: str | None = None,
    row_splits_dtype: TensorOperand = None,
) -> RaggedTensor:
    """Construct a constant `RaggedTensor` from a nested Python list.

    Args:
        pylist: Nested Python list, tuple, or scalar.
        dtype: Optional element dtype.
        ragged_rank: Optional number of ragged dimensions.
        inner_shape: Optional shape of inner dense dimensions.
        name: Optional operation name.
        row_splits_dtype: Optional dtype for row splits.

    Returns:
        RaggedTensor: Constructed ragged tensor.
    """
    del dtype, inner_shape, name, row_splits_dtype
    return RaggedTensor(pylist, ragged_rank=ragged_rank if ragged_rank is not None else 1)


def boolean_mask(
    data: RaggedTensor | Sequence[TensorOperand] | TensorOperand,
    mask: RaggedTensor | Sequence[TensorOperand] | TensorOperand,
    name: str | None = None,
) -> RaggedTensor:
    """Apply a boolean `mask` to `data` while preserving ragged row structure.

    Args:
        data: Input `RaggedTensor`, sequence, or scalar.
        mask: Boolean mask `RaggedTensor`, sequence, or scalar matching `data`.
        name: Optional operation name.

    Returns:
        RaggedTensor: Masked `RaggedTensor`.
    """
    del name
    if isinstance(data, RaggedTensor):
        d_list = data.to_list()
    elif isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        d_list = list(data)
    else:
        return RaggedTensor([data] if mask else [])

    if isinstance(mask, RaggedTensor):
        m_list = mask.to_list()
    elif isinstance(mask, Sequence) and not isinstance(mask, (str, bytes)):
        m_list = list(mask)
    else:
        m_list = [mask] * len(d_list)

    if (
        d_list
        and isinstance(d_list[0], Sequence)
        and not isinstance(d_list[0], (str, bytes))
        and m_list
        and isinstance(m_list[0], Sequence)
        and not isinstance(m_list[0], (str, bytes))
    ):
        nested_out = [
            [dv for dv, mv in zip(d_row, m_row) if mv]
            for d_row, m_row in zip(d_list, m_list)
            if isinstance(d_row, Sequence) and isinstance(m_row, Sequence)
        ]
        return RaggedTensor(nested_out)

    filtered = [d for d, m in zip(d_list, m_list) if m]
    return RaggedTensor(filtered)


def map_flat_values(
    op: CallableOperandProtocol,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> RaggedTensor:
    """Apply `op` to the `flat_values` of one or more `RaggedTensor`s while preserving `row_splits`.

    Args:
        op: Callable operation to apply to flat values.
        *args: Positional arguments (any `RaggedTensor` is unwrapped to `flat_values`).
        **kwargs: Keyword arguments passed to `op`.

    Returns:
        RaggedTensor: Result wrapped in a `RaggedTensor` with preserved `row_splits`.
    """
    preserved_splits = next((a.row_splits for a in args if isinstance(a, RaggedTensor)), None)
    unwrapped_args: list[TensorOperand] = [
        a.flat_values if isinstance(a, RaggedTensor) else a for a in args
    ]
    if unwrapped_args:
        return RaggedTensor(op(*unwrapped_args, **kwargs), row_splits=preserved_splits)
    return RaggedTensor(op(None, **kwargs), row_splits=preserved_splits)


def _to_ragged_rows(
    x: RaggedTensor | Sequence[TensorOperand] | TensorOperand,
) -> list[list[TensorOperand]]:
    """Convert `x` into a 2-D list of rows.

    Args:
        x: Input `RaggedTensor` or nested sequence.

    Returns:
        list[list[TensorOperand]]: 2-D list of rows.
    """
    raw = (
        x.to_list()
        if isinstance(x, RaggedTensor)
        else (list(x) if isinstance(x, Sequence) and not isinstance(x, (str, bytes)) else [[x]])
    )
    return [
        list(r) if isinstance(r, Sequence) and not isinstance(r, (str, bytes)) else [r] for r in raw
    ]


def _concat(
    values: Sequence[RaggedTensor | Sequence[TensorOperand] | TensorOperand] = (),
    axis: int = 0,
    name: str | None = None,
) -> RaggedTensor:
    """Concatenate `RaggedTensor`s along `axis`.

    Args:
        values: Sequence of `RaggedTensor`s or nested lists.
        axis: Dimension along which to concatenate (`0` for rows, `1`/`-1` for columns).
        name: Optional operation name.

    Returns:
        RaggedTensor: Concatenated `RaggedTensor`.
    """
    del name
    rt_rows = [_to_ragged_rows(v) for v in values]
    if not rt_rows:
        return RaggedTensor([])
    if axis == 0:
        out_rows: list[list[TensorOperand]] = []
        for r_list in rt_rows:
            out_rows.extend(r_list)
        return RaggedTensor(out_rows)
    nrows = max((len(r) for r in rt_rows), default=0)
    merged_rows: list[list[TensorOperand]] = []
    for i in builtins.range(nrows):
        combined: list[TensorOperand] = []
        for r_list in rt_rows:
            combined.extend(r_list[i] if i < len(r_list) else [])
        merged_rows.append(combined)
    return RaggedTensor(merged_rows)


def stack(
    values: Sequence[RaggedTensor | Sequence[TensorOperand] | TensorOperand] = (),
    axis: int = 0,
    name: str | None = None,
) -> RaggedTensor:
    """Stack a list of `RaggedTensor`s or tensors along `axis`.

    Args:
        values: Sequence of `RaggedTensor`s or sequences.
        axis: Axis along which to stack.
        name: Optional operation name.

    Returns:
        RaggedTensor: Stacked `RaggedTensor`.
    """
    return _concat(values=values, axis=axis, name=name)


def stack_dynamic_partitions(
    data: Sequence[TensorOperand] | Tensor | TensorOperand = (),
    partitions: Sequence[int] | Tensor | TensorOperand = (),
    num_partitions: int = 1,
    name: str | None = None,
) -> RaggedTensor:
    """Stack dynamic partitions of `data` into a `RaggedTensor` with `num_partitions` rows.

    Args:
        data: Values to partition.
        partitions: Partition index for each element of `data`.
        num_partitions: Total number of output rows.
        name: Optional operation name.

    Returns:
        RaggedTensor: Partitioned `RaggedTensor`.
    """
    del name
    d_seq: list[TensorOperand] = []
    _flatten(data.tolist() if isinstance(data, Tensor) else data, d_seq)
    p_seq: list[TensorOperand] = []
    _flatten(partitions.tolist() if isinstance(partitions, Tensor) else partitions, p_seq)
    n_parts = max(1, int(num_partitions))
    buckets: list[list[TensorOperand]] = [[] for _ in builtins.range(n_parts)]
    for val, part_idx in zip(d_seq, p_seq):
        idx = int(part_idx) if isinstance(part_idx, (int, float)) else 0
        buckets[idx % n_parts].append(val)
    return RaggedTensor(buckets)


def cross(
    inputs: Sequence[RaggedTensor | Sequence[TensorOperand] | TensorOperand] = (),
    name: str | None = None,
) -> RaggedTensor:
    """Generate feature cross from a list of ragged and dense tensors.

    Args:
        inputs: Sequence of 2-D `RaggedTensor`s or nested lists.
        name: Optional operation name.

    Returns:
        RaggedTensor: 2-D `RaggedTensor` of crossed string features (`'a_X_b'`).
    """
    del name
    if not inputs:
        return RaggedTensor([])
    input_rows = [_to_ragged_rows(inp) for inp in inputs]
    nrows = min((len(r) for r in input_rows), default=0)
    crossed_rows: list[list[str]] = []
    for r_idx in builtins.range(nrows):
        row_lists = [[str(v) for v in inp_r[r_idx]] for inp_r in input_rows]
        crossed_rows.append(["_X_".join(prod) for prod in itertools.product(*row_lists)])
    return RaggedTensor(crossed_rows)


def cross_hashed(
    inputs: Sequence[RaggedTensor | Sequence[TensorOperand] | TensorOperand] = (),
    num_buckets: int = 0,
    hash_key: int | None = None,
    name: str | None = None,
) -> RaggedTensor:
    """Generate hashed feature cross from a list of ragged and dense tensors.

    Args:
        inputs: Sequence of 2-D `RaggedTensor`s or nested lists.
        num_buckets: Non-negative number of hash buckets (`0` uses `2**63 - 1`).
        hash_key: Optional custom hash key seed.
        name: Optional operation name.

    Returns:
        RaggedTensor: 2-D `RaggedTensor` of hashed `int64` bucket IDs.
    """
    del hash_key
    crossed = cross(inputs=inputs, name=name)
    mod = max(1, int(num_buckets)) if num_buckets > 0 else (1 << 31)
    hashed_rows: list[list[int]] = [
        [abs(hash(str(item))) % mod for item in row] for row in _to_ragged_rows(crossed)
    ]
    return RaggedTensor(hashed_rows)


def range(
    starts: int | Sequence[int] | Tensor = 0,
    limits: int | Sequence[int] | Tensor | None = None,
    deltas: int | Sequence[int] | Tensor = 1,
    dtype: TensorOperand = None,
    name: str | None = None,
    row_splits_dtype: TensorOperand = None,
) -> RaggedTensor:
    """Return a `RaggedTensor` containing the specified sequences of numbers.

    Args:
        starts: Scalar or 1-D sequence of start values (or `limits` when `limits is None`).
        limits: Optional scalar or 1-D sequence of exclusive upper limits.
        deltas: Scalar or 1-D sequence of step increments.
        dtype: Optional element dtype.
        name: Optional operation name.
        row_splits_dtype: Optional dtype for `row_splits`.

    Returns:
        RaggedTensor: 2-D `RaggedTensor` where row `i` is `range(starts[i], limits[i], deltas[i])`.
    """
    del dtype, name, row_splits_dtype
    if limits is None:
        eff_limits = starts
        eff_starts: int | Sequence[int] | Tensor = 0
    else:
        eff_starts = starts
        eff_limits = limits

    def _to_ints(x: int | Sequence[int] | Tensor) -> list[int]:
        """Convert scalar, sequence, or `Tensor` to `list[int]`.

        Args:
            x: Input integer or sequence.

        Returns:
            list[int]: List of integers.
        """
        if isinstance(x, Tensor):
            raw_t = x.tolist()
            if isinstance(raw_t, builtins.list):
                return [int(v) if isinstance(v, (int, float)) else 0 for v in raw_t]
            return [int(raw_t) if isinstance(raw_t, (int, float)) else 0]
        if isinstance(x, Sequence):
            return [int(v) for v in x]
        return [int(x)]

    s_list = _to_ints(eff_starts)
    l_list = _to_ints(eff_limits)
    d_list = _to_ints(deltas)
    nrows = max(len(s_list), len(l_list), len(d_list))
    rows: list[list[int]] = []
    for i in builtins.range(nrows):
        s_val = s_list[i if i < len(s_list) else 0]
        l_val = l_list[i if i < len(l_list) else 0]
        d_val = d_list[i if i < len(d_list) else 0]
        step = d_val if d_val != 0 else 1
        rows.append(list(builtins.range(s_val, l_val, step)))
    return RaggedTensor(rows)


def row_splits_to_segment_ids(
    splits: Sequence[int] | Tensor = (0,),
    name: str | None = None,
    out_type: TensorOperand = None,
) -> Tensor:
    """Generate the segmentation corresponding to a `RaggedTensor` `row_splits`.

    Args:
        splits: Sorted 1-D integer sequence or `Tensor` starting with `0`.
        name: Optional operation name.
        out_type: Optional integer output dtype.

    Returns:
        Tensor: 1-D segment IDs tensor.
    """
    del name
    raw = splits.tolist() if isinstance(splits, Tensor) else splits
    sp_list = (
        [int(v) if isinstance(v, (int, float)) else 0 for v in raw]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
        else [0]
    )
    seg_ids: list[int] = []
    for row_id in builtins.range(len(sp_list) - 1):
        count = max(0, sp_list[row_id + 1] - sp_list[row_id])
        seg_ids.extend([row_id] * count)
    dt = to_dtype(out_type) if out_type is not None else DType.Int64
    return Tensor(seg_ids, dtype=dt)


def segment_ids_to_row_splits(
    segment_ids: Sequence[int] | Tensor = (),
    num_segments: int | None = None,
    out_type: TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Generate the `RaggedTensor` `row_splits` corresponding to a segmentation.

    Args:
        segment_ids: 1-D non-decreasing integer sequence or `Tensor` of segment IDs.
        num_segments: Optional total number of segments.
        out_type: Optional integer output dtype.
        name: Optional operation name.

    Returns:
        Tensor: 1-D cumulative `row_splits` tensor of length `num_segments + 1`.
    """
    del name
    raw = segment_ids.tolist() if isinstance(segment_ids, Tensor) else segment_ids
    seg_list = (
        [int(v) if isinstance(v, (int, float)) else 0 for v in raw]
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes))
        else []
    )
    nrows = (
        int(num_segments) if num_segments is not None else (max(seg_list) + 1 if seg_list else 0)
    )
    counts = [0] * nrows
    for sid in seg_list:
        counts[sid % max(1, nrows)] += 1
    splits: list[int] = [0]
    for c in counts:
        splits.append(splits[-1] + c)
    dt = to_dtype(out_type) if out_type is not None else DType.Int64
    return Tensor(splits, dtype=dt)
