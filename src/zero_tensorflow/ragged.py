"""TensorFlow `ragged` tensor module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_tensorflow._typing import (
    ArrayLikeProtocol,
    CallableOperandProtocol,
    NumpyConvertibleProtocol,
    TensorOperand,
    as_operand,
)
from zero_tensorflow.core_tensor import Tensor, to_dtype
from zero_tensorflow.sparse import SparseTensor

from ml_switcheroo_compiler.core.dtype import DType

__all__ = [
    "boolean_mask",
    "constant",
    "map_flat_values",
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
    ) -> None:
        """Initialize a `RaggedTensor` from nested values and optional `row_splits`.

        Args:
            values: Nested sequence, `RaggedTensor`, or flat values.
            row_splits: Optional explicit cumulative row split offsets.
            ragged_rank: Number of ragged dimensions.
        """
        flat: list[TensorOperand] = []
        _flatten(values, flat)
        self.flat_values: list[TensorOperand] = flat
        self._has_nested_rows: bool = (
            row_splits is not None
            or isinstance(values, RaggedTensor)
            or (
                isinstance(values, Sequence)
                and not isinstance(values, (str, bytes))
                and bool(values)
                and isinstance(values[0], Sequence)
                and not isinstance(values[0], (str, bytes))
            )
        )
        self._row_splits: tuple[int, ...] = (
            tuple(int(s) for s in row_splits)
            if row_splits is not None
            else _compute_row_splits(values)
        )
        self._ragged_rank: int = ragged_rank

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
        if self.flat_values and isinstance(self.flat_values[0], int):
            return to_dtype(int)
        return to_dtype(float)

    @classmethod
    def from_row_splits(
        cls,
        values: Sequence[TensorOperand],
        row_splits: Sequence[int],
        name: str | None = None,
        validate: bool = True,
    ) -> RaggedTensor:
        """Construct a `RaggedTensor` from `values` and `row_splits`.

        Args:
            values: Flat values sequence.
            row_splits: Cumulative row-split offsets starting with `0`.
            name: Optional operation name.
            validate: If `True`, validate `row_splits` monotonicity.

        Returns:
            RaggedTensor: Constructed ragged tensor.

        Raises:
            ValueError: If `validate` is `True` and `row_splits` is invalid.
        """
        del name
        if validate and (not row_splits or row_splits[0] != 0 or row_splits[-1] != len(values)):
            raise ValueError("Invalid row_splits for values.")
        return cls(list(values), row_splits=row_splits)

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
        """Reconstruct a Python list using `row_splits` and `flat_values`.

        Returns:
            list[TensorOperand]: Nested or flat Python list.
        """
        if not self._has_nested_rows:
            return list(self.flat_values)
        rows: list[TensorOperand] = []
        for i in range(len(self._row_splits) - 1):
            start = self._row_splits[i]
            end = self._row_splits[i + 1]
            rows.append(list(self.flat_values[start:end]))
        return rows

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
        del name, shape
        if not self.flat_values:
            return Tensor([])
        rows: list[list[TensorOperand]] = []
        for i in range(len(self._row_splits) - 1):
            start = self._row_splits[i]
            end = self._row_splits[i + 1]
            rows.append(list(self.flat_values[start:end]))
        max_cols = max((len(r) for r in rows), default=0)
        padded = [r + [default_value] * (max_cols - len(r)) for r in rows]
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
        for r_idx in range(nrows):
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
