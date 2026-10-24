"""TensorFlow `ragged` tensor module."""

from __future__ import annotations

from collections.abc import Sequence

from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand

__all__ = [
    "boolean_mask",
    "map_flat_values",
]


def _flatten_nested(val: TensorOperand | RaggedTensor) -> list[TensorOperand]:
    """Flatten a nested sequence or `RaggedTensor` into a 1-D list of flat values.

    Args:
        val: Nested sequence, scalar, or `RaggedTensor`.

    Returns:
        list[TensorOperand]: Flattened leaf values.
    """
    if isinstance(val, RaggedTensor):
        return list(val.flat_values)
    if isinstance(val, Sequence) and not isinstance(val, (str, bytes)):
        out: list[TensorOperand] = []
        for item in val:
            out.extend(_flatten_nested(item))
        return out
    if val is None:
        return []
    return [val]


class RaggedTensor:
    """Represents a tensor with ragged (variable-length) dimensions."""

    def __init__(
        self,
        *args: TensorOperand | RaggedTensor,
        **kwargs: TensorOperand | RaggedTensor,
    ) -> None:
        """Initialize a `RaggedTensor`.

        Args:
            *args: Positional ragged values and row-partition arguments.
            **kwargs: Keyword ragged construction arguments.
        """
        self._args = args
        self._kwargs = kwargs
        first = args[0] if args else kwargs.get("values")
        self.flat_values: list[TensorOperand] = _flatten_nested(first)

    def to_list(self) -> list[TensorOperand]:
        """Convert the `RaggedTensor` values to a Python list.

        Returns:
            list[TensorOperand]: List representation of the ragged tensor.
        """
        if (
            self._args
            and isinstance(self._args[0], Sequence)
            and not isinstance(self._args[0], (str, bytes))
        ):
            return list(self._args[0])
        return list(self.flat_values)


def boolean_mask(
    data: TensorOperand | RaggedTensor,
    mask: TensorOperand | RaggedTensor,
    name: str | None = None,
) -> RaggedTensor:
    """Apply a boolean mask to `data` without flattening the mask dimensions.

    Args:
        data: A potentially ragged tensor to mask.
        mask: A potentially ragged boolean mask tensor.
        name: Optional name prefix for the returned tensors.

    Returns:
        RaggedTensor: Masked ragged tensor.
    """
    del name
    if (
        isinstance(data, Sequence)
        and not isinstance(data, (str, bytes))
        and isinstance(mask, Sequence)
        and not isinstance(mask, (str, bytes))
    ):
        filtered: list[TensorOperand] = []
        for d_item, m_item in zip(data, mask):
            if (
                isinstance(d_item, Sequence)
                and not isinstance(d_item, (str, bytes))
                and isinstance(m_item, Sequence)
                and not isinstance(m_item, (str, bytes))
            ):
                filtered.append([v for v, keep in zip(d_item, m_item) if bool(keep)])
            elif bool(m_item):
                filtered.append(d_item)
        return RaggedTensor(filtered)
    return RaggedTensor(data, mask)


def map_flat_values(
    op: CallableOperandProtocol,
    *args: TensorOperand | RaggedTensor,
    **kwargs: TensorOperand,
) -> RaggedTensor:
    """Apply `op` to the `flat_values` of one or more `RaggedTensor`s.

    Args:
        op: Operation callable applied to the flat values.
        *args: Positional arguments forwarded to `op`.
        **kwargs: Keyword arguments forwarded to `op`.

    Returns:
        RaggedTensor: RaggedTensor wrapping the transformed flat values.
    """
    unwrapped_args: list[TensorOperand] = [
        a.flat_values if isinstance(a, RaggedTensor) else a for a in args
    ]
    if unwrapped_args:
        return RaggedTensor(op(*unwrapped_args, **kwargs))
    return RaggedTensor(op(None, **kwargs))
