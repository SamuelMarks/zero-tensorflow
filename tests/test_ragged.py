"""Unit tests for `test_ragged`."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.ragged import RaggedTensor, boolean_mask, map_flat_values


def test_ragged() -> None:
    """Execute `test_ragged`."""
    rt1 = RaggedTensor()
    assert isinstance(rt1, RaggedTensor)
    assert rt1.to_list() == []

    rt_nested = RaggedTensor([[1, 2], [3]])
    assert rt_nested.to_list() == [[1, 2], [3]]
    rt_copy = RaggedTensor(rt_nested)
    assert rt_copy.flat_values == [1, 2, 3]

    rt2 = boolean_mask([1, 2], [True, False])
    assert isinstance(rt2, RaggedTensor)
    assert rt2.to_list() == [1]

    rt2_2d = boolean_mask([[1, 2], [3, 4]], [[True, False], [False, True]])
    assert rt2_2d.to_list() == [[1], [4]]

    rt2_scalar = boolean_mask(10, True)
    assert isinstance(rt2_scalar, RaggedTensor)

    def dummy_op(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Execute `dummy_op`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOperand: Computed result.
        """
        del kwargs
        return args[0] if args else None

    rt3 = map_flat_values(dummy_op, [1, 2])
    assert isinstance(rt3, RaggedTensor)
    rt4 = map_flat_values(dummy_op)
    assert isinstance(rt4, RaggedTensor)
