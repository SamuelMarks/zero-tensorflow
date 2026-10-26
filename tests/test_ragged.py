"""Unit tests for `zero_tensorflow.ragged`."""

from __future__ import annotations

from zero_tensorflow import Tensor, debugging
from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.ragged import RaggedTensor, boolean_mask, constant, map_flat_values

import pytest


def test_ragged() -> None:
    """Verify `RaggedTensor` construction, factory methods, conversions, and ops."""
    rt1 = RaggedTensor()
    assert isinstance(rt1, RaggedTensor)
    assert rt1.to_list() == []
    assert rt1.to_tensor().shape == (0,)
    assert rt1.dtype.value == "float32"

    rt_nested = RaggedTensor([[1, 2], [3]])
    assert rt_nested.to_list() == [[1, 2], [3]]
    assert rt_nested.values == [1, 2, 3]
    assert rt_nested.row_splits == (0, 2, 3)
    assert rt_nested.ragged_rank == 1
    assert rt_nested.shape == (2, None)
    assert rt_nested.dtype.value == "int32"

    dense = rt_nested.to_tensor(default_value=0)
    debugging.assert_near(dense, [[1, 2], [3, 0]])

    sp = rt_nested.to_sparse()
    assert sp.indices == [[0, 0], [0, 1], [1, 0]]
    assert sp.dense_shape == [2, 2]

    rt_copy = RaggedTensor(rt_nested)
    assert rt_copy.flat_values == [1, 2, 3]
    assert rt_copy.row_splits == (0, 2, 3)

    rt_from_splits = RaggedTensor.from_row_splits([10, 20, 30], [0, 2, 3])
    assert rt_from_splits.to_list() == [[10, 20], [30]]
    with pytest.raises(ValueError):
        RaggedTensor.from_row_splits([10, 20], [0, 5])

    rt_from_lens = RaggedTensor.from_row_lengths([10, 20, 30], [1, 2])
    assert rt_from_lens.to_list() == [[10], [20, 30]]

    rt_from_rowids = RaggedTensor.from_value_rowids([10, 20, 30], [0, 0, 1])
    assert rt_from_rowids.to_list() == [[10, 20], [30]]

    rt_from_tensor = RaggedTensor.from_tensor(Tensor([[1, 2], [3, 4]]))
    assert rt_from_tensor.to_list() == [[1, 2], [3, 4]]

    rt_const = constant([[5, 6], [7]])
    assert rt_const.to_list() == [[5, 6], [7]]

    rt2 = boolean_mask([1, 2], [True, False])
    assert isinstance(rt2, RaggedTensor)
    assert rt2.to_list() == [1]

    rt2_2d = boolean_mask([[1, 2], [3, 4]], [[True, False], [False, True]])
    assert rt2_2d.to_list() == [[1], [4]]

    rt2_rt = boolean_mask(rt_nested, RaggedTensor([[True, False], [True]]))
    assert rt2_rt.to_list() == [[1], [3]]

    rt2_scalar = boolean_mask(10, True)
    assert rt2_scalar.to_list() == [10]
    rt2_scalar_false = boolean_mask(10, False)
    assert rt2_scalar_false.to_list() == []
    rt2_broadcast_mask = boolean_mask([1, 2], True)
    assert rt2_broadcast_mask.to_list() == [1, 2]

    def dummy_op(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Return first positional argument for `map_flat_values`.

        Args:
            *args: Positional arguments.
            **kwargs: Keyword arguments.

        Returns:
            TensorOperand: First argument or `None`.
        """
        del kwargs
        return args[0] if args else None

    rt3 = map_flat_values(dummy_op, rt_nested)
    assert rt3.row_splits == (0, 2, 3)
    rt4 = map_flat_values(dummy_op)
    assert isinstance(rt4, RaggedTensor)
