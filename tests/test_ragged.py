"""Unit tests for `zero_tensorflow.ragged`."""

from __future__ import annotations

from zero_tensorflow import Tensor, debugging, ragged
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

    rt_str = RaggedTensor([["a", "b"], ["c"]])
    assert rt_str.dtype.value == "string"

    rt_nested = RaggedTensor([[1, 2], [3]])
    assert rt_nested.to_list() == [[1, 2], [3]]
    assert rt_nested.values == [1, 2, 3]
    assert rt_nested.row_splits == (0, 2, 3)
    assert rt_nested.nested_row_splits == ((0, 2, 3),)
    assert rt_nested.ragged_rank == 1
    assert rt_nested.shape == (2, None)
    assert rt_nested.dtype.value == "int32"

    debugging.assert_equal(rt_nested[0], [1, 2])
    sl_row = rt_nested[0:1]
    assert isinstance(sl_row, RaggedTensor) and sl_row.to_list() == [[1, 2]]
    debugging.assert_equal(rt_nested[0, 1], 2)
    sl_2d = rt_nested[0:2, 0:1]
    assert isinstance(sl_2d, RaggedTensor) and sl_2d.to_list() == [[1], [3]]
    rt_1d = RaggedTensor([10, 20], row_splits=None)
    rt_1d._has_nested_rows = False
    debugging.assert_equal(rt_1d[0, 0], 10)

    dense = rt_nested.to_tensor(default_value=0)
    debugging.assert_near(dense, [[1, 2], [3, 0]])
    dense_shaped = rt_nested.to_tensor(default_value=9, shape=(3, 3))
    debugging.assert_near(dense_shaped, [[1, 2, 9], [3, 9, 9], [9, 9, 9]])

    sp = rt_nested.to_sparse()
    assert sp.indices == [[0, 0], [0, 1], [1, 0]]
    assert sp.dense_shape == [2, 2]

    rt_copy = RaggedTensor(rt_nested)
    assert rt_copy.flat_values == [1, 2, 3]
    assert rt_copy.row_splits == (0, 2, 3)

    rt_from_splits = RaggedTensor.from_row_splits([10, 20, 30], [0, 2, 3])
    assert rt_from_splits.to_list() == [[10, 20], [30]]
    rt_from_inner_rt = RaggedTensor.from_row_splits(rt_from_splits, [0, 1, 2])
    assert rt_from_inner_rt.ragged_rank == 2
    assert rt_from_inner_rt.to_list() == [[[10, 20]], [[30]]]
    with pytest.raises(ValueError):
        RaggedTensor.from_row_splits([10, 20], [0, 5])
    with pytest.raises(ValueError):
        RaggedTensor.from_nested_row_splits([10, 20], [[0, 5]])

    rt_n_len = RaggedTensor.from_nested_row_lengths([1, 2, 3], [[1, 1], [2, 1]])
    assert rt_n_len.to_list() == [[[1, 2]], [[3]]]

    rt_n_rid = RaggedTensor.from_nested_value_rowids([1, 2, 3], [[0, 1], [0, 0, 1]])
    assert rt_n_rid.to_list() == [[[1, 2]], [[3]]]

    rt_from_lens = RaggedTensor.from_row_lengths([10, 20, 30], [1, 2])
    assert rt_from_lens.to_list() == [[10], [20, 30]]

    rt_from_rowids = RaggedTensor.from_value_rowids([10, 20, 30], [0, 0, 1])
    assert rt_from_rowids.to_list() == [[10, 20], [30]]

    rt_from_tensor = RaggedTensor.from_tensor(Tensor([[1, 2], [3, 4]]))
    assert rt_from_tensor.to_list() == [[1, 2], [3, 4]]

    rt_const = constant([[5, 6], [7]])
    assert rt_const.to_list() == [[5, 6], [7]]

    assert ragged.stack([rt_nested, rt_const], axis=0).to_list() == [[1, 2], [3], [5, 6], [7]]
    assert ragged.stack([rt_nested, rt_const], axis=1).to_list() == [[1, 2, 5, 6], [3, 7]]
    assert ragged.stack([]).to_list() == []
    assert ragged.stack_dynamic_partitions([10, 20, 30], [1, 0, 1], 2).to_list() == [[20], [10, 30]]
    assert ragged.stack_dynamic_partitions(Tensor(10), Tensor(0), 1).to_list() == [[10]]
    assert ragged.cross(
        [RaggedTensor([["a"], ["b"]]), RaggedTensor([["1", "2"], ["3"]])]
    ).to_list() == [
        ["a_X_1", "a_X_2"],
        ["b_X_3"],
    ]
    assert ragged.cross([]).to_list() == []
    assert (
        len(
            ragged.cross_hashed(
                [RaggedTensor([["a"]]), RaggedTensor([["1"]])], num_buckets=10
            ).to_list()
        )
        == 1
    )
    assert ragged.range(3).to_list() == [[0, 1, 2]]
    assert ragged.range([0, 2], [2, 5], [1, 2]).to_list() == [[0, 1], [2, 4]]
    assert ragged.range(Tensor([0, 1]), Tensor([2, 3]), Tensor(1)).to_list() == [[0, 1], [1, 2]]
    debugging.assert_equal(ragged.row_splits_to_segment_ids([0, 2, 3]), [0, 0, 1])
    debugging.assert_equal(ragged.row_splits_to_segment_ids(Tensor(0)), [])
    debugging.assert_equal(
        ragged.segment_ids_to_row_splits([0, 0, 1], num_segments=3), [0, 2, 3, 3]
    )
    debugging.assert_equal(ragged.segment_ids_to_row_splits([0, 0, 1]), [0, 2, 3])
    debugging.assert_equal(ragged.segment_ids_to_row_splits(Tensor(0)), [0])

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
