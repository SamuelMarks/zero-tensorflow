"""Unit tests for `zero_tensorflow.sparse`."""

from __future__ import annotations

from zero_tensorflow import Tensor, debugging, sparse
from zero_tensorflow.sparse import (
    SparseTensor,
    _extract_values,
    _to_coord_list,
    _to_int_list,
    _to_scalar_list,
    add,
    concat,
    from_dense,
    sparse_dense_matmul,
    split,
    to_dense,
)


def test_sparse() -> None:
    """Verify `SparseTensor`, `from_dense`, `to_dense`, `sparse_dense_matmul`, `add`, `concat`, and `split`."""
    sp = from_dense([[1.0, 0.0], [0.0, 2.0]])
    assert isinstance(sp, SparseTensor)
    assert sp.indices == [[0, 0], [1, 1]]
    assert sp.shape == (2, 2)
    assert isinstance(sp.values, Tensor)
    debugging.assert_near(sp.values, [1.0, 2.0])

    dense = to_dense(sp)
    assert dense is not None
    debugging.assert_near(dense, [[1.0, 0.0], [0.0, 2.0]])

    b = Tensor([[3.0, 4.0], [5.0, 6.0]])
    mm = sparse_dense_matmul(sp, b)
    debugging.assert_near(mm, [[3.0, 4.0], [10.0, 12.0]])

    mm_adj = sparse_dense_matmul(sp, b, adjoint_a=True, adjoint_b=True)
    debugging.assert_near(mm_adj, [[3.0, 5.0], [8.0, 12.0]])

    added_dense = add(sp, b)
    assert isinstance(added_dense, Tensor)
    debugging.assert_near(added_dense, [[4.0, 4.0], [5.0, 8.0]])

    added_sp = add(sp, sp)
    assert isinstance(added_sp, SparseTensor)
    assert isinstance(added_sp.values, Tensor)
    debugging.assert_near(added_sp.values, [[2.0, 0.0], [0.0, 4.0]])

    cat = concat(0, [sp, sp])
    assert isinstance(cat, SparseTensor)
    assert cat.shape == (4, 2)
    cat_empty = concat(0, [Tensor([1.0]), Tensor([2.0])], axis_arg=0)
    assert cat_empty.indices is None

    parts = split(sp, 2, axis=0)
    assert len(parts) == 2
    assert parts[0].shape == (1, 2)
    parts_dense = split(Tensor([1.0, 2.0]), 2, axis=0)
    assert len(parts_dense) == 2

    assert to_dense(Tensor([1.0])) is not None
    assert to_dense(None) is None
    assert to_dense(SparseTensor(None, None, None)) is None
    sp_unstructured = SparseTensor(None, [5.0, 6.0], None)
    unstructured_dense = to_dense(sp_unstructured)
    assert unstructured_dense is not None
    debugging.assert_near(unstructured_dense, [5.0, 6.0])

    assert from_dense(None).indices == []

    class _Holder:
        """ValuesHolderProtocol test helper."""

        @property
        def values(self) -> list[float]:
            """Return underlying values list.

            Returns:
                list[float]: Numeric list.
            """
            return [1.0, 2.0]

    assert _extract_values(_Holder()).shape == (2,)
    assert _extract_values(SparseTensor(None, None, None)) is not None
    assert _to_int_list(Tensor([2, 3])) == [2, 3]
    assert _to_int_list("bad") == []
    assert _to_coord_list(Tensor([[0, 1]])) == [[0, 1]]
    assert _to_coord_list([1, 2]) == [[1], [2]]
    assert _to_coord_list("bad") == []
    assert _to_scalar_list(Tensor([1.0, 2.0])) == [1.0, 2.0]
    assert _to_scalar_list(3.5) == [3.5]
    assert _to_scalar_list("bad") == []
    assert SparseTensor([[0, 0]], [1.0], [1, 1]).dtype.value == "float32"

    bc = sparse.bincount([1, 1, 2], minlength=4, maxlength=3, binary_output=False)
    debugging.assert_near(to_dense(bc), [0.0, 2.0, 1.0, 0.0])
    assert sparse.cross([[[1, 2]], [[3]]]).shape == (1, 2)
    assert sparse.cross([]).shape == (0, 0)
    assert sparse.cross_hashed([[[1]], [[2]]], num_buckets=10).shape == (1, 1)
    assert sparse.expand_dims(sp, axis=0).shape == (1, 2, 2)
    assert sparse.expand_dims(sp, axis=-1).shape == (2, 2, 1)
    debugging.assert_near(to_dense(sparse.eye(2)), [[1.0, 0.0], [0.0, 1.0]])
    sp_gap = SparseTensor([[0, 0], [2, 1]], Tensor([1.0, 5.0]), [3, 2])
    filled_sp, emp_ind = sparse.fill_empty_rows(sp_gap, default_value=9.0)
    assert filled_sp.shape == (3, 2)
    debugging.assert_equal(emp_ind, [False, True, False])
    mapped_sp = sparse.map_values(lambda v: Tensor(v) * 3.0, sp)
    debugging.assert_near(mapped_sp.values, [3.0, 6.0])
    assert sparse.map_values(lambda v: Tensor(v) * 2.0, Tensor([1.0, 2.0])).shape == (2,)
    masked_sp = sparse.mask(sp, [0])
    assert masked_sp.indices == [[1, 1]]
    debugging.assert_near(
        to_dense(sparse.maximum(sp, from_dense([[0.0, 3.0], [0.0, 1.0]]))), [[1.0, 3.0], [0.0, 2.0]]
    )
    debugging.assert_near(
        to_dense(sparse.minimum(sp, from_dense([[0.5, 3.0], [0.0, 1.0]]))), [[0.5, 0.0], [0.0, 1.0]]
    )
    debugging.assert_near(sparse.reduce_max(sp, axis=0), [1.0, 2.0])
    assert isinstance(sparse.reduce_max(sp, axis=[0, 1], output_is_sparse=True), SparseTensor)
    debugging.assert_near(sparse.reduce_sum(sp, axis=1), [1.0, 2.0])
    assert isinstance(sparse.reduce_sum(sp, axis=[0], output_is_sparse=True), SparseTensor)
    unord_sp = SparseTensor([[1, 0], [0, 1]], Tensor([2.0, 1.0]), [2, 2])
    reord_sp = sparse.reorder(unord_sp)
    assert reord_sp.indices == [[0, 1], [1, 0]]
    assert sparse.reset_shape(sp, [4, 4]).shape == (4, 4)
    assert sparse.reset_shape(sp).shape == (2, 2)
    assert sparse.reshape(sp, [4, 1]).shape == (4, 1)
    ret_sp = sparse.retain(sp, [True, False])
    assert ret_sp.indices == [[0, 0]]
    assert sparse.retain(sp, "bad").indices == []
    seg_data = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    debugging.assert_near(
        sparse.segment_sum(seg_data, [0, 1, 2], [0, 0, 2]), [[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]]
    )
    debugging.assert_near(sparse.segment_mean(seg_data, [0, 1], [0, 0]), [[2.0, 3.0]])
    debugging.assert_near(sparse.segment_sqrt_n(seg_data, [0], [0]), [[1.0, 2.0]])
    assert sparse.segment_sum(seg_data, [], []).shape == (0, 2)
    sl_sp = sparse.slice(sp, [0, 0], [1, 2])
    assert sl_sp.indices == [[0, 0]] and sl_sp.shape == (1, 2)
    sm_sp = sparse.softmax(sp)
    debugging.assert_near(sm_sp.values, [1.0, 1.0])
    ind_t = sparse.to_indicator(
        SparseTensor([[0, 0], [1, 0]], Tensor([1, 2]), [2, 1]), vocab_size=3
    )
    debugging.assert_equal(ind_t, [[False, True, False], [False, False, True]])
    tr_sp = sparse.transpose(SparseTensor([[0, 1]], Tensor([7.0]), [2, 3]))
    assert tr_sp.indices == [[1, 0]] and tr_sp.shape == (3, 2)
