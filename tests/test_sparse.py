"""Unit tests for `zero_tensorflow.sparse`."""

from __future__ import annotations

from zero_tensorflow import Tensor, debugging
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
