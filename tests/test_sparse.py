"""Unit tests for `test_sparse`."""

from __future__ import annotations

from zero_tensorflow.sparse import (
    SparseTensor,
    add,
    concat,
    from_dense,
    sparse_dense_matmul,
    split,
    to_dense,
)

import ml_switcheroo_compiler.ops as msc_ops


def test_sparse() -> None:
    """Execute `test_sparse`."""
    t1 = msc_ops.zeros((2, 2))
    sp1 = SparseTensor([[0]], msc_ops.zeros((2, 2)), [1, 1])
    sparse_dense_matmul(sp1, t1)
    add(sp1, sp1)
    add(t1, t1)
    concat(0, [sp1, sp1])
    split(sp1, 2, 0)
    dense1 = to_dense(sp1)
    assert dense1 is not None and dense1.shape == (2, 2)
    dense2 = to_dense(t1)
    assert dense2 is not None and dense2.shape == (2, 2)
    assert to_dense(None) is None
    assert to_dense(SparseTensor(None, None, None)) is None
    sp2 = from_dense([[1.0, 0.0], [0.0, 2.0]])
    assert isinstance(sp2, SparseTensor)
    assert sp2.indices == [[0, 0], [1, 1]]
    assert from_dense(None).indices == []
