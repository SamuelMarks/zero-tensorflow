import ml_switcheroo_compiler.ops as msc_ops

from zero_tensorflow.sparse import (
    SparseTensor,
    add,
    concat,
    from_dense,
    sparse_dense_matmul,
    split,
    to_dense,
)


def test_sparse():
    t1 = msc_ops.zeros((2, 2))
    sp1 = SparseTensor([[0]], msc_ops.zeros((2, 2)), [1, 1])

    # Just asserting it executes without error in the facade
    res = sparse_dense_matmul(sp1, t1)

    add(sp1, sp1)
    add(t1, t1)

    concat(0, [sp1, sp1])

    split(sp1, 2, 0)

    res = to_dense(sp1)
    assert res.shape == (2, 2)

    res = to_dense(t1)
    assert res is t1

    res = from_dense(t1)
    assert isinstance(res, SparseTensor)
