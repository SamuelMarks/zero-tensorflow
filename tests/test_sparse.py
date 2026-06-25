import pytest
from zero_tensorflow import sparse, SparseTensor


def test_sparse_tensor():
    t = SparseTensor([[0, 0], [1, 2]], [1.0, 2.0], [3, 4])
    assert t.indices == [[0, 0], [1, 2]]
    assert t.values == [1.0, 2.0]
    assert t.dense_shape == [3, 4]


def test_sparse_ops():
    with pytest.raises(NotImplementedError):
        sparse.sparse_dense_matmul(1, 1)
    with pytest.raises(NotImplementedError):
        sparse.add(1, 1)
    with pytest.raises(NotImplementedError):
        sparse.concat(1, [])
    with pytest.raises(NotImplementedError):
        sparse.split(1, 1, 1)
    with pytest.raises(NotImplementedError):
        sparse.to_dense(1)
    with pytest.raises(NotImplementedError):
        sparse.from_dense(1)
