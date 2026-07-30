from zero_tensorflow.lookup import experimental


def test_lookup():
    assert experimental.DenseHashTable() is not None
    assert experimental.MutableHashTable() is not None
