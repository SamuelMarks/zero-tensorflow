import pytest
from zero_tensorflow import ragged, RaggedTensor


def test_ragged_tensor():
    t = RaggedTensor(1, 2, a=3)
    assert t._args == (1, 2)
    assert t._kwargs == {"a": 3}


def test_ragged_ops():
    with pytest.raises(NotImplementedError):
        ragged.boolean_mask(1, 1)
    with pytest.raises(NotImplementedError):
        ragged.map_flat_values(1)
