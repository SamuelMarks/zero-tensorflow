from zero_tensorflow.ragged import RaggedTensor, boolean_mask, map_flat_values


def test_ragged():
    rt1 = RaggedTensor()
    assert isinstance(rt1, RaggedTensor)

    rt2 = boolean_mask([1, 2], [True, False])
    assert isinstance(rt2, RaggedTensor)

    def dummy_op(x):
        return x

    rt3 = map_flat_values(dummy_op, [1, 2])
    assert isinstance(rt3, RaggedTensor)

    rt4 = map_flat_values(dummy_op)
    assert isinstance(rt4, RaggedTensor)
