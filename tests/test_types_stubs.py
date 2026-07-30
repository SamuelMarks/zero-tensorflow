import zero_tensorflow.types as tf_types


def call_safely(callable_obj):
    try:
        return callable_obj()
    except TypeError:
        try:
            return callable_obj(None)
        except TypeError:
            pass
    return None


def test_types_stubs():
    assert call_safely(tf_types.experimental) is not None
    assert call_safely(tf_types.experimental.AtomicFunction) is not None
    assert call_safely(tf_types.experimental.Callable) is not None
    assert call_safely(tf_types.experimental.ConcreteFunction) is not None
    assert call_safely(tf_types.experimental.FunctionType) is not None
    assert call_safely(tf_types.experimental.GenericFunction) is not None
    assert call_safely(tf_types.experimental.PolymorphicFunction) is not None
    assert call_safely(tf_types.experimental.SupportsTracingProtocol) is not None
    assert call_safely(tf_types.experimental.TensorLike) is not None
    assert call_safely(tf_types.experimental.TraceType) is not None
    assert call_safely(tf_types.experimental.distributed) is None
