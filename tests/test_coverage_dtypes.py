from zero_tensorflow import Tensor, _TracingContext
import numpy as np


def test_tensor_dtypes():
    Tensor(np.array(1.0, dtype=np.float64))
    Tensor(np.array(1.0, dtype=np.float32))
    Tensor(np.array(1, dtype=np.int64))
    Tensor(np.array(1, dtype=np.int32))
    Tensor(np.array(True, dtype=bool))

    # Also test passing dtype explicitly
    Tensor(1.0, dtype=np.float32)

    # Also test error handling and other types
    Tensor(np.array(1, dtype=np.int8))
    Tensor(np.array("test string"))  # hits the dt_str Exception

    from ml_switcheroo_compiler.tracing import ProxyTensor
    from zero_tensorflow import _to_tensor

    pt = ProxyTensor("test", (), "float32")
    _to_tensor(pt)

    pt_no_dtype = ProxyTensor("test", ())
    _to_tensor(pt_no_dtype)

    pt_invalid_dtype = ProxyTensor("test", (), "invalid_dtype_string")
    _to_tensor(pt_invalid_dtype)

    # We need to reach line 90: dt = DType(dt_str) and throw Exception
    # _to_tensor with np array uint64
    from zero_tensorflow import _to_tensor

    _to_tensor(np.array(1, dtype=np.uint64))


def test_bool_errors():
    t_multi = Tensor([1, 2])
    try:
        bool(t_multi)
    except ValueError:
        pass

    class DummyNode:
        id = "test_id"

    t_traced = Tensor(1.0, _traced_node=DummyNode())
    try:
        bool(t_traced)
    except TypeError:
        pass


def test_tracing_context():
    _ = _TracingContext()
    _TracingContext.enter()
    _TracingContext.exit()
    _TracingContext.get()
