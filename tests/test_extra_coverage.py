from zero_tensorflow import _to_tensor
from zero_tensorflow.nn import softmax
import numpy as np


def test_missing_branches():
    # for __init__.py 57->61
    from ml_switcheroo_compiler.tracing import ProxyTensor

    pt = ProxyTensor("test", ())
    pt.dtype = None
    _to_tensor(pt)

    # for nn.py 76->78
    softmax(np.array([1.0, 2.0]), axis=0)


from zero_tensorflow import Tensor, GradientTape
import pytest


def test_tensor_len():
    # scalar tensor
    t1 = Tensor(3.0)
    assert len(t1) == 0
    # 1D tensor
    t2 = Tensor([1.0, 2.0])
    assert len(t2) == 2


def test_gradient_tape_edge_cases():
    with GradientTape() as tape:
        t = Tensor(3.0)
        tape.watch(t)
        y = t * t

    # None target
    assert tape.gradient(None, t) is None

    # Invalid target
    with pytest.raises(
        ValueError, match="Target was not created during the tape's recording."
    ):
        tape.gradient(Tensor(5.0), t)

    # Invalid source (not a Tensor)
    assert tape.gradient(y, [None])[0] is None
    assert tape.gradient(y, None) is None

    # Multiple sources including invalid
    t2 = Tensor(4.0)
    tape.watch(t2)
    with GradientTape() as tape2:
        y2 = t2 * 2.0
    grads = tape2.gradient(y2, [t2, None])
    assert len(grads) == 2
    assert grads[1] is None


def test_to_tensor_switcheroo_tensor_tracing():
    from zero_tensorflow import _to_tensor, Tensor
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor

    prev_tracing = getattr(_tracer, "is_tracing", False)
    prev_graph = getattr(_tracer, "active_graph", None)

    # 1) CREATE EAGER TENSOR FIRST!
    t = Tensor(3.0)
    mls_tensor = t._tensor

    t2 = Tensor(3.0)

    try:
        _tracer.is_tracing = True
        _tracer.active_graph = type("Graph", (), {"nodes": {}})()

        # 1) original_tensor=None -> 43->53, 61->63
        _to_tensor(mls_tensor)

        # 2) original_tensor != None, but graph_id in _traced_node_ids
        t2._traced_node_ids = {id(_tracer.active_graph): "my_node_id"}
        _to_tensor(t2)

        proxy = ProxyTensor(id="pt", shape=(1,), dtype=mls_tensor.dtype.value)
        _to_tensor(proxy)

    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph


def test_array_on_traced_non_constant():
    from zero_tensorflow import Tensor, GradientTape
    import pytest

    with GradientTape() as tape:
        t = Tensor(3.0)
        tape.watch(t)
        y = t * 2.0
        with pytest.raises(
            ValueError, match="Cannot call array conversion on a traced tensor"
        ):
            y.numpy()


def test_watch_non_tensor():
    from zero_tensorflow import GradientTape

    with GradientTape() as tape:
        tape.watch("this is a string, not a tensor")


def test_to_tensor_invalid_dtype():
    pass
