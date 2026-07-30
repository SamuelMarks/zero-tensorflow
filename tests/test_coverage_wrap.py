import ml_switcheroo_compiler
import numpy as np

from zero_tensorflow import Tensor, _to_tensor, _wrap, function


def test_wrap():
    t = Tensor(1.0)
    _wrap(t)
    _wrap((t, t))
    _wrap([t, t])


def test_tensor_corner_cases():
    class DummyNode:
        id = "test_id"

    t_traced = Tensor(1.0, _traced_node=DummyNode())

    t_none = Tensor(None)
    assert t_none.shape == ()
    assert t_none.dtype is None

    # Test numpy on traced tensor
    try:
        t_traced.numpy()
    except ValueError:
        pass


def test_to_tensor_tracing():
    # Trigger line 31: passing a non-traced ml_switcheroo.Tensor to _to_tensor during tracing
    from ml_switcheroo_compiler.core.config import config

    # Create an eager ml_switcheroo.Tensor
    from ml_switcheroo_compiler.core.tensor import TensorConfig

    t = ml_switcheroo_compiler.Tensor(
        np.array(1.0),
        config=TensorConfig(
            shape=(), dtype=config.default_float_dtype, device=config.default_device
        ),
    )
    t_wrap = Tensor(t)
    del t_wrap._traced_node_ids

    @function
    def f():
        return _to_tensor(t_wrap)

    f()
