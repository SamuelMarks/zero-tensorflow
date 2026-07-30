import pytest
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_compiler.core.tensor import Tensor, TensorConfig
from ml_switcheroo_compiler.ops import control_flow
from ml_switcheroo_compiler.tracing import ProxyTensor
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer


def test_control_flow_tracing():
    prev_tracing = getattr(_tracer, "is_tracing", False)
    prev_graph = getattr(_tracer, "active_graph", None)
    prev_start = getattr(_tracer, "start_tracing", None)
    prev_stop = getattr(_tracer, "stop_tracing", None)
    prev_add = getattr(_tracer, "add_node", None)

    proxy_x = ProxyTensor("x_id", (2, 2), DType.Float32.value)
    x = Tensor(proxy_x, TensorConfig(shape=(2, 2), dtype=DType.Float32, device="cpu"))

    try:
        _tracer.is_tracing = True
        _tracer.active_graph = type(
            "Graph", (), {"nodes": {}, "add_node": lambda self, n: None, "parent": None}
        )()
        _tracer.start_tracing = lambda name="": type(
            "Graph", (), {"nodes": {}, "add_node": lambda self, n: None, "parent": None}
        )()
        _tracer.stop_tracing = lambda: None
        _tracer.add_node = lambda n: None

        def true_fn():
            return x

        def false_fn():
            return x

        control_flow.cond(x, true_fn, false_fn)

        def cond_fn(val):
            return x

        def body_fn(val):
            return val

        control_flow.while_loop(cond_fn, body_fn, x)

        def scan_fn(c, xs):
            return c, xs

        control_flow.scan(scan_fn, x, x)

        def vmap_fn(x):
            return x

        vmapped = control_flow.vmap(vmap_fn)
        vmapped(x)

        pmapped = control_flow.pmap(vmap_fn)
        pmapped(x)

        control_flow.stop_gradient(x)

        control_flow.stop_gradient(proxy_x)

        _tracer.is_tracing = False
        from ml_switcheroo_compiler.core.config import config

        prev_eager = config.eager_mode
        config.eager_mode = False
        try:
            with pytest.raises(Exception):  # noqa: B017
                control_flow.cond(x, true_fn, false_fn)
            with pytest.raises(Exception):  # noqa: B017
                control_flow.while_loop(cond_fn, body_fn, x)
            with pytest.raises(Exception):  # noqa: B017
                control_flow.scan(scan_fn, x, x)
            with pytest.raises(Exception):  # noqa: B017
                vmapped(x)
            with pytest.raises(Exception):  # noqa: B017
                pmapped(x)
        finally:
            config.eager_mode = prev_eager

    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph
        if prev_start:
            _tracer.start_tracing = prev_start
        if prev_stop:
            _tracer.stop_tracing = prev_stop
        if prev_add:
            _tracer.add_node = prev_add

    def true_fn_tuple():
        return (x, x)

    def false_fn_tuple():
        return (x, x)

    def cond_fn_tuple(*val):
        return x

    def body_fn_tuple(*val):
        return val

    # Try the tuple output version for full coverage
    try:
        _tracer.is_tracing = True
        _tracer.active_graph = type(
            "Graph", (), {"nodes": {}, "add_node": lambda self, n: None, "parent": None}
        )()
        _tracer.start_tracing = lambda name="": type(
            "Graph", (), {"nodes": {}, "add_node": lambda self, n: None, "parent": None}
        )()
        _tracer.stop_tracing = lambda: None
        _tracer.add_node = lambda n: None
        control_flow.cond(x, true_fn_tuple, false_fn_tuple)
        control_flow.while_loop(cond_fn_tuple, body_fn_tuple, (x, x))
    finally:
        _tracer.is_tracing = prev_tracing
