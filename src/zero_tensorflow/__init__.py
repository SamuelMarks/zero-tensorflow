"""zero_tensorflow API."""

import functools
from typing import Any, Optional
from ml_switcheroo.core.tensor_utils import to_array, to_numpy_dtype, ndarray
import ml_switcheroo
import ml_switcheroo.ops as _ops
from ml_switcheroo_ir import LogicalNode
import sys
import zero_keras as keras

sys.modules["zero_tensorflow.keras"] = keras

if not hasattr(keras.layers.Dense, "units"):
    keras.layers.Dense.units = property(lambda self: self._kwargs.get("units"))

__all__ = [
    "Variable",
    "function",
    "GradientTape",
    "data",
    "math",
    "nn",
    "keras",
    "Tensor",
]


def _to_tensor(x: Any, dtype: Optional[Any] = None) -> ml_switcheroo.Tensor:
    """_to_tensor docstring."""
    original_tensor = None
    if isinstance(x, Tensor):
        original_tensor = x
        x = x._tensor

    from ml_switcheroo.tracing import _tracer, ProxyTensor
    from ml_switcheroo.core.config import config
    import uuid

    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            graph_id = id(getattr(_tracer, "active_graph", None))
            if original_tensor is not None:
                if not hasattr(original_tensor, "_traced_node_ids"):
                    original_tensor._traced_node_ids = {}
                if graph_id in original_tensor._traced_node_ids:
                    out_id = original_tensor._traced_node_ids[graph_id]
                    pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
                    return ml_switcheroo.Tensor(
                        data=pt, shape=x.shape, dtype=x.dtype, device=x.device
                    )

            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": to_array(x.data).tolist()},
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            if original_tensor is not None:
                original_tensor._traced_node_ids[graph_id] = out_id
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            return ml_switcheroo.Tensor(
                data=pt, shape=x.shape, dtype=x.dtype, device=x.device
            )
        return x
    if isinstance(x, ProxyTensor):
        from ml_switcheroo.core.dtype import DType

        dt = config.default_float_dtype
        try:
            if x.dtype:
                dt = DType(x.dtype)
        except Exception:
            pass
        return ml_switcheroo.Tensor(
            data=x,
            shape=x.shape,
            dtype=dt,
            device=config.default_device,
        )

    arr = to_array(x)
    if dtype is not None:
        arr = arr.astype(dtype)

    from ml_switcheroo.core.dtype import DType

    dt_str = str(arr.dtype)
    dt = config.default_float_dtype
    try:
        if "float" in dt_str or "int" in dt_str or "bool" in dt_str:
            if dt_str == "float64":
                dt = DType.Float64
            elif dt_str == "float32":
                dt = DType.Float32
            elif dt_str == "int64":
                dt = DType.Int64
            elif dt_str == "int32":
                dt = DType.Int32
            elif dt_str == "bool":
                dt = DType.Bool
            else:
                dt = DType(dt_str)
    except Exception:
        pass

    res = ml_switcheroo.Tensor(
        data=arr, shape=arr.shape, dtype=dt, device=config.default_device
    )
    if _tracer.is_tracing:
        out_id = str(uuid.uuid4())
        node = LogicalNode(
            id=out_id,
            op_type="Constant",
            attributes={"value": to_array(res.data).tolist()},
            shape_metadata=res.shape,
        )
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=res.shape, dtype=res.dtype.value)
        res = ml_switcheroo.Tensor(
            data=pt, shape=res.shape, dtype=res.dtype, device=res.device
        )
    return res


def _wrap(x: Any) -> Any:
    """_wrap docstring."""
    if isinstance(x, Tensor):
        return x
    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)
    if isinstance(x, list):
        return list(_wrap(i) for i in x)
    return Tensor(x)


class Tensor:
    """Dual-state Tensor Primitive (Eager NumPy + Traced LogicalNode)."""

    def __init__(self, value: Any, dtype=None, _traced_node=None):
        """__init__ docstring."""
        self._traced_node_ids: dict[int, str] = {}
        if _traced_node is not None:
            from ml_switcheroo.tracing import ProxyTensor
            from ml_switcheroo.core.config import config

            pt = ProxyTensor(id=_traced_node.id, shape=())
            self._tensor = ml_switcheroo.Tensor(
                data=pt,
                shape=(),
                dtype=config.default_float_dtype,
                device=config.default_device,
            )
        elif isinstance(value, Tensor):
            self._tensor = value._tensor
        elif value is None:
            self._tensor = None
        else:
            self._tensor = _to_tensor(value, dtype=dtype)

    @property
    def shape(self):
        """shape docstring."""
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self):
        """dtype docstring."""
        return (
            to_numpy_dtype(self._tensor.dtype.value)
            if self._tensor is not None
            else None
        )

    def numpy(self):
        """numpy docstring."""
        if hasattr(self._tensor.data, "id"):
            from ml_switcheroo.tracing import _tracer

            current_graph = getattr(_tracer, "active_graph", None)
            if current_graph and self._tensor.data.id in current_graph.nodes:
                node = current_graph.nodes[self._tensor.data.id]
                if node.op_type == "Constant":
                    return to_array(node.attributes["value"])
            raise ValueError("Cannot call numpy on a traced tensor")
        return to_array(self._tensor.data)

    def __add__(self, other):
        """__add__ docstring."""
        return _wrap(_ops.add(_to_tensor(self), _to_tensor(other)))

    def __sub__(self, other):
        """__sub__ docstring."""
        return _wrap(_ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __mul__(self, other):
        """__mul__ docstring."""
        return _wrap(_ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __truediv__(self, other):
        """__truediv__ docstring."""
        return _wrap(_ops.divide(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other):
        """__radd__ docstring."""
        return _wrap(_ops.add(_to_tensor(other), _to_tensor(self)))

    def __rsub__(self, other):
        """__rsub__ docstring."""
        return _wrap(_ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __rmul__(self, other):
        """__rmul__ docstring."""
        return _wrap(_ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __rtruediv__(self, other):
        """__rtruediv__ docstring."""
        return _wrap(_ops.divide(_to_tensor(other), _to_tensor(self)))

    def __eq__(self, other):
        """__eq__ docstring."""
        return _wrap(_ops.equal(_to_tensor(self), _to_tensor(other)))

    def __ne__(self, other):
        """__ne__ docstring."""
        return _wrap(_ops.not_equal(_to_tensor(self), _to_tensor(other)))

    def __lt__(self, other):
        """__lt__ docstring."""
        return _wrap(_ops.less(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other):
        """__le__ docstring."""
        return _wrap(_ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other):
        """__gt__ docstring."""
        return _wrap(_ops.greater(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other):
        """__ge__ docstring."""
        return _wrap(_ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __bool__(self):
        """__bool__ docstring."""
        if hasattr(self._tensor.data, "id"):
            raise TypeError(
                "Using a `tf.Tensor` as a Python `bool` is not allowed in Graph execution."
            )
        arr = self.numpy()
        if arr.size == 1:
            return bool(arr.item())
        raise ValueError(
            "The truth value of an array with more than one element is ambiguous."
        )

    def __nonzero__(self):
        """__nonzero__ docstring."""
        return self.__bool__()

    def __len__(self):
        """__len__ docstring."""
        return self.shape[0] if len(self.shape) > 0 else 0


class Variable(Tensor):
    """Variable docstring."""

    def __init__(self, initial_value, trainable=True):
        """__init__ docstring."""
        super().__init__(initial_value)
        self.trainable = trainable

    @property
    def value(self):
        """value docstring."""
        return self

    def assign(self, value):
        """assign docstring."""
        self._tensor = _to_tensor(value)
        return self

    def assign_add(self, delta):
        """assign_add docstring."""
        self._tensor = _ops.add(self._tensor, _to_tensor(delta))
        return self

    def assign_sub(self, delta):
        """assign_sub docstring."""
        self._tensor = _ops.subtract(self._tensor, _to_tensor(delta))
        return self


class _TracingContext:
    """_TracingContext docstring."""

    _current_context = None

    def __init__(self):
        """__init__ docstring."""
        pass

    @classmethod
    def enter(cls):
        """enter docstring."""
        pass

    @classmethod
    def exit(cls):
        """exit docstring."""
        pass

    @classmethod
    def get(cls):
        """get docstring."""
        pass


def function(func):
    """function docstring."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """wrapped docstring."""

        def _to_tensor_if_possible(x):
            """_to_tensor_if_possible docstring."""
            if isinstance(x, (int, float, list, ndarray, Tensor, Variable)):
                return _wrap(_to_tensor(x))
            return x

        t_args = [_to_tensor_if_possible(a) for a in args]
        t_kwargs = {k: _to_tensor_if_possible(v) for k, v in kwargs.items()}

        # We need to set tracing mode ON
        from ml_switcheroo.tracing import _tracer
        from ml_switcheroo_ir import LogicalGraph

        print("IN FUNCTION, id=", id(_tracer))
        prev_is_tracing = _tracer.is_tracing
        prev_graph = _tracer.active_graph
        _tracer.is_tracing = True
        _tracer.active_graph = LogicalGraph(name="tf_function")

        try:
            res = func(*t_args, **t_kwargs)
        finally:
            _tracer.is_tracing = prev_is_tracing
            _tracer.active_graph = prev_graph

        if isinstance(res, tuple):
            return tuple(_to_tensor_if_possible(r) for r in res)
        return _to_tensor_if_possible(res)

    return wrapped


class GradientTape:
    """GradientTape docstring."""

    def __init__(self, persistent=False):
        """__init__ docstring."""
        self.persistent = persistent
        self.watched = []
        self._tape = None

    def __enter__(self):
        """__enter__ docstring."""
        from ml_switcheroo.tracing import TracerTape, _tracer

        self._prev_tracer_graph = getattr(_tracer, "active_graph", None)
        self._prev_is_tracing = getattr(_tracer, "is_tracing", False)

        if self._prev_is_tracing and self._prev_tracer_graph is not None:
            self._tape = None
            self._graph = self._prev_tracer_graph
        else:
            self._tape = TracerTape()
            self._graph = self._tape.start_tracing("GradientTape")
            _tracer.active_graph = self._graph
            _tracer.is_tracing = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__ docstring."""
        from ml_switcheroo.tracing import _tracer

        _tracer.active_graph = self._prev_tracer_graph
        _tracer.is_tracing = self._prev_is_tracing

    def watch(self, tensor):
        """watch docstring."""
        self.watched.append(tensor)
        if isinstance(tensor, Tensor):
            # Evaluate it so that it gets a node ID assigned to the current graph
            _to_tensor(tensor)

    def gradient(self, target, sources):
        """gradient docstring."""
        from ml_switcheroo.grad import grad
        from ml_switcheroo.interpreter import evaluate_graph

        if target is None:
            return None

        target_id = getattr(getattr(target._tensor, "data", None), "id", None)
        if target_id is None:
            raise ValueError("Target was not created during the tape's recording.")

        is_single = not isinstance(sources, (list, tuple))
        sources_list = [sources] if is_single else sources

        wrt_ids = []
        for s in sources_list:
            if not isinstance(s, Tensor):
                wrt_ids.append(None)
                continue
            graph_id = id(self._graph)
            if hasattr(s, "_traced_node_ids") and graph_id in s._traced_node_ids:
                wrt_ids.append(s._traced_node_ids[graph_id])
            else:
                data_id = getattr(getattr(s._tensor, "data", None), "id", None)
                wrt_ids.append(data_id)

        valid_wrt_ids = [w for w in wrt_ids if w is not None]
        if not valid_wrt_ids:
            return [None for _ in sources_list] if not is_single else None

        grad_graph = grad(self._graph, valid_wrt_ids, target_id)
        out_vals = evaluate_graph(grad_graph, {})

        res = []
        valid_idx = 0
        for w in wrt_ids:
            if w is None:
                res.append(None)
            else:
                grad_val = out_vals[grad_graph.outputs[valid_idx]]
                res.append(Tensor(grad_val))
                valid_idx += 1

        return res[0] if is_single else res


class math:
    """math docstring."""

    @staticmethod
    def abs(*args, **kwargs):
        """abs docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "abs")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acos(*args, **kwargs):
        """acos docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acosh(*args, **kwargs):
        """acosh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def add(*args, **kwargs):
        """add docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def all(*args, **kwargs):
        """all docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "all")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def allclose(*args, **kwargs):
        """allclose docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "allclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def any(*args, **kwargs):
        """any docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "any")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def arange(*args, **kwargs):
        """arange docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "arange")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmax(*args, **kwargs):
        """argmax docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmin(*args, **kwargs):
        """argmin docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asin(*args, **kwargs):
        """asin docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asinh(*args, **kwargs):
        """asinh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan(*args, **kwargs):
        """atan docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan2(*args, **kwargs):
        """atan2 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atanh(*args, **kwargs):
        """atanh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def binary(*args, **kwargs):
        """binary docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "binary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitcast(*args, **kwargs):
        """bitcast docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitcast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_and(*args, **kwargs):
        """bitwise_and docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_not(*args, **kwargs):
        """bitwise_not docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_or(*args, **kwargs):
        """bitwise_or docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_xor(*args, **kwargs):
        """bitwise_xor docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def broadcast_to(*args, **kwargs):
        """broadcast_to docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "broadcast_to")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cast(*args, **kwargs):
        """cast docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cbrt(*args, **kwargs):
        """cbrt docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cbrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ceil(*args, **kwargs):
        """ceil docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ceil")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cholesky(*args, **kwargs):
        """cholesky docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cholesky")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def concatenate(*args, **kwargs):
        """concatenate docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "concatenate")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def conj(*args, **kwargs):
        """conj docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "conj")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def copysign(*args, **kwargs):
        """copysign docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "copysign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cos(*args, **kwargs):
        """cos docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cosh(*args, **kwargs):
        """cosh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def count_nonzero(*args, **kwargs):
        """count_nonzero docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "count_nonzero")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def creation(*args, **kwargs):
        """creation docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "creation")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def deg2rad(*args, **kwargs):
        """deg2rad docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "deg2rad")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def det(*args, **kwargs):
        """det docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "det")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def diag(*args, **kwargs):
        """diag docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "diag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def digamma(*args, **kwargs):
        """digamma docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "digamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divide(*args, **kwargs):
        """divide docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divmod(*args, **kwargs):
        """divmod docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dot(*args, **kwargs):
        """dot docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dynamic_slice(*args, **kwargs):
        """dynamic_slice docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dynamic_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigh(*args, **kwargs):
        """eigh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigvalsh(*args, **kwargs):
        """eigvalsh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigvalsh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def einsum(*args, **kwargs):
        """einsum docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "einsum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def empty(*args, **kwargs):
        """empty docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "empty")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def equal(*args, **kwargs):
        """equal docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erf(*args, **kwargs):
        """erf docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfc(*args, **kwargs):
        """erfc docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfinv(*args, **kwargs):
        """erfinv docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp(*args, **kwargs):
        """exp docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp2(*args, **kwargs):
        """exp2 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expand(*args, **kwargs):
        """expand docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expand")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expm1(*args, **kwargs):
        """expm1 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expm1")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eye(*args, **kwargs):
        """eye docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eye")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fix(*args, **kwargs):
        """fix docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fix")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def flatten(*args, **kwargs):
        """flatten docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "flatten")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def float_power(*args, **kwargs):
        """float_power docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "float_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor(*args, **kwargs):
        """floor docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor_divide(*args, **kwargs):
        """floor_divide docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor_divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmax(*args, **kwargs):
        """fmax docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmin(*args, **kwargs):
        """fmin docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmod(*args, **kwargs):
        """fmod docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def frexp(*args, **kwargs):
        """frexp docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "frexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full(*args, **kwargs):
        """full docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full_like(*args, **kwargs):
        """full_like docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather(*args, **kwargs):
        """gather docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather_nd(*args, **kwargs):
        """gather_nd docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gcd(*args, **kwargs):
        """gcd docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gcd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater(*args, **kwargs):
        """greater docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater_equal(*args, **kwargs):
        """greater_equal docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def heaviside(*args, **kwargs):
        """heaviside docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "heaviside")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def hypot(*args, **kwargs):
        """hypot docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "hypot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def identity(*args, **kwargs):
        """identity docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "identity")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def imag(*args, **kwargs):
        """imag docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "imag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inner(*args, **kwargs):
        """inner docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inner")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inv(*args, **kwargs):
        """inv docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isclose(*args, **kwargs):
        """isclose docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isfinite(*args, **kwargs):
        """isfinite docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isfinite")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isinf(*args, **kwargs):
        """isinf docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isinf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isnan(*args, **kwargs):
        """isnan docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isnan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lcm(*args, **kwargs):
        """lcm docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lcm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ldexp(*args, **kwargs):
        """ldexp docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ldexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def left_shift(*args, **kwargs):
        """left_shift docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "left_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less(*args, **kwargs):
        """less docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less_equal(*args, **kwargs):
        """less_equal docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lgamma(*args, **kwargs):
        """lgamma docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lgamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linalg(*args, **kwargs):
        """linalg docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linalg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linspace(*args, **kwargs):
        """linspace docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linspace")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log(*args, **kwargs):
        """log docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log10(*args, **kwargs):
        """log10 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log10")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log1p(*args, **kwargs):
        """log1p docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log1p")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log2(*args, **kwargs):
        """log2 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp(*args, **kwargs):
        """logaddexp docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp2(*args, **kwargs):
        """logaddexp2 docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_and(*args, **kwargs):
        """logical_and docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_not(*args, **kwargs):
        """logical_not docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_or(*args, **kwargs):
        """logical_or docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_xor(*args, **kwargs):
        """logical_xor docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logsumexp(*args, **kwargs):
        """logsumexp docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logsumexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matmul(*args, **kwargs):
        """matmul docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matmul")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matrix_power(*args, **kwargs):
        """matrix_power docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matrix_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_max(*args, **kwargs):
        """reduce_max docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "max")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def maximum(*args, **kwargs):
        """maximum docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "maximum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_mean(*args, **kwargs):
        """reduce_mean docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mean")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def meshgrid(*args, **kwargs):
        """meshgrid docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "meshgrid")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_min(*args, **kwargs):
        """reduce_min docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "min")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def minimum(*args, **kwargs):
        """minimum docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "minimum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def mod(*args, **kwargs):
        """mod docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def moveaxis(*args, **kwargs):
        """moveaxis docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "moveaxis")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def multiply(*args, **kwargs):
        """multiply docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "multiply")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def negative(*args, **kwargs):
        """negative docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "negative")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def nextafter(*args, **kwargs):
        """nextafter docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "nextafter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def norm(*args, **kwargs):
        """norm docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "norm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def not_equal(*args, **kwargs):
        """not_equal docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "not_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones(*args, **kwargs):
        """ones docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones_like(*args, **kwargs):
        """ones_like docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def outer(*args, **kwargs):
        """outer docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "outer")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def permute(*args, **kwargs):
        """permute docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "permute")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pinv(*args, **kwargs):
        """pinv docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "pinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def positive(*args, **kwargs):
        """positive docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "positive")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def power(*args, **kwargs):
        """power docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def prod(*args, **kwargs):
        """prod docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "prod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def qr(*args, **kwargs):
        """qr docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "qr")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rad2deg(*args, **kwargs):
        """rad2deg docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rad2deg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def real(*args, **kwargs):
        """real docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "real")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reciprocal(*args, **kwargs):
        """reciprocal docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reciprocal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reductions(*args, **kwargs):
        """reductions docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reductions")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def remainder(*args, **kwargs):
        """remainder docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "remainder")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def repeat(*args, **kwargs):
        """repeat docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "repeat")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reshape(*args, **kwargs):
        """reshape docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reshape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def right_shift(*args, **kwargs):
        """right_shift docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "right_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def roll(*args, **kwargs):
        """roll docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "roll")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def round(*args, **kwargs):
        """round docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "round")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rsqrt(*args, **kwargs):
        """rsqrt docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rsqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter(*args, **kwargs):
        """scatter docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_add(*args, **kwargs):
        """scatter_add docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_nd(*args, **kwargs):
        """scatter_nd docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def shape(*args, **kwargs):
        """shape docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "shape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sign(*args, **kwargs):
        """sign docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sin(*args, **kwargs):
        """sin docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinc(*args, **kwargs):
        """sinc docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinh(*args, **kwargs):
        """sinh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slice(*args, **kwargs):
        """slice docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slogdet(*args, **kwargs):
        """slogdet docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slogdet")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def split(*args, **kwargs):
        """split docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "split")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sqrt(*args, **kwargs):
        """sqrt docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def square(*args, **kwargs):
        """square docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "square")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def squeeze(*args, **kwargs):
        """squeeze docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "squeeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def stack(*args, **kwargs):
        """stack docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "stack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def std(*args, **kwargs):
        """std docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "std")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def strided_slice(*args, **kwargs):
        """strided_slice docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "strided_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def subtract(*args, **kwargs):
        """subtract docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "subtract")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_sum(*args, **kwargs):
        """reduce_sum docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def svd(*args, **kwargs):
        """svd docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "svd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def swapaxes(*args, **kwargs):
        """swapaxes docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "swapaxes")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def take(*args, **kwargs):
        """take docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "take")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tan(*args, **kwargs):
        """tan docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tanh(*args, **kwargs):
        """tanh docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tensordot(*args, **kwargs):
        """tensordot docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tile(*args, **kwargs):
        """tile docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tile")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def transpose(*args, **kwargs):
        """transpose docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "transpose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tril(*args, **kwargs):
        """tril docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tril")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def triu(*args, **kwargs):
        """triu docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "triu")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def trunc(*args, **kwargs):
        """trunc docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "trunc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unary(*args, **kwargs):
        """unary docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsqueeze(*args, **kwargs):
        """unsqueeze docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unsqueeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unstack(*args, **kwargs):
        """unstack docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unstack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def update_slice(*args, **kwargs):
        """update_slice docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "update_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def variance(*args, **kwargs):
        """variance docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "variance")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def vdot(*args, **kwargs):
        """vdot docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "vdot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def where(*args, **kwargs):
        """where docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "where")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros(*args, **kwargs):
        """zeros docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros_like(*args, **kwargs):
        """zeros_like docstring."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pow(x, y):
        """pow docstring."""
        return _wrap(_ops.power(_to_tensor(x), _to_tensor(y)))


from . import data
from . import nn
