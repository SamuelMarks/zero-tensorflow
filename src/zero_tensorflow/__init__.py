"""zero_tensorflow API."""

import functools
from typing import Any, Optional
import numpy as np
import ml_switcheroo
import ml_switcheroo.ops as _ops
from ml_switcheroo_ir import LogicalNode
import sys
import zero_keras as keras

sys.modules["zero_tensorflow.keras"] = keras


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
    if isinstance(x, Tensor):
        x = x._tensor
    if isinstance(x, Variable):
        x = x._tensor

    from ml_switcheroo.tracing import _tracer, ProxyTensor
    from ml_switcheroo.core.config import config
    import uuid

    print(
        "IN _TO_TENSOR, is_tracing=",
        getattr(_tracer, "is_tracing", None),
        "id=",
        id(_tracer),
    )
    if isinstance(x, ml_switcheroo.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={"value": np.array(x.data).tolist()},
                shape_metadata=x.shape,
            )
            print("GRAPH:", getattr(_tracer, "active_graph", None))
            _tracer.add_node(node)
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

    arr = np.array(x)
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
            attributes={"value": np.array(res.data).tolist()},
            shape_metadata=res.shape,
        )
        print("GRAPH:", getattr(_tracer, "active_graph", None))
        _tracer.add_node(node)
        pt = ProxyTensor(id=out_id, shape=res.shape, dtype=res.dtype.value)
        res = ml_switcheroo.Tensor(
            data=pt, shape=res.shape, dtype=res.dtype, device=res.device
        )
    return res


def _wrap(x: Any) -> "Tensor":
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
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self):
        return np.dtype(self._tensor.dtype.value) if self._tensor is not None else None

    def numpy(self):
        if hasattr(self._tensor.data, "id"):
            raise ValueError("Cannot call numpy on a traced tensor")
        return np.array(self._tensor.data)

    def __add__(self, other):
        return _wrap(_ops.add(_to_tensor(self), _to_tensor(other)))

    def __sub__(self, other):
        return _wrap(_ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __mul__(self, other):
        return _wrap(_ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __truediv__(self, other):
        return _wrap(_ops.divide(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other):
        return _wrap(_ops.add(_to_tensor(other), _to_tensor(self)))

    def __rsub__(self, other):
        return _wrap(_ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __rmul__(self, other):
        return _wrap(_ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __rtruediv__(self, other):
        return _wrap(_ops.divide(_to_tensor(other), _to_tensor(self)))

    def __eq__(self, other):
        return _wrap(_ops.equal(_to_tensor(self), _to_tensor(other)))

    def __ne__(self, other):
        return _wrap(_ops.not_equal(_to_tensor(self), _to_tensor(other)))

    def __lt__(self, other):
        return _wrap(_ops.less(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other):
        return _wrap(_ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other):
        return _wrap(_ops.greater(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other):
        return _wrap(_ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __bool__(self):
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
        return self.__bool__()

    def __len__(self):
        return self.shape[0] if len(self.shape) > 0 else 0


class Variable(Tensor):
    """Variable docstring."""

    def __init__(self, initial_value, trainable=True):
        super().__init__(initial_value)
        self.trainable = trainable

    @property
    def value(self):
        return self

    def assign(self, value):
        self._tensor = _to_tensor(value)
        return self

    def assign_add(self, delta):
        self._tensor = _ops.add(self._tensor, _to_tensor(delta))
        return self

    def assign_sub(self, delta):
        self._tensor = _ops.subtract(self._tensor, _to_tensor(delta))
        return self


class _TracingContext:
    """_TracingContext docstring."""

    _current_context = None

    def __init__(self):
        pass

    @classmethod
    def enter(cls):
        pass

    @classmethod
    def exit(cls):
        pass

    @classmethod
    def get(cls):
        pass


def function(func):
    """function docstring."""

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        def _to_tensor_if_possible(x):
            if isinstance(x, (int, float, list, np.ndarray, Tensor, Variable)):
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
        self.persistent = persistent
        self.watched = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def watch(self, tensor):
        self.watched.append(tensor)

    def gradient(self, target, sources):
        if isinstance(sources, (list, tuple)):
            return [Tensor([1.0]) for _ in sources]
        return Tensor([1.0])


class math:
    """math docstring."""

    @staticmethod
    def abs(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "abs")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acos(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acosh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def add(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def all(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "all")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def allclose(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "allclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def any(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "any")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def arange(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "arange")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmax(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmin(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asin(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asinh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan2(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atanh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def binary(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "binary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitcast(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitcast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_and(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_not(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_or(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_xor(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def broadcast_to(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "broadcast_to")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cast(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cbrt(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cbrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ceil(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ceil")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cholesky(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cholesky")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def concatenate(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "concatenate")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def conj(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "conj")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def copysign(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "copysign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cos(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cosh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def count_nonzero(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "count_nonzero")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def creation(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "creation")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def deg2rad(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "deg2rad")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def det(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "det")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def diag(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "diag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def digamma(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "digamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divide(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divmod(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dot(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dynamic_slice(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dynamic_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigvalsh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigvalsh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def einsum(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "einsum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def empty(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "empty")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def equal(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erf(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfc(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfinv(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp2(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expand(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expand")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expm1(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expm1")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eye(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eye")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fix(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fix")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def flatten(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "flatten")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def float_power(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "float_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor_divide(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor_divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmax(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmin(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmod(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def frexp(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "frexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full_like(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather_nd(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gcd(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gcd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater_equal(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def heaviside(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "heaviside")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def hypot(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "hypot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def identity(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "identity")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def imag(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "imag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inner(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inner")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inv(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isclose(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isfinite(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isfinite")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isinf(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isinf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isnan(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isnan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lcm(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lcm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ldexp(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ldexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def left_shift(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "left_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less_equal(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lgamma(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lgamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linalg(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linalg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linspace(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linspace")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log10(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log10")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log1p(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log1p")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log2(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp2(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_and(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_not(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_or(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_xor(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logsumexp(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logsumexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matmul(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matmul")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matrix_power(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matrix_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_max(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "max")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def maximum(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "maximum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_mean(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mean")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def meshgrid(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "meshgrid")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_min(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "min")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def minimum(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "minimum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def mod(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def moveaxis(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "moveaxis")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def multiply(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "multiply")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def negative(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "negative")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def nextafter(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "nextafter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def norm(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "norm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def not_equal(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "not_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones_like(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def outer(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "outer")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def permute(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "permute")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pinv(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "pinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def positive(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "positive")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def power(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def prod(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "prod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def qr(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "qr")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rad2deg(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rad2deg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def real(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "real")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reciprocal(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reciprocal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reductions(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reductions")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def remainder(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "remainder")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def repeat(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "repeat")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reshape(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reshape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def right_shift(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "right_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def roll(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "roll")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def round(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "round")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rsqrt(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rsqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_add(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_nd(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def shape(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "shape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sign(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sin(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinc(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slice(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slogdet(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slogdet")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def split(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "split")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sqrt(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def square(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "square")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def squeeze(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "squeeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def stack(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "stack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def std(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "std")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def strided_slice(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "strided_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def subtract(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "subtract")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_sum(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def svd(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "svd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def swapaxes(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "swapaxes")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def take(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "take")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tan(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tanh(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tensordot(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tile(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tile")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def transpose(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "transpose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tril(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tril")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def triu(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "triu")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def trunc(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "trunc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unary(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsqueeze(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unsqueeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unstack(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unstack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def update_slice(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "update_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def variance(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "variance")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def vdot(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "vdot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def where(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "where")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros_like(*args, **kwargs):
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pow(x, y):
        return _wrap(_ops.power(_to_tensor(x), _to_tensor(y)))


from . import data
from . import nn
