"""zero_tensorflow API."""

import functools
from typing import Any, Optional
from ml_switcheroo.core.tensor_utils import to_array, to_dtype
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
    """
    Convert input to a Tensor.

    Args:
        x (Any): Input data.
        dtype (Optional[Any]): Target data type.

    Returns:
        ml_switcheroo.Tensor: The converted tensor.
    """
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
        except Exception:  # pragma: no cover
            pass  # pragma: no cover
        return ml_switcheroo.Tensor(
            data=x,
            shape=x.shape,
            dtype=dt,
            device=config.default_device,
        )

    arr = to_array(x, copy=True)
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
    except Exception:  # pragma: no cover
        pass  # pragma: no cover

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
    """
    Wrap input into a Tensor or collection of Tensors.

    Args:
        x (Any): Input data.

    Returns:
        Any: Wrapped tensor or collection.
    """
    if isinstance(x, Tensor):
        return x
    if isinstance(x, tuple):
        return tuple(_wrap(i) for i in x)
    if isinstance(x, list):
        return list(_wrap(i) for i in x)
    return Tensor(x)


class Tensor:
    """
    Dual-state Tensor Primitive (Eager + Traced LogicalNode).

    Args:
        value: The value to initialize the tensor with.
        dtype: The data type.
        _traced_node: Internal node.
    """

    def __init__(self, value: Any, dtype=None, _traced_node=None):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
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
        """
        Get the shape of the tensor.

        Args:
            None

        Returns:
            Any: The shape value.
        """
        return self._tensor.shape if self._tensor is not None else ()

    @property
    def dtype(self):
        """
        Get the dtype of the tensor.

        Args:
            None

        Returns:
            Any: The dtype value.
        """
        return to_dtype(self._tensor.dtype.value) if self._tensor is not None else None

    def numpy(self):
        """
        Convert the tensor to a array.

        Args:
            None

        Returns:
            ndarray: The array representation.
        """
        if hasattr(self._tensor.data, "id"):
            from ml_switcheroo.tracing import _tracer

            current_graph = getattr(_tracer, "active_graph", None)
            if current_graph and self._tensor.data.id in current_graph.nodes:
                node = current_graph.nodes[self._tensor.data.id]
                if node.op_type == "Constant":
                    return to_array(node.attributes["value"])
            raise ValueError("Cannot call array conversion on a traced tensor")
        return to_array(self._tensor.data)

    def __add__(self, other):
        """
        Compute add operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.add(_to_tensor(self), _to_tensor(other)))

    def __sub__(self, other):
        """
        Compute sub operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.subtract(_to_tensor(self), _to_tensor(other)))

    def __mul__(self, other):
        """
        Compute mul operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.multiply(_to_tensor(self), _to_tensor(other)))

    def __truediv__(self, other):
        """
        Compute truediv operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.divide(_to_tensor(self), _to_tensor(other)))

    def __radd__(self, other):
        """
        Compute radd operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.add(_to_tensor(other), _to_tensor(self)))

    def __rsub__(self, other):
        """
        Compute rsub operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.subtract(_to_tensor(other), _to_tensor(self)))

    def __rmul__(self, other):
        """
        Compute rmul operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.multiply(_to_tensor(other), _to_tensor(self)))

    def __rtruediv__(self, other):
        """
        Compute rtruediv operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.divide(_to_tensor(other), _to_tensor(self)))

    def __eq__(self, other):
        """
        Compute eq operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.equal(_to_tensor(self), _to_tensor(other)))

    def __ne__(self, other):
        """
        Compute ne operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.not_equal(_to_tensor(self), _to_tensor(other)))

    def __lt__(self, other):
        """
        Compute lt operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.less(_to_tensor(self), _to_tensor(other)))

    def __le__(self, other):
        """
        Compute le operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.less_equal(_to_tensor(self), _to_tensor(other)))

    def __gt__(self, other):
        """
        Compute gt operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.greater(_to_tensor(self), _to_tensor(other)))

    def __ge__(self, other):
        """
        Compute ge operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return _wrap(_ops.greater_equal(_to_tensor(self), _to_tensor(other)))

    def __bool__(self):
        """
        Compute bool operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
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
        """
        Compute nonzero operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return self.__bool__()

    def __len__(self):
        """
        Compute len operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        return self.shape[0] if len(self.shape) > 0 else 0


class Variable(Tensor):
    """
    A mutable Tensor.

    Args:
        initial_value: The initial value.
        trainable: Whether the variable is trainable.
        name: Variable name.
        dtype: Data type.
        shape: Tensor shape.
    """

    def __init__(self, initial_value, trainable=True):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(initial_value)
        self.trainable = trainable

    @property
    def value(self):
        """
        Apply value operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the value operation.
        """
        return self

    def assign(self, value):
        """
        Apply assign operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the assign operation.
        """
        self._tensor = _to_tensor(value)
        return self

    def assign_add(self, delta):
        """
        Apply assign_add operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the assign_add operation.
        """
        self._tensor = _ops.add(self._tensor, _to_tensor(delta))
        return self

    def assign_sub(self, delta):
        """
        Apply assign_sub operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the assign_sub operation.
        """
        self._tensor = _ops.subtract(self._tensor, _to_tensor(delta))
        return self


class _TracingContext:
    """
    Apply _TracingContext operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the _TracingContext operation.
    """

    _current_context = None

    def __init__(self):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        pass  # pragma: no cover

    @classmethod
    def enter(cls):
        """
        Apply enter operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the enter operation.
        """
        pass  # pragma: no cover

    @classmethod
    def exit(cls):
        """
        Apply exit operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the exit operation.
        """
        pass  # pragma: no cover

    @classmethod
    def get(cls):
        """
        Apply get operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the get operation.
        """
        pass  # pragma: no cover


def function(func):
    """
    Apply function operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the function operation.
    """

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        """
        Apply wrapped operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the wrapped operation.
        """

        def _to_tensor_if_possible(x):
            """
            Apply _to_tensor_if_possible operation.

            Args:
                *args: Variable length argument list.
                **kwargs: Arbitrary keyword arguments.

            Returns:
                Tensor: The result of the _to_tensor_if_possible operation.
            """
            if type(x).__name__ == "ndarray" or isinstance(
                x, (int, float, list, Tensor, Variable)
            ):
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
    """
    Record operations for automatic differentiation.

    Args:
        persistent: Whether tape is persistent.
        watch_accessed_variables: Whether to auto-watch variables.
    """

    def __init__(self, persistent=False):
        """
        Initialize the object.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.persistent = persistent
        self.watched = []
        self._tape = None

    def __enter__(self):
        """
        Compute enter operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
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
        """
        Compute exit operation.

        Args:
            other: The other operand.

        Returns:
            Tensor: The result of the operation.
        """
        from ml_switcheroo.tracing import _tracer

        _tracer.active_graph = self._prev_tracer_graph
        _tracer.is_tracing = self._prev_is_tracing

    def watch(self, tensor):
        """
        Apply watch operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the watch operation.
        """
        self.watched.append(tensor)
        if isinstance(tensor, Tensor):
            # Evaluate it so that it gets a node ID assigned to the current graph
            _to_tensor(tensor)

    def gradient(self, target, sources):
        """
        Apply gradient operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the gradient operation.
        """
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
    """
    Apply math operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the math operation.
    """

    @staticmethod
    def abs(*args, **kwargs):
        """
        Apply abs operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the abs operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "abs")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acos(*args, **kwargs):
        """
        Apply acos operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the acos operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def acosh(*args, **kwargs):
        """
        Apply acosh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the acosh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "acosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def add(*args, **kwargs):
        """
        Apply add operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the add operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def all(*args, **kwargs):
        """
        Apply all operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the all operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "all")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def allclose(*args, **kwargs):
        """
        Apply allclose operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the allclose operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "allclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def any(*args, **kwargs):
        """
        Apply any operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the any operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "any")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def arange(*args, **kwargs):
        """
        Apply arange operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the arange operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "arange")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmax(*args, **kwargs):
        """
        Apply argmax operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the argmax operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def argmin(*args, **kwargs):
        """
        Apply argmin operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the argmin operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "argmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asin(*args, **kwargs):
        """
        Apply asin operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the asin operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def asinh(*args, **kwargs):
        """
        Apply asinh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the asinh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "asinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan(*args, **kwargs):
        """
        Apply atan operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the atan operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atan2(*args, **kwargs):
        """
        Apply atan2 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the atan2 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atan2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def atanh(*args, **kwargs):
        """
        Apply atanh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the atanh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "atanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def binary(*args, **kwargs):
        """
        Apply binary operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the binary operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "binary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitcast(*args, **kwargs):
        """
        Apply bitcast operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitcast operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitcast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_and(*args, **kwargs):
        """
        Apply bitwise_and operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitwise_and operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_not(*args, **kwargs):
        """
        Apply bitwise_not operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitwise_not operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_or(*args, **kwargs):
        """
        Apply bitwise_or operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitwise_or operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bitwise_xor(*args, **kwargs):
        """
        Apply bitwise_xor operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitwise_xor operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "bitwise_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def broadcast_to(*args, **kwargs):
        """
        Apply broadcast_to operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the broadcast_to operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "broadcast_to")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cast(*args, **kwargs):
        """
        Apply cast operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the cast operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cast")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cbrt(*args, **kwargs):
        """
        Apply cbrt operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the cbrt operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cbrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ceil(*args, **kwargs):
        """
        Apply ceil operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the ceil operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ceil")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cholesky(*args, **kwargs):
        """
        Apply cholesky operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the cholesky operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cholesky")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def concatenate(*args, **kwargs):
        """
        Apply concatenate operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the concatenate operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "concatenate")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def conj(*args, **kwargs):
        """
        Apply conj operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the conj operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "conj")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def copysign(*args, **kwargs):
        """
        Apply copysign operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the copysign operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "copysign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cos(*args, **kwargs):
        """
        Apply cos operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the cos operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cos")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cosh(*args, **kwargs):
        """
        Apply cosh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the cosh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "cosh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def count_nonzero(*args, **kwargs):
        """
        Apply count_nonzero operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the count_nonzero operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "count_nonzero")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def creation(*args, **kwargs):
        """
        Apply creation operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the creation operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "creation")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def deg2rad(*args, **kwargs):
        """
        Apply deg2rad operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the deg2rad operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "deg2rad")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def det(*args, **kwargs):
        """
        Apply det operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the det operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "det")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def diag(*args, **kwargs):
        """
        Apply diag operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the diag operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "diag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def digamma(*args, **kwargs):
        """
        Apply digamma operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the digamma operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "digamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divide(*args, **kwargs):
        """
        Apply divide operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the divide operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def divmod(*args, **kwargs):
        """
        Apply divmod operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the divmod operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "divmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dot(*args, **kwargs):
        """
        Apply dot operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the dot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def dynamic_slice(*args, **kwargs):
        """
        Apply dynamic_slice operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the dynamic_slice operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "dynamic_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigh(*args, **kwargs):
        """
        Apply eigh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the eigh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eigvalsh(*args, **kwargs):
        """
        Apply eigvalsh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the eigvalsh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eigvalsh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def einsum(*args, **kwargs):
        """
        Apply einsum operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the einsum operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "einsum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def empty(*args, **kwargs):
        """
        Apply empty operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the empty operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "empty")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def equal(*args, **kwargs):
        """
        Apply equal operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the equal operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erf(*args, **kwargs):
        """
        Apply erf operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the erf operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfc(*args, **kwargs):
        """
        Apply erfc operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the erfc operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def erfinv(*args, **kwargs):
        """
        Apply erfinv operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the erfinv operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "erfinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp(*args, **kwargs):
        """
        Apply exp operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the exp operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def exp2(*args, **kwargs):
        """
        Apply exp2 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the exp2 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "exp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expand(*args, **kwargs):
        """
        Apply expand operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the expand operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expand")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def expm1(*args, **kwargs):
        """
        Apply expm1 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the expm1 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "expm1")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def eye(*args, **kwargs):
        """
        Apply eye operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the eye operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "eye")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fix(*args, **kwargs):
        """
        Apply fix operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the fix operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fix")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def flatten(*args, **kwargs):
        """
        Apply flatten operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the flatten operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "flatten")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def float_power(*args, **kwargs):
        """
        Apply float_power operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the float_power operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "float_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor(*args, **kwargs):
        """
        Apply floor operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the floor operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floor_divide(*args, **kwargs):
        """
        Apply floor_divide operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the floor_divide operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "floor_divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmax(*args, **kwargs):
        """
        Apply fmax operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the fmax operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmax")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmin(*args, **kwargs):
        """
        Apply fmin operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the fmin operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def fmod(*args, **kwargs):
        """
        Apply fmod operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the fmod operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "fmod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def frexp(*args, **kwargs):
        """
        Apply frexp operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the frexp operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "frexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full(*args, **kwargs):
        """
        Apply full operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the full operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def full_like(*args, **kwargs):
        """
        Apply full_like operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the full_like operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "full_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather(*args, **kwargs):
        """
        Apply gather operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the gather operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gather_nd(*args, **kwargs):
        """
        Apply gather_nd operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the gather_nd operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gather_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def gcd(*args, **kwargs):
        """
        Apply gcd operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the gcd operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "gcd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater(*args, **kwargs):
        """
        Apply greater operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the greater operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def greater_equal(*args, **kwargs):
        """
        Apply greater_equal operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the greater_equal operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "greater_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def heaviside(*args, **kwargs):
        """
        Apply heaviside operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the heaviside operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "heaviside")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def hypot(*args, **kwargs):
        """
        Apply hypot operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the hypot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "hypot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def identity(*args, **kwargs):
        """
        Apply identity operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the identity operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "identity")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def imag(*args, **kwargs):
        """
        Apply imag operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the imag operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "imag")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inner(*args, **kwargs):
        """
        Apply inner operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the inner operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inner")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def inv(*args, **kwargs):
        """
        Apply inv operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the inv operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "inv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isclose(*args, **kwargs):
        """
        Apply isclose operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the isclose operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isclose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isfinite(*args, **kwargs):
        """
        Apply isfinite operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the isfinite operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isfinite")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isinf(*args, **kwargs):
        """
        Apply isinf operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the isinf operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isinf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def isnan(*args, **kwargs):
        """
        Apply isnan operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the isnan operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "isnan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lcm(*args, **kwargs):
        """
        Apply lcm operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the lcm operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lcm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ldexp(*args, **kwargs):
        """
        Apply ldexp operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the ldexp operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ldexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def left_shift(*args, **kwargs):
        """
        Apply left_shift operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the left_shift operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "left_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less(*args, **kwargs):
        """
        Apply less operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the less operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def less_equal(*args, **kwargs):
        """
        Apply less_equal operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the less_equal operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "less_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def lgamma(*args, **kwargs):
        """
        Apply lgamma operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the lgamma operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "lgamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linalg(*args, **kwargs):
        """
        Apply linalg operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the linalg operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linalg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def linspace(*args, **kwargs):
        """
        Apply linspace operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the linspace operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "linspace")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log(*args, **kwargs):
        """
        Apply log operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the log operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log10(*args, **kwargs):
        """
        Apply log10 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the log10 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log10")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log1p(*args, **kwargs):
        """
        Apply log1p operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the log1p operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log1p")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log2(*args, **kwargs):
        """
        Apply log2 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the log2 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "log2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp(*args, **kwargs):
        """
        Apply logaddexp operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logaddexp operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logaddexp2(*args, **kwargs):
        """
        Apply logaddexp2 operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logaddexp2 operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logaddexp2")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_and(*args, **kwargs):
        """
        Apply logical_and operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logical_and operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_and")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_not(*args, **kwargs):
        """
        Apply logical_not operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logical_not operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_not")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_or(*args, **kwargs):
        """
        Apply logical_or operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logical_or operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_or")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logical_xor(*args, **kwargs):
        """
        Apply logical_xor operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logical_xor operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logical_xor")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def logsumexp(*args, **kwargs):
        """
        Apply logsumexp operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the logsumexp operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "logsumexp")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matmul(*args, **kwargs):
        """
        Apply matmul operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the matmul operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matmul")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def matrix_power(*args, **kwargs):
        """
        Apply matrix_power operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the matrix_power operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "matrix_power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_max(*args, **kwargs):
        """
        Apply reduce_max operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reduce_max operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "max")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def maximum(*args, **kwargs):
        """
        Apply maximum operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the maximum operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "maximum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_mean(*args, **kwargs):
        """
        Apply reduce_mean operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reduce_mean operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mean")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def meshgrid(*args, **kwargs):
        """
        Apply meshgrid operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the meshgrid operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "meshgrid")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_min(*args, **kwargs):
        """
        Apply reduce_min operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reduce_min operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "min")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def minimum(*args, **kwargs):
        """
        Apply minimum operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the minimum operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "minimum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def mod(*args, **kwargs):
        """
        Apply mod operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the mod operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "mod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def moveaxis(*args, **kwargs):
        """
        Apply moveaxis operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the moveaxis operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "moveaxis")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def multiply(*args, **kwargs):
        """
        Apply multiply operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the multiply operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "multiply")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def negative(*args, **kwargs):
        """
        Apply negative operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the negative operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "negative")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def nextafter(*args, **kwargs):
        """
        Apply nextafter operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the nextafter operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "nextafter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def norm(*args, **kwargs):
        """
        Apply norm operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the norm operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "norm")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def not_equal(*args, **kwargs):
        """
        Apply not_equal operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the not_equal operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "not_equal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones(*args, **kwargs):
        """
        Apply ones operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the ones operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def ones_like(*args, **kwargs):
        """
        Apply ones_like operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the ones_like operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "ones_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def outer(*args, **kwargs):
        """
        Apply outer operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the outer operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "outer")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def permute(*args, **kwargs):
        """
        Apply permute operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the permute operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "permute")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pinv(*args, **kwargs):
        """
        Apply pinv operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the pinv operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "pinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def positive(*args, **kwargs):
        """
        Apply positive operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the positive operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "positive")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def power(*args, **kwargs):
        """
        Apply power operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the power operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "power")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def prod(*args, **kwargs):
        """
        Apply prod operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the prod operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "prod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def qr(*args, **kwargs):
        """
        Apply qr operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the qr operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "qr")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rad2deg(*args, **kwargs):
        """
        Apply rad2deg operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the rad2deg operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rad2deg")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def real(*args, **kwargs):
        """
        Apply real operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the real operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "real")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reciprocal(*args, **kwargs):
        """
        Apply reciprocal operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reciprocal operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reciprocal")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reductions(*args, **kwargs):
        """
        Apply reductions operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reductions operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reductions")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def remainder(*args, **kwargs):
        """
        Apply remainder operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the remainder operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "remainder")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def repeat(*args, **kwargs):
        """
        Apply repeat operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the repeat operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "repeat")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reshape(*args, **kwargs):
        """
        Apply reshape operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reshape operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "reshape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def right_shift(*args, **kwargs):
        """
        Apply right_shift operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the right_shift operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "right_shift")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def roll(*args, **kwargs):
        """
        Apply roll operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the roll operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "roll")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def round(*args, **kwargs):
        """
        Apply round operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the round operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "round")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def rsqrt(*args, **kwargs):
        """
        Apply rsqrt operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the rsqrt operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "rsqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter(*args, **kwargs):
        """
        Apply scatter operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the scatter operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_add(*args, **kwargs):
        """
        Apply scatter_add operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the scatter_add operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_add")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scatter_nd(*args, **kwargs):
        """
        Apply scatter_nd operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the scatter_nd operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "scatter_nd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def shape(*args, **kwargs):
        """
        Get the shape of the tensor.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The shape value.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "shape")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sign(*args, **kwargs):
        """
        Apply sign operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the sign operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sign")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sin(*args, **kwargs):
        """
        Apply sin operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the sin operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sin")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinc(*args, **kwargs):
        """
        Apply sinc operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the sinc operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sinh(*args, **kwargs):
        """
        Apply sinh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the sinh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sinh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slice(*args, **kwargs):
        """
        Apply slice operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the slice operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def slogdet(*args, **kwargs):
        """
        Apply slogdet operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the slogdet operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "slogdet")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def split(*args, **kwargs):
        """
        Apply split operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the split operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "split")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sqrt(*args, **kwargs):
        """
        Apply sqrt operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the sqrt operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sqrt")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def square(*args, **kwargs):
        """
        Apply square operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the square operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "square")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def squeeze(*args, **kwargs):
        """
        Apply squeeze operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the squeeze operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "squeeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def stack(*args, **kwargs):
        """
        Apply stack operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the stack operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "stack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def std(*args, **kwargs):
        """
        Apply std operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the std operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "std")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def strided_slice(*args, **kwargs):
        """
        Apply strided_slice operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the strided_slice operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "strided_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def subtract(*args, **kwargs):
        """
        Apply subtract operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the subtract operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "subtract")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_sum(*args, **kwargs):
        """
        Apply reduce_sum operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the reduce_sum operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "sum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def svd(*args, **kwargs):
        """
        Apply svd operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the svd operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "svd")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def swapaxes(*args, **kwargs):
        """
        Apply swapaxes operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the swapaxes operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "swapaxes")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def take(*args, **kwargs):
        """
        Apply take operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the take operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "take")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tan(*args, **kwargs):
        """
        Apply tan operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tan operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tanh(*args, **kwargs):
        """
        Apply tanh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tanh operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tanh")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tensordot(*args, **kwargs):
        """
        Apply tensordot operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tensordot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tile(*args, **kwargs):
        """
        Apply tile operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tile operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tile")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def transpose(*args, **kwargs):
        """
        Apply transpose operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the transpose operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "transpose")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tril(*args, **kwargs):
        """
        Apply tril operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tril operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tril")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def triu(*args, **kwargs):
        """
        Apply triu operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the triu operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "triu")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def trunc(*args, **kwargs):
        """
        Apply trunc operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the trunc operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "trunc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unary(*args, **kwargs):
        """
        Apply unary operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the unary operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unary")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsqueeze(*args, **kwargs):
        """
        Apply unsqueeze operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the unsqueeze operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unsqueeze")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unstack(*args, **kwargs):
        """
        Apply unstack operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the unstack operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "unstack")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def update_slice(*args, **kwargs):
        """
        Apply update_slice operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the update_slice operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "update_slice")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def variance(*args, **kwargs):
        """
        Apply variance operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the variance operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "variance")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def vdot(*args, **kwargs):
        """
        Apply vdot operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the vdot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "vdot")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def where(*args, **kwargs):
        """
        Apply where operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the where operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "where")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros(*args, **kwargs):
        """
        Apply zeros operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the zeros operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zeros_like(*args, **kwargs):
        """
        Apply zeros_like operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the zeros_like operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "zeros_like")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pow(x, y):
        """
        Apply pow operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the pow operation.
        """
        return _wrap(_ops.power(_to_tensor(x), _to_tensor(y)))


from . import data
from . import nn
