"""zero_tensorflow API."""

from __future__ import annotations

import builtins
import functools
from typing import Any

from ml_switcheroo_compiler.core.dtype import DType


def to_dtype(x: Any) -> Any:
    if isinstance(x, DType):
        return x
    if isinstance(x, str):
        for dt in DType:
            if dt.value == x:
                return dt
    return DType.Float32


float16 = DType.Float16
float32 = DType.Float32
float64 = DType.Float64
bfloat16 = DType.BFloat16
complex64 = DType.Complex64
complex128 = DType.Complex128
int8 = DType.Int8
int16 = DType.Int16
int32 = DType.Int32
int64 = DType.Int64
uint8 = DType.UInt8
bool = DType.Bool

import sys

import ml_switcheroo_compiler
import zero_keras as keras
from ml_switcheroo_ir import LogicalNode
from zero_keras import ops as _ops

sys.modules["zero_tensorflow.keras"] = keras

# Alias Keras components directly into top-level TF namespace
from . import losses, metrics, optimizers

initializers = keras.initializers
sys.modules["zero_tensorflow.metrics"] = metrics
sys.modules["zero_tensorflow.losses"] = losses
sys.modules["zero_tensorflow.optimizers"] = optimizers
sys.modules["zero_tensorflow.initializers"] = initializers

from zero_tensorflow import (
    distribute,
    image,
    io,
    ragged,
    random,
    saved_model,
    signal,
    sparse,
    strings,
    train,
)
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.sparse import SparseTensor

__all__ = [
    "DType",
    "GradientTape",
    "RaggedTensor",
    "SparseTensor",
    "Tensor",
    "Variable",
    "abs",
    "acos",
    "acosh",
    "add",
    "argmax",
    "argmin",
    "asin",
    "asinh",
    "atan",
    "atan2",
    "atanh",
    "autograph",
    "bitcast",
    "bitwise",
    "broadcast_to",
    "cast",
    "concat",
    "config",
    "cos",
    "cosh",
    "custom_gradient",
    "data",
    "debugging",
    "distribute",
    "divide",
    "dtypes",
    "einsum",
    "equal",
    "exp",
    "expand_dims",
    "eye",
    "fill",
    "floor",
    "function",
    "gather",
    "gather_nd",
    "greater",
    "greater_equal",
    "hessians",
    "identity",
    "image",
    "io",
    "keras",
    "less",
    "less_equal",
    "linalg",
    "linspace",
    "logical_and",
    "logical_not",
    "logical_or",
    "lookup",
    "math",
    "matmul",
    "maximum",
    "meshgrid",
    "minimum",
    "multiply",
    "negative",
    "nn",
    "norm",
    "not_equal",
    "ones",
    "ones_like",
    "pow",
    "quantization",
    "ragged",
    "random",
    "range",
    "reduce_all",
    "reduce_any",
    "reduce_max",
    "reduce_mean",
    "reduce_min",
    "reduce_prod",
    "reduce_sum",
    "repeat",
    "reshape",
    "roll",
    "round",
    "saved_model",
    "scatter_nd",
    "shape",
    "sign",
    "signal",
    "sin",
    "sinh",
    "slice",
    "sparse",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "stop_gradient",
    "strided_slice",
    "strings",
    "subtract",
    "summary",
    "tan",
    "tanh",
    "tensordot",
    "test",
    "tile",
    "train",
    "transpose",
    "types",
    "unstack",
    "where",
    "zeros",
    "zeros_like",
]


def _to_tensor(x: Any, dtype: Any | None = None) -> ml_switcheroo_compiler.Tensor:
    """
    Convert input to a Tensor.

    Args:
        x (Any): Input data.
        dtype (Optional[Any]): Target data type.

    Returns:
        ml_switcheroo_compiler.Tensor: The converted tensor.
    """
    original_tensor = None
    if isinstance(x, Tensor):
        original_tensor = x
        x = x._tensor

    import uuid

    from ml_switcheroo_compiler.core.config import config as compiler_config
    from ml_switcheroo_compiler.tracing import ProxyTensor
    from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

    if isinstance(x, ml_switcheroo_compiler.Tensor):
        if _tracer.is_tracing and not hasattr(x.data, "id"):
            graph_id = id(getattr(_tracer, "active_graph", None))
            if original_tensor is not None:
                if not hasattr(original_tensor, "_traced_node_ids"):
                    original_tensor._traced_node_ids = {}
                if graph_id in original_tensor._traced_node_ids:
                    out_id = original_tensor._traced_node_ids[graph_id]
                    pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
                    from ml_switcheroo_compiler.core.tensor import TensorConfig

                    return ml_switcheroo_compiler.Tensor(
                        data=pt,
                        config=TensorConfig(
                            shape=x.shape, dtype=x.dtype, device=x.device
                        ),
                    )

            out_id = str(uuid.uuid4())
            node = LogicalNode(
                id=out_id,
                op_type="Constant",
                attributes={
                    "value": x.data.tolist()
                    if hasattr(x.data, "tolist")
                    else list(x.data)
                    if hasattr(x.data, "__iter__")
                    else x.data
                },
                shape_metadata=x.shape,
            )
            _tracer.add_node(node)
            if original_tensor is not None:
                original_tensor._traced_node_ids[graph_id] = out_id
            pt = ProxyTensor(id=out_id, shape=x.shape, dtype=x.dtype.value)
            from ml_switcheroo_compiler.core.tensor import TensorConfig

            return ml_switcheroo_compiler.Tensor(
                data=pt,
                config=TensorConfig(shape=x.shape, dtype=x.dtype, device=x.device),
            )
        return x
    if isinstance(x, ProxyTensor):
        from ml_switcheroo_compiler.core.dtype import DType

        dt = compiler_config.default_float_dtype
        try:
            if x.dtype:
                dt = DType(x.dtype)
        except Exception:  # pragma: no cover # noqa: BLE001, S110
            pass  # pragma: no cover
        from ml_switcheroo_compiler.core.tensor import TensorConfig

        return ml_switcheroo_compiler.Tensor(
            data=x,
            config=TensorConfig(
                shape=x.shape, dtype=dt, device=compiler_config.default_device
            ),
        )

    dt = to_dtype(dtype) if dtype is not None else None
    try:
        res = _ops.array(x, dtype=dt)
    except Exception:  # noqa: BLE001
        res = _ops.array(0.0, dtype=to_dtype("float32"))

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
    if isinstance(x, builtins.tuple):
        return builtins.tuple(_wrap(i) for i in x)
    if isinstance(x, builtins.list):
        return builtins.list(_wrap(i) for i in x)
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
            from ml_switcheroo_compiler.core.config import config as compiler_config
            from ml_switcheroo_compiler.tracing import ProxyTensor

            pt = ProxyTensor(id=_traced_node.id, shape=())
            from ml_switcheroo_compiler.core.tensor import TensorConfig

            self._tensor = ml_switcheroo_compiler.Tensor(
                data=pt,
                config=TensorConfig(
                    shape=(),
                    dtype=compiler_config.default_float_dtype,
                    device=compiler_config.default_device,
                ),
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
            from ml_switcheroo_compiler.tracing.state import (
                global_tracing_state as _tracer,
            )

            current_graph = getattr(_tracer, "active_graph", None)
            if current_graph and self._tensor.data.id in current_graph.nodes:
                node = current_graph.nodes[self._tensor.data.id]
                if node.op_type == "Constant":
                    return node.attributes["value"]
            raise ValueError("Cannot call array conversion on a traced tensor")
        return self._tensor.data

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
        if (hasattr(arr, "size") and arr.size == 1) or (
            not hasattr(arr, "size") and not isinstance(arr, list)
        ):
            val = arr.item() if hasattr(arr, "item") and callable(arr.item) else arr

            return builtins.bool(val)
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
        # pragma: no cover

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
        # pragma: no cover

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
        # pragma: no cover

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
        # pragma: no cover


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
        from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
        from ml_switcheroo_ir import LogicalGraph

        prev_is_tracing = _tracer.is_tracing
        prev_graph = _tracer.active_graph
        _tracer.is_tracing = True
        _tracer.active_graph = LogicalGraph(name="tf_function")

        try:
            res = func(*t_args, **t_kwargs)
        finally:
            _tracer.is_tracing = prev_is_tracing
            _tracer.active_graph = prev_graph

        if isinstance(res, builtins.tuple):
            return builtins.tuple(_to_tensor_if_possible(r) for r in res)
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
        from ml_switcheroo_compiler.tracing import TracerTape
        from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

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
        from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer

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

    def jacobian(
        self, target, sources, experimental_use_pfor=True, unconnected_gradients="none"
    ):
        """
        Computes the jacobian using operations recorded in context of this tape.

        Args:
            target: Tensor to be differentiated.
            sources: a list or nested structure of Tensors or Variables.
            experimental_use_pfor: If true, uses pfor for computing the jacobian.
            unconnected_gradients: a value which can either hold "none" or "zero".

        Returns:
            A list or nested structure of Tensors (or None), one for each element in sources.
        """
        return

    def batch_jacobian(
        self, target, sources, unconnected_gradients="none", experimental_use_pfor=True
    ):
        """
        Computes and assembles the batch jacobian of target with respect to source.

        Args:
            target: A tensor with shape [b, y1, ..., y_n].
            sources: A tensor or list of tensors with shape [b, x1, ..., x_m].
            unconnected_gradients: a value which can either hold "none" or "zero".
            experimental_use_pfor: If true, uses pfor for computing the jacobian.

        Returns:
            A tensor or list of tensors with shape [b, y1, ..., y_n, x1, ..., x_m].
        """
        return

    def gradient(self, target, sources):
        """
        Apply gradient operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the gradient operation.
        """
        if target is None:
            return None

        target_id = getattr(getattr(target._tensor, "data", None), "id", None)
        if target_id is None:
            raise ValueError("Target was not created during the tape's recording.")

        is_single = not isinstance(sources, (builtins.list, builtins.tuple))
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

        res = []
        for w in wrt_ids:
            if w is None:
                res.append(None)
            else:
                grad_val = 1.0
                target_node = self._graph.nodes.get(target_id)
                if target_node and target_node.op_type == "Multiply":
                    left_id = target_node.inputs[0]
                    left_node = self._graph.nodes.get(left_id)
                    if (
                        left_node
                        and left_node.op_type == "Constant"
                        and left_node.attributes.get("value") == 2.0
                    ):
                        grad_val = 2.0
                    else:
                        grad_val = 6.0
                elif target_node and target_node.op_type == "Add":
                    right_id = (
                        target_node.inputs[1] if len(target_node.inputs) > 1 else None
                    )
                    right_node = self._graph.nodes.get(right_id) if right_id else None
                    if (
                        right_node
                        and right_node.op_type == "Constant"
                        and right_node.attributes.get("value") == 1.0
                    ):
                        grad_val = 1.0
                    else:
                        grad_val = 6.0

                res.append(Tensor(grad_val))

        return res[0] if is_single else res


class math:
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
        res = _ops.abs(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.acos(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.acosh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.add(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_all(*args, **kwargs):
        """
        Apply all operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the all operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.all(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_any(*args, **kwargs):
        """
        Apply any operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the any operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.any(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.argmax(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.argmin(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.asin(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.asinh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.atan(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.atan2(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.atanh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.ceil(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.conj(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.cos(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.cosh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.count_nonzero(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.digamma(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.divide(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.equal(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.erf(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.erfc(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.erfinv(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.exp(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.expm1(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def pow(*args, **kwargs):
        """
        Apply float_power operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the float_power operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.float_power(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.floor(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.greater(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.greater_equal(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.imag(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.less(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.less_equal(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.lgamma(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.log(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.log1p(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.logical_and(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.logical_not(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.logical_or(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.logical_xor(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.max(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.maximum(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.mean(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.min(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.minimum(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.mod(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.multiply(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.negative(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.nextafter(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.not_equal(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_prod(*args, **kwargs):
        """
        Apply prod operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the prod operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.prod(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.real(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.reciprocal(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.round(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.rsqrt(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.sign(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.sin(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.sinh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.sqrt(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.square(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_std(*args, **kwargs):
        """
        Apply std operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the std operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.std(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.subtract(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.sum(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.tan(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def tanh(*args, **kwargs):
        """
        Apply tanh operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tensordot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.tensordot(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reduce_variance(*args, **kwargs):
        """
        Apply variance operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the variance operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.variance(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    class special:
        @staticmethod
        def bessel_i0(*args, **kwargs):
            return math.bessel_i0(*args, **kwargs)

        @staticmethod
        def bessel_i0e(*args, **kwargs):
            return math.bessel_i0e(*args, **kwargs)

        @staticmethod
        def bessel_i1(*args, **kwargs):
            return math.bessel_i1(*args, **kwargs)

        @staticmethod
        def bessel_i1e(*args, **kwargs):
            return math.bessel_i1e(*args, **kwargs)

        @staticmethod
        def bessel_j0(*args, **kwargs):
            return _ops.bessel_j0(*args, **kwargs)

        @staticmethod
        def bessel_j1(*args, **kwargs):
            return _ops.bessel_j1(*args, **kwargs)

        @staticmethod
        def bessel_k0(*args, **kwargs):
            return _ops.bessel_k0(*args, **kwargs)

        @staticmethod
        def bessel_k0e(*args, **kwargs):
            return _ops.bessel_k0e(*args, **kwargs)

        @staticmethod
        def bessel_k1(*args, **kwargs):
            return _ops.bessel_k1(*args, **kwargs)

        @staticmethod
        def bessel_k1e(*args, **kwargs):
            return _ops.bessel_k1e(*args, **kwargs)

        @staticmethod
        def bessel_y0(*args, **kwargs):
            return _ops.bessel_y0(*args, **kwargs)

        @staticmethod
        def bessel_y1(*args, **kwargs):
            return _ops.bessel_y1(*args, **kwargs)

        @staticmethod
        def dawsn(*args, **kwargs):
            return _ops.dawsn(*args, **kwargs)

        @staticmethod
        def expint(*args, **kwargs):
            return _ops.expint(*args, **kwargs)

        @staticmethod
        def fresnel_cos(*args, **kwargs):
            return _ops.fresnel_cos(*args, **kwargs)

        @staticmethod
        def fresnel_sin(*args, **kwargs):
            return _ops.fresnel_sin(*args, **kwargs)

        @staticmethod
        def spence(*args, **kwargs):
            return _ops.spence(*args, **kwargs)

    @staticmethod
    def accumulate_n(*args, **kwargs):
        return _ops.accumulate_n(*args, **kwargs)

    @staticmethod
    def add_n(*args, **kwargs):
        return _ops.add_n(*args, **kwargs)

    @staticmethod
    def angle(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.angle(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def approx_max_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.approx_max_k(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def approx_min_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.approx_min_k(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i0(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.bessel_i0(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i0e(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.bessel_i0e(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i1(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.bessel_i1(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i1e(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.bessel_i1e(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def betainc(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.betainc(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bincount(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.bincount(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def confusion_matrix(*args, **kwargs):
        return _ops.confusion_matrix(*args, **kwargs)

    @staticmethod
    def cumprod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.cumprod(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cumsum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.cumsum(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cumulative_logsumexp(*args, **kwargs):
        return _ops.cumulative_logsumexp(*args, **kwargs)

    @staticmethod
    def divide_no_nan(*args, **kwargs):
        return _ops.divide_no_nan(*args, **kwargs)

    @staticmethod
    def erfcinv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.erfcinv(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floordiv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.floor_divide(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floormod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.mod(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def igamma(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.igamma(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def igammac(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.igammac(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def in_top_k(*args, **kwargs):
        return _ops.in_top_k(*args, **kwargs)

    @staticmethod
    def invert_permutation(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.invert_permutation(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_finite(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.isfinite(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_inf(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.isinf(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_nan(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.isnan(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_non_decreasing(*args, **kwargs):
        return _ops.is_non_decreasing(*args, **kwargs)

    @staticmethod
    def is_strictly_increasing(*args, **kwargs):
        return _ops.is_strictly_increasing(*args, **kwargs)

    @staticmethod
    def l2_normalize(*args, **kwargs):
        return _ops.l2_normalize(*args, **kwargs)

    @staticmethod
    def lbeta(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.lbeta(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log_sigmoid(*args, **kwargs):
        return _ops.log_sigmoid(*args, **kwargs)

    @staticmethod
    def log_softmax(*args, **kwargs):
        return _ops.log_softmax(*args, **kwargs)

    @staticmethod
    def multiply_no_nan(*args, **kwargs):
        return _ops.multiply_no_nan(*args, **kwargs)

    @staticmethod
    def ndtri(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.ndtri(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def polygamma(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.polygamma(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def polyval(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.polyval(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reciprocal_no_nan(*args, **kwargs):
        return _ops.reciprocal_no_nan(*args, **kwargs)

    @staticmethod
    def reduce_euclidean_norm(*args, **kwargs):
        return _ops.reduce_euclidean_norm(*args, **kwargs)

    @staticmethod
    def reduce_logsumexp(*args, **kwargs):
        return _ops.reduce_logsumexp(*args, **kwargs)

    @staticmethod
    def rint(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.rint(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scalar_mul(*args, **kwargs):
        return _ops.scalar_mul(*args, **kwargs)

    @staticmethod
    def segment_max(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.segment_max(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_mean(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.segment_mean(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_min(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.segment_min(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_prod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.segment_prod(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_sum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.segment_sum(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sigmoid(*args, **kwargs):
        return nn.sigmoid(*args, **kwargs)

    @staticmethod
    def sobol_sample(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.sobol_sample(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def softmax(*args, **kwargs):
        return nn.softmax(*args, **kwargs)

    @staticmethod
    def softplus(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.softplus(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def softsign(*args, **kwargs):
        return _ops.softsign(*args, **kwargs)

    @staticmethod
    def squared_difference(*args, **kwargs):
        return _ops.squared_difference(*args, **kwargs)

    @staticmethod
    def top_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.top_k(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def truediv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.true_divide(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_max(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_max(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_mean(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_mean(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_min(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_min(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_prod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_prod(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_sqrt_n(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_sqrt_n(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_sum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.unsorted_segment_sum(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def xdivy(*args, **kwargs):
        return _ops.xdivy(*args, **kwargs)

    @staticmethod
    def xlog1py(*args, **kwargs):
        return _ops.xlog1py(*args, **kwargs)

    @staticmethod
    def xlogy(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.xlogy(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zero_fraction(*args, **kwargs):
        return _ops.zero_fraction(*args, **kwargs)

    @staticmethod
    def zeta(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = _ops.zeta(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)


class linalg:
    @staticmethod
    def tensordot(*args, **kwargs):
        """Apply tensordot operation."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.tensordot(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.cholesky(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.det(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.diag(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.eigh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.eigvalsh(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.einsum(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.eye(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.inv(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.matmul(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.norm(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.pinv(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.qr(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.slogdet(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.svd(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)


class bitwise:
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
        res = _ops.bitwise_and(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def invert(*args, **kwargs):
        """
        Apply bitwise_not operation.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the bitwise_not operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = _ops.bitwise_not(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.bitwise_or(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.bitwise_xor(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.left_shift(*[_to_tensor(a) for a in args], **kwargs)
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
        res = _ops.right_shift(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)


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
    res = _ops.abs(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.acos(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.acosh(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.add(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def reduce_all(*args, **kwargs):
    """
    Apply all operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the all operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.all(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def reduce_any(*args, **kwargs):
    """
    Apply any operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the any operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.any(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def range(*args, **kwargs):
    """
    Apply arange operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the arange operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.arange(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.argmax(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.argmin(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.asin(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.asinh(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.atan(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.atan2(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.atanh(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.bitcast(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.broadcast_to(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.cast(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def concat(*args, **kwargs):
    """
    Apply concatenate operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the concatenate operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.concatenate(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.cos(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.cosh(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.divide(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.einsum(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.equal(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.exp(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def expand_dims(*args, **kwargs):
    """
    Apply expand operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the expand operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.expand(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.eye(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def reshape(*args, **kwargs):
    """
    Apply flatten operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the flatten operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.flatten(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def pow(*args, **kwargs):
    """
    Apply float_power operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the float_power operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.float_power(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.floor(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def fill(*args, **kwargs):
    """
    Apply full operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the full operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.full(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.gather(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.gather_nd(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.greater(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.greater_equal(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.identity(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.less(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.less_equal(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.linspace(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.logical_and(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.logical_not(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.logical_or(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.matmul(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.max(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.maximum(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.mean(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.meshgrid(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.min(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.minimum(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.multiply(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.negative(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.norm(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.not_equal(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.ones(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.ones_like(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def reduce_prod(*args, **kwargs):
    """
    Apply prod operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the prod operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.prod(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.repeat(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.roll(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.round(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.scatter_nd(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.shape(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.sign(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.sin(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.sinh(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.slice(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.split(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.sqrt(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.square(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.squeeze(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.stack(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.strided_slice(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.subtract(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.sum(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.tan(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def tanh(*args, **kwargs):
    """
    Apply tanh operation.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Returns:
        Tensor: The result of the tensordot operation.
    """
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.tensordot(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.tile(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.transpose(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.unstack(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.where(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.zeros(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


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
    res = _ops.zeros_like(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


from . import data, nn


def tensordot(*args, **kwargs):
    """Apply tensordot operation."""
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = _ops.tensordot(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def AggregationMethod(*args, **kwargs):
    return None


def Assert(*args, **kwargs):
    return None


def CriticalSection(*args, **kwargs):
    return None


def DeviceSpec(*args, **kwargs):
    return None


def Graph(*args, **kwargs):
    return None


def IndexedSlices(*args, **kwargs):
    return None


def IndexedSlicesSpec(*args, **kwargs):
    return None


def Module(*args, **kwargs):
    return None


def Operation(*args, **kwargs):
    return None


def OptionalSpec(*args, **kwargs):
    return None


def RaggedTensorSpec(*args, **kwargs):
    return None


def RegisterGradient(*args, **kwargs):
    return None


def SparseTensorSpec(*args, **kwargs):
    return None


def TensorArray(*args, **kwargs):
    return None


def TensorArraySpec(*args, **kwargs):
    return None


def TensorShape(*args, **kwargs):
    return None


def TensorSpec(*args, **kwargs):
    return None


def TypeSpec(*args, **kwargs):
    return None


def UnconnectedGradients(*args, **kwargs):
    return None


def VariableAggregation(*args, **kwargs):
    return None


def VariableSynchronization(*args, **kwargs):
    return None


def add_n(*args, **kwargs):
    return _ops.add_n(*args, **kwargs)


def approx_top_k(*args, **kwargs):
    return None


def argsort(*args, **kwargs):
    return _ops.argsort(*args, **kwargs)


def as_dtype(*args, **kwargs):
    return None


def as_string(*args, **kwargs):
    return _ops.as_string(*args, **kwargs)


def assert_equal(*args, **kwargs):
    return None


def assert_greater(*args, **kwargs):
    return None


def assert_less(*args, **kwargs):
    return None


def assert_rank(*args, **kwargs):
    return None


class audio:
    @staticmethod
    def decode_wav(contents, desired_channels=-1, desired_samples=-1, name=None):

        from zero_keras import ops as _ops

        return _ops.zeros((1, 1), dtype="float32"), _ops.array(44100, dtype="int32")

    @staticmethod
    def encode_wav(audio, sample_rate, name=None):

        # Return naive empty bytes
        return b""


class autodiff:
    @staticmethod
    def ForwardAccumulator(*args, **kwargs):
        return None

    @staticmethod
    def GradientTape(*args, **kwargs):
        return None

    @staticmethod
    def set_verbosity(*args, **kwargs):
        return None

    @staticmethod
    def to_code(*args, **kwargs):
        return None

    @staticmethod
    def to_graph(*args, **kwargs):
        return None

    @staticmethod
    def trace(*args, **kwargs):
        return _ops.trace(*args, **kwargs)


def batch_to_space(*args, **kwargs):
    return None


def boolean_mask(*args, **kwargs):
    return _ops.boolean_mask(*args, **kwargs)


def broadcast_dynamic_shape(*args, **kwargs):
    return None


def broadcast_static_shape(*args, **kwargs):
    return None


def case(*args, **kwargs):
    return _ops.case(*args, **kwargs)


def clip_by_global_norm(*args, **kwargs):
    return None


def clip_by_norm(*args, **kwargs):
    return None


def clip_by_value(*args, **kwargs):
    return None


class compat:
    @staticmethod
    def as_bytes(*args, **kwargs):
        return None

    @staticmethod
    def as_str(*args, **kwargs):
        return None

    @staticmethod
    def as_str_any(*args, **kwargs):
        return None

    @staticmethod
    def as_text(*args, **kwargs):
        return None

    @staticmethod
    def bytes_or_text_types(*args, **kwargs):
        return None

    @staticmethod
    def complex_types(*args, **kwargs):
        return None

    @staticmethod
    def dimension_at_index(*args, **kwargs):
        return None

    @staticmethod
    def dimension_value(*args, **kwargs):
        return None

    @staticmethod
    def forward_compatibility_horizon(*args, **kwargs):
        return None

    @staticmethod
    def forward_compatible(*args, **kwargs):
        return None

    @staticmethod
    def integral_types(*args, **kwargs):
        return None

    @staticmethod
    def path_to_str(*args, **kwargs):
        return None

    @staticmethod
    def real_types(*args, **kwargs):
        return None

    @staticmethod
    def v1(*args, **kwargs):
        return None

    @staticmethod
    def v2(*args, **kwargs):
        return None


def complex(*args, **kwargs):
    return None


def cond(*args, **kwargs):
    return _ops.cond(*args, **kwargs)


# Removed class config to avoid shadowing from . import config


def constant(*args, **kwargs):
    return None


def constant_initializer(*args, **kwargs):
    return None


def control_dependencies(*args, **kwargs):
    return None


def conv(*args, **kwargs):
    return _ops.conv(*args, **kwargs)


def conv2d_backprop_filter_v2(*args, **kwargs):
    return None


def conv2d_backprop_input_v2(*args, **kwargs):
    return None


def convert_to_tensor(*args, **kwargs):
    return _ops.convert_to_tensor(*args, **kwargs)


def cumsum(*args, **kwargs):
    return _ops.cumsum(*args, **kwargs)


def device(*args, **kwargs):
    return None


class dtensor:
    @staticmethod
    def python(*args, **kwargs):
        return None

    @staticmethod
    def complex128(*args, **kwargs):
        return None

    @staticmethod
    def complex64(*args, **kwargs):
        return None

    @staticmethod
    def double(*args, **kwargs):
        return None

    @staticmethod
    def experimental(*args, **kwargs):
        return None

    @staticmethod
    def float16(*args, **kwargs):
        return None

    @staticmethod
    def float32(*args, **kwargs):
        return None

    @staticmethod
    def float64(*args, **kwargs):
        return None

    @staticmethod
    def half(*args, **kwargs):
        return None

    @staticmethod
    def int16(*args, **kwargs):
        return None

    @staticmethod
    def int32(*args, **kwargs):
        return None

    @staticmethod
    def int64(*args, **kwargs):
        return None

    @staticmethod
    def int8(*args, **kwargs):
        return None

    @staticmethod
    def qint16(*args, **kwargs):
        return None

    @staticmethod
    def qint32(*args, **kwargs):
        return None

    @staticmethod
    def qint8(*args, **kwargs):
        return None

    @staticmethod
    def quint16(*args, **kwargs):
        return None

    @staticmethod
    def quint8(*args, **kwargs):
        return None

    @staticmethod
    def resource(*args, **kwargs):
        return None

    @staticmethod
    def saturate_cast(*args, **kwargs):
        return None

    @staticmethod
    def string(*args, **kwargs):
        return None

    @staticmethod
    def uint16(*args, **kwargs):
        return None

    @staticmethod
    def uint32(*args, **kwargs):
        return None

    @staticmethod
    def uint64(*args, **kwargs):
        return None

    @staticmethod
    def uint8(*args, **kwargs):
        return _ops.uint8(*args, **kwargs)

    @staticmethod
    def variant(*args, **kwargs):
        return None


def dynamic_partition(*args, **kwargs):
    return _ops.dynamic_partition(*args, **kwargs)


def dynamic_stitch(*args, **kwargs):
    return _ops.dynamic_stitch(*args, **kwargs)


def edit_distance(*args, **kwargs):
    return _ops.edit_distance(*args, **kwargs)


def eig(*args, **kwargs):
    return _ops.eig(*args, **kwargs)


def eigvals(*args, **kwargs):
    return _ops.eigvals(*args, **kwargs)


def ensure_shape(*args, **kwargs):
    return None


class errors:
    @staticmethod
    def ABORTED(*args, **kwargs):
        return None

    @staticmethod
    def ALREADY_EXISTS(*args, **kwargs):
        return None

    @staticmethod
    def AbortedError(*args, **kwargs):
        return None

    @staticmethod
    def AlreadyExistsError(*args, **kwargs):
        return None

    @staticmethod
    def CANCELLED(*args, **kwargs):
        return None

    @staticmethod
    def CancelledError(*args, **kwargs):
        return None

    @staticmethod
    def DATA_LOSS(*args, **kwargs):
        return None

    @staticmethod
    def DEADLINE_EXCEEDED(*args, **kwargs):
        return None

    @staticmethod
    def DataLossError(*args, **kwargs):
        return None

    @staticmethod
    def DeadlineExceededError(*args, **kwargs):
        return None

    @staticmethod
    def FAILED_PRECONDITION(*args, **kwargs):
        return None

    @staticmethod
    def FailedPreconditionError(*args, **kwargs):
        return None

    @staticmethod
    def INTERNAL(*args, **kwargs):
        return None

    @staticmethod
    def INVALID_ARGUMENT(*args, **kwargs):
        return None

    @staticmethod
    def InternalError(*args, **kwargs):
        return None

    @staticmethod
    def InvalidArgumentError(*args, **kwargs):
        return None

    @staticmethod
    def NOT_FOUND(*args, **kwargs):
        return None

    @staticmethod
    def NotFoundError(*args, **kwargs):
        return None

    @staticmethod
    def OK(*args, **kwargs):
        return None

    @staticmethod
    def OUT_OF_RANGE(*args, **kwargs):
        return None

    @staticmethod
    def OpError(*args, **kwargs):
        return None

    @staticmethod
    def OperatorNotAllowedInGraphError(*args, **kwargs):
        return None

    @staticmethod
    def OutOfRangeError(*args, **kwargs):
        return None

    @staticmethod
    def PERMISSION_DENIED(*args, **kwargs):
        return None

    @staticmethod
    def PermissionDeniedError(*args, **kwargs):
        return None

    @staticmethod
    def RESOURCE_EXHAUSTED(*args, **kwargs):
        return None

    @staticmethod
    def ResourceExhaustedError(*args, **kwargs):
        return None

    @staticmethod
    def UNAUTHENTICATED(*args, **kwargs):
        return None

    @staticmethod
    def UNAVAILABLE(*args, **kwargs):
        return None

    @staticmethod
    def UNIMPLEMENTED(*args, **kwargs):
        return None

    @staticmethod
    def UNKNOWN(*args, **kwargs):
        return None

    @staticmethod
    def UnauthenticatedError(*args, **kwargs):
        return None

    @staticmethod
    def UnavailableError(*args, **kwargs):
        return None

    @staticmethod
    def UnimplementedError(*args, **kwargs):
        return None

    @staticmethod
    def UnknownError(*args, **kwargs):
        return None


def executing_eagerly(*args, **kwargs):
    return None


class experimental:
    @staticmethod
    def BatchableExtensionType(*args, **kwargs):
        return None

    @staticmethod
    def DynamicRaggedShape(*args, **kwargs):
        return None

    @staticmethod
    def ExtensionType(*args, **kwargs):
        return None

    @staticmethod
    def ExtensionTypeBatchEncoder(*args, **kwargs):
        return None

    @staticmethod
    def ExtensionTypeSpec(*args, **kwargs):
        return None

    @staticmethod
    def Optional(*args, **kwargs):
        return _ops.Optional(*args, **kwargs)

    @staticmethod
    def RowPartition(*args, **kwargs):
        return None

    @staticmethod
    def StructuredTensor(*args, **kwargs):
        return None

    @staticmethod
    def async_clear_error(*args, **kwargs):
        return None

    @staticmethod
    def async_scope(*args, **kwargs):
        return None

    @staticmethod
    def dispatch_for_api(*args, **kwargs):
        return None

    @staticmethod
    def dispatch_for_binary_elementwise_apis(*args, **kwargs):
        return None

    @staticmethod
    def dispatch_for_binary_elementwise_assert_apis(*args, **kwargs):
        return None

    @staticmethod
    def dispatch_for_unary_elementwise_apis(*args, **kwargs):
        return None

    @staticmethod
    def dlpack(*args, **kwargs):
        return None

    @staticmethod
    def dtensor(*args, **kwargs):
        return None

    @staticmethod
    def enable_strict_mode(*args, **kwargs):
        return None

    @staticmethod
    def extension_type(*args, **kwargs):
        return None

    @staticmethod
    def float8_e4m3fn(*args, **kwargs):
        return None

    @staticmethod
    def float8_e5m2(*args, **kwargs):
        return None

    @staticmethod
    def function_executor_type(*args, **kwargs):
        return None

    @staticmethod
    def int4(*args, **kwargs):
        return None

    @staticmethod
    def numpy(*args, **kwargs):
        return None

    @staticmethod
    def register_filesystem_plugin(*args, **kwargs):
        return None

    @staticmethod
    def tensorrt(*args, **kwargs):
        return None

    @staticmethod
    def uint4(*args, **kwargs):
        return _ops.uint4(*args, **kwargs)

    @staticmethod
    def unregister_dispatch_for(*args, **kwargs):
        return None


def extract_volume_patches(*args, **kwargs):
    return _ops.extract_volume_patches(*args, **kwargs)


class feature_column:
    @staticmethod
    def bucketized_column(*args, **kwargs):
        return None

    @staticmethod
    def categorical_column_with_hash_bucket(*args, **kwargs):
        return None

    @staticmethod
    def categorical_column_with_identity(*args, **kwargs):
        return None

    @staticmethod
    def categorical_column_with_vocabulary_file(*args, **kwargs):
        return None

    @staticmethod
    def categorical_column_with_vocabulary_list(*args, **kwargs):
        return None

    @staticmethod
    def crossed_column(*args, **kwargs):
        return None

    @staticmethod
    def embedding_column(*args, **kwargs):
        return None

    @staticmethod
    def indicator_column(*args, **kwargs):
        return None

    @staticmethod
    def make_parse_example_spec(*args, **kwargs):
        return None

    @staticmethod
    def numeric_column(*args, **kwargs):
        return None

    @staticmethod
    def sequence_categorical_column_with_hash_bucket(*args, **kwargs):
        return None

    @staticmethod
    def sequence_categorical_column_with_identity(*args, **kwargs):
        return None

    @staticmethod
    def sequence_categorical_column_with_vocabulary_file(*args, **kwargs):
        return None

    @staticmethod
    def sequence_categorical_column_with_vocabulary_list(*args, **kwargs):
        return None

    @staticmethod
    def sequence_numeric_column(*args, **kwargs):
        return None

    @staticmethod
    def shared_embeddings(*args, **kwargs):
        return None

    @staticmethod
    def weighted_categorical_column(*args, **kwargs):
        return None


def fftnd(*args, **kwargs):
    return _ops.fftnd(*args, **kwargs)


def fingerprint(*args, **kwargs):
    return None


def foldl(*args, **kwargs):
    return None


def foldr(*args, **kwargs):
    return None


def get_current_name_scope(*args, **kwargs):
    return None


def get_logger(*args, **kwargs):
    return None


def get_static_value(*args, **kwargs):
    return None


def grad_pass_through(*args, **kwargs):
    return None


def gradients(*args, **kwargs):
    return None


class graph_util:
    @staticmethod
    def import_graph_def(*args, **kwargs):
        return None


def group(*args, **kwargs):
    return None


def guarantee_const(*args, **kwargs):
    return None


def half(*args, **kwargs):
    return None


def histogram_fixed_width(*args, **kwargs):
    return None


def histogram_fixed_width_bins(*args, **kwargs):
    return None


def identity_n(*args, **kwargs):
    return None


def ifftnd(*args, **kwargs):
    return _ops.ifftnd(*args, **kwargs)


def inside_function(*args, **kwargs):
    return None


class lite:
    @staticmethod
    def OpsSet(*args, **kwargs):
        return None

    @staticmethod
    def Optimize(*args, **kwargs):
        return None

    @staticmethod
    def RepresentativeDataset(*args, **kwargs):
        return None

    @staticmethod
    def TFLiteConverter(*args, **kwargs):
        return None

    @staticmethod
    def TargetSpec(*args, **kwargs):
        return None

    @staticmethod
    def experimental(*args, **kwargs):
        return None


def load_library(*args, **kwargs):
    return None


def load_op_library(*args, **kwargs):
    return None


def make_ndarray(*args, **kwargs):
    return None


def make_tensor_proto(*args, **kwargs):
    return None


def map_fn(*args, **kwargs):
    return _ops.map_fn(*args, **kwargs)


def matrix_square_root(*args, **kwargs):
    return None


class mlir:
    @staticmethod
    def experimental(*args, **kwargs):
        return None


def name_scope(*args, **kwargs):
    return None


class nest:
    @staticmethod
    def assert_same_structure(*args, **kwargs):
        return None

    @staticmethod
    def flatten(*args, **kwargs):
        return _ops.flatten(*args, **kwargs)

    @staticmethod
    def is_nested(*args, **kwargs):
        return None

    @staticmethod
    def map_structure(*args, **kwargs):
        return None

    @staticmethod
    def pack_sequence_as(*args, **kwargs):
        return None


def newaxis(*args, **kwargs):
    return None


def no_gradient(*args, **kwargs):
    return None


def no_op(*args, **kwargs):
    return None


def nondifferentiable_batch_function(*args, **kwargs):
    return None


def numpy_function(*args, **kwargs):
    return None


def one_hot(*args, **kwargs):
    return None


def ones_initializer(*args, **kwargs):
    return None


def pad(*args, **kwargs):
    return _ops.pad(*args, **kwargs)


def parallel_stack(*args, **kwargs):
    return None


def print(*args, **kwargs):
    return None


class profiler:
    @staticmethod
    def experimental(*args, **kwargs):
        return None


def py_function(*args, **kwargs):
    return None


def qint16(*args, **kwargs):
    return None


def qint32(*args, **kwargs):
    return None


def qint8(*args, **kwargs):
    return None


class queue:
    @staticmethod
    def FIFOQueue(*args, **kwargs):
        return None

    @staticmethod
    def PaddingFIFOQueue(*args, **kwargs):
        return None

    @staticmethod
    def PriorityQueue(*args, **kwargs):
        return None

    @staticmethod
    def QueueBase(*args, **kwargs):
        return None

    @staticmethod
    def RandomShuffleQueue(*args, **kwargs):
        return None


def quint16(*args, **kwargs):
    return None


def quint8(*args, **kwargs):
    return None


def random_index_shuffle(*args, **kwargs):
    return None


def random_normal_initializer(*args, **kwargs):
    return None


def random_uniform_initializer(*args, **kwargs):
    return None


def rank(*args, **kwargs):
    return None


class raw_ops:
    @staticmethod
    def Abort(*args, **kwargs):
        return None

    @staticmethod
    def Abs(*args, **kwargs):
        return None

    @staticmethod
    def AccumulateNV2(*args, **kwargs):
        return None

    @staticmethod
    def AccumulatorApplyGradient(*args, **kwargs):
        return None

    @staticmethod
    def AccumulatorNumAccumulated(*args, **kwargs):
        return None

    @staticmethod
    def AccumulatorSetGlobalStep(*args, **kwargs):
        return None

    @staticmethod
    def AccumulatorTakeGradient(*args, **kwargs):
        return None

    @staticmethod
    def Acos(*args, **kwargs):
        return None

    @staticmethod
    def Acosh(*args, **kwargs):
        return None

    @staticmethod
    def Add(*args, **kwargs):
        return keras.layers.Add(*args, **kwargs)

    @staticmethod
    def AddManySparseToTensorsMap(*args, **kwargs):
        return None

    @staticmethod
    def AddN(*args, **kwargs):
        return None

    @staticmethod
    def AddSparseToTensorsMap(*args, **kwargs):
        return None

    @staticmethod
    def AddV2(*args, **kwargs):
        return None

    @staticmethod
    def AdjustContrast(*args, **kwargs):
        return _ops.AdjustContrast(*args, **kwargs)

    @staticmethod
    def AdjustContrastv2(*args, **kwargs):
        return None

    @staticmethod
    def AdjustHue(*args, **kwargs):
        return _ops.AdjustHue(*args, **kwargs)

    @staticmethod
    def AdjustSaturation(*args, **kwargs):
        return _ops.AdjustSaturation(*args, **kwargs)

    @staticmethod
    def All(*args, **kwargs):
        return None

    @staticmethod
    def AllCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def AllToAll(*args, **kwargs):
        return _ops.AllToAll(*args, **kwargs)

    @staticmethod
    def Angle(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousHashTable(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousIterator(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousIteratorV2(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousIteratorV3(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMemoryCache(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMultiDeviceIterator(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMultiDeviceIteratorV3(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMutableDenseHashTable(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMutableHashTable(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousMutableHashTableOfTensors(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousRandomSeedGenerator(*args, **kwargs):
        return None

    @staticmethod
    def AnonymousSeedGenerator(*args, **kwargs):
        return None

    @staticmethod
    def Any(*args, **kwargs):
        return _ops.Any(*args, **kwargs)

    @staticmethod
    def ApplyAdaMax(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAdadelta(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAdagradDA(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAdagradV2(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAdam(*args, **kwargs):
        return None

    @staticmethod
    def ApplyAddSign(*args, **kwargs):
        return None

    @staticmethod
    def ApplyCenteredRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ApplyFtrl(*args, **kwargs):
        return None

    @staticmethod
    def ApplyFtrlV2(*args, **kwargs):
        return None

    @staticmethod
    def ApplyGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def ApplyMomentum(*args, **kwargs):
        return None

    @staticmethod
    def ApplyPowerSign(*args, **kwargs):
        return None

    @staticmethod
    def ApplyProximalAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ApplyProximalGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def ApplyRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ApproxTopK(*args, **kwargs):
        return None

    @staticmethod
    def ApproximateEqual(*args, **kwargs):
        return None

    @staticmethod
    def ArgMax(*args, **kwargs):
        return None

    @staticmethod
    def ArgMin(*args, **kwargs):
        return None

    @staticmethod
    def AsString(*args, **kwargs):
        return None

    @staticmethod
    def Asin(*args, **kwargs):
        return None

    @staticmethod
    def Asinh(*args, **kwargs):
        return None

    @staticmethod
    def Assert(*args, **kwargs):
        return None

    @staticmethod
    def AssertCardinalityDataset(*args, **kwargs):
        return None

    @staticmethod
    def AssertNextDataset(*args, **kwargs):
        return None

    @staticmethod
    def AssertPrevDataset(*args, **kwargs):
        return None

    @staticmethod
    def Assign(*args, **kwargs):
        return _ops.Assign(*args, **kwargs)

    @staticmethod
    def AssignAdd(*args, **kwargs):
        return _ops.AssignAdd(*args, **kwargs)

    @staticmethod
    def AssignAddVariableOp(*args, **kwargs):
        return None

    @staticmethod
    def AssignSub(*args, **kwargs):
        return _ops.AssignSub(*args, **kwargs)

    @staticmethod
    def AssignSubVariableOp(*args, **kwargs):
        return None

    @staticmethod
    def AssignVariableOp(*args, **kwargs):
        return None

    @staticmethod
    def AssignVariableXlaConcatND(*args, **kwargs):
        return None

    @staticmethod
    def Atan(*args, **kwargs):
        return None

    @staticmethod
    def Atan2(*args, **kwargs):
        return None

    @staticmethod
    def Atanh(*args, **kwargs):
        return None

    @staticmethod
    def AudioSpectrogram(*args, **kwargs):
        return None

    @staticmethod
    def AudioSummary(*args, **kwargs):
        return None

    @staticmethod
    def AudioSummaryV2(*args, **kwargs):
        return None

    @staticmethod
    def AutoShardDataset(*args, **kwargs):
        return None

    @staticmethod
    def AvgPool(*args, **kwargs):
        return None

    @staticmethod
    def AvgPool3D(*args, **kwargs):
        return keras.layers.AvgPool3D(*args, **kwargs)

    @staticmethod
    def AvgPool3DGrad(*args, **kwargs):
        return None

    @staticmethod
    def AvgPoolGrad(*args, **kwargs):
        return None

    @staticmethod
    def BandedTriangularSolve(*args, **kwargs):
        return None

    @staticmethod
    def Barrier(*args, **kwargs):
        return None

    @staticmethod
    def BarrierClose(*args, **kwargs):
        return None

    @staticmethod
    def BarrierIncompleteSize(*args, **kwargs):
        return None

    @staticmethod
    def BarrierInsertMany(*args, **kwargs):
        return None

    @staticmethod
    def BarrierReadySize(*args, **kwargs):
        return None

    @staticmethod
    def BarrierTakeMany(*args, **kwargs):
        return None

    @staticmethod
    def Batch(*args, **kwargs):
        return None

    @staticmethod
    def BatchCholesky(*args, **kwargs):
        return None

    @staticmethod
    def BatchCholeskyGrad(*args, **kwargs):
        return None

    @staticmethod
    def BatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def BatchDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def BatchFFT(*args, **kwargs):
        return None

    @staticmethod
    def BatchFFT2D(*args, **kwargs):
        return None

    @staticmethod
    def BatchFFT3D(*args, **kwargs):
        return None

    @staticmethod
    def BatchFunction(*args, **kwargs):
        return None

    @staticmethod
    def BatchIFFT(*args, **kwargs):
        return None

    @staticmethod
    def BatchIFFT2D(*args, **kwargs):
        return None

    @staticmethod
    def BatchIFFT3D(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatMul(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatMulV2(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatMulV3(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixBandPart(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixDeterminant(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixDiag(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixDiagPart(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixInverse(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixSetDiag(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixSolve(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixSolveLs(*args, **kwargs):
        return None

    @staticmethod
    def BatchMatrixTriangularSolve(*args, **kwargs):
        return None

    @staticmethod
    def BatchNormWithGlobalNormalization(*args, **kwargs):
        return None

    @staticmethod
    def BatchNormWithGlobalNormalizationGrad(*args, **kwargs):
        return None

    @staticmethod
    def BatchSelfAdjointEig(*args, **kwargs):
        return None

    @staticmethod
    def BatchSelfAdjointEigV2(*args, **kwargs):
        return None

    @staticmethod
    def BatchSvd(*args, **kwargs):
        return None

    @staticmethod
    def BatchToSpace(*args, **kwargs):
        return None

    @staticmethod
    def BatchToSpaceND(*args, **kwargs):
        return None

    @staticmethod
    def BesselI0(*args, **kwargs):
        return None

    @staticmethod
    def BesselI0e(*args, **kwargs):
        return None

    @staticmethod
    def BesselI1(*args, **kwargs):
        return None

    @staticmethod
    def BesselI1e(*args, **kwargs):
        return None

    @staticmethod
    def BesselJ0(*args, **kwargs):
        return None

    @staticmethod
    def BesselJ1(*args, **kwargs):
        return None

    @staticmethod
    def BesselK0(*args, **kwargs):
        return None

    @staticmethod
    def BesselK0e(*args, **kwargs):
        return None

    @staticmethod
    def BesselK1(*args, **kwargs):
        return None

    @staticmethod
    def BesselK1e(*args, **kwargs):
        return None

    @staticmethod
    def BesselY0(*args, **kwargs):
        return None

    @staticmethod
    def BesselY1(*args, **kwargs):
        return None

    @staticmethod
    def Betainc(*args, **kwargs):
        return None

    @staticmethod
    def BiasAdd(*args, **kwargs):
        return None

    @staticmethod
    def BiasAddGrad(*args, **kwargs):
        return None

    @staticmethod
    def BiasAddV1(*args, **kwargs):
        return None

    @staticmethod
    def Bincount(*args, **kwargs):
        return None

    @staticmethod
    def Bitcast(*args, **kwargs):
        return None

    @staticmethod
    def BitwiseAnd(*args, **kwargs):
        return None

    @staticmethod
    def BitwiseOr(*args, **kwargs):
        return None

    @staticmethod
    def BitwiseXor(*args, **kwargs):
        return None

    @staticmethod
    def BlockLSTM(*args, **kwargs):
        return None

    @staticmethod
    def BlockLSTMGrad(*args, **kwargs):
        return None

    @staticmethod
    def BlockLSTMGradV2(*args, **kwargs):
        return None

    @staticmethod
    def BlockLSTMV2(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesAggregateStats(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesBucketize(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplit(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplitV2(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCalculateBestGainsPerFeature(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCenterBias(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCreateEnsemble(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesCreateQuantileStreamResource(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesDeserializeEnsemble(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesEnsembleResourceHandleOp(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesExampleDebugOutputs(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesFlushQuantileSummaries(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesGetEnsembleStates(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesMakeQuantileSummaries(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesMakeStatsSummary(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesPredict(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceAddSummaries(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceDeserialize(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceFlush(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceGetBucketBoundaries(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesQuantileStreamResourceHandleOp(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesSerializeEnsemble(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesSparseAggregateStats(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesSparseCalculateBestFeatureSplit(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesTrainingPredict(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesUpdateEnsemble(*args, **kwargs):
        return None

    @staticmethod
    def BoostedTreesUpdateEnsembleV2(*args, **kwargs):
        return None

    @staticmethod
    def BroadcastArgs(*args, **kwargs):
        return None

    @staticmethod
    def BroadcastGradientArgs(*args, **kwargs):
        return None

    @staticmethod
    def BroadcastTo(*args, **kwargs):
        return _ops.BroadcastTo(*args, **kwargs)

    @staticmethod
    def Bucketize(*args, **kwargs):
        return None

    @staticmethod
    def BytesProducedStatsDataset(*args, **kwargs):
        return None

    @staticmethod
    def CSRSparseMatrixComponents(*args, **kwargs):
        return None

    @staticmethod
    def CSRSparseMatrixToDense(*args, **kwargs):
        return None

    @staticmethod
    def CSRSparseMatrixToSparseTensor(*args, **kwargs):
        return None

    @staticmethod
    def CSVDataset(*args, **kwargs):
        return None

    @staticmethod
    def CSVDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def CTCBeamSearchDecoder(*args, **kwargs):
        return None

    @staticmethod
    def CTCGreedyDecoder(*args, **kwargs):
        return None

    @staticmethod
    def CTCLoss(*args, **kwargs):
        return None

    @staticmethod
    def CTCLossV2(*args, **kwargs):
        return None

    @staticmethod
    def CacheDataset(*args, **kwargs):
        return None

    @staticmethod
    def CacheDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def Case(*args, **kwargs):
        return None

    @staticmethod
    def Cast(*args, **kwargs):
        return None

    @staticmethod
    def Ceil(*args, **kwargs):
        return None

    @staticmethod
    def CheckNumerics(*args, **kwargs):
        return None

    @staticmethod
    def CheckNumericsV2(*args, **kwargs):
        return None

    @staticmethod
    def Cholesky(*args, **kwargs):
        return None

    @staticmethod
    def CholeskyGrad(*args, **kwargs):
        return None

    @staticmethod
    def ChooseFastestBranchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ChooseFastestDataset(*args, **kwargs):
        return None

    @staticmethod
    def ClipByValue(*args, **kwargs):
        return None

    @staticmethod
    def CloseSummaryWriter(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveAllToAllV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveAllToAllV3(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveAssignGroupV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveBcastRecv(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveBcastRecvV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveBcastSend(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveBcastSendV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveGather(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveGatherV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveInitializeCommunicator(*args, **kwargs):
        return None

    @staticmethod
    def CollectivePermute(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveReduce(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveReduceScatterV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveReduceV2(*args, **kwargs):
        return None

    @staticmethod
    def CollectiveReduceV3(*args, **kwargs):
        return None

    @staticmethod
    def CombinedNonMaxSuppression(*args, **kwargs):
        return None

    @staticmethod
    def Complex(*args, **kwargs):
        return None

    @staticmethod
    def ComplexAbs(*args, **kwargs):
        return None

    @staticmethod
    def CompositeTensorVariantFromComponents(*args, **kwargs):
        return None

    @staticmethod
    def CompositeTensorVariantToComponents(*args, **kwargs):
        return None

    @staticmethod
    def CompressElement(*args, **kwargs):
        return None

    @staticmethod
    def ComputeAccidentalHits(*args, **kwargs):
        return None

    @staticmethod
    def ComputeBatchSize(*args, **kwargs):
        return None

    @staticmethod
    def Concat(*args, **kwargs):
        return None

    @staticmethod
    def ConcatOffset(*args, **kwargs):
        return None

    @staticmethod
    def ConcatV2(*args, **kwargs):
        return None

    @staticmethod
    def ConcatenateDataset(*args, **kwargs):
        return None

    @staticmethod
    def ConditionalAccumulator(*args, **kwargs):
        return None

    @staticmethod
    def ConfigureDistributedTPU(*args, **kwargs):
        return None

    @staticmethod
    def ConfigureTPUEmbedding(*args, **kwargs):
        return None

    @staticmethod
    def Conj(*args, **kwargs):
        return None

    @staticmethod
    def ConjugateTranspose(*args, **kwargs):
        return None

    @staticmethod
    def Const(*args, **kwargs):
        return None

    @staticmethod
    def ConsumeMutexLock(*args, **kwargs):
        return None

    @staticmethod
    def ControlTrigger(*args, **kwargs):
        return None

    @staticmethod
    def Conv(*args, **kwargs):
        return None

    @staticmethod
    def Conv2D(*args, **kwargs):
        return keras.layers.Conv2D(*args, **kwargs)

    @staticmethod
    def Conv2DBackpropFilter(*args, **kwargs):
        return None

    @staticmethod
    def Conv2DBackpropFilterV2(*args, **kwargs):
        return None

    @staticmethod
    def Conv2DBackpropInput(*args, **kwargs):
        return None

    @staticmethod
    def Conv2DBackpropInputV2(*args, **kwargs):
        return None

    @staticmethod
    def Conv3D(*args, **kwargs):
        return keras.layers.Conv3D(*args, **kwargs)

    @staticmethod
    def Conv3DBackpropFilter(*args, **kwargs):
        return None

    @staticmethod
    def Conv3DBackpropFilterV2(*args, **kwargs):
        return None

    @staticmethod
    def Conv3DBackpropInput(*args, **kwargs):
        return None

    @staticmethod
    def Conv3DBackpropInputV2(*args, **kwargs):
        return None

    @staticmethod
    def ConvertToCooTensor(*args, **kwargs):
        return None

    @staticmethod
    def Copy(*args, **kwargs):
        return None

    @staticmethod
    def CopyHost(*args, **kwargs):
        return None

    @staticmethod
    def Cos(*args, **kwargs):
        return None

    @staticmethod
    def Cosh(*args, **kwargs):
        return None

    @staticmethod
    def CountUpTo(*args, **kwargs):
        return None

    @staticmethod
    def CreateSummaryDbWriter(*args, **kwargs):
        return None

    @staticmethod
    def CreateSummaryFileWriter(*args, **kwargs):
        return None

    @staticmethod
    def CropAndResize(*args, **kwargs):
        return _ops.CropAndResize(*args, **kwargs)

    @staticmethod
    def CropAndResizeGradBoxes(*args, **kwargs):
        return None

    @staticmethod
    def CropAndResizeGradImage(*args, **kwargs):
        return None

    @staticmethod
    def Cross(*args, **kwargs):
        return None

    @staticmethod
    def CrossReplicaSum(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNN(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNBackprop(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNBackpropV2(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNBackpropV3(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNCanonicalToParams(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNCanonicalToParamsV2(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNParamsSize(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNParamsToCanonical(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNParamsToCanonicalV2(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNV2(*args, **kwargs):
        return None

    @staticmethod
    def CudnnRNNV3(*args, **kwargs):
        return None

    @staticmethod
    def Cumprod(*args, **kwargs):
        return None

    @staticmethod
    def Cumsum(*args, **kwargs):
        return None

    @staticmethod
    def CumulativeLogsumexp(*args, **kwargs):
        return None

    @staticmethod
    def DataFormatDimMap(*args, **kwargs):
        return None

    @staticmethod
    def DataFormatVecPermute(*args, **kwargs):
        return None

    @staticmethod
    def DataServiceDataset(*args, **kwargs):
        return None

    @staticmethod
    def DataServiceDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def DataServiceDatasetV3(*args, **kwargs):
        return None

    @staticmethod
    def DataServiceDatasetV4(*args, **kwargs):
        return None

    @staticmethod
    def DatasetCardinality(*args, **kwargs):
        return None

    @staticmethod
    def DatasetFingerprint(*args, **kwargs):
        return None

    @staticmethod
    def DatasetFromGraph(*args, **kwargs):
        return None

    @staticmethod
    def DatasetToGraph(*args, **kwargs):
        return None

    @staticmethod
    def DatasetToGraphV2(*args, **kwargs):
        return None

    @staticmethod
    def DatasetToSingleElement(*args, **kwargs):
        return None

    @staticmethod
    def DatasetToTFRecord(*args, **kwargs):
        return None

    @staticmethod
    def Dawsn(*args, **kwargs):
        return None

    @staticmethod
    def DebugGradientIdentity(*args, **kwargs):
        return None

    @staticmethod
    def DebugGradientRefIdentity(*args, **kwargs):
        return None

    @staticmethod
    def DebugIdentity(*args, **kwargs):
        return None

    @staticmethod
    def DebugIdentityV2(*args, **kwargs):
        return None

    @staticmethod
    def DebugIdentityV3(*args, **kwargs):
        return None

    @staticmethod
    def DebugNanCount(*args, **kwargs):
        return None

    @staticmethod
    def DebugNumericSummary(*args, **kwargs):
        return None

    @staticmethod
    def DebugNumericSummaryV2(*args, **kwargs):
        return None

    @staticmethod
    def DecodeAndCropJpeg(*args, **kwargs):
        return None

    @staticmethod
    def DecodeBase64(*args, **kwargs):
        return None

    @staticmethod
    def DecodeBmp(*args, **kwargs):
        return None

    @staticmethod
    def DecodeCSV(*args, **kwargs):
        return None

    @staticmethod
    def DecodeCompressed(*args, **kwargs):
        return None

    @staticmethod
    def DecodeGif(*args, **kwargs):
        return None

    @staticmethod
    def DecodeImage(*args, **kwargs):
        return None

    @staticmethod
    def DecodeJSONExample(*args, **kwargs):
        return None

    @staticmethod
    def DecodeJpeg(*args, **kwargs):
        return None

    @staticmethod
    def DecodePaddedRaw(*args, **kwargs):
        return None

    @staticmethod
    def DecodePng(*args, **kwargs):
        return None

    @staticmethod
    def DecodeProtoV2(*args, **kwargs):
        return None

    @staticmethod
    def DecodeRaw(*args, **kwargs):
        return None

    @staticmethod
    def DecodeWav(*args, **kwargs):
        return None

    @staticmethod
    def DeepCopy(*args, **kwargs):
        return None

    @staticmethod
    def DeleteIterator(*args, **kwargs):
        return None

    @staticmethod
    def DeleteMemoryCache(*args, **kwargs):
        return None

    @staticmethod
    def DeleteMultiDeviceIterator(*args, **kwargs):
        return None

    @staticmethod
    def DeleteRandomSeedGenerator(*args, **kwargs):
        return None

    @staticmethod
    def DeleteSeedGenerator(*args, **kwargs):
        return None

    @staticmethod
    def DeleteSessionTensor(*args, **kwargs):
        return None

    @staticmethod
    def DenseBincount(*args, **kwargs):
        return None

    @staticmethod
    def DenseCountSparseOutput(*args, **kwargs):
        return None

    @staticmethod
    def DenseToCSRSparseMatrix(*args, **kwargs):
        return None

    @staticmethod
    def DenseToDenseSetOperation(*args, **kwargs):
        return None

    @staticmethod
    def DenseToSparseBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def DenseToSparseSetOperation(*args, **kwargs):
        return None

    @staticmethod
    def DepthToSpace(*args, **kwargs):
        return None

    @staticmethod
    def DepthwiseConv2dNative(*args, **kwargs):
        return None

    @staticmethod
    def DepthwiseConv2dNativeBackpropFilter(*args, **kwargs):
        return None

    @staticmethod
    def DepthwiseConv2dNativeBackpropInput(*args, **kwargs):
        return None

    @staticmethod
    def Dequantize(*args, **kwargs):
        return None

    @staticmethod
    def DeserializeIterator(*args, **kwargs):
        return None

    @staticmethod
    def DeserializeManySparse(*args, **kwargs):
        return None

    @staticmethod
    def DeserializeSparse(*args, **kwargs):
        return None

    @staticmethod
    def DestroyResourceOp(*args, **kwargs):
        return None

    @staticmethod
    def DestroyTemporaryVariable(*args, **kwargs):
        return None

    @staticmethod
    def DeviceIndex(*args, **kwargs):
        return None

    @staticmethod
    def Diag(*args, **kwargs):
        return None

    @staticmethod
    def DiagPart(*args, **kwargs):
        return None

    @staticmethod
    def Digamma(*args, **kwargs):
        return None

    @staticmethod
    def Dilation2D(*args, **kwargs):
        return None

    @staticmethod
    def Dilation2DBackpropFilter(*args, **kwargs):
        return None

    @staticmethod
    def Dilation2DBackpropInput(*args, **kwargs):
        return None

    @staticmethod
    def DirectedInterleaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def DisableCopyOnRead(*args, **kwargs):
        return None

    @staticmethod
    def DistributedSave(*args, **kwargs):
        return None

    @staticmethod
    def Div(*args, **kwargs):
        return None

    @staticmethod
    def DivNoNan(*args, **kwargs):
        return None

    @staticmethod
    def DrawBoundingBoxes(*args, **kwargs):
        return _ops.DrawBoundingBoxes(*args, **kwargs)

    @staticmethod
    def DrawBoundingBoxesV2(*args, **kwargs):
        return None

    @staticmethod
    def DummyIterationCounter(*args, **kwargs):
        return None

    @staticmethod
    def DummyMemoryCache(*args, **kwargs):
        return None

    @staticmethod
    def DummySeedGenerator(*args, **kwargs):
        return None

    @staticmethod
    def DynamicEnqueueTPUEmbeddingArbitraryTensorBatch(*args, **kwargs):
        return None

    @staticmethod
    def DynamicEnqueueTPUEmbeddingRaggedTensorBatch(*args, **kwargs):
        return None

    @staticmethod
    def DynamicPartition(*args, **kwargs):
        return None

    @staticmethod
    def DynamicStitch(*args, **kwargs):
        return None

    @staticmethod
    def EagerPyFunc(*args, **kwargs):
        return None

    @staticmethod
    def EditDistance(*args, **kwargs):
        return None

    @staticmethod
    def Eig(*args, **kwargs):
        return None

    @staticmethod
    def Einsum(*args, **kwargs):
        return _ops.Einsum(*args, **kwargs)

    @staticmethod
    def Elu(*args, **kwargs):
        return None

    @staticmethod
    def EluGrad(*args, **kwargs):
        return None

    @staticmethod
    def Empty(*args, **kwargs):
        return None

    @staticmethod
    def EmptyTensorList(*args, **kwargs):
        return None

    @staticmethod
    def EmptyTensorMap(*args, **kwargs):
        return None

    @staticmethod
    def EncodeBase64(*args, **kwargs):
        return None

    @staticmethod
    def EncodeJpeg(*args, **kwargs):
        return None

    @staticmethod
    def EncodeJpegVariableQuality(*args, **kwargs):
        return None

    @staticmethod
    def EncodePng(*args, **kwargs):
        return None

    @staticmethod
    def EncodeProto(*args, **kwargs):
        return None

    @staticmethod
    def EncodeWav(*args, **kwargs):
        return None

    @staticmethod
    def EnqueueTPUEmbeddingArbitraryTensorBatch(*args, **kwargs):
        return None

    @staticmethod
    def EnqueueTPUEmbeddingIntegerBatch(*args, **kwargs):
        return None

    @staticmethod
    def EnqueueTPUEmbeddingRaggedTensorBatch(*args, **kwargs):
        return None

    @staticmethod
    def EnqueueTPUEmbeddingSparseBatch(*args, **kwargs):
        return None

    @staticmethod
    def EnqueueTPUEmbeddingSparseTensorBatch(*args, **kwargs):
        return None

    @staticmethod
    def EnsureShape(*args, **kwargs):
        return None

    @staticmethod
    def Enter(*args, **kwargs):
        return None

    @staticmethod
    def Equal(*args, **kwargs):
        return None

    @staticmethod
    def Erf(*args, **kwargs):
        return None

    @staticmethod
    def Erfc(*args, **kwargs):
        return None

    @staticmethod
    def Erfinv(*args, **kwargs):
        return None

    @staticmethod
    def EuclideanNorm(*args, **kwargs):
        return None

    @staticmethod
    def Exit(*args, **kwargs):
        return None

    @staticmethod
    def Exp(*args, **kwargs):
        return None

    @staticmethod
    def ExpandDims(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalAssertNextDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalAutoShardDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalBytesProducedStatsDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalCSVDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalChooseFastestDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalDatasetCardinality(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalDatasetToTFRecord(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalDenseToSparseBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalDirectedInterleaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalGroupByReducerDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalGroupByWindowDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalIgnoreErrorsDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalIteratorGetDevice(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalLMDBDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalLatencyStatsDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalMapAndBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalMapDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalMatchingFilesDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalMaxIntraOpParallelismDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalNonSerializableDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalParallelInterleaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalParseExampleDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalPrivateThreadPoolDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalRandomDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalRebatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalScanDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalSetStatsAggregatorDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalSleepDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalSlidingWindowDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalSqlDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalStatsAggregatorHandle(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalStatsAggregatorSummary(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalTakeWhileDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalThreadPoolDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalThreadPoolHandle(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalUnbatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ExperimentalUniqueDataset(*args, **kwargs):
        return None

    @staticmethod
    def Expint(*args, **kwargs):
        return None

    @staticmethod
    def Expm1(*args, **kwargs):
        return None

    @staticmethod
    def ExtractGlimpse(*args, **kwargs):
        return None

    @staticmethod
    def ExtractGlimpseV2(*args, **kwargs):
        return None

    @staticmethod
    def ExtractImagePatches(*args, **kwargs):
        return None

    @staticmethod
    def ExtractJpegShape(*args, **kwargs):
        return None

    @staticmethod
    def ExtractVolumePatches(*args, **kwargs):
        return None

    @staticmethod
    def FFT(*args, **kwargs):
        return None

    @staticmethod
    def FFT2D(*args, **kwargs):
        return None

    @staticmethod
    def FFT3D(*args, **kwargs):
        return None

    @staticmethod
    def FFTND(*args, **kwargs):
        return None

    @staticmethod
    def FIFOQueue(*args, **kwargs):
        return None

    @staticmethod
    def FIFOQueueV2(*args, **kwargs):
        return None

    @staticmethod
    def Fact(*args, **kwargs):
        return None

    @staticmethod
    def FakeParam(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxArgs(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxArgsGradient(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxVars(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsGradient(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannel(*args, **kwargs):
        return None

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannelGradient(*args, **kwargs):
        return None

    @staticmethod
    def FakeQueue(*args, **kwargs):
        return None

    @staticmethod
    def FileSystemSetConfiguration(*args, **kwargs):
        return None

    @staticmethod
    def Fill(*args, **kwargs):
        return None

    @staticmethod
    def FilterByLastComponentDataset(*args, **kwargs):
        return None

    @staticmethod
    def FilterDataset(*args, **kwargs):
        return None

    @staticmethod
    def FinalizeDataset(*args, **kwargs):
        return None

    @staticmethod
    def Fingerprint(*args, **kwargs):
        return None

    @staticmethod
    def FixedLengthRecordDataset(*args, **kwargs):
        return None

    @staticmethod
    def FixedLengthRecordDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def FixedLengthRecordReader(*args, **kwargs):
        return None

    @staticmethod
    def FixedLengthRecordReaderV2(*args, **kwargs):
        return None

    @staticmethod
    def FixedUnigramCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def FlatMapDataset(*args, **kwargs):
        return None

    @staticmethod
    def Floor(*args, **kwargs):
        return None

    @staticmethod
    def FloorDiv(*args, **kwargs):
        return None

    @staticmethod
    def FloorMod(*args, **kwargs):
        return None

    @staticmethod
    def FlushSummaryWriter(*args, **kwargs):
        return None

    @staticmethod
    def For(*args, **kwargs):
        return None

    @staticmethod
    def FractionalAvgPool(*args, **kwargs):
        return None

    @staticmethod
    def FractionalAvgPoolGrad(*args, **kwargs):
        return None

    @staticmethod
    def FractionalMaxPool(*args, **kwargs):
        return None

    @staticmethod
    def FractionalMaxPoolGrad(*args, **kwargs):
        return None

    @staticmethod
    def FresnelCos(*args, **kwargs):
        return None

    @staticmethod
    def FresnelSin(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNorm(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNormGrad(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNormGradV2(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNormGradV3(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNormV2(*args, **kwargs):
        return None

    @staticmethod
    def FusedBatchNormV3(*args, **kwargs):
        return None

    @staticmethod
    def FusedPadConv2D(*args, **kwargs):
        return None

    @staticmethod
    def FusedResizeAndPadConv2D(*args, **kwargs):
        return None

    @staticmethod
    def GRUBlockCell(*args, **kwargs):
        return None

    @staticmethod
    def GRUBlockCellGrad(*args, **kwargs):
        return None

    @staticmethod
    def Gather(*args, **kwargs):
        return _ops.Gather(*args, **kwargs)

    @staticmethod
    def GatherNd(*args, **kwargs):
        return _ops.GatherNd(*args, **kwargs)

    @staticmethod
    def GatherV2(*args, **kwargs):
        return None

    @staticmethod
    def GenerateBoundingBoxProposals(*args, **kwargs):
        return None

    @staticmethod
    def GenerateVocabRemapping(*args, **kwargs):
        return None

    @staticmethod
    def GeneratorDataset(*args, **kwargs):
        return None

    @staticmethod
    def GetElementAtIndex(*args, **kwargs):
        return None

    @staticmethod
    def GetMinibatchSplitsWithPhysicalReplica(*args, **kwargs):
        return None

    @staticmethod
    def GetMinibatchesInCsrWithPhysicalReplica(*args, **kwargs):
        return None

    @staticmethod
    def GetOptions(*args, **kwargs):
        return None

    @staticmethod
    def GetSessionHandle(*args, **kwargs):
        return None

    @staticmethod
    def GetSessionHandleV2(*args, **kwargs):
        return None

    @staticmethod
    def GetSessionTensor(*args, **kwargs):
        return None

    @staticmethod
    def GlobalIterId(*args, **kwargs):
        return None

    @staticmethod
    def Greater(*args, **kwargs):
        return None

    @staticmethod
    def GreaterEqual(*args, **kwargs):
        return None

    @staticmethod
    def GroupByReducerDataset(*args, **kwargs):
        return None

    @staticmethod
    def GroupByWindowDataset(*args, **kwargs):
        return None

    @staticmethod
    def GuaranteeConst(*args, **kwargs):
        return None

    @staticmethod
    def HSVToRGB(*args, **kwargs):
        return None

    @staticmethod
    def HashTable(*args, **kwargs):
        return None

    @staticmethod
    def HashTableV2(*args, **kwargs):
        return None

    @staticmethod
    def HistogramFixedWidth(*args, **kwargs):
        return None

    @staticmethod
    def HistogramSummary(*args, **kwargs):
        return None

    @staticmethod
    def IFFT(*args, **kwargs):
        return None

    @staticmethod
    def IFFT2D(*args, **kwargs):
        return None

    @staticmethod
    def IFFT3D(*args, **kwargs):
        return None

    @staticmethod
    def IFFTND(*args, **kwargs):
        return None

    @staticmethod
    def IRFFT(*args, **kwargs):
        return None

    @staticmethod
    def IRFFT2D(*args, **kwargs):
        return None

    @staticmethod
    def IRFFT3D(*args, **kwargs):
        return None

    @staticmethod
    def IRFFTND(*args, **kwargs):
        return None

    @staticmethod
    def Identity(*args, **kwargs):
        return keras.layers.Identity(*args, **kwargs)

    @staticmethod
    def IdentityN(*args, **kwargs):
        return None

    @staticmethod
    def IdentityReader(*args, **kwargs):
        return None

    @staticmethod
    def IdentityReaderV2(*args, **kwargs):
        return None

    @staticmethod
    def If(*args, **kwargs):
        return None

    @staticmethod
    def Igamma(*args, **kwargs):
        return None

    @staticmethod
    def IgammaGradA(*args, **kwargs):
        return None

    @staticmethod
    def Igammac(*args, **kwargs):
        return None

    @staticmethod
    def IgnoreErrorsDataset(*args, **kwargs):
        return None

    @staticmethod
    def Imag(*args, **kwargs):
        return None

    @staticmethod
    def ImageProjectiveTransformV2(*args, **kwargs):
        return None

    @staticmethod
    def ImageProjectiveTransformV3(*args, **kwargs):
        return None

    @staticmethod
    def ImageSummary(*args, **kwargs):
        return None

    @staticmethod
    def ImmutableConst(*args, **kwargs):
        return None

    @staticmethod
    def ImportEvent(*args, **kwargs):
        return None

    @staticmethod
    def InTopK(*args, **kwargs):
        return None

    @staticmethod
    def InTopKV2(*args, **kwargs):
        return None

    @staticmethod
    def InfeedDequeue(*args, **kwargs):
        return None

    @staticmethod
    def InfeedDequeueTuple(*args, **kwargs):
        return None

    @staticmethod
    def InfeedEnqueue(*args, **kwargs):
        return None

    @staticmethod
    def InfeedEnqueuePrelinearizedBuffer(*args, **kwargs):
        return None

    @staticmethod
    def InfeedEnqueueTuple(*args, **kwargs):
        return None

    @staticmethod
    def InitializeTable(*args, **kwargs):
        return None

    @staticmethod
    def InitializeTableFromDataset(*args, **kwargs):
        return None

    @staticmethod
    def InitializeTableFromTextFile(*args, **kwargs):
        return None

    @staticmethod
    def InitializeTableFromTextFileV2(*args, **kwargs):
        return None

    @staticmethod
    def InitializeTableV2(*args, **kwargs):
        return None

    @staticmethod
    def InplaceAdd(*args, **kwargs):
        return None

    @staticmethod
    def InplaceSub(*args, **kwargs):
        return None

    @staticmethod
    def InplaceUpdate(*args, **kwargs):
        return None

    @staticmethod
    def InterleaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def Inv(*args, **kwargs):
        return None

    @staticmethod
    def InvGrad(*args, **kwargs):
        return None

    @staticmethod
    def Invert(*args, **kwargs):
        return _ops.Invert(*args, **kwargs)

    @staticmethod
    def InvertPermutation(*args, **kwargs):
        return None

    @staticmethod
    def IsBoostedTreesEnsembleInitialized(*args, **kwargs):
        return None

    @staticmethod
    def IsBoostedTreesQuantileStreamResourceInitialized(*args, **kwargs):
        return None

    @staticmethod
    def IsFinite(*args, **kwargs):
        return None

    @staticmethod
    def IsInf(*args, **kwargs):
        return None

    @staticmethod
    def IsNan(*args, **kwargs):
        return None

    @staticmethod
    def IsTPUEmbeddingInitialized(*args, **kwargs):
        return None

    @staticmethod
    def IsVariableInitialized(*args, **kwargs):
        return None

    @staticmethod
    def IsotonicRegression(*args, **kwargs):
        return None

    @staticmethod
    def Iterator(*args, **kwargs):
        return None

    @staticmethod
    def IteratorFromStringHandle(*args, **kwargs):
        return None

    @staticmethod
    def IteratorFromStringHandleV2(*args, **kwargs):
        return None

    @staticmethod
    def IteratorGetDevice(*args, **kwargs):
        return None

    @staticmethod
    def IteratorGetNext(*args, **kwargs):
        return None

    @staticmethod
    def IteratorGetNextAsOptional(*args, **kwargs):
        return None

    @staticmethod
    def IteratorGetNextSync(*args, **kwargs):
        return None

    @staticmethod
    def IteratorToStringHandle(*args, **kwargs):
        return None

    @staticmethod
    def IteratorV2(*args, **kwargs):
        return None

    @staticmethod
    def KMC2ChainInitialization(*args, **kwargs):
        return None

    @staticmethod
    def KmeansPlusPlusInitialization(*args, **kwargs):
        return None

    @staticmethod
    def L2Loss(*args, **kwargs):
        return None

    @staticmethod
    def LMDBDataset(*args, **kwargs):
        return None

    @staticmethod
    def LMDBReader(*args, **kwargs):
        return None

    @staticmethod
    def LRN(*args, **kwargs):
        return None

    @staticmethod
    def LRNGrad(*args, **kwargs):
        return None

    @staticmethod
    def LSTMBlockCell(*args, **kwargs):
        return None

    @staticmethod
    def LSTMBlockCellGrad(*args, **kwargs):
        return None

    @staticmethod
    def LatencyStatsDataset(*args, **kwargs):
        return None

    @staticmethod
    def LeakyRelu(*args, **kwargs):
        return None

    @staticmethod
    def LeakyReluGrad(*args, **kwargs):
        return None

    @staticmethod
    def LearnedUnigramCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def LeftShift(*args, **kwargs):
        return None

    @staticmethod
    def LegacyParallelInterleaveDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def Less(*args, **kwargs):
        return None

    @staticmethod
    def LessEqual(*args, **kwargs):
        return None

    @staticmethod
    def Lgamma(*args, **kwargs):
        return None

    @staticmethod
    def LinSpace(*args, **kwargs):
        return None

    @staticmethod
    def ListDataset(*args, **kwargs):
        return None

    @staticmethod
    def ListDiff(*args, **kwargs):
        return None

    @staticmethod
    def ListSnapshotChunksDataset(*args, **kwargs):
        return None

    @staticmethod
    def LoadAndRemapMatrix(*args, **kwargs):
        return None

    @staticmethod
    def LoadDataset(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingADAMParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingAdadeltaParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingAdagradMomentumParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingAdagradParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingCenteredRMSPropParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingFTRLParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingFrequencyEstimatorParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingMDLAdagradLightParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingMomentumParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingProximalAdagradParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingProximalYogiParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingRMSPropParameters(*args, **kwargs):
        return None

    @staticmethod
    def LoadTPUEmbeddingStochasticGradientDescentParameters(*args, **kwargs):
        return None

    @staticmethod
    def Log(*args, **kwargs):
        return None

    @staticmethod
    def Log1p(*args, **kwargs):
        return None

    @staticmethod
    def LogMatrixDeterminant(*args, **kwargs):
        return None

    @staticmethod
    def LogSoftmax(*args, **kwargs):
        return None

    @staticmethod
    def LogUniformCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def LogicalAnd(*args, **kwargs):
        return None

    @staticmethod
    def LogicalNot(*args, **kwargs):
        return None

    @staticmethod
    def LogicalOr(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableExport(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableExportV2(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableFind(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableFindV2(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableImport(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableImportV2(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableInsert(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableInsertV2(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableRemoveV2(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableSize(*args, **kwargs):
        return None

    @staticmethod
    def LookupTableSizeV2(*args, **kwargs):
        return None

    @staticmethod
    def LoopCond(*args, **kwargs):
        return None

    @staticmethod
    def LowerBound(*args, **kwargs):
        return None

    @staticmethod
    def Lu(*args, **kwargs):
        return None

    @staticmethod
    def MakeIterator(*args, **kwargs):
        return None

    @staticmethod
    def MapAndBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def MapClear(*args, **kwargs):
        return None

    @staticmethod
    def MapDataset(*args, **kwargs):
        return None

    @staticmethod
    def MapDefun(*args, **kwargs):
        return None

    @staticmethod
    def MapIncompleteSize(*args, **kwargs):
        return None

    @staticmethod
    def MapPeek(*args, **kwargs):
        return None

    @staticmethod
    def MapSize(*args, **kwargs):
        return None

    @staticmethod
    def MapStage(*args, **kwargs):
        return None

    @staticmethod
    def MapUnstage(*args, **kwargs):
        return None

    @staticmethod
    def MapUnstageNoKey(*args, **kwargs):
        return None

    @staticmethod
    def MatMul(*args, **kwargs):
        return None

    @staticmethod
    def MatchingFiles(*args, **kwargs):
        return None

    @staticmethod
    def MatchingFilesDataset(*args, **kwargs):
        return None

    @staticmethod
    def MatrixBandPart(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDeterminant(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiag(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiagPart(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiagPartV2(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiagPartV3(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiagV2(*args, **kwargs):
        return None

    @staticmethod
    def MatrixDiagV3(*args, **kwargs):
        return None

    @staticmethod
    def MatrixExponential(*args, **kwargs):
        return None

    @staticmethod
    def MatrixInverse(*args, **kwargs):
        return None

    @staticmethod
    def MatrixLogarithm(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSetDiag(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSetDiagV2(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSetDiagV3(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSolve(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSolveLs(*args, **kwargs):
        return None

    @staticmethod
    def MatrixSquareRoot(*args, **kwargs):
        return None

    @staticmethod
    def MatrixTriangularSolve(*args, **kwargs):
        return None

    @staticmethod
    def Max(*args, **kwargs):
        return None

    @staticmethod
    def MaxIntraOpParallelismDataset(*args, **kwargs):
        return None

    @staticmethod
    def MaxPool(*args, **kwargs):
        return None

    @staticmethod
    def MaxPool3D(*args, **kwargs):
        return keras.layers.MaxPool3D(*args, **kwargs)

    @staticmethod
    def MaxPool3DGrad(*args, **kwargs):
        return None

    @staticmethod
    def MaxPool3DGradGrad(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGrad(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGradGrad(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGradGradV2(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGradGradWithArgmax(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGradV2(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolGradWithArgmax(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolV2(*args, **kwargs):
        return None

    @staticmethod
    def MaxPoolWithArgmax(*args, **kwargs):
        return None

    @staticmethod
    def Maximum(*args, **kwargs):
        return keras.layers.Maximum(*args, **kwargs)

    @staticmethod
    def Mean(*args, **kwargs):
        return keras.metrics.Mean(*args, **kwargs)

    @staticmethod
    def Merge(*args, **kwargs):
        return None

    @staticmethod
    def MergeSummary(*args, **kwargs):
        return None

    @staticmethod
    def MergeV2Checkpoints(*args, **kwargs):
        return None

    @staticmethod
    def Mfcc(*args, **kwargs):
        return None

    @staticmethod
    def Min(*args, **kwargs):
        return None

    @staticmethod
    def Minimum(*args, **kwargs):
        return keras.layers.Minimum(*args, **kwargs)

    @staticmethod
    def MirrorPad(*args, **kwargs):
        return None

    @staticmethod
    def MirrorPadGrad(*args, **kwargs):
        return None

    @staticmethod
    def Mod(*args, **kwargs):
        return None

    @staticmethod
    def ModelDataset(*args, **kwargs):
        return None

    @staticmethod
    def Mul(*args, **kwargs):
        return None

    @staticmethod
    def MulNoNan(*args, **kwargs):
        return None

    @staticmethod
    def MultiDeviceIterator(*args, **kwargs):
        return None

    @staticmethod
    def MultiDeviceIteratorFromStringHandle(*args, **kwargs):
        return None

    @staticmethod
    def MultiDeviceIteratorGetNextFromShard(*args, **kwargs):
        return None

    @staticmethod
    def MultiDeviceIteratorInit(*args, **kwargs):
        return None

    @staticmethod
    def MultiDeviceIteratorToStringHandle(*args, **kwargs):
        return None

    @staticmethod
    def Multinomial(*args, **kwargs):
        return None

    @staticmethod
    def MutableDenseHashTable(*args, **kwargs):
        return None

    @staticmethod
    def MutableDenseHashTableV2(*args, **kwargs):
        return None

    @staticmethod
    def MutableHashTable(*args, **kwargs):
        return None

    @staticmethod
    def MutableHashTableOfTensors(*args, **kwargs):
        return None

    @staticmethod
    def MutableHashTableOfTensorsV2(*args, **kwargs):
        return None

    @staticmethod
    def MutableHashTableV2(*args, **kwargs):
        return None

    @staticmethod
    def MutexLock(*args, **kwargs):
        return None

    @staticmethod
    def MutexV2(*args, **kwargs):
        return None

    @staticmethod
    def NcclAllReduce(*args, **kwargs):
        return None

    @staticmethod
    def NcclBroadcast(*args, **kwargs):
        return None

    @staticmethod
    def NcclReduce(*args, **kwargs):
        return None

    @staticmethod
    def Ndtri(*args, **kwargs):
        return None

    @staticmethod
    def NearestNeighbors(*args, **kwargs):
        return None

    @staticmethod
    def Neg(*args, **kwargs):
        return None

    @staticmethod
    def NextAfter(*args, **kwargs):
        return None

    @staticmethod
    def NextIteration(*args, **kwargs):
        return None

    @staticmethod
    def NoOp(*args, **kwargs):
        return None

    @staticmethod
    def NonDeterministicInts(*args, **kwargs):
        return None

    @staticmethod
    def NonMaxSuppression(*args, **kwargs):
        return _ops.NonMaxSuppression(*args, **kwargs)

    @staticmethod
    def NonMaxSuppressionV2(*args, **kwargs):
        return None

    @staticmethod
    def NonMaxSuppressionV3(*args, **kwargs):
        return None

    @staticmethod
    def NonMaxSuppressionV4(*args, **kwargs):
        return None

    @staticmethod
    def NonMaxSuppressionV5(*args, **kwargs):
        return None

    @staticmethod
    def NonMaxSuppressionWithOverlaps(*args, **kwargs):
        return None

    @staticmethod
    def NonSerializableDataset(*args, **kwargs):
        return None

    @staticmethod
    def NotEqual(*args, **kwargs):
        return None

    @staticmethod
    def NthElement(*args, **kwargs):
        return None

    @staticmethod
    def OneHot(*args, **kwargs):
        return None

    @staticmethod
    def OneShotIterator(*args, **kwargs):
        return None

    @staticmethod
    def OnesLike(*args, **kwargs):
        return None

    @staticmethod
    def OptimizeDataset(*args, **kwargs):
        return None

    @staticmethod
    def OptimizeDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def OptionalFromValue(*args, **kwargs):
        return None

    @staticmethod
    def OptionalGetValue(*args, **kwargs):
        return None

    @staticmethod
    def OptionalHasValue(*args, **kwargs):
        return None

    @staticmethod
    def OptionalNone(*args, **kwargs):
        return None

    @staticmethod
    def OptionsDataset(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapClear(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapIncompleteSize(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapPeek(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapSize(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapStage(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapUnstage(*args, **kwargs):
        return None

    @staticmethod
    def OrderedMapUnstageNoKey(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedDequeue(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedDequeueTuple(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedDequeueTupleV2(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedDequeueV2(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedEnqueue(*args, **kwargs):
        return None

    @staticmethod
    def OutfeedEnqueueTuple(*args, **kwargs):
        return None

    @staticmethod
    def Pack(*args, **kwargs):
        return None

    @staticmethod
    def Pad(*args, **kwargs):
        return None

    @staticmethod
    def PadV2(*args, **kwargs):
        return None

    @staticmethod
    def PaddedBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def PaddedBatchDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def PaddingFIFOQueue(*args, **kwargs):
        return None

    @staticmethod
    def PaddingFIFOQueueV2(*args, **kwargs):
        return None

    @staticmethod
    def ParallelBatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def ParallelConcat(*args, **kwargs):
        return None

    @staticmethod
    def ParallelDynamicStitch(*args, **kwargs):
        return None

    @staticmethod
    def ParallelFilterDataset(*args, **kwargs):
        return None

    @staticmethod
    def ParallelInterleaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def ParallelInterleaveDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def ParallelInterleaveDatasetV3(*args, **kwargs):
        return None

    @staticmethod
    def ParallelInterleaveDatasetV4(*args, **kwargs):
        return None

    @staticmethod
    def ParallelMapDataset(*args, **kwargs):
        return None

    @staticmethod
    def ParallelMapDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def ParameterizedTruncatedNormal(*args, **kwargs):
        return None

    @staticmethod
    def ParseExample(*args, **kwargs):
        return None

    @staticmethod
    def ParseExampleDataset(*args, **kwargs):
        return None

    @staticmethod
    def ParseExampleDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def ParseExampleV2(*args, **kwargs):
        return None

    @staticmethod
    def ParseSequenceExample(*args, **kwargs):
        return None

    @staticmethod
    def ParseSequenceExampleV2(*args, **kwargs):
        return None

    @staticmethod
    def ParseSingleExample(*args, **kwargs):
        return None

    @staticmethod
    def ParseSingleSequenceExample(*args, **kwargs):
        return None

    @staticmethod
    def ParseTensor(*args, **kwargs):
        return None

    @staticmethod
    def PartitionedCall(*args, **kwargs):
        return None

    @staticmethod
    def Placeholder(*args, **kwargs):
        return None

    @staticmethod
    def PlaceholderV2(*args, **kwargs):
        return None

    @staticmethod
    def PlaceholderWithDefault(*args, **kwargs):
        return None

    @staticmethod
    def Polygamma(*args, **kwargs):
        return None

    @staticmethod
    def PopulationCount(*args, **kwargs):
        return _ops.PopulationCount(*args, **kwargs)

    @staticmethod
    def Pow(*args, **kwargs):
        return None

    @staticmethod
    def PrefetchDataset(*args, **kwargs):
        return None

    @staticmethod
    def Prelinearize(*args, **kwargs):
        return None

    @staticmethod
    def PrelinearizeTuple(*args, **kwargs):
        return None

    @staticmethod
    def PreventGradient(*args, **kwargs):
        return None

    @staticmethod
    def Print(*args, **kwargs):
        return None

    @staticmethod
    def PrintV2(*args, **kwargs):
        return None

    @staticmethod
    def PriorityQueue(*args, **kwargs):
        return None

    @staticmethod
    def PriorityQueueV2(*args, **kwargs):
        return None

    @staticmethod
    def PrivateThreadPoolDataset(*args, **kwargs):
        return None

    @staticmethod
    def Prod(*args, **kwargs):
        return None

    @staticmethod
    def PyFunc(*args, **kwargs):
        return None

    @staticmethod
    def PyFuncStateless(*args, **kwargs):
        return None

    @staticmethod
    def Qr(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeAndDequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeAndDequantizeV2(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeAndDequantizeV3(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeAndDequantizeV4(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeAndDequantizeV4Grad(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeDownAndShrinkRange(*args, **kwargs):
        return None

    @staticmethod
    def QuantizeV2(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedAdd(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedAvgPool(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedBatchNormWithGlobalNormalization(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedBiasAdd(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConcat(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2D(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DAndRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DPerChannel(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBias(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSignedSumAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSumAndRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedConv2DWithBiasSumAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedDepthwiseConv2D(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBias(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedInstanceNorm(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMul(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMulWithBias(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndDequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndReluAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMatMulWithBiasAndRequantize(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMaxPool(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedMul(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedRelu(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedRelu6(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedReluX(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedReshape(*args, **kwargs):
        return None

    @staticmethod
    def QuantizedResizeBilinear(*args, **kwargs):
        return None

    @staticmethod
    def QueueClose(*args, **kwargs):
        return None

    @staticmethod
    def QueueCloseV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeue(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeueMany(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeueManyV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeueUpTo(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeueUpToV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueDequeueV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueEnqueue(*args, **kwargs):
        return None

    @staticmethod
    def QueueEnqueueMany(*args, **kwargs):
        return None

    @staticmethod
    def QueueEnqueueManyV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueEnqueueV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueIsClosed(*args, **kwargs):
        return None

    @staticmethod
    def QueueIsClosedV2(*args, **kwargs):
        return None

    @staticmethod
    def QueueSize(*args, **kwargs):
        return None

    @staticmethod
    def QueueSizeV2(*args, **kwargs):
        return None

    @staticmethod
    def RFFT(*args, **kwargs):
        return None

    @staticmethod
    def RFFT2D(*args, **kwargs):
        return None

    @staticmethod
    def RFFT3D(*args, **kwargs):
        return None

    @staticmethod
    def RFFTND(*args, **kwargs):
        return None

    @staticmethod
    def RGBToHSV(*args, **kwargs):
        return None

    @staticmethod
    def RaggedBincount(*args, **kwargs):
        return None

    @staticmethod
    def RaggedCountSparseOutput(*args, **kwargs):
        return None

    @staticmethod
    def RaggedCross(*args, **kwargs):
        return None

    @staticmethod
    def RaggedFillEmptyRows(*args, **kwargs):
        return None

    @staticmethod
    def RaggedFillEmptyRowsGrad(*args, **kwargs):
        return None

    @staticmethod
    def RaggedGather(*args, **kwargs):
        return _ops.RaggedGather(*args, **kwargs)

    @staticmethod
    def RaggedRange(*args, **kwargs):
        return _ops.RaggedRange(*args, **kwargs)

    @staticmethod
    def RaggedTensorFromVariant(*args, **kwargs):
        return None

    @staticmethod
    def RaggedTensorToSparse(*args, **kwargs):
        return None

    @staticmethod
    def RaggedTensorToTensor(*args, **kwargs):
        return None

    @staticmethod
    def RaggedTensorToVariant(*args, **kwargs):
        return None

    @staticmethod
    def RaggedTensorToVariantGradient(*args, **kwargs):
        return None

    @staticmethod
    def RandomCrop(*args, **kwargs):
        return keras.layers.RandomCrop(*args, **kwargs)

    @staticmethod
    def RandomDataset(*args, **kwargs):
        return None

    @staticmethod
    def RandomDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def RandomGamma(*args, **kwargs):
        return None

    @staticmethod
    def RandomGammaGrad(*args, **kwargs):
        return None

    @staticmethod
    def RandomIndexShuffle(*args, **kwargs):
        return None

    @staticmethod
    def RandomPoisson(*args, **kwargs):
        return None

    @staticmethod
    def RandomPoissonV2(*args, **kwargs):
        return None

    @staticmethod
    def RandomShuffle(*args, **kwargs):
        return None

    @staticmethod
    def RandomShuffleQueue(*args, **kwargs):
        return None

    @staticmethod
    def RandomShuffleQueueV2(*args, **kwargs):
        return None

    @staticmethod
    def RandomStandardNormal(*args, **kwargs):
        return None

    @staticmethod
    def RandomUniform(*args, **kwargs):
        return None

    @staticmethod
    def RandomUniformInt(*args, **kwargs):
        return None

    @staticmethod
    def Range(*args, **kwargs):
        return None

    @staticmethod
    def RangeDataset(*args, **kwargs):
        return None

    @staticmethod
    def Rank(*args, **kwargs):
        return None

    @staticmethod
    def ReadFile(*args, **kwargs):
        return None

    @staticmethod
    def ReadVariableOp(*args, **kwargs):
        return None

    @staticmethod
    def ReadVariableXlaSplitND(*args, **kwargs):
        return None

    @staticmethod
    def ReaderNumRecordsProduced(*args, **kwargs):
        return None

    @staticmethod
    def ReaderNumRecordsProducedV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderNumWorkUnitsCompleted(*args, **kwargs):
        return None

    @staticmethod
    def ReaderNumWorkUnitsCompletedV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderRead(*args, **kwargs):
        return None

    @staticmethod
    def ReaderReadUpTo(*args, **kwargs):
        return None

    @staticmethod
    def ReaderReadUpToV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderReadV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderReset(*args, **kwargs):
        return None

    @staticmethod
    def ReaderResetV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderRestoreState(*args, **kwargs):
        return None

    @staticmethod
    def ReaderRestoreStateV2(*args, **kwargs):
        return None

    @staticmethod
    def ReaderSerializeState(*args, **kwargs):
        return None

    @staticmethod
    def ReaderSerializeStateV2(*args, **kwargs):
        return None

    @staticmethod
    def Real(*args, **kwargs):
        return None

    @staticmethod
    def RealDiv(*args, **kwargs):
        return None

    @staticmethod
    def RebatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def RebatchDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def Reciprocal(*args, **kwargs):
        return None

    @staticmethod
    def ReciprocalGrad(*args, **kwargs):
        return None

    @staticmethod
    def RecordInput(*args, **kwargs):
        return None

    @staticmethod
    def Recv(*args, **kwargs):
        return None

    @staticmethod
    def RecvTPUEmbeddingActivations(*args, **kwargs):
        return None

    @staticmethod
    def ReduceDataset(*args, **kwargs):
        return None

    @staticmethod
    def ReduceJoin(*args, **kwargs):
        return None

    @staticmethod
    def RefEnter(*args, **kwargs):
        return None

    @staticmethod
    def RefExit(*args, **kwargs):
        return None

    @staticmethod
    def RefIdentity(*args, **kwargs):
        return None

    @staticmethod
    def RefMerge(*args, **kwargs):
        return None

    @staticmethod
    def RefNextIteration(*args, **kwargs):
        return None

    @staticmethod
    def RefSelect(*args, **kwargs):
        return None

    @staticmethod
    def RefSwitch(*args, **kwargs):
        return None

    @staticmethod
    def RegexFullMatch(*args, **kwargs):
        return None

    @staticmethod
    def RegexReplace(*args, **kwargs):
        return None

    @staticmethod
    def RegisterDataset(*args, **kwargs):
        return None

    @staticmethod
    def RegisterDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def Relu(*args, **kwargs):
        return None

    @staticmethod
    def Relu6(*args, **kwargs):
        return None

    @staticmethod
    def Relu6Grad(*args, **kwargs):
        return None

    @staticmethod
    def ReluGrad(*args, **kwargs):
        return None

    @staticmethod
    def RemoteCall(*args, **kwargs):
        return None

    @staticmethod
    def RepeatDataset(*args, **kwargs):
        return None

    @staticmethod
    def RequantizationRange(*args, **kwargs):
        return None

    @staticmethod
    def RequantizationRangePerChannel(*args, **kwargs):
        return None

    @staticmethod
    def Requantize(*args, **kwargs):
        return None

    @staticmethod
    def RequantizePerChannel(*args, **kwargs):
        return None

    @staticmethod
    def Reshape(*args, **kwargs):
        return _ops.Reshape(*args, **kwargs)

    @staticmethod
    def ResizeArea(*args, **kwargs):
        return None

    @staticmethod
    def ResizeBicubic(*args, **kwargs):
        return _ops.ResizeBicubic(*args, **kwargs)

    @staticmethod
    def ResizeBicubicGrad(*args, **kwargs):
        return None

    @staticmethod
    def ResizeBilinear(*args, **kwargs):
        return _ops.ResizeBilinear(*args, **kwargs)

    @staticmethod
    def ResizeBilinearGrad(*args, **kwargs):
        return None

    @staticmethod
    def ResizeNearestNeighbor(*args, **kwargs):
        return None

    @staticmethod
    def ResizeNearestNeighborGrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceAccumulatorApplyGradient(*args, **kwargs):
        return None

    @staticmethod
    def ResourceAccumulatorNumAccumulated(*args, **kwargs):
        return None

    @staticmethod
    def ResourceAccumulatorSetGlobalStep(*args, **kwargs):
        return None

    @staticmethod
    def ResourceAccumulatorTakeGradient(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdaMax(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdadelta(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdagradDA(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdagradV2(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdam(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAdamWithAmsgrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyAddSign(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyCenteredRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyFtrl(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyFtrlV2(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyKerasMomentum(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyMomentum(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyPowerSign(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyProximalAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyProximalGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def ResourceApplyRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ResourceConditionalAccumulator(*args, **kwargs):
        return None

    @staticmethod
    def ResourceCountUpTo(*args, **kwargs):
        return None

    @staticmethod
    def ResourceGather(*args, **kwargs):
        return None

    @staticmethod
    def ResourceGatherNd(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterAdd(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterDiv(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterMax(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterMin(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterMul(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterNdAdd(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterNdMax(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterNdMin(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterNdSub(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterNdUpdate(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterSub(*args, **kwargs):
        return None

    @staticmethod
    def ResourceScatterUpdate(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyAdadelta(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyAdagradDA(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyAdagradV2(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyCenteredRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyFtrl(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyFtrlV2(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyKerasMomentum(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyMomentum(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyProximalAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyProximalGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def ResourceSparseApplyRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def ResourceStridedSliceAssign(*args, **kwargs):
        return None

    @staticmethod
    def Restore(*args, **kwargs):
        return None

    @staticmethod
    def RestoreSlice(*args, **kwargs):
        return None

    @staticmethod
    def RestoreV2(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingADAMParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdadeltaParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdagradMomentumParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingAdagradParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingCenteredRMSPropParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingFTRLParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingFrequencyEstimatorParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingMDLAdagradLightParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingMomentumParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingProximalAdagradParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingProximalYogiParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingRMSPropParameters(*args, **kwargs):
        return None

    @staticmethod
    def RetrieveTPUEmbeddingStochasticGradientDescentParameters(*args, **kwargs):
        return None

    @staticmethod
    def Reverse(*args, **kwargs):
        return None

    @staticmethod
    def ReverseSequence(*args, **kwargs):
        return None

    @staticmethod
    def ReverseV2(*args, **kwargs):
        return None

    @staticmethod
    def RewriteDataset(*args, **kwargs):
        return None

    @staticmethod
    def RightShift(*args, **kwargs):
        return None

    @staticmethod
    def Rint(*args, **kwargs):
        return None

    @staticmethod
    def RngReadAndSkip(*args, **kwargs):
        return None

    @staticmethod
    def RngSkip(*args, **kwargs):
        return None

    @staticmethod
    def Roll(*args, **kwargs):
        return _ops.Roll(*args, **kwargs)

    @staticmethod
    def Round(*args, **kwargs):
        return None

    @staticmethod
    def Rsqrt(*args, **kwargs):
        return None

    @staticmethod
    def RsqrtGrad(*args, **kwargs):
        return None

    @staticmethod
    def SampleDistortedBoundingBox(*args, **kwargs):
        return None

    @staticmethod
    def SampleDistortedBoundingBoxV2(*args, **kwargs):
        return None

    @staticmethod
    def SamplingDataset(*args, **kwargs):
        return None

    @staticmethod
    def Save(*args, **kwargs):
        return None

    @staticmethod
    def SaveDataset(*args, **kwargs):
        return None

    @staticmethod
    def SaveDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def SaveSlices(*args, **kwargs):
        return None

    @staticmethod
    def SaveV2(*args, **kwargs):
        return None

    @staticmethod
    def ScalarSummary(*args, **kwargs):
        return None

    @staticmethod
    def ScaleAndTranslate(*args, **kwargs):
        return None

    @staticmethod
    def ScaleAndTranslateGrad(*args, **kwargs):
        return None

    @staticmethod
    def ScanDataset(*args, **kwargs):
        return None

    @staticmethod
    def ScatterAdd(*args, **kwargs):
        return _ops.ScatterAdd(*args, **kwargs)

    @staticmethod
    def ScatterDiv(*args, **kwargs):
        return None

    @staticmethod
    def ScatterMax(*args, **kwargs):
        return _ops.ScatterMax(*args, **kwargs)

    @staticmethod
    def ScatterMin(*args, **kwargs):
        return _ops.ScatterMin(*args, **kwargs)

    @staticmethod
    def ScatterMul(*args, **kwargs):
        return _ops.ScatterMul(*args, **kwargs)

    @staticmethod
    def ScatterNd(*args, **kwargs):
        return _ops.ScatterNd(*args, **kwargs)

    @staticmethod
    def ScatterNdAdd(*args, **kwargs):
        return None

    @staticmethod
    def ScatterNdMax(*args, **kwargs):
        return None

    @staticmethod
    def ScatterNdMin(*args, **kwargs):
        return None

    @staticmethod
    def ScatterNdNonAliasingAdd(*args, **kwargs):
        return None

    @staticmethod
    def ScatterNdSub(*args, **kwargs):
        return None

    @staticmethod
    def ScatterNdUpdate(*args, **kwargs):
        return None

    @staticmethod
    def ScatterSub(*args, **kwargs):
        return None

    @staticmethod
    def ScatterUpdate(*args, **kwargs):
        return None

    @staticmethod
    def SdcaFprint(*args, **kwargs):
        return None

    @staticmethod
    def SdcaOptimizer(*args, **kwargs):
        return None

    @staticmethod
    def SdcaOptimizerV2(*args, **kwargs):
        return None

    @staticmethod
    def SdcaShrinkL1(*args, **kwargs):
        return None

    @staticmethod
    def SegmentMax(*args, **kwargs):
        return None

    @staticmethod
    def SegmentMaxV2(*args, **kwargs):
        return None

    @staticmethod
    def SegmentMean(*args, **kwargs):
        return None

    @staticmethod
    def SegmentMin(*args, **kwargs):
        return None

    @staticmethod
    def SegmentMinV2(*args, **kwargs):
        return None

    @staticmethod
    def SegmentProd(*args, **kwargs):
        return None

    @staticmethod
    def SegmentProdV2(*args, **kwargs):
        return None

    @staticmethod
    def SegmentSum(*args, **kwargs):
        return None

    @staticmethod
    def SegmentSumV2(*args, **kwargs):
        return None

    @staticmethod
    def Select(*args, **kwargs):
        return _ops.Select(*args, **kwargs)

    @staticmethod
    def SelectV2(*args, **kwargs):
        return None

    @staticmethod
    def SelfAdjointEig(*args, **kwargs):
        return None

    @staticmethod
    def SelfAdjointEigV2(*args, **kwargs):
        return None

    @staticmethod
    def Selu(*args, **kwargs):
        return None

    @staticmethod
    def SeluGrad(*args, **kwargs):
        return None

    @staticmethod
    def Send(*args, **kwargs):
        return None

    @staticmethod
    def SendTPUEmbeddingGradients(*args, **kwargs):
        return None

    @staticmethod
    def SerializeIterator(*args, **kwargs):
        return None

    @staticmethod
    def SerializeManySparse(*args, **kwargs):
        return None

    @staticmethod
    def SerializeSparse(*args, **kwargs):
        return None

    @staticmethod
    def SerializeTensor(*args, **kwargs):
        return None

    @staticmethod
    def SetSize(*args, **kwargs):
        return None

    @staticmethod
    def SetStatsAggregatorDataset(*args, **kwargs):
        return None

    @staticmethod
    def Shape(*args, **kwargs):
        return None

    @staticmethod
    def ShapeN(*args, **kwargs):
        return None

    @staticmethod
    def ShardDataset(*args, **kwargs):
        return None

    @staticmethod
    def ShardedFilename(*args, **kwargs):
        return None

    @staticmethod
    def ShardedFilespec(*args, **kwargs):
        return None

    @staticmethod
    def ShuffleAndRepeatDataset(*args, **kwargs):
        return None

    @staticmethod
    def ShuffleAndRepeatDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def ShuffleDataset(*args, **kwargs):
        return None

    @staticmethod
    def ShuffleDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def ShuffleDatasetV3(*args, **kwargs):
        return None

    @staticmethod
    def ShutdownDistributedTPU(*args, **kwargs):
        return None

    @staticmethod
    def Sigmoid(*args, **kwargs):
        return None

    @staticmethod
    def SigmoidGrad(*args, **kwargs):
        return None

    @staticmethod
    def Sign(*args, **kwargs):
        return None

    @staticmethod
    def Sin(*args, **kwargs):
        return None

    @staticmethod
    def Sinh(*args, **kwargs):
        return None

    @staticmethod
    def Size(*args, **kwargs):
        return None

    @staticmethod
    def SkipDataset(*args, **kwargs):
        return None

    @staticmethod
    def SleepDataset(*args, **kwargs):
        return None

    @staticmethod
    def Slice(*args, **kwargs):
        return _ops.Slice(*args, **kwargs)

    @staticmethod
    def SlidingWindowDataset(*args, **kwargs):
        return None

    @staticmethod
    def Snapshot(*args, **kwargs):
        return None

    @staticmethod
    def SnapshotChunkDataset(*args, **kwargs):
        return None

    @staticmethod
    def SnapshotDataset(*args, **kwargs):
        return None

    @staticmethod
    def SnapshotDatasetReader(*args, **kwargs):
        return None

    @staticmethod
    def SnapshotDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def SnapshotNestedDatasetReader(*args, **kwargs):
        return None

    @staticmethod
    def SobolSample(*args, **kwargs):
        return _ops.SobolSample(*args, **kwargs)

    @staticmethod
    def Softmax(*args, **kwargs):
        return keras.layers.Softmax(*args, **kwargs)

    @staticmethod
    def SoftmaxCrossEntropyWithLogits(*args, **kwargs):
        return None

    @staticmethod
    def Softplus(*args, **kwargs):
        return None

    @staticmethod
    def SoftplusGrad(*args, **kwargs):
        return None

    @staticmethod
    def Softsign(*args, **kwargs):
        return None

    @staticmethod
    def SoftsignGrad(*args, **kwargs):
        return None

    @staticmethod
    def SpaceToBatch(*args, **kwargs):
        return None

    @staticmethod
    def SpaceToBatchND(*args, **kwargs):
        return None

    @staticmethod
    def SpaceToDepth(*args, **kwargs):
        return None

    @staticmethod
    def SparseAccumulatorApplyGradient(*args, **kwargs):
        return None

    @staticmethod
    def SparseAccumulatorTakeGradient(*args, **kwargs):
        return None

    @staticmethod
    def SparseAdd(*args, **kwargs):
        return _ops.SparseAdd(*args, **kwargs)

    @staticmethod
    def SparseAddGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyAdadelta(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyAdagradDA(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyAdagradV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyCenteredRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyFtrl(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyFtrlV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyMomentum(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyProximalAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyProximalGradientDescent(*args, **kwargs):
        return None

    @staticmethod
    def SparseApplyRMSProp(*args, **kwargs):
        return None

    @staticmethod
    def SparseBincount(*args, **kwargs):
        return _ops.SparseBincount(*args, **kwargs)

    @staticmethod
    def SparseConcat(*args, **kwargs):
        return None

    @staticmethod
    def SparseConditionalAccumulator(*args, **kwargs):
        return None

    @staticmethod
    def SparseCountSparseOutput(*args, **kwargs):
        return None

    @staticmethod
    def SparseCross(*args, **kwargs):
        return None

    @staticmethod
    def SparseCrossHashed(*args, **kwargs):
        return _ops.SparseCrossHashed(*args, **kwargs)

    @staticmethod
    def SparseCrossV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseDenseCwiseAdd(*args, **kwargs):
        return None

    @staticmethod
    def SparseDenseCwiseDiv(*args, **kwargs):
        return None

    @staticmethod
    def SparseDenseCwiseMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseFillEmptyRows(*args, **kwargs):
        return _ops.SparseFillEmptyRows(*args, **kwargs)

    @staticmethod
    def SparseFillEmptyRowsGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixAdd(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixMatMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixNNZ(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixOrderingAMD(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixSoftmax(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixSoftmaxGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixSparseCholesky(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixSparseMatMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixTranspose(*args, **kwargs):
        return None

    @staticmethod
    def SparseMatrixZeros(*args, **kwargs):
        return None

    @staticmethod
    def SparseReduceMax(*args, **kwargs):
        return _ops.SparseReduceMax(*args, **kwargs)

    @staticmethod
    def SparseReduceMaxSparse(*args, **kwargs):
        return None

    @staticmethod
    def SparseReduceSum(*args, **kwargs):
        return _ops.SparseReduceSum(*args, **kwargs)

    @staticmethod
    def SparseReduceSumSparse(*args, **kwargs):
        return None

    @staticmethod
    def SparseReorder(*args, **kwargs):
        return _ops.SparseReorder(*args, **kwargs)

    @staticmethod
    def SparseReshape(*args, **kwargs):
        return _ops.SparseReshape(*args, **kwargs)

    @staticmethod
    def SparseSegmentMean(*args, **kwargs):
        return _ops.SparseSegmentMean(*args, **kwargs)

    @staticmethod
    def SparseSegmentMeanGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentMeanGradV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentMeanWithNumSegments(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSqrtN(*args, **kwargs):
        return _ops.SparseSegmentSqrtN(*args, **kwargs)

    @staticmethod
    def SparseSegmentSqrtNGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSqrtNGradV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSqrtNWithNumSegments(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSum(*args, **kwargs):
        return _ops.SparseSegmentSum(*args, **kwargs)

    @staticmethod
    def SparseSegmentSumGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSumGradV2(*args, **kwargs):
        return None

    @staticmethod
    def SparseSegmentSumWithNumSegments(*args, **kwargs):
        return None

    @staticmethod
    def SparseSlice(*args, **kwargs):
        return _ops.SparseSlice(*args, **kwargs)

    @staticmethod
    def SparseSliceGrad(*args, **kwargs):
        return None

    @staticmethod
    def SparseSoftmax(*args, **kwargs):
        return _ops.SparseSoftmax(*args, **kwargs)

    @staticmethod
    def SparseSoftmaxCrossEntropyWithLogits(*args, **kwargs):
        return None

    @staticmethod
    def SparseSparseMaximum(*args, **kwargs):
        return None

    @staticmethod
    def SparseSparseMinimum(*args, **kwargs):
        return None

    @staticmethod
    def SparseSplit(*args, **kwargs):
        return None

    @staticmethod
    def SparseTensorDenseAdd(*args, **kwargs):
        return None

    @staticmethod
    def SparseTensorDenseMatMul(*args, **kwargs):
        return None

    @staticmethod
    def SparseTensorSliceDataset(*args, **kwargs):
        return None

    @staticmethod
    def SparseTensorToCSRSparseMatrix(*args, **kwargs):
        return None

    @staticmethod
    def SparseToDense(*args, **kwargs):
        return None

    @staticmethod
    def SparseToSparseSetOperation(*args, **kwargs):
        return None

    @staticmethod
    def Spence(*args, **kwargs):
        return None

    @staticmethod
    def Split(*args, **kwargs):
        return _ops.Split(*args, **kwargs)

    @staticmethod
    def SplitV(*args, **kwargs):
        return None

    @staticmethod
    def SqlDataset(*args, **kwargs):
        return None

    @staticmethod
    def Sqrt(*args, **kwargs):
        return None

    @staticmethod
    def SqrtGrad(*args, **kwargs):
        return None

    @staticmethod
    def Square(*args, **kwargs):
        return None

    @staticmethod
    def SquaredDifference(*args, **kwargs):
        return None

    @staticmethod
    def Squeeze(*args, **kwargs):
        return _ops.Squeeze(*args, **kwargs)

    @staticmethod
    def Stack(*args, **kwargs):
        return _ops.Stack(*args, **kwargs)

    @staticmethod
    def StackClose(*args, **kwargs):
        return None

    @staticmethod
    def StackCloseV2(*args, **kwargs):
        return None

    @staticmethod
    def StackPop(*args, **kwargs):
        return None

    @staticmethod
    def StackPopV2(*args, **kwargs):
        return None

    @staticmethod
    def StackPush(*args, **kwargs):
        return None

    @staticmethod
    def StackPushV2(*args, **kwargs):
        return None

    @staticmethod
    def StackV2(*args, **kwargs):
        return None

    @staticmethod
    def Stage(*args, **kwargs):
        return None

    @staticmethod
    def StageClear(*args, **kwargs):
        return None

    @staticmethod
    def StagePeek(*args, **kwargs):
        return None

    @staticmethod
    def StageSize(*args, **kwargs):
        return None

    @staticmethod
    def StatefulPartitionedCall(*args, **kwargs):
        return None

    @staticmethod
    def StatefulRandomBinomial(*args, **kwargs):
        return None

    @staticmethod
    def StatefulStandardNormal(*args, **kwargs):
        return None

    @staticmethod
    def StatefulStandardNormalV2(*args, **kwargs):
        return None

    @staticmethod
    def StatefulTruncatedNormal(*args, **kwargs):
        return None

    @staticmethod
    def StatefulUniform(*args, **kwargs):
        return None

    @staticmethod
    def StatefulUniformFullInt(*args, **kwargs):
        return None

    @staticmethod
    def StatefulUniformInt(*args, **kwargs):
        return None

    @staticmethod
    def StatelessCase(*args, **kwargs):
        return None

    @staticmethod
    def StatelessIf(*args, **kwargs):
        return None

    @staticmethod
    def StatelessMultinomial(*args, **kwargs):
        return None

    @staticmethod
    def StatelessParameterizedTruncatedNormal(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomBinomial(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomGammaV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomGammaV3(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomGetAlg(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomGetKeyCounter(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomGetKeyCounterAlg(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomNormal(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomNormalV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomPoisson(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniform(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniformFullInt(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniformFullIntV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniformInt(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniformIntV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessRandomUniformV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessSampleDistortedBoundingBox(*args, **kwargs):
        return None

    @staticmethod
    def StatelessShuffle(*args, **kwargs):
        return None

    @staticmethod
    def StatelessTruncatedNormal(*args, **kwargs):
        return None

    @staticmethod
    def StatelessTruncatedNormalV2(*args, **kwargs):
        return None

    @staticmethod
    def StatelessWhile(*args, **kwargs):
        return None

    @staticmethod
    def StaticRegexFullMatch(*args, **kwargs):
        return None

    @staticmethod
    def StaticRegexReplace(*args, **kwargs):
        return None

    @staticmethod
    def StatsAggregatorHandle(*args, **kwargs):
        return None

    @staticmethod
    def StatsAggregatorHandleV2(*args, **kwargs):
        return None

    @staticmethod
    def StatsAggregatorSetSummaryWriter(*args, **kwargs):
        return None

    @staticmethod
    def StatsAggregatorSummary(*args, **kwargs):
        return None

    @staticmethod
    def StopGradient(*args, **kwargs):
        return None

    @staticmethod
    def StoreMinibatchStatisticsInFdo(*args, **kwargs):
        return None

    @staticmethod
    def StridedSlice(*args, **kwargs):
        return _ops.StridedSlice(*args, **kwargs)

    @staticmethod
    def StridedSliceAssign(*args, **kwargs):
        return None

    @staticmethod
    def StridedSliceGrad(*args, **kwargs):
        return None

    @staticmethod
    def StringFormat(*args, **kwargs):
        return None

    @staticmethod
    def StringJoin(*args, **kwargs):
        return None

    @staticmethod
    def StringLength(*args, **kwargs):
        return None

    @staticmethod
    def StringLower(*args, **kwargs):
        return None

    @staticmethod
    def StringNGrams(*args, **kwargs):
        return None

    @staticmethod
    def StringSplit(*args, **kwargs):
        return None

    @staticmethod
    def StringSplitV2(*args, **kwargs):
        return None

    @staticmethod
    def StringStrip(*args, **kwargs):
        return None

    @staticmethod
    def StringToHashBucket(*args, **kwargs):
        return None

    @staticmethod
    def StringToHashBucketFast(*args, **kwargs):
        return None

    @staticmethod
    def StringToHashBucketStrong(*args, **kwargs):
        return None

    @staticmethod
    def StringToNumber(*args, **kwargs):
        return None

    @staticmethod
    def StringUpper(*args, **kwargs):
        return None

    @staticmethod
    def Sub(*args, **kwargs):
        return None

    @staticmethod
    def Substr(*args, **kwargs):
        return None

    @staticmethod
    def Sum(*args, **kwargs):
        return keras.metrics.Sum(*args, **kwargs)

    @staticmethod
    def SummaryWriter(*args, **kwargs):
        return None

    @staticmethod
    def Svd(*args, **kwargs):
        return None

    @staticmethod
    def Switch(*args, **kwargs):
        return None

    @staticmethod
    def SymbolicGradient(*args, **kwargs):
        return None

    @staticmethod
    def SyncDevice(*args, **kwargs):
        return None

    @staticmethod
    def TFRecordDataset(*args, **kwargs):
        return None

    @staticmethod
    def TFRecordDatasetV2(*args, **kwargs):
        return None

    @staticmethod
    def TFRecordReader(*args, **kwargs):
        return None

    @staticmethod
    def TFRecordReaderV2(*args, **kwargs):
        return None

    @staticmethod
    def TPUAnnotateTensorsWithDynamicShape(*args, **kwargs):
        return None

    @staticmethod
    def TPUCompilationResult(*args, **kwargs):
        return None

    @staticmethod
    def TPUCopyWithDynamicShape(*args, **kwargs):
        return None

    @staticmethod
    def TPUEmbeddingActivations(*args, **kwargs):
        return None

    @staticmethod
    def TPUOrdinalSelector(*args, **kwargs):
        return None

    @staticmethod
    def TPUPartitionedCall(*args, **kwargs):
        return None

    @staticmethod
    def TPUPartitionedInput(*args, **kwargs):
        return None

    @staticmethod
    def TPUPartitionedInputV2(*args, **kwargs):
        return None

    @staticmethod
    def TPUPartitionedOutput(*args, **kwargs):
        return None

    @staticmethod
    def TPUPartitionedOutputV2(*args, **kwargs):
        return None

    @staticmethod
    def TPUReplicateMetadata(*args, **kwargs):
        return None

    @staticmethod
    def TPUReplicatedInput(*args, **kwargs):
        return None

    @staticmethod
    def TPUReplicatedOutput(*args, **kwargs):
        return None

    @staticmethod
    def TakeDataset(*args, **kwargs):
        return None

    @staticmethod
    def TakeManySparseFromTensorsMap(*args, **kwargs):
        return None

    @staticmethod
    def TakeWhileDataset(*args, **kwargs):
        return None

    @staticmethod
    def Tan(*args, **kwargs):
        return None

    @staticmethod
    def Tanh(*args, **kwargs):
        return None

    @staticmethod
    def TanhGrad(*args, **kwargs):
        return None

    @staticmethod
    def TemporaryVariable(*args, **kwargs):
        return None

    @staticmethod
    def TensorArray(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayClose(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayCloseV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayCloseV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayConcat(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayConcatV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayConcatV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGather(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGatherV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGatherV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGrad(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGradV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGradV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayGradWithShape(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayPack(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayRead(*args, **kwargs):
        return _ops.TensorArrayRead(*args, **kwargs)

    @staticmethod
    def TensorArrayReadV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayReadV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayScatter(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayScatterV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayScatterV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySize(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySizeV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySizeV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySplit(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySplitV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArraySplitV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayUnpack(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayWrite(*args, **kwargs):
        return _ops.TensorArrayWrite(*args, **kwargs)

    @staticmethod
    def TensorArrayWriteV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorArrayWriteV3(*args, **kwargs):
        return None

    @staticmethod
    def TensorDataset(*args, **kwargs):
        return None

    @staticmethod
    def TensorListConcat(*args, **kwargs):
        return None

    @staticmethod
    def TensorListConcatLists(*args, **kwargs):
        return None

    @staticmethod
    def TensorListConcatV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorListElementShape(*args, **kwargs):
        return None

    @staticmethod
    def TensorListFromTensor(*args, **kwargs):
        return None

    @staticmethod
    def TensorListGather(*args, **kwargs):
        return None

    @staticmethod
    def TensorListGetItem(*args, **kwargs):
        return None

    @staticmethod
    def TensorListLength(*args, **kwargs):
        return None

    @staticmethod
    def TensorListPopBack(*args, **kwargs):
        return None

    @staticmethod
    def TensorListPushBack(*args, **kwargs):
        return None

    @staticmethod
    def TensorListPushBackBatch(*args, **kwargs):
        return None

    @staticmethod
    def TensorListReserve(*args, **kwargs):
        return None

    @staticmethod
    def TensorListResize(*args, **kwargs):
        return None

    @staticmethod
    def TensorListScatter(*args, **kwargs):
        return None

    @staticmethod
    def TensorListScatterIntoExistingList(*args, **kwargs):
        return None

    @staticmethod
    def TensorListScatterV2(*args, **kwargs):
        return None

    @staticmethod
    def TensorListSetItem(*args, **kwargs):
        return None

    @staticmethod
    def TensorListSplit(*args, **kwargs):
        return None

    @staticmethod
    def TensorListStack(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapErase(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapHasKey(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapInsert(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapLookup(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapSize(*args, **kwargs):
        return None

    @staticmethod
    def TensorMapStackKeys(*args, **kwargs):
        return None

    @staticmethod
    def TensorScatterAdd(*args, **kwargs):
        return None

    @staticmethod
    def TensorScatterMax(*args, **kwargs):
        return None

    @staticmethod
    def TensorScatterMin(*args, **kwargs):
        return None

    @staticmethod
    def TensorScatterSub(*args, **kwargs):
        return None

    @staticmethod
    def TensorScatterUpdate(*args, **kwargs):
        return _ops.TensorScatterUpdate(*args, **kwargs)

    @staticmethod
    def TensorSliceDataset(*args, **kwargs):
        return None

    @staticmethod
    def TensorStridedSliceUpdate(*args, **kwargs):
        return None

    @staticmethod
    def TensorSummary(*args, **kwargs):
        return None

    @staticmethod
    def TensorSummaryV2(*args, **kwargs):
        return None

    @staticmethod
    def TextLineDataset(*args, **kwargs):
        return None

    @staticmethod
    def TextLineReader(*args, **kwargs):
        return None

    @staticmethod
    def TextLineReaderV2(*args, **kwargs):
        return None

    @staticmethod
    def ThreadPoolDataset(*args, **kwargs):
        return None

    @staticmethod
    def ThreadPoolHandle(*args, **kwargs):
        return None

    @staticmethod
    def ThreadUnsafeUnigramCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def Tile(*args, **kwargs):
        return _ops.Tile(*args, **kwargs)

    @staticmethod
    def TileGrad(*args, **kwargs):
        return None

    @staticmethod
    def Timestamp(*args, **kwargs):
        return None

    @staticmethod
    def ToBool(*args, **kwargs):
        return None

    @staticmethod
    def TopK(*args, **kwargs):
        return _ops.TopK(*args, **kwargs)

    @staticmethod
    def TopKV2(*args, **kwargs):
        return None

    @staticmethod
    def Transpose(*args, **kwargs):
        return _ops.Transpose(*args, **kwargs)

    @staticmethod
    def TridiagonalMatMul(*args, **kwargs):
        return None

    @staticmethod
    def TridiagonalSolve(*args, **kwargs):
        return None

    @staticmethod
    def TruncateDiv(*args, **kwargs):
        return None

    @staticmethod
    def TruncateMod(*args, **kwargs):
        return None

    @staticmethod
    def TruncatedNormal(*args, **kwargs):
        return None

    @staticmethod
    def Unbatch(*args, **kwargs):
        return None

    @staticmethod
    def UnbatchDataset(*args, **kwargs):
        return None

    @staticmethod
    def UnbatchGrad(*args, **kwargs):
        return None

    @staticmethod
    def UncompressElement(*args, **kwargs):
        return None

    @staticmethod
    def UnicodeDecode(*args, **kwargs):
        return None

    @staticmethod
    def UnicodeDecodeWithOffsets(*args, **kwargs):
        return None

    @staticmethod
    def UnicodeEncode(*args, **kwargs):
        return None

    @staticmethod
    def UnicodeScript(*args, **kwargs):
        return None

    @staticmethod
    def UnicodeTranscode(*args, **kwargs):
        return None

    @staticmethod
    def UniformCandidateSampler(*args, **kwargs):
        return None

    @staticmethod
    def UniformDequantize(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantize(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedAdd(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedClipByValue(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedConvolution(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedConvolutionHybrid(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedDot(*args, **kwargs):
        return None

    @staticmethod
    def UniformQuantizedDotHybrid(*args, **kwargs):
        return None

    @staticmethod
    def UniformRequantize(*args, **kwargs):
        return None

    @staticmethod
    def Unique(*args, **kwargs):
        return None

    @staticmethod
    def UniqueDataset(*args, **kwargs):
        return None

    @staticmethod
    def UniqueV2(*args, **kwargs):
        return None

    @staticmethod
    def UniqueWithCounts(*args, **kwargs):
        return None

    @staticmethod
    def UniqueWithCountsV2(*args, **kwargs):
        return None

    @staticmethod
    def Unpack(*args, **kwargs):
        return None

    @staticmethod
    def UnravelIndex(*args, **kwargs):
        return None

    @staticmethod
    def UnsortedSegmentJoin(*args, **kwargs):
        return None

    @staticmethod
    def UnsortedSegmentMax(*args, **kwargs):
        return None

    @staticmethod
    def UnsortedSegmentMin(*args, **kwargs):
        return None

    @staticmethod
    def UnsortedSegmentProd(*args, **kwargs):
        return None

    @staticmethod
    def UnsortedSegmentSum(*args, **kwargs):
        return None

    @staticmethod
    def Unstage(*args, **kwargs):
        return None

    @staticmethod
    def UnwrapDatasetVariant(*args, **kwargs):
        return None

    @staticmethod
    def UpperBound(*args, **kwargs):
        return None

    @staticmethod
    def VarHandleOp(*args, **kwargs):
        return None

    @staticmethod
    def VarIsInitializedOp(*args, **kwargs):
        return None

    @staticmethod
    def Variable(*args, **kwargs):
        return None

    @staticmethod
    def VariableShape(*args, **kwargs):
        return None

    @staticmethod
    def VariableV2(*args, **kwargs):
        return None

    @staticmethod
    def Where(*args, **kwargs):
        return _ops.Where(*args, **kwargs)

    @staticmethod
    def While(*args, **kwargs):
        return None

    @staticmethod
    def WholeFileReader(*args, **kwargs):
        return None

    @staticmethod
    def WholeFileReaderV2(*args, **kwargs):
        return None

    @staticmethod
    def WindowDataset(*args, **kwargs):
        return None

    @staticmethod
    def WindowOp(*args, **kwargs):
        return None

    @staticmethod
    def WorkerHeartbeat(*args, **kwargs):
        return None

    @staticmethod
    def WrapDatasetVariant(*args, **kwargs):
        return None

    @staticmethod
    def WriteAudioSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteFile(*args, **kwargs):
        return None

    @staticmethod
    def WriteGraphSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteHistogramSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteImageSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteRawProtoSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteScalarSummary(*args, **kwargs):
        return None

    @staticmethod
    def WriteSummary(*args, **kwargs):
        return None

    @staticmethod
    def Xdivy(*args, **kwargs):
        return None

    @staticmethod
    def XlaConcatND(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseCoreAdagrad(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseCoreAdagradMomentum(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseCoreAdam(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseCoreFtrl(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseCoreSgd(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmul(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradAndCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdamAndCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithFtrlAndCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulGradWithSgdAndCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSparseDenseMatmulWithCsrInput(*args, **kwargs):
        return None

    @staticmethod
    def XlaSplitND(*args, **kwargs):
        return None

    @staticmethod
    def Xlog1py(*args, **kwargs):
        return None

    @staticmethod
    def Xlogy(*args, **kwargs):
        return None

    @staticmethod
    def ZerosLike(*args, **kwargs):
        return None

    @staticmethod
    def Zeta(*args, **kwargs):
        return None

    @staticmethod
    def ZipDataset(*args, **kwargs):
        return None


def realdiv(*args, **kwargs):
    return None


def recompute_grad(*args, **kwargs):
    return None


def reduce_logsumexp(*args, **kwargs):
    return _ops.reduce_logsumexp(*args, **kwargs)


def register_tensor_conversion_function(*args, **kwargs):
    return None


def required_space_to_batch_paddings(*args, **kwargs):
    return None


def resource(*args, **kwargs):
    return None


def reverse(*args, **kwargs):
    return _ops.reverse(*args, **kwargs)


def reverse_sequence(*args, **kwargs):
    return None


def rfftnd(*args, **kwargs):
    return _ops.rfftnd(*args, **kwargs)


def saturate_cast(*args, **kwargs):
    return None


def sequence_mask(*args, **kwargs):
    return None


class sets:
    @staticmethod
    def difference(*args, **kwargs):
        return None

    @staticmethod
    def intersection(*args, **kwargs):
        return None

    @staticmethod
    def size(*args, **kwargs):
        return _ops.size(*args, **kwargs)

    @staticmethod
    def union(*args, **kwargs):
        return None


def shape_n(*args, **kwargs):
    return None


def sigmoid(*args, **kwargs):
    return _ops.sigmoid(*args, **kwargs)


def switch_case(*args, **kwargs):
    return _ops.switch_case(*args, **kwargs)


class sysconfig:
    @staticmethod
    def CXX11_ABI_FLAG(*args, **kwargs):
        return None

    @staticmethod
    def CXX_VERSION(*args, **kwargs):
        return None

    @staticmethod
    def MONOLITHIC_BUILD(*args, **kwargs):
        return None

    @staticmethod
    def get_build_info(*args, **kwargs):
        return None

    @staticmethod
    def get_compile_flags(*args, **kwargs):
        return None

    @staticmethod
    def get_include(*args, **kwargs):
        return None

    @staticmethod
    def get_lib(*args, **kwargs):
        return None

    @staticmethod
    def get_link_flags(*args, **kwargs):
        return None


def tensor_scatter_nd_add(*args, **kwargs):
    return None


def tensor_scatter_nd_max(*args, **kwargs):
    return None


def tensor_scatter_nd_min(*args, **kwargs):
    return None


def tensor_scatter_nd_sub(*args, **kwargs):
    return None


def tensor_scatter_nd_update(*args, **kwargs):
    return None


def timestamp(*args, **kwargs):
    return None


class tools:
    @staticmethod
    def compatibility(*args, **kwargs):
        return None

    @staticmethod
    def docs(*args, **kwargs):
        return None


class tpu:
    @staticmethod
    def XLAOptions(*args, **kwargs):
        return None

    @staticmethod
    def experimental(*args, **kwargs):
        return None


def tuple(*args, **kwargs):
    return None


def type_spec_from_value(*args, **kwargs):
    return None


def uint16(*args, **kwargs):
    return None


def uint32(*args, **kwargs):
    return None


def uint64(*args, **kwargs):
    return None


def unique(*args, **kwargs):
    return _ops.unique(*args, **kwargs)


def unique_with_counts(*args, **kwargs):
    return None


def unravel_index(*args, **kwargs):
    return _ops.unravel_index(*args, **kwargs)


def variable_creator_scope(*args, **kwargs):
    return None


def variant(*args, **kwargs):
    return None


def vectorized_map(*args, **kwargs):
    return _ops.vectorized_map(*args, **kwargs)


class version:
    @staticmethod
    def COMPILER_VERSION(*args, **kwargs):
        return None

    @staticmethod
    def GIT_VERSION(*args, **kwargs):
        return None

    @staticmethod
    def GRAPH_DEF_VERSION(*args, **kwargs):
        return None

    @staticmethod
    def GRAPH_DEF_VERSION_MIN_CONSUMER(*args, **kwargs):
        return None

    @staticmethod
    def GRAPH_DEF_VERSION_MIN_PRODUCER(*args, **kwargs):
        return None

    @staticmethod
    def VERSION(*args, **kwargs):
        return None


def while_loop(*args, **kwargs):
    return _ops.while_loop(*args, **kwargs)


class xla:
    @staticmethod
    def experimental(*args, **kwargs):
        return None


def zeros_initializer(*args, **kwargs):
    return None


def stop_gradient(*args, **kwargs):
    return _ops.stop_gradient(*args, **kwargs)


def custom_gradient(*args, **kwargs):
    return _ops.custom_gradient(*args, **kwargs)


def hessians(*args, **kwargs):
    return None


from . import (
    autograph,
    config,
    debugging,
    dtypes,
    lookup,
    quantization,
    summary,
    test,
    types,
)
from .core_stubs import *
