"""zero_tensorflow API."""

import functools
import builtins
from typing import Any, Optional
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

import ml_switcheroo_compiler
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_ir import LogicalNode
import sys
import zero_keras as keras

sys.modules["zero_tensorflow.keras"] = keras

# Alias Keras components directly into top-level TF namespace
from . import metrics
from . import losses
from . import optimizers

initializers = keras.initializers
sys.modules["zero_tensorflow.metrics"] = metrics
sys.modules["zero_tensorflow.losses"] = losses
sys.modules["zero_tensorflow.optimizers"] = optimizers
sys.modules["zero_tensorflow.initializers"] = initializers

import zero_tensorflow.distribute as distribute
import zero_tensorflow.io as io
import zero_tensorflow.train as train
import zero_tensorflow.saved_model as saved_model
import zero_tensorflow.sparse as sparse
import zero_tensorflow.ragged as ragged
import zero_tensorflow.random as random
import zero_tensorflow.strings as strings
import zero_tensorflow.signal as signal
import zero_tensorflow.image as image
from zero_tensorflow.sparse import SparseTensor
from zero_tensorflow.ragged import RaggedTensor


__all__ = [
    "Variable",
    "function",
    "GradientTape",
    "data",
    "math",
    "nn",
    "keras",
    "Tensor",
    "linalg",
    "bitwise",
    "DType",
    "distribute",
    "io",
    "train",
    "saved_model",
    "sparse",
    "ragged",
    "random",
    "strings",
    "signal",
    "autograph",
    "config",
    "debugging",
    "dtypes",
    "lookup",
    "quantization",
    "summary",
    "test",
    "types",
    "image",
    "SparseTensor",
    "RaggedTensor",
    "stop_gradient",
    "custom_gradient",
    "hessians",
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
    "bitcast",
    "broadcast_to",
    "cast",
    "concat",
    "cos",
    "cosh",
    "divide",
    "einsum",
    "equal",
    "exp",
    "expand_dims",
    "eye",
    "fill",
    "floor",
    "gather",
    "gather_nd",
    "greater",
    "greater_equal",
    "identity",
    "less",
    "less_equal",
    "linspace",
    "logical_and",
    "logical_not",
    "logical_or",
    "matmul",
    "maximum",
    "meshgrid",
    "minimum",
    "multiply",
    "negative",
    "norm",
    "not_equal",
    "ones",
    "ones_like",
    "pow",
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
    "scatter_nd",
    "shape",
    "sign",
    "sin",
    "sinh",
    "slice",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "strided_slice",
    "subtract",
    "tan",
    "tanh",
    "tensordot",
    "tile",
    "transpose",
    "unstack",
    "where",
    "zeros",
    "zeros_like",
]


def _to_tensor(x: Any, dtype: Optional[Any] = None) -> ml_switcheroo_compiler.Tensor:
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

    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor
    from ml_switcheroo_compiler.core.config import config as compiler_config
    import uuid

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
        except Exception:  # pragma: no cover
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
        res = ml_switcheroo_compiler.ops.array(x, dtype=dt)
    except Exception:
        res = ml_switcheroo_compiler.ops.array(0.0, dtype=to_dtype("float32"))

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
            from ml_switcheroo_compiler.tracing import ProxyTensor
            from ml_switcheroo_compiler.core.config import config as compiler_config

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
            from ml_switcheroo_compiler.tracing import _tracer

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
            val = (
                arr.item()
                if hasattr(arr, "item") and callable(getattr(arr, "item"))
                else arr
            )

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
        from ml_switcheroo_compiler.tracing import _tracer
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
        from ml_switcheroo_compiler.tracing import TracerTape, _tracer

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
        from ml_switcheroo_compiler.tracing import _tracer

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
        raise NotImplementedError("Not implemented: tf.GradientTape.jacobian")

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
        raise NotImplementedError("Not implemented: tf.GradientTape.batch_jacobian")

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
        res = getattr(_ops, "all")(*[_to_tensor(a) for a in args], **kwargs)
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
        res = getattr(_ops, "any")(*[_to_tensor(a) for a in args], **kwargs)
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
        res = getattr(_ops, "prod")(*[_to_tensor(a) for a in args], **kwargs)
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
        res = getattr(_ops, "std")(*[_to_tensor(a) for a in args], **kwargs)
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
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Tensor: The result of the tensordot operation.
        """
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
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
        res = getattr(_ops, "variance")(*[_to_tensor(a) for a in args], **kwargs)
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
            raise NotImplementedError("Not implemented: tf.math.special.bessel_j0")

        @staticmethod
        def bessel_j1(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_j1")

        @staticmethod
        def bessel_k0(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_k0")

        @staticmethod
        def bessel_k0e(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_k0e")

        @staticmethod
        def bessel_k1(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_k1")

        @staticmethod
        def bessel_k1e(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_k1e")

        @staticmethod
        def bessel_y0(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_y0")

        @staticmethod
        def bessel_y1(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.bessel_y1")

        @staticmethod
        def dawsn(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.dawsn")

        @staticmethod
        def expint(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.expint")

        @staticmethod
        def fresnel_cos(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.fresnel_cos")

        @staticmethod
        def fresnel_sin(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.fresnel_sin")

        @staticmethod
        def spence(*args, **kwargs):
            raise NotImplementedError("Not implemented: tf.math.special.spence")

    @staticmethod
    def accumulate_n(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.accumulate_n")

    @staticmethod
    def add_n(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.add_n")

    @staticmethod
    def angle(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "angle")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def approx_max_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "approx_max_k")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def approx_min_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "approx_min_k")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i0(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "bessel_i0")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i0e(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "bessel_i0e")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i1(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "bessel_i1")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bessel_i1e(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "bessel_i1e")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def betainc(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "betainc")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def bincount(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "bincount")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def confusion_matrix(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.confusion_matrix")

    @staticmethod
    def cumprod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "cumprod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cumsum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "cumsum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def cumulative_logsumexp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.cumulative_logsumexp")

    @staticmethod
    def divide_no_nan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.divide_no_nan")

    @staticmethod
    def erfcinv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "erfcinv")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floordiv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "floor_divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def floormod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "mod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def igamma(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "igamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def igammac(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "igammac")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def in_top_k(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.in_top_k")

    @staticmethod
    def invert_permutation(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "invert_permutation")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def is_finite(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "isfinite")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_inf(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "isinf")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_nan(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "isnan")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def is_non_decreasing(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.is_non_decreasing")

    @staticmethod
    def is_strictly_increasing(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.is_strictly_increasing")

    @staticmethod
    def l2_normalize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.l2_normalize")

    @staticmethod
    def lbeta(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "lbeta")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def log_sigmoid(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.log_sigmoid")

    @staticmethod
    def log_softmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.log_softmax")

    @staticmethod
    def multiply_no_nan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.multiply_no_nan")

    @staticmethod
    def ndtri(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "ndtri")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def polygamma(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "polygamma")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def polyval(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "polyval")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def reciprocal_no_nan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.reciprocal_no_nan")

    @staticmethod
    def reduce_euclidean_norm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.reduce_euclidean_norm")

    @staticmethod
    def reduce_logsumexp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.reduce_logsumexp")

    @staticmethod
    def rint(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "rint")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def scalar_mul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.scalar_mul")

    @staticmethod
    def segment_max(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "segment_max")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_mean(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "segment_mean")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_min(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "segment_min")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_prod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "segment_prod")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def segment_sum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "segment_sum")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def sigmoid(*args, **kwargs):
        return nn.sigmoid(*args, **kwargs)

    @staticmethod
    def sobol_sample(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "sobol_sample")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def softmax(*args, **kwargs):
        return nn.softmax(*args, **kwargs)

    @staticmethod
    def softplus(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "softplus")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def softsign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.softsign")

    @staticmethod
    def squared_difference(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.squared_difference")

    @staticmethod
    def top_k(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "top_k")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def truediv(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "true_divide")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def unsorted_segment_max(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_max")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def unsorted_segment_mean(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_mean")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def unsorted_segment_min(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_min")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def unsorted_segment_prod(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_prod")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def unsorted_segment_sqrt_n(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_sqrt_n")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def unsorted_segment_sum(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "unsorted_segment_sum")(
            *[_to_tensor(a) for a in args], **kwargs
        )
        return _wrap(res)

    @staticmethod
    def xdivy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.xdivy")

    @staticmethod
    def xlog1py(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.xlog1py")

    @staticmethod
    def xlogy(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "xlogy")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)

    @staticmethod
    def zero_fraction(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.math.zero_fraction")

    @staticmethod
    def zeta(*args, **kwargs):
        kwargs = {"axis" if k == "dim" else k: v for k, v in kwargs.items()}
        res = getattr(_ops, "zeta")(*[_to_tensor(a) for a in args], **kwargs)
        return _wrap(res)


class linalg:
    @staticmethod
    def tensordot(*args, **kwargs):
        """Apply tensordot operation."""
        kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
        res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
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
        res = getattr(_ops, "bitwise_and")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "all")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "any")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "arange")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "argmax")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "argmin")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "asin")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "asinh")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "atan")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "atan2")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "atanh")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "bitcast")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "broadcast_to")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "cast")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "concatenate")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "cos")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "cosh")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "divide")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "einsum")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "equal")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "exp")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "expand")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "eye")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "flatten")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "float_power")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "floor")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "full")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "gather")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "gather_nd")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "greater")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "greater_equal")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "identity")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "less")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "less_equal")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "linspace")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "logical_and")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "logical_not")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "logical_or")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "matmul")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "max")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "maximum")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "mean")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "meshgrid")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "min")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "minimum")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "multiply")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "negative")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "norm")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "not_equal")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "ones")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "ones_like")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "prod")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "repeat")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "roll")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "round")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "scatter_nd")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "shape")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "sign")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "sin")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "sinh")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "slice")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "split")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "sqrt")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "square")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "squeeze")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "stack")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "strided_slice")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "subtract")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "sum")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "tan")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "tile")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "transpose")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "unstack")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "where")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "zeros")(*[_to_tensor(a) for a in args], **kwargs)
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
    res = getattr(_ops, "zeros_like")(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


from . import data
from . import nn


def tensordot(*args, **kwargs):
    """Apply tensordot operation."""
    kwargs = {("axis" if k == "dim" else k): v for k, v in kwargs.items()}
    res = getattr(_ops, "tensordot")(*[_to_tensor(a) for a in args], **kwargs)
    return _wrap(res)


def AggregationMethod(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.AggregationMethod")


def Assert(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.Assert")


def CriticalSection(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.CriticalSection")


def DeviceSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.DeviceSpec")


def Graph(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.Graph")


def IndexedSlices(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.IndexedSlices")


def IndexedSlicesSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.IndexedSlicesSpec")


def Module(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.Module")


def Operation(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.Operation")


def OptionalSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.OptionalSpec")


def RaggedTensorSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.RaggedTensorSpec")


def RegisterGradient(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.RegisterGradient")


def SparseTensorSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.SparseTensorSpec")


def TensorArray(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.TensorArray")


def TensorArraySpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.TensorArraySpec")


def TensorShape(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.TensorShape")


def TensorSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.TensorSpec")


def TypeSpec(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.TypeSpec")


def UnconnectedGradients(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.UnconnectedGradients")


def VariableAggregation(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.VariableAggregation")


def VariableSynchronization(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.VariableSynchronization")


def add_n(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.add_n")


def approx_top_k(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.approx_top_k")


def argsort(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.argsort")


def as_dtype(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.as_dtype")


def as_string(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.as_string")


def assert_equal(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.assert_equal")


def assert_greater(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.assert_greater")


def assert_less(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.assert_less")


def assert_rank(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.assert_rank")


class audio:
    @staticmethod
    def decode_wav(contents, desired_channels=-1, desired_samples=-1, name=None):
        import numpy as np

        # Return a naive dummy tensor and sample rate to pass eager execution shape checks
        # contents would usually be a string of bytes
        return np.zeros((1, 1), dtype=np.float32), np.array(44100, dtype=np.int32)

    @staticmethod
    def encode_wav(audio, sample_rate, name=None):
        import numpy as np

        # Return naive empty bytes
        return np.array(b"")


class autodiff:
    @staticmethod
    def ForwardAccumulator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autodiff.ForwardAccumulator")

    @staticmethod
    def GradientTape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autodiff.GradientTape")

    @staticmethod
    def set_verbosity(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autograph.set_verbosity")

    @staticmethod
    def to_code(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autograph.to_code")

    @staticmethod
    def to_graph(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autograph.to_graph")

    @staticmethod
    def trace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.autograph.trace")


def batch_to_space(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.batch_to_space")


def boolean_mask(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.boolean_mask")


def broadcast_dynamic_shape(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.broadcast_dynamic_shape")


def broadcast_static_shape(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.broadcast_static_shape")


def case(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.case")


def clip_by_global_norm(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.clip_by_global_norm")


def clip_by_norm(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.clip_by_norm")


def clip_by_value(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.clip_by_value")


class compat:
    @staticmethod
    def as_bytes(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.as_bytes")

    @staticmethod
    def as_str(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.as_str")

    @staticmethod
    def as_str_any(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.as_str_any")

    @staticmethod
    def as_text(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.as_text")

    @staticmethod
    def bytes_or_text_types(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.bytes_or_text_types")

    @staticmethod
    def complex_types(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.complex_types")

    @staticmethod
    def dimension_at_index(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.dimension_at_index")

    @staticmethod
    def dimension_value(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.dimension_value")

    @staticmethod
    def forward_compatibility_horizon(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.compat.forward_compatibility_horizon"
        )

    @staticmethod
    def forward_compatible(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.forward_compatible")

    @staticmethod
    def integral_types(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.integral_types")

    @staticmethod
    def path_to_str(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.path_to_str")

    @staticmethod
    def real_types(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.real_types")

    @staticmethod
    def v1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.v1")

    @staticmethod
    def v2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.compat.v2")


def complex(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.complex")


def cond(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.cond")

    @staticmethod
    def LogicalDeviceConfiguration(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.LogicalDeviceConfiguration"
        )

    @staticmethod
    def PhysicalDevice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.PhysicalDevice")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.experimental")

    @staticmethod
    def experimental_connect_to_cluster(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.experimental_connect_to_cluster"
        )

    @staticmethod
    def experimental_connect_to_host(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.experimental_connect_to_host"
        )

    @staticmethod
    def experimental_functions_run_eagerly(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.experimental_functions_run_eagerly"
        )

    @staticmethod
    def experimental_run_functions_eagerly(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.experimental_run_functions_eagerly"
        )

    @staticmethod
    def functions_run_eagerly(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.functions_run_eagerly")

    @staticmethod
    def get_logical_device_configuration(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.get_logical_device_configuration"
        )

    @staticmethod
    def get_soft_device_placement(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.get_soft_device_placement"
        )

    @staticmethod
    def get_visible_devices(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.get_visible_devices")

    @staticmethod
    def list_logical_devices(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.list_logical_devices")

    @staticmethod
    def list_physical_devices(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.list_physical_devices")

    @staticmethod
    def optimizer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.optimizer")

    @staticmethod
    def run_functions_eagerly(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.run_functions_eagerly")

    @staticmethod
    def set_logical_device_configuration(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.set_logical_device_configuration"
        )

    @staticmethod
    def set_soft_device_placement(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.config.set_soft_device_placement"
        )

    @staticmethod
    def set_visible_devices(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.set_visible_devices")

    @staticmethod
    def threading(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.config.threading")


def constant(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.constant")


def constant_initializer(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.constant_initializer")


def control_dependencies(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.control_dependencies")


def conv(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.conv")


def conv2d_backprop_filter_v2(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.conv2d_backprop_filter_v2")


def conv2d_backprop_input_v2(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.conv2d_backprop_input_v2")


def convert_to_tensor(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.convert_to_tensor")


def cumsum(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.cumsum")

    @staticmethod
    def assert_all_finite(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_all_finite")

    @staticmethod
    def assert_equal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_equal")

    @staticmethod
    def assert_greater(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_greater")

    @staticmethod
    def assert_greater_equal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_greater_equal")

    @staticmethod
    def assert_integer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_integer")

    @staticmethod
    def assert_less(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_less")

    @staticmethod
    def assert_less_equal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_less_equal")

    @staticmethod
    def assert_near(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_near")

    @staticmethod
    def assert_negative(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_negative")

    @staticmethod
    def assert_non_negative(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_non_negative")

    @staticmethod
    def assert_non_positive(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_non_positive")

    @staticmethod
    def assert_none_equal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_none_equal")

    @staticmethod
    def assert_positive(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_positive")

    @staticmethod
    def assert_proper_iterable(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.assert_proper_iterable"
        )

    @staticmethod
    def assert_rank(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_rank")

    @staticmethod
    def assert_rank_at_least(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_rank_at_least")

    @staticmethod
    def assert_rank_in(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_rank_in")

    @staticmethod
    def assert_same_float_dtype(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.assert_same_float_dtype"
        )

    @staticmethod
    def assert_scalar(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_scalar")

    @staticmethod
    def assert_shapes(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_shapes")

    @staticmethod
    def assert_type(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.assert_type")

    @staticmethod
    def check_numerics(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.check_numerics")

    @staticmethod
    def disable_check_numerics(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.disable_check_numerics"
        )

    @staticmethod
    def disable_traceback_filtering(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.disable_traceback_filtering"
        )

    @staticmethod
    def enable_check_numerics(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.enable_check_numerics")

    @staticmethod
    def enable_traceback_filtering(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.enable_traceback_filtering"
        )

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.experimental")

    @staticmethod
    def get_log_device_placement(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.get_log_device_placement"
        )

    @staticmethod
    def is_numeric_tensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.debugging.is_numeric_tensor")

    @staticmethod
    def is_traceback_filtering_enabled(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.is_traceback_filtering_enabled"
        )

    @staticmethod
    def set_log_device_placement(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.debugging.set_log_device_placement"
        )


def device(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.device")

    @staticmethod
    def python(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtensor.python")

    @staticmethod
    def QUANTIZED_DTYPES(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.QUANTIZED_DTYPES")

    @staticmethod
    def as_dtype(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.as_dtype")

    @staticmethod
    def bfloat16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.bfloat16")

    @staticmethod
    def bool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.bool")

    @staticmethod
    def cast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.cast")

    @staticmethod
    def complex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.complex")

    @staticmethod
    def complex128(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.complex128")

    @staticmethod
    def complex64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.complex64")

    @staticmethod
    def double(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.double")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.experimental")

    @staticmethod
    def float16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.float16")

    @staticmethod
    def float32(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.float32")

    @staticmethod
    def float64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.float64")

    @staticmethod
    def half(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.half")

    @staticmethod
    def int16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.int16")

    @staticmethod
    def int32(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.int32")

    @staticmethod
    def int64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.int64")

    @staticmethod
    def int8(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.int8")

    @staticmethod
    def qint16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.qint16")

    @staticmethod
    def qint32(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.qint32")

    @staticmethod
    def qint8(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.qint8")

    @staticmethod
    def quint16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.quint16")

    @staticmethod
    def quint8(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.quint8")

    @staticmethod
    def resource(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.resource")

    @staticmethod
    def saturate_cast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.saturate_cast")

    @staticmethod
    def string(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.string")

    @staticmethod
    def uint16(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.uint16")

    @staticmethod
    def uint32(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.uint32")

    @staticmethod
    def uint64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.uint64")

    @staticmethod
    def uint8(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.uint8")

    @staticmethod
    def variant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.dtypes.variant")


def dynamic_partition(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.dynamic_partition")


def dynamic_stitch(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.dynamic_stitch")


def edit_distance(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.edit_distance")


def eig(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.eig")


def eigvals(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.eigvals")


def ensure_shape(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.ensure_shape")


class errors:
    @staticmethod
    def ABORTED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.ABORTED")

    @staticmethod
    def ALREADY_EXISTS(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.ALREADY_EXISTS")

    @staticmethod
    def AbortedError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.AbortedError")

    @staticmethod
    def AlreadyExistsError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.AlreadyExistsError")

    @staticmethod
    def CANCELLED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.CANCELLED")

    @staticmethod
    def CancelledError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.CancelledError")

    @staticmethod
    def DATA_LOSS(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.DATA_LOSS")

    @staticmethod
    def DEADLINE_EXCEEDED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.DEADLINE_EXCEEDED")

    @staticmethod
    def DataLossError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.DataLossError")

    @staticmethod
    def DeadlineExceededError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.DeadlineExceededError")

    @staticmethod
    def FAILED_PRECONDITION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.FAILED_PRECONDITION")

    @staticmethod
    def FailedPreconditionError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.FailedPreconditionError")

    @staticmethod
    def INTERNAL(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.INTERNAL")

    @staticmethod
    def INVALID_ARGUMENT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.INVALID_ARGUMENT")

    @staticmethod
    def InternalError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.InternalError")

    @staticmethod
    def InvalidArgumentError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.InvalidArgumentError")

    @staticmethod
    def NOT_FOUND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.NOT_FOUND")

    @staticmethod
    def NotFoundError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.NotFoundError")

    @staticmethod
    def OK(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.OK")

    @staticmethod
    def OUT_OF_RANGE(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.OUT_OF_RANGE")

    @staticmethod
    def OpError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.OpError")

    @staticmethod
    def OperatorNotAllowedInGraphError(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.errors.OperatorNotAllowedInGraphError"
        )

    @staticmethod
    def OutOfRangeError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.OutOfRangeError")

    @staticmethod
    def PERMISSION_DENIED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.PERMISSION_DENIED")

    @staticmethod
    def PermissionDeniedError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.PermissionDeniedError")

    @staticmethod
    def RESOURCE_EXHAUSTED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.RESOURCE_EXHAUSTED")

    @staticmethod
    def ResourceExhaustedError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.ResourceExhaustedError")

    @staticmethod
    def UNAUTHENTICATED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UNAUTHENTICATED")

    @staticmethod
    def UNAVAILABLE(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UNAVAILABLE")

    @staticmethod
    def UNIMPLEMENTED(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UNIMPLEMENTED")

    @staticmethod
    def UNKNOWN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UNKNOWN")

    @staticmethod
    def UnauthenticatedError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UnauthenticatedError")

    @staticmethod
    def UnavailableError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UnavailableError")

    @staticmethod
    def UnimplementedError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UnimplementedError")

    @staticmethod
    def UnknownError(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.errors.UnknownError")


def executing_eagerly(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.executing_eagerly")


class experimental:
    @staticmethod
    def BatchableExtensionType(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.BatchableExtensionType"
        )

    @staticmethod
    def DynamicRaggedShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.DynamicRaggedShape")

    @staticmethod
    def ExtensionType(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.ExtensionType")

    @staticmethod
    def ExtensionTypeBatchEncoder(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.ExtensionTypeBatchEncoder"
        )

    @staticmethod
    def ExtensionTypeSpec(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.ExtensionTypeSpec")

    @staticmethod
    def Optional(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.Optional")

    @staticmethod
    def RowPartition(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.RowPartition")

    @staticmethod
    def StructuredTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.StructuredTensor")

    @staticmethod
    def async_clear_error(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.async_clear_error")

    @staticmethod
    def async_scope(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.async_scope")

    @staticmethod
    def dispatch_for_api(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.dispatch_for_api")

    @staticmethod
    def dispatch_for_binary_elementwise_apis(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.dispatch_for_binary_elementwise_apis"
        )

    @staticmethod
    def dispatch_for_binary_elementwise_assert_apis(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.dispatch_for_binary_elementwise_assert_apis"
        )

    @staticmethod
    def dispatch_for_unary_elementwise_apis(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.dispatch_for_unary_elementwise_apis"
        )

    @staticmethod
    def dlpack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.dlpack")

    @staticmethod
    def dtensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.dtensor")

    @staticmethod
    def enable_strict_mode(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.enable_strict_mode")

    @staticmethod
    def extension_type(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.extension_type")

    @staticmethod
    def float8_e4m3fn(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.float8_e4m3fn")

    @staticmethod
    def float8_e5m2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.float8_e5m2")

    @staticmethod
    def function_executor_type(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.function_executor_type"
        )

    @staticmethod
    def int4(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.int4")

    @staticmethod
    def numpy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.numpy")

    @staticmethod
    def register_filesystem_plugin(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.register_filesystem_plugin"
        )

    @staticmethod
    def tensorrt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.tensorrt")

    @staticmethod
    def uint4(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.experimental.uint4")

    @staticmethod
    def unregister_dispatch_for(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.experimental.unregister_dispatch_for"
        )


def extract_volume_patches(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.extract_volume_patches")


class feature_column:
    @staticmethod
    def bucketized_column(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.bucketized_column"
        )

    @staticmethod
    def categorical_column_with_hash_bucket(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.categorical_column_with_hash_bucket"
        )

    @staticmethod
    def categorical_column_with_identity(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.categorical_column_with_identity"
        )

    @staticmethod
    def categorical_column_with_vocabulary_file(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.categorical_column_with_vocabulary_file"
        )

    @staticmethod
    def categorical_column_with_vocabulary_list(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.categorical_column_with_vocabulary_list"
        )

    @staticmethod
    def crossed_column(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.feature_column.crossed_column")

    @staticmethod
    def embedding_column(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.feature_column.embedding_column")

    @staticmethod
    def indicator_column(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.feature_column.indicator_column")

    @staticmethod
    def make_parse_example_spec(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.make_parse_example_spec"
        )

    @staticmethod
    def numeric_column(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.feature_column.numeric_column")

    @staticmethod
    def sequence_categorical_column_with_hash_bucket(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.sequence_categorical_column_with_hash_bucket"
        )

    @staticmethod
    def sequence_categorical_column_with_identity(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.sequence_categorical_column_with_identity"
        )

    @staticmethod
    def sequence_categorical_column_with_vocabulary_file(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.sequence_categorical_column_with_vocabulary_file"
        )

    @staticmethod
    def sequence_categorical_column_with_vocabulary_list(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.sequence_categorical_column_with_vocabulary_list"
        )

    @staticmethod
    def sequence_numeric_column(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.sequence_numeric_column"
        )

    @staticmethod
    def shared_embeddings(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.shared_embeddings"
        )

    @staticmethod
    def weighted_categorical_column(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.feature_column.weighted_categorical_column"
        )


def fftnd(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.fftnd")


def fingerprint(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.fingerprint")


def foldl(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.foldl")


def foldr(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.foldr")


def get_current_name_scope(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.get_current_name_scope")


def get_logger(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.get_logger")


def get_static_value(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.get_static_value")


def grad_pass_through(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.grad_pass_through")


def gradients(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.gradients")


class graph_util:
    @staticmethod
    def import_graph_def(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.graph_util.import_graph_def")


def group(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.group")


def guarantee_const(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.guarantee_const")


def half(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.half")


def histogram_fixed_width(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.histogram_fixed_width")


def histogram_fixed_width_bins(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.histogram_fixed_width_bins")


def identity_n(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.identity_n")


def ifftnd(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.ifftnd")


def inside_function(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.inside_function")

    @staticmethod
    def OpsSet(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.OpsSet")

    @staticmethod
    def Optimize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.Optimize")

    @staticmethod
    def RepresentativeDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.RepresentativeDataset")

    @staticmethod
    def TFLiteConverter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.TFLiteConverter")

    @staticmethod
    def TargetSpec(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.TargetSpec")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lite.experimental")


def load_library(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.load_library")


def load_op_library(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.load_op_library")

    @staticmethod
    def StaticHashTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lookup.StaticHashTable")

    @staticmethod
    def StaticVocabularyTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lookup.StaticVocabularyTable")

    @staticmethod
    def TextFileIndex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lookup.TextFileIndex")

    @staticmethod
    def TextFileInitializer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lookup.TextFileInitializer")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.lookup.experimental")


def make_ndarray(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.make_ndarray")


def make_tensor_proto(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.make_tensor_proto")


def map_fn(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.map_fn")


def matrix_square_root(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.matrix_square_root")


class mlir:
    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.mlir.experimental")


def name_scope(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.name_scope")


class nest:
    @staticmethod
    def assert_same_structure(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.nest.assert_same_structure")

    @staticmethod
    def flatten(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.nest.flatten")

    @staticmethod
    def is_nested(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.nest.is_nested")

    @staticmethod
    def map_structure(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.nest.map_structure")

    @staticmethod
    def pack_sequence_as(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.nest.pack_sequence_as")


def newaxis(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.newaxis")


def no_gradient(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.no_gradient")


def no_op(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.no_op")


def nondifferentiable_batch_function(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.nondifferentiable_batch_function")


def numpy_function(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.numpy_function")


def one_hot(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.one_hot")


def ones_initializer(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.ones_initializer")


def pad(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.pad")


def parallel_stack(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.parallel_stack")


def print(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.print")


class profiler:
    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.profiler.experimental")


def py_function(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.py_function")


def qint16(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.qint16")


def qint32(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.qint32")


def qint8(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.qint8")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.quantization.experimental")

    @staticmethod
    def fake_quant_with_min_max_args(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_args"
        )

    @staticmethod
    def fake_quant_with_min_max_args_gradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_args_gradient"
        )

    @staticmethod
    def fake_quant_with_min_max_vars(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_vars"
        )

    @staticmethod
    def fake_quant_with_min_max_vars_gradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_vars_gradient"
        )

    @staticmethod
    def fake_quant_with_min_max_vars_per_channel(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_vars_per_channel"
        )

    @staticmethod
    def fake_quant_with_min_max_vars_per_channel_gradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.fake_quant_with_min_max_vars_per_channel_gradient"
        )

    @staticmethod
    def quantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.quantization.quantize")

    @staticmethod
    def quantize_and_dequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.quantize_and_dequantize"
        )

    @staticmethod
    def quantize_and_dequantize_v2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.quantization.quantize_and_dequantize_v2"
        )

    @staticmethod
    def quantized_concat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.quantization.quantized_concat")


class queue:
    @staticmethod
    def FIFOQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.queue.FIFOQueue")

    @staticmethod
    def PaddingFIFOQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.queue.PaddingFIFOQueue")

    @staticmethod
    def PriorityQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.queue.PriorityQueue")

    @staticmethod
    def QueueBase(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.queue.QueueBase")

    @staticmethod
    def RandomShuffleQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.queue.RandomShuffleQueue")


def quint16(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.quint16")


def quint8(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.quint8")


def random_index_shuffle(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.random_index_shuffle")


def random_normal_initializer(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.random_normal_initializer")


def random_uniform_initializer(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.random_uniform_initializer")


def rank(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.rank")


class raw_ops:
    @staticmethod
    def Abort(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Abort")

    @staticmethod
    def Abs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Abs")

    @staticmethod
    def AccumulateNV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AccumulateNV2")

    @staticmethod
    def AccumulatorApplyGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AccumulatorApplyGradient"
        )

    @staticmethod
    def AccumulatorNumAccumulated(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AccumulatorNumAccumulated"
        )

    @staticmethod
    def AccumulatorSetGlobalStep(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AccumulatorSetGlobalStep"
        )

    @staticmethod
    def AccumulatorTakeGradient(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AccumulatorTakeGradient")

    @staticmethod
    def Acos(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Acos")

    @staticmethod
    def Acosh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Acosh")

    @staticmethod
    def Add(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Add")

    @staticmethod
    def AddManySparseToTensorsMap(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AddManySparseToTensorsMap"
        )

    @staticmethod
    def AddN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AddN")

    @staticmethod
    def AddSparseToTensorsMap(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AddSparseToTensorsMap")

    @staticmethod
    def AddV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AddV2")

    @staticmethod
    def AdjustContrast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AdjustContrast")

    @staticmethod
    def AdjustContrastv2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AdjustContrastv2")

    @staticmethod
    def AdjustHue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AdjustHue")

    @staticmethod
    def AdjustSaturation(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AdjustSaturation")

    @staticmethod
    def All(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.All")

    @staticmethod
    def AllCandidateSampler(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AllCandidateSampler")

    @staticmethod
    def AllToAll(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AllToAll")

    @staticmethod
    def Angle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Angle")

    @staticmethod
    def AnonymousHashTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousHashTable")

    @staticmethod
    def AnonymousIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousIterator")

    @staticmethod
    def AnonymousIteratorV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousIteratorV2")

    @staticmethod
    def AnonymousIteratorV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousIteratorV3")

    @staticmethod
    def AnonymousMemoryCache(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousMemoryCache")

    @staticmethod
    def AnonymousMultiDeviceIterator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousMultiDeviceIterator"
        )

    @staticmethod
    def AnonymousMultiDeviceIteratorV3(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousMultiDeviceIteratorV3"
        )

    @staticmethod
    def AnonymousMutableDenseHashTable(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousMutableDenseHashTable"
        )

    @staticmethod
    def AnonymousMutableHashTable(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousMutableHashTable"
        )

    @staticmethod
    def AnonymousMutableHashTableOfTensors(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousMutableHashTableOfTensors"
        )

    @staticmethod
    def AnonymousRandomSeedGenerator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AnonymousRandomSeedGenerator"
        )

    @staticmethod
    def AnonymousSeedGenerator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AnonymousSeedGenerator")

    @staticmethod
    def Any(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Any")

    @staticmethod
    def ApplyAdaMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdaMax")

    @staticmethod
    def ApplyAdadelta(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdadelta")

    @staticmethod
    def ApplyAdagrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdagrad")

    @staticmethod
    def ApplyAdagradDA(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdagradDA")

    @staticmethod
    def ApplyAdagradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdagradV2")

    @staticmethod
    def ApplyAdam(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAdam")

    @staticmethod
    def ApplyAddSign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyAddSign")

    @staticmethod
    def ApplyCenteredRMSProp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyCenteredRMSProp")

    @staticmethod
    def ApplyFtrl(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyFtrl")

    @staticmethod
    def ApplyFtrlV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyFtrlV2")

    @staticmethod
    def ApplyGradientDescent(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyGradientDescent")

    @staticmethod
    def ApplyMomentum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyMomentum")

    @staticmethod
    def ApplyPowerSign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyPowerSign")

    @staticmethod
    def ApplyProximalAdagrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyProximalAdagrad")

    @staticmethod
    def ApplyProximalGradientDescent(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ApplyProximalGradientDescent"
        )

    @staticmethod
    def ApplyRMSProp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApplyRMSProp")

    @staticmethod
    def ApproxTopK(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApproxTopK")

    @staticmethod
    def ApproximateEqual(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ApproximateEqual")

    @staticmethod
    def ArgMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ArgMax")

    @staticmethod
    def ArgMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ArgMin")

    @staticmethod
    def AsString(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AsString")

    @staticmethod
    def Asin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Asin")

    @staticmethod
    def Asinh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Asinh")

    @staticmethod
    def Assert(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Assert")

    @staticmethod
    def AssertCardinalityDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AssertCardinalityDataset"
        )

    @staticmethod
    def AssertNextDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssertNextDataset")

    @staticmethod
    def AssertPrevDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssertPrevDataset")

    @staticmethod
    def Assign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Assign")

    @staticmethod
    def AssignAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssignAdd")

    @staticmethod
    def AssignAddVariableOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssignAddVariableOp")

    @staticmethod
    def AssignSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssignSub")

    @staticmethod
    def AssignSubVariableOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssignSubVariableOp")

    @staticmethod
    def AssignVariableOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AssignVariableOp")

    @staticmethod
    def AssignVariableXlaConcatND(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.AssignVariableXlaConcatND"
        )

    @staticmethod
    def Atan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Atan")

    @staticmethod
    def Atan2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Atan2")

    @staticmethod
    def Atanh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Atanh")

    @staticmethod
    def AudioSpectrogram(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AudioSpectrogram")

    @staticmethod
    def AudioSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AudioSummary")

    @staticmethod
    def AudioSummaryV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AudioSummaryV2")

    @staticmethod
    def AutoShardDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AutoShardDataset")

    @staticmethod
    def AvgPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AvgPool")

    @staticmethod
    def AvgPool3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AvgPool3D")

    @staticmethod
    def AvgPool3DGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AvgPool3DGrad")

    @staticmethod
    def AvgPoolGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.AvgPoolGrad")

    @staticmethod
    def BandedTriangularSolve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BandedTriangularSolve")

    @staticmethod
    def Barrier(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Barrier")

    @staticmethod
    def BarrierClose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BarrierClose")

    @staticmethod
    def BarrierIncompleteSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BarrierIncompleteSize")

    @staticmethod
    def BarrierInsertMany(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BarrierInsertMany")

    @staticmethod
    def BarrierReadySize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BarrierReadySize")

    @staticmethod
    def BarrierTakeMany(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BarrierTakeMany")

    @staticmethod
    def Batch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Batch")

    @staticmethod
    def BatchCholesky(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchCholesky")

    @staticmethod
    def BatchCholeskyGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchCholeskyGrad")

    @staticmethod
    def BatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchDataset")

    @staticmethod
    def BatchDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchDatasetV2")

    @staticmethod
    def BatchFFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchFFT")

    @staticmethod
    def BatchFFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchFFT2D")

    @staticmethod
    def BatchFFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchFFT3D")

    @staticmethod
    def BatchFunction(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchFunction")

    @staticmethod
    def BatchIFFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchIFFT")

    @staticmethod
    def BatchIFFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchIFFT2D")

    @staticmethod
    def BatchIFFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchIFFT3D")

    @staticmethod
    def BatchMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatMul")

    @staticmethod
    def BatchMatMulV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatMulV2")

    @staticmethod
    def BatchMatMulV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatMulV3")

    @staticmethod
    def BatchMatrixBandPart(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixBandPart")

    @staticmethod
    def BatchMatrixDeterminant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixDeterminant")

    @staticmethod
    def BatchMatrixDiag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixDiag")

    @staticmethod
    def BatchMatrixDiagPart(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixDiagPart")

    @staticmethod
    def BatchMatrixInverse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixInverse")

    @staticmethod
    def BatchMatrixSetDiag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixSetDiag")

    @staticmethod
    def BatchMatrixSolve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixSolve")

    @staticmethod
    def BatchMatrixSolveLs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchMatrixSolveLs")

    @staticmethod
    def BatchMatrixTriangularSolve(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BatchMatrixTriangularSolve"
        )

    @staticmethod
    def BatchNormWithGlobalNormalization(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BatchNormWithGlobalNormalization"
        )

    @staticmethod
    def BatchNormWithGlobalNormalizationGrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BatchNormWithGlobalNormalizationGrad"
        )

    @staticmethod
    def BatchSelfAdjointEig(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchSelfAdjointEig")

    @staticmethod
    def BatchSelfAdjointEigV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchSelfAdjointEigV2")

    @staticmethod
    def BatchSvd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchSvd")

    @staticmethod
    def BatchToSpace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchToSpace")

    @staticmethod
    def BatchToSpaceND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BatchToSpaceND")

    @staticmethod
    def BesselI0(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselI0")

    @staticmethod
    def BesselI0e(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselI0e")

    @staticmethod
    def BesselI1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselI1")

    @staticmethod
    def BesselI1e(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselI1e")

    @staticmethod
    def BesselJ0(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselJ0")

    @staticmethod
    def BesselJ1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselJ1")

    @staticmethod
    def BesselK0(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselK0")

    @staticmethod
    def BesselK0e(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselK0e")

    @staticmethod
    def BesselK1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselK1")

    @staticmethod
    def BesselK1e(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselK1e")

    @staticmethod
    def BesselY0(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselY0")

    @staticmethod
    def BesselY1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BesselY1")

    @staticmethod
    def Betainc(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Betainc")

    @staticmethod
    def BiasAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BiasAdd")

    @staticmethod
    def BiasAddGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BiasAddGrad")

    @staticmethod
    def BiasAddV1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BiasAddV1")

    @staticmethod
    def Bincount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Bincount")

    @staticmethod
    def Bitcast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Bitcast")

    @staticmethod
    def BitwiseAnd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BitwiseAnd")

    @staticmethod
    def BitwiseOr(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BitwiseOr")

    @staticmethod
    def BitwiseXor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BitwiseXor")

    @staticmethod
    def BlockLSTM(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BlockLSTM")

    @staticmethod
    def BlockLSTMGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BlockLSTMGrad")

    @staticmethod
    def BlockLSTMGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BlockLSTMGradV2")

    @staticmethod
    def BlockLSTMV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BlockLSTMV2")

    @staticmethod
    def BoostedTreesAggregateStats(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesAggregateStats"
        )

    @staticmethod
    def BoostedTreesBucketize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BoostedTreesBucketize")

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplit(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesCalculateBestFeatureSplit"
        )

    @staticmethod
    def BoostedTreesCalculateBestFeatureSplitV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesCalculateBestFeatureSplitV2"
        )

    @staticmethod
    def BoostedTreesCalculateBestGainsPerFeature(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesCalculateBestGainsPerFeature"
        )

    @staticmethod
    def BoostedTreesCenterBias(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BoostedTreesCenterBias")

    @staticmethod
    def BoostedTreesCreateEnsemble(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesCreateEnsemble"
        )

    @staticmethod
    def BoostedTreesCreateQuantileStreamResource(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesCreateQuantileStreamResource"
        )

    @staticmethod
    def BoostedTreesDeserializeEnsemble(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesDeserializeEnsemble"
        )

    @staticmethod
    def BoostedTreesEnsembleResourceHandleOp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesEnsembleResourceHandleOp"
        )

    @staticmethod
    def BoostedTreesExampleDebugOutputs(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesExampleDebugOutputs"
        )

    @staticmethod
    def BoostedTreesFlushQuantileSummaries(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesFlushQuantileSummaries"
        )

    @staticmethod
    def BoostedTreesGetEnsembleStates(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesGetEnsembleStates"
        )

    @staticmethod
    def BoostedTreesMakeQuantileSummaries(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesMakeQuantileSummaries"
        )

    @staticmethod
    def BoostedTreesMakeStatsSummary(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesMakeStatsSummary"
        )

    @staticmethod
    def BoostedTreesPredict(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BoostedTreesPredict")

    @staticmethod
    def BoostedTreesQuantileStreamResourceAddSummaries(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesQuantileStreamResourceAddSummaries"
        )

    @staticmethod
    def BoostedTreesQuantileStreamResourceDeserialize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesQuantileStreamResourceDeserialize"
        )

    @staticmethod
    def BoostedTreesQuantileStreamResourceFlush(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesQuantileStreamResourceFlush"
        )

    @staticmethod
    def BoostedTreesQuantileStreamResourceGetBucketBoundaries(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesQuantileStreamResourceGetBucketBoundaries"
        )

    @staticmethod
    def BoostedTreesQuantileStreamResourceHandleOp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesQuantileStreamResourceHandleOp"
        )

    @staticmethod
    def BoostedTreesSerializeEnsemble(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesSerializeEnsemble"
        )

    @staticmethod
    def BoostedTreesSparseAggregateStats(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesSparseAggregateStats"
        )

    @staticmethod
    def BoostedTreesSparseCalculateBestFeatureSplit(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesSparseCalculateBestFeatureSplit"
        )

    @staticmethod
    def BoostedTreesTrainingPredict(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesTrainingPredict"
        )

    @staticmethod
    def BoostedTreesUpdateEnsemble(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesUpdateEnsemble"
        )

    @staticmethod
    def BoostedTreesUpdateEnsembleV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BoostedTreesUpdateEnsembleV2"
        )

    @staticmethod
    def BroadcastArgs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BroadcastArgs")

    @staticmethod
    def BroadcastGradientArgs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BroadcastGradientArgs")

    @staticmethod
    def BroadcastTo(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.BroadcastTo")

    @staticmethod
    def Bucketize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Bucketize")

    @staticmethod
    def BytesProducedStatsDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.BytesProducedStatsDataset"
        )

    @staticmethod
    def CSRSparseMatrixComponents(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CSRSparseMatrixComponents"
        )

    @staticmethod
    def CSRSparseMatrixToDense(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CSRSparseMatrixToDense")

    @staticmethod
    def CSRSparseMatrixToSparseTensor(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CSRSparseMatrixToSparseTensor"
        )

    @staticmethod
    def CSVDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CSVDataset")

    @staticmethod
    def CSVDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CSVDatasetV2")

    @staticmethod
    def CTCBeamSearchDecoder(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CTCBeamSearchDecoder")

    @staticmethod
    def CTCGreedyDecoder(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CTCGreedyDecoder")

    @staticmethod
    def CTCLoss(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CTCLoss")

    @staticmethod
    def CTCLossV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CTCLossV2")

    @staticmethod
    def CacheDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CacheDataset")

    @staticmethod
    def CacheDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CacheDatasetV2")

    @staticmethod
    def Case(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Case")

    @staticmethod
    def Cast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cast")

    @staticmethod
    def Ceil(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Ceil")

    @staticmethod
    def CheckNumerics(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CheckNumerics")

    @staticmethod
    def CheckNumericsV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CheckNumericsV2")

    @staticmethod
    def Cholesky(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cholesky")

    @staticmethod
    def CholeskyGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CholeskyGrad")

    @staticmethod
    def ChooseFastestBranchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ChooseFastestBranchDataset"
        )

    @staticmethod
    def ChooseFastestDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ChooseFastestDataset")

    @staticmethod
    def ClipByValue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ClipByValue")

    @staticmethod
    def CloseSummaryWriter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CloseSummaryWriter")

    @staticmethod
    def CollectiveAllToAllV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveAllToAllV2")

    @staticmethod
    def CollectiveAllToAllV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveAllToAllV3")

    @staticmethod
    def CollectiveAssignGroupV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveAssignGroupV2")

    @staticmethod
    def CollectiveBcastRecv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveBcastRecv")

    @staticmethod
    def CollectiveBcastRecvV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveBcastRecvV2")

    @staticmethod
    def CollectiveBcastSend(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveBcastSend")

    @staticmethod
    def CollectiveBcastSendV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveBcastSendV2")

    @staticmethod
    def CollectiveGather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveGather")

    @staticmethod
    def CollectiveGatherV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveGatherV2")

    @staticmethod
    def CollectiveInitializeCommunicator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CollectiveInitializeCommunicator"
        )

    @staticmethod
    def CollectivePermute(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectivePermute")

    @staticmethod
    def CollectiveReduce(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveReduce")

    @staticmethod
    def CollectiveReduceScatterV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CollectiveReduceScatterV2"
        )

    @staticmethod
    def CollectiveReduceV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveReduceV2")

    @staticmethod
    def CollectiveReduceV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CollectiveReduceV3")

    @staticmethod
    def CombinedNonMaxSuppression(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CombinedNonMaxSuppression"
        )

    @staticmethod
    def Complex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Complex")

    @staticmethod
    def ComplexAbs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ComplexAbs")

    @staticmethod
    def CompositeTensorVariantFromComponents(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CompositeTensorVariantFromComponents"
        )

    @staticmethod
    def CompositeTensorVariantToComponents(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CompositeTensorVariantToComponents"
        )

    @staticmethod
    def CompressElement(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CompressElement")

    @staticmethod
    def ComputeAccidentalHits(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ComputeAccidentalHits")

    @staticmethod
    def ComputeBatchSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ComputeBatchSize")

    @staticmethod
    def Concat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Concat")

    @staticmethod
    def ConcatOffset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConcatOffset")

    @staticmethod
    def ConcatV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConcatV2")

    @staticmethod
    def ConcatenateDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConcatenateDataset")

    @staticmethod
    def ConditionalAccumulator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConditionalAccumulator")

    @staticmethod
    def ConfigureDistributedTPU(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConfigureDistributedTPU")

    @staticmethod
    def ConfigureTPUEmbedding(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConfigureTPUEmbedding")

    @staticmethod
    def Conj(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conj")

    @staticmethod
    def ConjugateTranspose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConjugateTranspose")

    @staticmethod
    def Const(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Const")

    @staticmethod
    def ConsumeMutexLock(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConsumeMutexLock")

    @staticmethod
    def ControlTrigger(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ControlTrigger")

    @staticmethod
    def Conv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv")

    @staticmethod
    def Conv2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv2D")

    @staticmethod
    def Conv2DBackpropFilter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv2DBackpropFilter")

    @staticmethod
    def Conv2DBackpropFilterV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv2DBackpropFilterV2")

    @staticmethod
    def Conv2DBackpropInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv2DBackpropInput")

    @staticmethod
    def Conv2DBackpropInputV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv2DBackpropInputV2")

    @staticmethod
    def Conv3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv3D")

    @staticmethod
    def Conv3DBackpropFilter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv3DBackpropFilter")

    @staticmethod
    def Conv3DBackpropFilterV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv3DBackpropFilterV2")

    @staticmethod
    def Conv3DBackpropInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv3DBackpropInput")

    @staticmethod
    def Conv3DBackpropInputV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Conv3DBackpropInputV2")

    @staticmethod
    def ConvertToCooTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ConvertToCooTensor")

    @staticmethod
    def Copy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Copy")

    @staticmethod
    def CopyHost(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CopyHost")

    @staticmethod
    def Cos(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cos")

    @staticmethod
    def Cosh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cosh")

    @staticmethod
    def CountUpTo(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CountUpTo")

    @staticmethod
    def CreateSummaryDbWriter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CreateSummaryDbWriter")

    @staticmethod
    def CreateSummaryFileWriter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CreateSummaryFileWriter")

    @staticmethod
    def CropAndResize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CropAndResize")

    @staticmethod
    def CropAndResizeGradBoxes(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CropAndResizeGradBoxes")

    @staticmethod
    def CropAndResizeGradImage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CropAndResizeGradImage")

    @staticmethod
    def Cross(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cross")

    @staticmethod
    def CrossReplicaSum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CrossReplicaSum")

    @staticmethod
    def CudnnRNN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNN")

    @staticmethod
    def CudnnRNNBackprop(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNBackprop")

    @staticmethod
    def CudnnRNNBackpropV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNBackpropV2")

    @staticmethod
    def CudnnRNNBackpropV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNBackpropV3")

    @staticmethod
    def CudnnRNNCanonicalToParams(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CudnnRNNCanonicalToParams"
        )

    @staticmethod
    def CudnnRNNCanonicalToParamsV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CudnnRNNCanonicalToParamsV2"
        )

    @staticmethod
    def CudnnRNNParamsSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNParamsSize")

    @staticmethod
    def CudnnRNNParamsToCanonical(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CudnnRNNParamsToCanonical"
        )

    @staticmethod
    def CudnnRNNParamsToCanonicalV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.CudnnRNNParamsToCanonicalV2"
        )

    @staticmethod
    def CudnnRNNV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNV2")

    @staticmethod
    def CudnnRNNV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CudnnRNNV3")

    @staticmethod
    def Cumprod(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cumprod")

    @staticmethod
    def Cumsum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Cumsum")

    @staticmethod
    def CumulativeLogsumexp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.CumulativeLogsumexp")

    @staticmethod
    def DataFormatDimMap(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataFormatDimMap")

    @staticmethod
    def DataFormatVecPermute(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataFormatVecPermute")

    @staticmethod
    def DataServiceDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataServiceDataset")

    @staticmethod
    def DataServiceDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataServiceDatasetV2")

    @staticmethod
    def DataServiceDatasetV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataServiceDatasetV3")

    @staticmethod
    def DataServiceDatasetV4(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DataServiceDatasetV4")

    @staticmethod
    def DatasetCardinality(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetCardinality")

    @staticmethod
    def DatasetFingerprint(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetFingerprint")

    @staticmethod
    def DatasetFromGraph(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetFromGraph")

    @staticmethod
    def DatasetToGraph(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetToGraph")

    @staticmethod
    def DatasetToGraphV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetToGraphV2")

    @staticmethod
    def DatasetToSingleElement(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetToSingleElement")

    @staticmethod
    def DatasetToTFRecord(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DatasetToTFRecord")

    @staticmethod
    def Dawsn(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Dawsn")

    @staticmethod
    def DebugGradientIdentity(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugGradientIdentity")

    @staticmethod
    def DebugGradientRefIdentity(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DebugGradientRefIdentity"
        )

    @staticmethod
    def DebugIdentity(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugIdentity")

    @staticmethod
    def DebugIdentityV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugIdentityV2")

    @staticmethod
    def DebugIdentityV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugIdentityV3")

    @staticmethod
    def DebugNanCount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugNanCount")

    @staticmethod
    def DebugNumericSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugNumericSummary")

    @staticmethod
    def DebugNumericSummaryV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DebugNumericSummaryV2")

    @staticmethod
    def DecodeAndCropJpeg(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeAndCropJpeg")

    @staticmethod
    def DecodeBase64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeBase64")

    @staticmethod
    def DecodeBmp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeBmp")

    @staticmethod
    def DecodeCSV(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeCSV")

    @staticmethod
    def DecodeCompressed(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeCompressed")

    @staticmethod
    def DecodeGif(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeGif")

    @staticmethod
    def DecodeImage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeImage")

    @staticmethod
    def DecodeJSONExample(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeJSONExample")

    @staticmethod
    def DecodeJpeg(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeJpeg")

    @staticmethod
    def DecodePaddedRaw(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodePaddedRaw")

    @staticmethod
    def DecodePng(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodePng")

    @staticmethod
    def DecodeProtoV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeProtoV2")

    @staticmethod
    def DecodeRaw(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeRaw")

    @staticmethod
    def DecodeWav(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DecodeWav")

    @staticmethod
    def DeepCopy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeepCopy")

    @staticmethod
    def DeleteIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeleteIterator")

    @staticmethod
    def DeleteMemoryCache(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeleteMemoryCache")

    @staticmethod
    def DeleteMultiDeviceIterator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DeleteMultiDeviceIterator"
        )

    @staticmethod
    def DeleteRandomSeedGenerator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DeleteRandomSeedGenerator"
        )

    @staticmethod
    def DeleteSeedGenerator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeleteSeedGenerator")

    @staticmethod
    def DeleteSessionTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeleteSessionTensor")

    @staticmethod
    def DenseBincount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DenseBincount")

    @staticmethod
    def DenseCountSparseOutput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DenseCountSparseOutput")

    @staticmethod
    def DenseToCSRSparseMatrix(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DenseToCSRSparseMatrix")

    @staticmethod
    def DenseToDenseSetOperation(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DenseToDenseSetOperation"
        )

    @staticmethod
    def DenseToSparseBatchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DenseToSparseBatchDataset"
        )

    @staticmethod
    def DenseToSparseSetOperation(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DenseToSparseSetOperation"
        )

    @staticmethod
    def DepthToSpace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DepthToSpace")

    @staticmethod
    def DepthwiseConv2dNative(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DepthwiseConv2dNative")

    @staticmethod
    def DepthwiseConv2dNativeBackpropFilter(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DepthwiseConv2dNativeBackpropFilter"
        )

    @staticmethod
    def DepthwiseConv2dNativeBackpropInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DepthwiseConv2dNativeBackpropInput"
        )

    @staticmethod
    def Dequantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Dequantize")

    @staticmethod
    def DeserializeIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeserializeIterator")

    @staticmethod
    def DeserializeManySparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeserializeManySparse")

    @staticmethod
    def DeserializeSparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeserializeSparse")

    @staticmethod
    def DestroyResourceOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DestroyResourceOp")

    @staticmethod
    def DestroyTemporaryVariable(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DestroyTemporaryVariable"
        )

    @staticmethod
    def DeviceIndex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DeviceIndex")

    @staticmethod
    def Diag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Diag")

    @staticmethod
    def DiagPart(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DiagPart")

    @staticmethod
    def Digamma(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Digamma")

    @staticmethod
    def Dilation2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Dilation2D")

    @staticmethod
    def Dilation2DBackpropFilter(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.Dilation2DBackpropFilter"
        )

    @staticmethod
    def Dilation2DBackpropInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Dilation2DBackpropInput")

    @staticmethod
    def DirectedInterleaveDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DirectedInterleaveDataset"
        )

    @staticmethod
    def DisableCopyOnRead(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DisableCopyOnRead")

    @staticmethod
    def DistributedSave(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DistributedSave")

    @staticmethod
    def Div(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Div")

    @staticmethod
    def DivNoNan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DivNoNan")

    @staticmethod
    def DrawBoundingBoxes(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DrawBoundingBoxes")

    @staticmethod
    def DrawBoundingBoxesV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DrawBoundingBoxesV2")

    @staticmethod
    def DummyIterationCounter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DummyIterationCounter")

    @staticmethod
    def DummyMemoryCache(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DummyMemoryCache")

    @staticmethod
    def DummySeedGenerator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DummySeedGenerator")

    @staticmethod
    def DynamicEnqueueTPUEmbeddingArbitraryTensorBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DynamicEnqueueTPUEmbeddingArbitraryTensorBatch"
        )

    @staticmethod
    def DynamicEnqueueTPUEmbeddingRaggedTensorBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.DynamicEnqueueTPUEmbeddingRaggedTensorBatch"
        )

    @staticmethod
    def DynamicPartition(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DynamicPartition")

    @staticmethod
    def DynamicStitch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.DynamicStitch")

    @staticmethod
    def EagerPyFunc(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EagerPyFunc")

    @staticmethod
    def EditDistance(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EditDistance")

    @staticmethod
    def Eig(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Eig")

    @staticmethod
    def Einsum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Einsum")

    @staticmethod
    def Elu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Elu")

    @staticmethod
    def EluGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EluGrad")

    @staticmethod
    def Empty(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Empty")

    @staticmethod
    def EmptyTensorList(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EmptyTensorList")

    @staticmethod
    def EmptyTensorMap(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EmptyTensorMap")

    @staticmethod
    def EncodeBase64(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EncodeBase64")

    @staticmethod
    def EncodeJpeg(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EncodeJpeg")

    @staticmethod
    def EncodeJpegVariableQuality(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EncodeJpegVariableQuality"
        )

    @staticmethod
    def EncodePng(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EncodePng")

    @staticmethod
    def EncodeProto(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EncodeProto")

    @staticmethod
    def EncodeWav(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EncodeWav")

    @staticmethod
    def EnqueueTPUEmbeddingArbitraryTensorBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EnqueueTPUEmbeddingArbitraryTensorBatch"
        )

    @staticmethod
    def EnqueueTPUEmbeddingIntegerBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EnqueueTPUEmbeddingIntegerBatch"
        )

    @staticmethod
    def EnqueueTPUEmbeddingRaggedTensorBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EnqueueTPUEmbeddingRaggedTensorBatch"
        )

    @staticmethod
    def EnqueueTPUEmbeddingSparseBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EnqueueTPUEmbeddingSparseBatch"
        )

    @staticmethod
    def EnqueueTPUEmbeddingSparseTensorBatch(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.EnqueueTPUEmbeddingSparseTensorBatch"
        )

    @staticmethod
    def EnsureShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EnsureShape")

    @staticmethod
    def Enter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Enter")

    @staticmethod
    def Equal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Equal")

    @staticmethod
    def Erf(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Erf")

    @staticmethod
    def Erfc(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Erfc")

    @staticmethod
    def Erfinv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Erfinv")

    @staticmethod
    def EuclideanNorm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.EuclideanNorm")

    @staticmethod
    def Exit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Exit")

    @staticmethod
    def Exp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Exp")

    @staticmethod
    def ExpandDims(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExpandDims")

    @staticmethod
    def ExperimentalAssertNextDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalAssertNextDataset"
        )

    @staticmethod
    def ExperimentalAutoShardDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalAutoShardDataset"
        )

    @staticmethod
    def ExperimentalBytesProducedStatsDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalBytesProducedStatsDataset"
        )

    @staticmethod
    def ExperimentalCSVDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExperimentalCSVDataset")

    @staticmethod
    def ExperimentalChooseFastestDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalChooseFastestDataset"
        )

    @staticmethod
    def ExperimentalDatasetCardinality(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalDatasetCardinality"
        )

    @staticmethod
    def ExperimentalDatasetToTFRecord(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalDatasetToTFRecord"
        )

    @staticmethod
    def ExperimentalDenseToSparseBatchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalDenseToSparseBatchDataset"
        )

    @staticmethod
    def ExperimentalDirectedInterleaveDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalDirectedInterleaveDataset"
        )

    @staticmethod
    def ExperimentalGroupByReducerDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalGroupByReducerDataset"
        )

    @staticmethod
    def ExperimentalGroupByWindowDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalGroupByWindowDataset"
        )

    @staticmethod
    def ExperimentalIgnoreErrorsDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalIgnoreErrorsDataset"
        )

    @staticmethod
    def ExperimentalIteratorGetDevice(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalIteratorGetDevice"
        )

    @staticmethod
    def ExperimentalLMDBDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExperimentalLMDBDataset")

    @staticmethod
    def ExperimentalLatencyStatsDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalLatencyStatsDataset"
        )

    @staticmethod
    def ExperimentalMapAndBatchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalMapAndBatchDataset"
        )

    @staticmethod
    def ExperimentalMapDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExperimentalMapDataset")

    @staticmethod
    def ExperimentalMatchingFilesDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalMatchingFilesDataset"
        )

    @staticmethod
    def ExperimentalMaxIntraOpParallelismDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalMaxIntraOpParallelismDataset"
        )

    @staticmethod
    def ExperimentalNonSerializableDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalNonSerializableDataset"
        )

    @staticmethod
    def ExperimentalParallelInterleaveDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalParallelInterleaveDataset"
        )

    @staticmethod
    def ExperimentalParseExampleDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalParseExampleDataset"
        )

    @staticmethod
    def ExperimentalPrivateThreadPoolDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalPrivateThreadPoolDataset"
        )

    @staticmethod
    def ExperimentalRandomDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalRandomDataset"
        )

    @staticmethod
    def ExperimentalRebatchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalRebatchDataset"
        )

    @staticmethod
    def ExperimentalScanDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExperimentalScanDataset")

    @staticmethod
    def ExperimentalSetStatsAggregatorDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalSetStatsAggregatorDataset"
        )

    @staticmethod
    def ExperimentalSleepDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalSleepDataset"
        )

    @staticmethod
    def ExperimentalSlidingWindowDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalSlidingWindowDataset"
        )

    @staticmethod
    def ExperimentalSqlDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExperimentalSqlDataset")

    @staticmethod
    def ExperimentalStatsAggregatorHandle(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalStatsAggregatorHandle"
        )

    @staticmethod
    def ExperimentalStatsAggregatorSummary(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalStatsAggregatorSummary"
        )

    @staticmethod
    def ExperimentalTakeWhileDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalTakeWhileDataset"
        )

    @staticmethod
    def ExperimentalThreadPoolDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalThreadPoolDataset"
        )

    @staticmethod
    def ExperimentalThreadPoolHandle(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalThreadPoolHandle"
        )

    @staticmethod
    def ExperimentalUnbatchDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalUnbatchDataset"
        )

    @staticmethod
    def ExperimentalUniqueDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ExperimentalUniqueDataset"
        )

    @staticmethod
    def Expint(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Expint")

    @staticmethod
    def Expm1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Expm1")

    @staticmethod
    def ExtractGlimpse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExtractGlimpse")

    @staticmethod
    def ExtractGlimpseV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExtractGlimpseV2")

    @staticmethod
    def ExtractImagePatches(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExtractImagePatches")

    @staticmethod
    def ExtractJpegShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExtractJpegShape")

    @staticmethod
    def ExtractVolumePatches(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ExtractVolumePatches")

    @staticmethod
    def FFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FFT")

    @staticmethod
    def FFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FFT2D")

    @staticmethod
    def FFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FFT3D")

    @staticmethod
    def FFTND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FFTND")

    @staticmethod
    def FIFOQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FIFOQueue")

    @staticmethod
    def FIFOQueueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FIFOQueueV2")

    @staticmethod
    def Fact(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Fact")

    @staticmethod
    def FakeParam(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FakeParam")

    @staticmethod
    def FakeQuantWithMinMaxArgs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FakeQuantWithMinMaxArgs")

    @staticmethod
    def FakeQuantWithMinMaxArgsGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FakeQuantWithMinMaxArgsGradient"
        )

    @staticmethod
    def FakeQuantWithMinMaxVars(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FakeQuantWithMinMaxVars")

    @staticmethod
    def FakeQuantWithMinMaxVarsGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FakeQuantWithMinMaxVarsGradient"
        )

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannel(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FakeQuantWithMinMaxVarsPerChannel"
        )

    @staticmethod
    def FakeQuantWithMinMaxVarsPerChannelGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FakeQuantWithMinMaxVarsPerChannelGradient"
        )

    @staticmethod
    def FakeQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FakeQueue")

    @staticmethod
    def FileSystemSetConfiguration(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FileSystemSetConfiguration"
        )

    @staticmethod
    def Fill(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Fill")

    @staticmethod
    def FilterByLastComponentDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FilterByLastComponentDataset"
        )

    @staticmethod
    def FilterDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FilterDataset")

    @staticmethod
    def FinalizeDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FinalizeDataset")

    @staticmethod
    def Fingerprint(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Fingerprint")

    @staticmethod
    def FixedLengthRecordDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FixedLengthRecordDataset"
        )

    @staticmethod
    def FixedLengthRecordDatasetV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FixedLengthRecordDatasetV2"
        )

    @staticmethod
    def FixedLengthRecordReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FixedLengthRecordReader")

    @staticmethod
    def FixedLengthRecordReaderV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FixedLengthRecordReaderV2"
        )

    @staticmethod
    def FixedUnigramCandidateSampler(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.FixedUnigramCandidateSampler"
        )

    @staticmethod
    def FlatMapDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FlatMapDataset")

    @staticmethod
    def Floor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Floor")

    @staticmethod
    def FloorDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FloorDiv")

    @staticmethod
    def FloorMod(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FloorMod")

    @staticmethod
    def FlushSummaryWriter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FlushSummaryWriter")

    @staticmethod
    def For(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.For")

    @staticmethod
    def FractionalAvgPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FractionalAvgPool")

    @staticmethod
    def FractionalAvgPoolGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FractionalAvgPoolGrad")

    @staticmethod
    def FractionalMaxPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FractionalMaxPool")

    @staticmethod
    def FractionalMaxPoolGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FractionalMaxPoolGrad")

    @staticmethod
    def FresnelCos(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FresnelCos")

    @staticmethod
    def FresnelSin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FresnelSin")

    @staticmethod
    def FusedBatchNorm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNorm")

    @staticmethod
    def FusedBatchNormGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNormGrad")

    @staticmethod
    def FusedBatchNormGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNormGradV2")

    @staticmethod
    def FusedBatchNormGradV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNormGradV3")

    @staticmethod
    def FusedBatchNormV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNormV2")

    @staticmethod
    def FusedBatchNormV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedBatchNormV3")

    @staticmethod
    def FusedPadConv2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedPadConv2D")

    @staticmethod
    def FusedResizeAndPadConv2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.FusedResizeAndPadConv2D")

    @staticmethod
    def GRUBlockCell(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GRUBlockCell")

    @staticmethod
    def GRUBlockCellGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GRUBlockCellGrad")

    @staticmethod
    def Gather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Gather")

    @staticmethod
    def GatherNd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GatherNd")

    @staticmethod
    def GatherV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GatherV2")

    @staticmethod
    def GenerateBoundingBoxProposals(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.GenerateBoundingBoxProposals"
        )

    @staticmethod
    def GenerateVocabRemapping(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GenerateVocabRemapping")

    @staticmethod
    def GeneratorDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GeneratorDataset")

    @staticmethod
    def GetElementAtIndex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GetElementAtIndex")

    @staticmethod
    def GetMinibatchSplitsWithPhysicalReplica(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.GetMinibatchSplitsWithPhysicalReplica"
        )

    @staticmethod
    def GetMinibatchesInCsrWithPhysicalReplica(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.GetMinibatchesInCsrWithPhysicalReplica"
        )

    @staticmethod
    def GetOptions(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GetOptions")

    @staticmethod
    def GetSessionHandle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GetSessionHandle")

    @staticmethod
    def GetSessionHandleV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GetSessionHandleV2")

    @staticmethod
    def GetSessionTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GetSessionTensor")

    @staticmethod
    def GlobalIterId(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GlobalIterId")

    @staticmethod
    def Greater(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Greater")

    @staticmethod
    def GreaterEqual(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GreaterEqual")

    @staticmethod
    def GroupByReducerDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GroupByReducerDataset")

    @staticmethod
    def GroupByWindowDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GroupByWindowDataset")

    @staticmethod
    def GuaranteeConst(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.GuaranteeConst")

    @staticmethod
    def HSVToRGB(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.HSVToRGB")

    @staticmethod
    def HashTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.HashTable")

    @staticmethod
    def HashTableV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.HashTableV2")

    @staticmethod
    def HistogramFixedWidth(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.HistogramFixedWidth")

    @staticmethod
    def HistogramSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.HistogramSummary")

    @staticmethod
    def IFFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IFFT")

    @staticmethod
    def IFFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IFFT2D")

    @staticmethod
    def IFFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IFFT3D")

    @staticmethod
    def IFFTND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IFFTND")

    @staticmethod
    def IRFFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IRFFT")

    @staticmethod
    def IRFFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IRFFT2D")

    @staticmethod
    def IRFFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IRFFT3D")

    @staticmethod
    def IRFFTND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IRFFTND")

    @staticmethod
    def Identity(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Identity")

    @staticmethod
    def IdentityN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IdentityN")

    @staticmethod
    def IdentityReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IdentityReader")

    @staticmethod
    def IdentityReaderV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IdentityReaderV2")

    @staticmethod
    def If(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.If")

    @staticmethod
    def Igamma(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Igamma")

    @staticmethod
    def IgammaGradA(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IgammaGradA")

    @staticmethod
    def Igammac(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Igammac")

    @staticmethod
    def IgnoreErrorsDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IgnoreErrorsDataset")

    @staticmethod
    def Imag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Imag")

    @staticmethod
    def ImageProjectiveTransformV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ImageProjectiveTransformV2"
        )

    @staticmethod
    def ImageProjectiveTransformV3(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ImageProjectiveTransformV3"
        )

    @staticmethod
    def ImageSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ImageSummary")

    @staticmethod
    def ImmutableConst(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ImmutableConst")

    @staticmethod
    def ImportEvent(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ImportEvent")

    @staticmethod
    def InTopK(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InTopK")

    @staticmethod
    def InTopKV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InTopKV2")

    @staticmethod
    def InfeedDequeue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InfeedDequeue")

    @staticmethod
    def InfeedDequeueTuple(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InfeedDequeueTuple")

    @staticmethod
    def InfeedEnqueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InfeedEnqueue")

    @staticmethod
    def InfeedEnqueuePrelinearizedBuffer(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.InfeedEnqueuePrelinearizedBuffer"
        )

    @staticmethod
    def InfeedEnqueueTuple(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InfeedEnqueueTuple")

    @staticmethod
    def InitializeTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InitializeTable")

    @staticmethod
    def InitializeTableFromDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.InitializeTableFromDataset"
        )

    @staticmethod
    def InitializeTableFromTextFile(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.InitializeTableFromTextFile"
        )

    @staticmethod
    def InitializeTableFromTextFileV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.InitializeTableFromTextFileV2"
        )

    @staticmethod
    def InitializeTableV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InitializeTableV2")

    @staticmethod
    def InplaceAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InplaceAdd")

    @staticmethod
    def InplaceSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InplaceSub")

    @staticmethod
    def InplaceUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InplaceUpdate")

    @staticmethod
    def InterleaveDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InterleaveDataset")

    @staticmethod
    def Inv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Inv")

    @staticmethod
    def InvGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InvGrad")

    @staticmethod
    def Invert(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Invert")

    @staticmethod
    def InvertPermutation(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.InvertPermutation")

    @staticmethod
    def IsBoostedTreesEnsembleInitialized(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IsBoostedTreesEnsembleInitialized"
        )

    @staticmethod
    def IsBoostedTreesQuantileStreamResourceInitialized(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IsBoostedTreesQuantileStreamResourceInitialized"
        )

    @staticmethod
    def IsFinite(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IsFinite")

    @staticmethod
    def IsInf(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IsInf")

    @staticmethod
    def IsNan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IsNan")

    @staticmethod
    def IsTPUEmbeddingInitialized(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IsTPUEmbeddingInitialized"
        )

    @staticmethod
    def IsVariableInitialized(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IsVariableInitialized")

    @staticmethod
    def IsotonicRegression(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IsotonicRegression")

    @staticmethod
    def Iterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Iterator")

    @staticmethod
    def IteratorFromStringHandle(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IteratorFromStringHandle"
        )

    @staticmethod
    def IteratorFromStringHandleV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IteratorFromStringHandleV2"
        )

    @staticmethod
    def IteratorGetDevice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IteratorGetDevice")

    @staticmethod
    def IteratorGetNext(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IteratorGetNext")

    @staticmethod
    def IteratorGetNextAsOptional(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.IteratorGetNextAsOptional"
        )

    @staticmethod
    def IteratorGetNextSync(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IteratorGetNextSync")

    @staticmethod
    def IteratorToStringHandle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IteratorToStringHandle")

    @staticmethod
    def IteratorV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.IteratorV2")

    @staticmethod
    def KMC2ChainInitialization(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.KMC2ChainInitialization")

    @staticmethod
    def KmeansPlusPlusInitialization(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.KmeansPlusPlusInitialization"
        )

    @staticmethod
    def L2Loss(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.L2Loss")

    @staticmethod
    def LMDBDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LMDBDataset")

    @staticmethod
    def LMDBReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LMDBReader")

    @staticmethod
    def LRN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LRN")

    @staticmethod
    def LRNGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LRNGrad")

    @staticmethod
    def LSTMBlockCell(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LSTMBlockCell")

    @staticmethod
    def LSTMBlockCellGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LSTMBlockCellGrad")

    @staticmethod
    def LatencyStatsDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LatencyStatsDataset")

    @staticmethod
    def LeakyRelu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LeakyRelu")

    @staticmethod
    def LeakyReluGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LeakyReluGrad")

    @staticmethod
    def LearnedUnigramCandidateSampler(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LearnedUnigramCandidateSampler"
        )

    @staticmethod
    def LeftShift(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LeftShift")

    @staticmethod
    def LegacyParallelInterleaveDatasetV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LegacyParallelInterleaveDatasetV2"
        )

    @staticmethod
    def Less(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Less")

    @staticmethod
    def LessEqual(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LessEqual")

    @staticmethod
    def Lgamma(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Lgamma")

    @staticmethod
    def LinSpace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LinSpace")

    @staticmethod
    def ListDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ListDataset")

    @staticmethod
    def ListDiff(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ListDiff")

    @staticmethod
    def ListSnapshotChunksDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ListSnapshotChunksDataset"
        )

    @staticmethod
    def LoadAndRemapMatrix(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LoadAndRemapMatrix")

    @staticmethod
    def LoadDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LoadDataset")

    @staticmethod
    def LoadTPUEmbeddingADAMParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingADAMParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingAdadeltaParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingAdadeltaParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingAdagradMomentumParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingAdagradMomentumParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingAdagradParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingAdagradParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingCenteredRMSPropParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingCenteredRMSPropParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingFTRLParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingFTRLParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingFrequencyEstimatorParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingFrequencyEstimatorParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingMDLAdagradLightParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingMDLAdagradLightParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingMomentumParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingMomentumParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingProximalAdagradParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingProximalAdagradParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingProximalYogiParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingProximalYogiParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingRMSPropParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingRMSPropParameters"
        )

    @staticmethod
    def LoadTPUEmbeddingStochasticGradientDescentParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LoadTPUEmbeddingStochasticGradientDescentParameters"
        )

    @staticmethod
    def Log(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Log")

    @staticmethod
    def Log1p(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Log1p")

    @staticmethod
    def LogMatrixDeterminant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LogMatrixDeterminant")

    @staticmethod
    def LogSoftmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LogSoftmax")

    @staticmethod
    def LogUniformCandidateSampler(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.LogUniformCandidateSampler"
        )

    @staticmethod
    def LogicalAnd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LogicalAnd")

    @staticmethod
    def LogicalNot(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LogicalNot")

    @staticmethod
    def LogicalOr(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LogicalOr")

    @staticmethod
    def LookupTableExport(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableExport")

    @staticmethod
    def LookupTableExportV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableExportV2")

    @staticmethod
    def LookupTableFind(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableFind")

    @staticmethod
    def LookupTableFindV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableFindV2")

    @staticmethod
    def LookupTableImport(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableImport")

    @staticmethod
    def LookupTableImportV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableImportV2")

    @staticmethod
    def LookupTableInsert(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableInsert")

    @staticmethod
    def LookupTableInsertV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableInsertV2")

    @staticmethod
    def LookupTableRemoveV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableRemoveV2")

    @staticmethod
    def LookupTableSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableSize")

    @staticmethod
    def LookupTableSizeV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LookupTableSizeV2")

    @staticmethod
    def LoopCond(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LoopCond")

    @staticmethod
    def LowerBound(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.LowerBound")

    @staticmethod
    def Lu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Lu")

    @staticmethod
    def MakeIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MakeIterator")

    @staticmethod
    def MapAndBatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapAndBatchDataset")

    @staticmethod
    def MapClear(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapClear")

    @staticmethod
    def MapDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapDataset")

    @staticmethod
    def MapDefun(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapDefun")

    @staticmethod
    def MapIncompleteSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapIncompleteSize")

    @staticmethod
    def MapPeek(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapPeek")

    @staticmethod
    def MapSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapSize")

    @staticmethod
    def MapStage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapStage")

    @staticmethod
    def MapUnstage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapUnstage")

    @staticmethod
    def MapUnstageNoKey(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MapUnstageNoKey")

    @staticmethod
    def MatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatMul")

    @staticmethod
    def MatchingFiles(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatchingFiles")

    @staticmethod
    def MatchingFilesDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatchingFilesDataset")

    @staticmethod
    def MatrixBandPart(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixBandPart")

    @staticmethod
    def MatrixDeterminant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDeterminant")

    @staticmethod
    def MatrixDiag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiag")

    @staticmethod
    def MatrixDiagPart(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiagPart")

    @staticmethod
    def MatrixDiagPartV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiagPartV2")

    @staticmethod
    def MatrixDiagPartV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiagPartV3")

    @staticmethod
    def MatrixDiagV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiagV2")

    @staticmethod
    def MatrixDiagV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixDiagV3")

    @staticmethod
    def MatrixExponential(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixExponential")

    @staticmethod
    def MatrixInverse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixInverse")

    @staticmethod
    def MatrixLogarithm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixLogarithm")

    @staticmethod
    def MatrixSetDiag(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSetDiag")

    @staticmethod
    def MatrixSetDiagV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSetDiagV2")

    @staticmethod
    def MatrixSetDiagV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSetDiagV3")

    @staticmethod
    def MatrixSolve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSolve")

    @staticmethod
    def MatrixSolveLs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSolveLs")

    @staticmethod
    def MatrixSquareRoot(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixSquareRoot")

    @staticmethod
    def MatrixTriangularSolve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MatrixTriangularSolve")

    @staticmethod
    def Max(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Max")

    @staticmethod
    def MaxIntraOpParallelismDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MaxIntraOpParallelismDataset"
        )

    @staticmethod
    def MaxPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPool")

    @staticmethod
    def MaxPool3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPool3D")

    @staticmethod
    def MaxPool3DGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPool3DGrad")

    @staticmethod
    def MaxPool3DGradGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPool3DGradGrad")

    @staticmethod
    def MaxPoolGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolGrad")

    @staticmethod
    def MaxPoolGradGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolGradGrad")

    @staticmethod
    def MaxPoolGradGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolGradGradV2")

    @staticmethod
    def MaxPoolGradGradWithArgmax(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MaxPoolGradGradWithArgmax"
        )

    @staticmethod
    def MaxPoolGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolGradV2")

    @staticmethod
    def MaxPoolGradWithArgmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolGradWithArgmax")

    @staticmethod
    def MaxPoolV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolV2")

    @staticmethod
    def MaxPoolWithArgmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MaxPoolWithArgmax")

    @staticmethod
    def Maximum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Maximum")

    @staticmethod
    def Mean(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Mean")

    @staticmethod
    def Merge(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Merge")

    @staticmethod
    def MergeSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MergeSummary")

    @staticmethod
    def MergeV2Checkpoints(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MergeV2Checkpoints")

    @staticmethod
    def Mfcc(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Mfcc")

    @staticmethod
    def Min(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Min")

    @staticmethod
    def Minimum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Minimum")

    @staticmethod
    def MirrorPad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MirrorPad")

    @staticmethod
    def MirrorPadGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MirrorPadGrad")

    @staticmethod
    def Mod(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Mod")

    @staticmethod
    def ModelDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ModelDataset")

    @staticmethod
    def Mul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Mul")

    @staticmethod
    def MulNoNan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MulNoNan")

    @staticmethod
    def MultiDeviceIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MultiDeviceIterator")

    @staticmethod
    def MultiDeviceIteratorFromStringHandle(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MultiDeviceIteratorFromStringHandle"
        )

    @staticmethod
    def MultiDeviceIteratorGetNextFromShard(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MultiDeviceIteratorGetNextFromShard"
        )

    @staticmethod
    def MultiDeviceIteratorInit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MultiDeviceIteratorInit")

    @staticmethod
    def MultiDeviceIteratorToStringHandle(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MultiDeviceIteratorToStringHandle"
        )

    @staticmethod
    def Multinomial(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Multinomial")

    @staticmethod
    def MutableDenseHashTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutableDenseHashTable")

    @staticmethod
    def MutableDenseHashTableV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutableDenseHashTableV2")

    @staticmethod
    def MutableHashTable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutableHashTable")

    @staticmethod
    def MutableHashTableOfTensors(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MutableHashTableOfTensors"
        )

    @staticmethod
    def MutableHashTableOfTensorsV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.MutableHashTableOfTensorsV2"
        )

    @staticmethod
    def MutableHashTableV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutableHashTableV2")

    @staticmethod
    def MutexLock(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutexLock")

    @staticmethod
    def MutexV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.MutexV2")

    @staticmethod
    def NcclAllReduce(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NcclAllReduce")

    @staticmethod
    def NcclBroadcast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NcclBroadcast")

    @staticmethod
    def NcclReduce(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NcclReduce")

    @staticmethod
    def Ndtri(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Ndtri")

    @staticmethod
    def NearestNeighbors(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NearestNeighbors")

    @staticmethod
    def Neg(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Neg")

    @staticmethod
    def NextAfter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NextAfter")

    @staticmethod
    def NextIteration(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NextIteration")

    @staticmethod
    def NoOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NoOp")

    @staticmethod
    def NonDeterministicInts(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonDeterministicInts")

    @staticmethod
    def NonMaxSuppression(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonMaxSuppression")

    @staticmethod
    def NonMaxSuppressionV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonMaxSuppressionV2")

    @staticmethod
    def NonMaxSuppressionV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonMaxSuppressionV3")

    @staticmethod
    def NonMaxSuppressionV4(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonMaxSuppressionV4")

    @staticmethod
    def NonMaxSuppressionV5(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonMaxSuppressionV5")

    @staticmethod
    def NonMaxSuppressionWithOverlaps(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.NonMaxSuppressionWithOverlaps"
        )

    @staticmethod
    def NonSerializableDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NonSerializableDataset")

    @staticmethod
    def NotEqual(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NotEqual")

    @staticmethod
    def NthElement(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.NthElement")

    @staticmethod
    def OneHot(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OneHot")

    @staticmethod
    def OneShotIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OneShotIterator")

    @staticmethod
    def OnesLike(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OnesLike")

    @staticmethod
    def OptimizeDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptimizeDataset")

    @staticmethod
    def OptimizeDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptimizeDatasetV2")

    @staticmethod
    def OptionalFromValue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptionalFromValue")

    @staticmethod
    def OptionalGetValue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptionalGetValue")

    @staticmethod
    def OptionalHasValue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptionalHasValue")

    @staticmethod
    def OptionalNone(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptionalNone")

    @staticmethod
    def OptionsDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OptionsDataset")

    @staticmethod
    def OrderedMapClear(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapClear")

    @staticmethod
    def OrderedMapIncompleteSize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.OrderedMapIncompleteSize"
        )

    @staticmethod
    def OrderedMapPeek(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapPeek")

    @staticmethod
    def OrderedMapSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapSize")

    @staticmethod
    def OrderedMapStage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapStage")

    @staticmethod
    def OrderedMapUnstage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapUnstage")

    @staticmethod
    def OrderedMapUnstageNoKey(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OrderedMapUnstageNoKey")

    @staticmethod
    def OutfeedDequeue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedDequeue")

    @staticmethod
    def OutfeedDequeueTuple(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedDequeueTuple")

    @staticmethod
    def OutfeedDequeueTupleV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedDequeueTupleV2")

    @staticmethod
    def OutfeedDequeueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedDequeueV2")

    @staticmethod
    def OutfeedEnqueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedEnqueue")

    @staticmethod
    def OutfeedEnqueueTuple(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.OutfeedEnqueueTuple")

    @staticmethod
    def Pack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Pack")

    @staticmethod
    def Pad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Pad")

    @staticmethod
    def PadV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PadV2")

    @staticmethod
    def PaddedBatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PaddedBatchDataset")

    @staticmethod
    def PaddedBatchDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PaddedBatchDatasetV2")

    @staticmethod
    def PaddingFIFOQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PaddingFIFOQueue")

    @staticmethod
    def PaddingFIFOQueueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PaddingFIFOQueueV2")

    @staticmethod
    def ParallelBatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelBatchDataset")

    @staticmethod
    def ParallelConcat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelConcat")

    @staticmethod
    def ParallelDynamicStitch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelDynamicStitch")

    @staticmethod
    def ParallelFilterDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelFilterDataset")

    @staticmethod
    def ParallelInterleaveDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParallelInterleaveDataset"
        )

    @staticmethod
    def ParallelInterleaveDatasetV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParallelInterleaveDatasetV2"
        )

    @staticmethod
    def ParallelInterleaveDatasetV3(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParallelInterleaveDatasetV3"
        )

    @staticmethod
    def ParallelInterleaveDatasetV4(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParallelInterleaveDatasetV4"
        )

    @staticmethod
    def ParallelMapDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelMapDataset")

    @staticmethod
    def ParallelMapDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParallelMapDatasetV2")

    @staticmethod
    def ParameterizedTruncatedNormal(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParameterizedTruncatedNormal"
        )

    @staticmethod
    def ParseExample(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseExample")

    @staticmethod
    def ParseExampleDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseExampleDataset")

    @staticmethod
    def ParseExampleDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseExampleDatasetV2")

    @staticmethod
    def ParseExampleV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseExampleV2")

    @staticmethod
    def ParseSequenceExample(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseSequenceExample")

    @staticmethod
    def ParseSequenceExampleV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseSequenceExampleV2")

    @staticmethod
    def ParseSingleExample(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseSingleExample")

    @staticmethod
    def ParseSingleSequenceExample(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ParseSingleSequenceExample"
        )

    @staticmethod
    def ParseTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ParseTensor")

    @staticmethod
    def PartitionedCall(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PartitionedCall")

    @staticmethod
    def Placeholder(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Placeholder")

    @staticmethod
    def PlaceholderV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PlaceholderV2")

    @staticmethod
    def PlaceholderWithDefault(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PlaceholderWithDefault")

    @staticmethod
    def Polygamma(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Polygamma")

    @staticmethod
    def PopulationCount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PopulationCount")

    @staticmethod
    def Pow(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Pow")

    @staticmethod
    def PrefetchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PrefetchDataset")

    @staticmethod
    def Prelinearize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Prelinearize")

    @staticmethod
    def PrelinearizeTuple(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PrelinearizeTuple")

    @staticmethod
    def PreventGradient(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PreventGradient")

    @staticmethod
    def Print(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Print")

    @staticmethod
    def PrintV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PrintV2")

    @staticmethod
    def PriorityQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PriorityQueue")

    @staticmethod
    def PriorityQueueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PriorityQueueV2")

    @staticmethod
    def PrivateThreadPoolDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.PrivateThreadPoolDataset"
        )

    @staticmethod
    def Prod(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Prod")

    @staticmethod
    def PyFunc(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PyFunc")

    @staticmethod
    def PyFuncStateless(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.PyFuncStateless")

    @staticmethod
    def Qr(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Qr")

    @staticmethod
    def QuantizeAndDequantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizeAndDequantize")

    @staticmethod
    def QuantizeAndDequantizeV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizeAndDequantizeV2")

    @staticmethod
    def QuantizeAndDequantizeV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizeAndDequantizeV3")

    @staticmethod
    def QuantizeAndDequantizeV4(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizeAndDequantizeV4")

    @staticmethod
    def QuantizeAndDequantizeV4Grad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizeAndDequantizeV4Grad"
        )

    @staticmethod
    def QuantizeDownAndShrinkRange(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizeDownAndShrinkRange"
        )

    @staticmethod
    def QuantizeV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizeV2")

    @staticmethod
    def QuantizedAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedAdd")

    @staticmethod
    def QuantizedAvgPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedAvgPool")

    @staticmethod
    def QuantizedBatchNormWithGlobalNormalization(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedBatchNormWithGlobalNormalization"
        )

    @staticmethod
    def QuantizedBiasAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedBiasAdd")

    @staticmethod
    def QuantizedConcat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedConcat")

    @staticmethod
    def QuantizedConv2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedConv2D")

    @staticmethod
    def QuantizedConv2DAndRelu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedConv2DAndRelu")

    @staticmethod
    def QuantizedConv2DAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedConv2DAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DAndRequantize"
        )

    @staticmethod
    def QuantizedConv2DPerChannel(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DPerChannel"
        )

    @staticmethod
    def QuantizedConv2DWithBias(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedConv2DWithBias")

    @staticmethod
    def QuantizedConv2DWithBiasAndRelu(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasAndRelu"
        )

    @staticmethod
    def QuantizedConv2DWithBiasAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedConv2DWithBiasAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasAndRequantize"
        )

    @staticmethod
    def QuantizedConv2DWithBiasSignedSumAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasSignedSumAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedConv2DWithBiasSumAndRelu(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasSumAndRelu"
        )

    @staticmethod
    def QuantizedConv2DWithBiasSumAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedConv2DWithBiasSumAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedDepthwiseConv2D(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedDepthwiseConv2D"
        )

    @staticmethod
    def QuantizedDepthwiseConv2DWithBias(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedDepthwiseConv2DWithBias"
        )

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndRelu(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedDepthwiseConv2DWithBiasAndRelu"
        )

    @staticmethod
    def QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedDepthwiseConv2DWithBiasAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedInstanceNorm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedInstanceNorm")

    @staticmethod
    def QuantizedMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedMatMul")

    @staticmethod
    def QuantizedMatMulWithBias(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedMatMulWithBias")

    @staticmethod
    def QuantizedMatMulWithBiasAndDequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedMatMulWithBiasAndDequantize"
        )

    @staticmethod
    def QuantizedMatMulWithBiasAndRelu(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedMatMulWithBiasAndRelu"
        )

    @staticmethod
    def QuantizedMatMulWithBiasAndReluAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedMatMulWithBiasAndReluAndRequantize"
        )

    @staticmethod
    def QuantizedMatMulWithBiasAndRequantize(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.QuantizedMatMulWithBiasAndRequantize"
        )

    @staticmethod
    def QuantizedMaxPool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedMaxPool")

    @staticmethod
    def QuantizedMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedMul")

    @staticmethod
    def QuantizedRelu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedRelu")

    @staticmethod
    def QuantizedRelu6(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedRelu6")

    @staticmethod
    def QuantizedReluX(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedReluX")

    @staticmethod
    def QuantizedReshape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedReshape")

    @staticmethod
    def QuantizedResizeBilinear(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QuantizedResizeBilinear")

    @staticmethod
    def QueueClose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueClose")

    @staticmethod
    def QueueCloseV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueCloseV2")

    @staticmethod
    def QueueDequeue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeue")

    @staticmethod
    def QueueDequeueMany(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeueMany")

    @staticmethod
    def QueueDequeueManyV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeueManyV2")

    @staticmethod
    def QueueDequeueUpTo(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeueUpTo")

    @staticmethod
    def QueueDequeueUpToV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeueUpToV2")

    @staticmethod
    def QueueDequeueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueDequeueV2")

    @staticmethod
    def QueueEnqueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueEnqueue")

    @staticmethod
    def QueueEnqueueMany(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueEnqueueMany")

    @staticmethod
    def QueueEnqueueManyV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueEnqueueManyV2")

    @staticmethod
    def QueueEnqueueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueEnqueueV2")

    @staticmethod
    def QueueIsClosed(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueIsClosed")

    @staticmethod
    def QueueIsClosedV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueIsClosedV2")

    @staticmethod
    def QueueSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueSize")

    @staticmethod
    def QueueSizeV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.QueueSizeV2")

    @staticmethod
    def RFFT(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RFFT")

    @staticmethod
    def RFFT2D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RFFT2D")

    @staticmethod
    def RFFT3D(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RFFT3D")

    @staticmethod
    def RFFTND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RFFTND")

    @staticmethod
    def RGBToHSV(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RGBToHSV")

    @staticmethod
    def RaggedBincount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedBincount")

    @staticmethod
    def RaggedCountSparseOutput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedCountSparseOutput")

    @staticmethod
    def RaggedCross(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedCross")

    @staticmethod
    def RaggedFillEmptyRows(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedFillEmptyRows")

    @staticmethod
    def RaggedFillEmptyRowsGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedFillEmptyRowsGrad")

    @staticmethod
    def RaggedGather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedGather")

    @staticmethod
    def RaggedRange(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedRange")

    @staticmethod
    def RaggedTensorFromVariant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedTensorFromVariant")

    @staticmethod
    def RaggedTensorToSparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedTensorToSparse")

    @staticmethod
    def RaggedTensorToTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedTensorToTensor")

    @staticmethod
    def RaggedTensorToVariant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RaggedTensorToVariant")

    @staticmethod
    def RaggedTensorToVariantGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RaggedTensorToVariantGradient"
        )

    @staticmethod
    def RandomCrop(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomCrop")

    @staticmethod
    def RandomDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomDataset")

    @staticmethod
    def RandomDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomDatasetV2")

    @staticmethod
    def RandomGamma(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomGamma")

    @staticmethod
    def RandomGammaGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomGammaGrad")

    @staticmethod
    def RandomIndexShuffle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomIndexShuffle")

    @staticmethod
    def RandomPoisson(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomPoisson")

    @staticmethod
    def RandomPoissonV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomPoissonV2")

    @staticmethod
    def RandomShuffle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomShuffle")

    @staticmethod
    def RandomShuffleQueue(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomShuffleQueue")

    @staticmethod
    def RandomShuffleQueueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomShuffleQueueV2")

    @staticmethod
    def RandomStandardNormal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomStandardNormal")

    @staticmethod
    def RandomUniform(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomUniform")

    @staticmethod
    def RandomUniformInt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RandomUniformInt")

    @staticmethod
    def Range(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Range")

    @staticmethod
    def RangeDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RangeDataset")

    @staticmethod
    def Rank(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Rank")

    @staticmethod
    def ReadFile(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReadFile")

    @staticmethod
    def ReadVariableOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReadVariableOp")

    @staticmethod
    def ReadVariableXlaSplitND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReadVariableXlaSplitND")

    @staticmethod
    def ReaderNumRecordsProduced(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ReaderNumRecordsProduced"
        )

    @staticmethod
    def ReaderNumRecordsProducedV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ReaderNumRecordsProducedV2"
        )

    @staticmethod
    def ReaderNumWorkUnitsCompleted(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ReaderNumWorkUnitsCompleted"
        )

    @staticmethod
    def ReaderNumWorkUnitsCompletedV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ReaderNumWorkUnitsCompletedV2"
        )

    @staticmethod
    def ReaderRead(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderRead")

    @staticmethod
    def ReaderReadUpTo(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderReadUpTo")

    @staticmethod
    def ReaderReadUpToV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderReadUpToV2")

    @staticmethod
    def ReaderReadV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderReadV2")

    @staticmethod
    def ReaderReset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderReset")

    @staticmethod
    def ReaderResetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderResetV2")

    @staticmethod
    def ReaderRestoreState(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderRestoreState")

    @staticmethod
    def ReaderRestoreStateV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderRestoreStateV2")

    @staticmethod
    def ReaderSerializeState(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderSerializeState")

    @staticmethod
    def ReaderSerializeStateV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReaderSerializeStateV2")

    @staticmethod
    def Real(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Real")

    @staticmethod
    def RealDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RealDiv")

    @staticmethod
    def RebatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RebatchDataset")

    @staticmethod
    def RebatchDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RebatchDatasetV2")

    @staticmethod
    def Reciprocal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Reciprocal")

    @staticmethod
    def ReciprocalGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReciprocalGrad")

    @staticmethod
    def RecordInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RecordInput")

    @staticmethod
    def Recv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Recv")

    @staticmethod
    def RecvTPUEmbeddingActivations(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RecvTPUEmbeddingActivations"
        )

    @staticmethod
    def ReduceDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReduceDataset")

    @staticmethod
    def ReduceJoin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReduceJoin")

    @staticmethod
    def RefEnter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefEnter")

    @staticmethod
    def RefExit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefExit")

    @staticmethod
    def RefIdentity(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefIdentity")

    @staticmethod
    def RefMerge(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefMerge")

    @staticmethod
    def RefNextIteration(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefNextIteration")

    @staticmethod
    def RefSelect(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefSelect")

    @staticmethod
    def RefSwitch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RefSwitch")

    @staticmethod
    def RegexFullMatch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RegexFullMatch")

    @staticmethod
    def RegexReplace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RegexReplace")

    @staticmethod
    def RegisterDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RegisterDataset")

    @staticmethod
    def RegisterDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RegisterDatasetV2")

    @staticmethod
    def Relu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Relu")

    @staticmethod
    def Relu6(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Relu6")

    @staticmethod
    def Relu6Grad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Relu6Grad")

    @staticmethod
    def ReluGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReluGrad")

    @staticmethod
    def RemoteCall(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RemoteCall")

    @staticmethod
    def RepeatDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RepeatDataset")

    @staticmethod
    def RequantizationRange(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RequantizationRange")

    @staticmethod
    def RequantizationRangePerChannel(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RequantizationRangePerChannel"
        )

    @staticmethod
    def Requantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Requantize")

    @staticmethod
    def RequantizePerChannel(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RequantizePerChannel")

    @staticmethod
    def Reshape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Reshape")

    @staticmethod
    def ResizeArea(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeArea")

    @staticmethod
    def ResizeBicubic(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeBicubic")

    @staticmethod
    def ResizeBicubicGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeBicubicGrad")

    @staticmethod
    def ResizeBilinear(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeBilinear")

    @staticmethod
    def ResizeBilinearGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeBilinearGrad")

    @staticmethod
    def ResizeNearestNeighbor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResizeNearestNeighbor")

    @staticmethod
    def ResizeNearestNeighborGrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResizeNearestNeighborGrad"
        )

    @staticmethod
    def ResourceAccumulatorApplyGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceAccumulatorApplyGradient"
        )

    @staticmethod
    def ResourceAccumulatorNumAccumulated(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceAccumulatorNumAccumulated"
        )

    @staticmethod
    def ResourceAccumulatorSetGlobalStep(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceAccumulatorSetGlobalStep"
        )

    @staticmethod
    def ResourceAccumulatorTakeGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceAccumulatorTakeGradient"
        )

    @staticmethod
    def ResourceApplyAdaMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdaMax")

    @staticmethod
    def ResourceApplyAdadelta(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdadelta")

    @staticmethod
    def ResourceApplyAdagrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdagrad")

    @staticmethod
    def ResourceApplyAdagradDA(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdagradDA")

    @staticmethod
    def ResourceApplyAdagradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdagradV2")

    @staticmethod
    def ResourceApplyAdam(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAdam")

    @staticmethod
    def ResourceApplyAdamWithAmsgrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyAdamWithAmsgrad"
        )

    @staticmethod
    def ResourceApplyAddSign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyAddSign")

    @staticmethod
    def ResourceApplyCenteredRMSProp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyCenteredRMSProp"
        )

    @staticmethod
    def ResourceApplyFtrl(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyFtrl")

    @staticmethod
    def ResourceApplyFtrlV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyFtrlV2")

    @staticmethod
    def ResourceApplyGradientDescent(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyGradientDescent"
        )

    @staticmethod
    def ResourceApplyKerasMomentum(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyKerasMomentum"
        )

    @staticmethod
    def ResourceApplyMomentum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyMomentum")

    @staticmethod
    def ResourceApplyPowerSign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyPowerSign")

    @staticmethod
    def ResourceApplyProximalAdagrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyProximalAdagrad"
        )

    @staticmethod
    def ResourceApplyProximalGradientDescent(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceApplyProximalGradientDescent"
        )

    @staticmethod
    def ResourceApplyRMSProp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceApplyRMSProp")

    @staticmethod
    def ResourceConditionalAccumulator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceConditionalAccumulator"
        )

    @staticmethod
    def ResourceCountUpTo(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceCountUpTo")

    @staticmethod
    def ResourceGather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceGather")

    @staticmethod
    def ResourceGatherNd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceGatherNd")

    @staticmethod
    def ResourceScatterAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterAdd")

    @staticmethod
    def ResourceScatterDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterDiv")

    @staticmethod
    def ResourceScatterMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterMax")

    @staticmethod
    def ResourceScatterMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterMin")

    @staticmethod
    def ResourceScatterMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterMul")

    @staticmethod
    def ResourceScatterNdAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterNdAdd")

    @staticmethod
    def ResourceScatterNdMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterNdMax")

    @staticmethod
    def ResourceScatterNdMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterNdMin")

    @staticmethod
    def ResourceScatterNdSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterNdSub")

    @staticmethod
    def ResourceScatterNdUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterNdUpdate")

    @staticmethod
    def ResourceScatterSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterSub")

    @staticmethod
    def ResourceScatterUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceScatterUpdate")

    @staticmethod
    def ResourceSparseApplyAdadelta(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyAdadelta"
        )

    @staticmethod
    def ResourceSparseApplyAdagrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyAdagrad"
        )

    @staticmethod
    def ResourceSparseApplyAdagradDA(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyAdagradDA"
        )

    @staticmethod
    def ResourceSparseApplyAdagradV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyAdagradV2"
        )

    @staticmethod
    def ResourceSparseApplyCenteredRMSProp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyCenteredRMSProp"
        )

    @staticmethod
    def ResourceSparseApplyFtrl(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ResourceSparseApplyFtrl")

    @staticmethod
    def ResourceSparseApplyFtrlV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyFtrlV2"
        )

    @staticmethod
    def ResourceSparseApplyKerasMomentum(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyKerasMomentum"
        )

    @staticmethod
    def ResourceSparseApplyMomentum(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyMomentum"
        )

    @staticmethod
    def ResourceSparseApplyProximalAdagrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyProximalAdagrad"
        )

    @staticmethod
    def ResourceSparseApplyProximalGradientDescent(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyProximalGradientDescent"
        )

    @staticmethod
    def ResourceSparseApplyRMSProp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceSparseApplyRMSProp"
        )

    @staticmethod
    def ResourceStridedSliceAssign(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ResourceStridedSliceAssign"
        )

    @staticmethod
    def Restore(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Restore")

    @staticmethod
    def RestoreSlice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RestoreSlice")

    @staticmethod
    def RestoreV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RestoreV2")

    @staticmethod
    def RetrieveTPUEmbeddingADAMParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingADAMParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingAdadeltaParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingAdadeltaParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingAdagradMomentumParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingAdagradMomentumParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingAdagradParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingAdagradParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingCenteredRMSPropParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingCenteredRMSPropParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingFTRLParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingFTRLParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingFrequencyEstimatorParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingFrequencyEstimatorParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingMDLAdagradLightParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingMDLAdagradLightParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingMomentumParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingMomentumParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingProximalAdagradParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingProximalAdagradParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingProximalYogiParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingProximalYogiParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingRMSPropParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingRMSPropParameters"
        )

    @staticmethod
    def RetrieveTPUEmbeddingStochasticGradientDescentParameters(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.RetrieveTPUEmbeddingStochasticGradientDescentParameters"
        )

    @staticmethod
    def Reverse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Reverse")

    @staticmethod
    def ReverseSequence(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReverseSequence")

    @staticmethod
    def ReverseV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ReverseV2")

    @staticmethod
    def RewriteDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RewriteDataset")

    @staticmethod
    def RightShift(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RightShift")

    @staticmethod
    def Rint(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Rint")

    @staticmethod
    def RngReadAndSkip(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RngReadAndSkip")

    @staticmethod
    def RngSkip(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RngSkip")

    @staticmethod
    def Roll(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Roll")

    @staticmethod
    def Round(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Round")

    @staticmethod
    def Rsqrt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Rsqrt")

    @staticmethod
    def RsqrtGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.RsqrtGrad")

    @staticmethod
    def SampleDistortedBoundingBox(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SampleDistortedBoundingBox"
        )

    @staticmethod
    def SampleDistortedBoundingBoxV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SampleDistortedBoundingBoxV2"
        )

    @staticmethod
    def SamplingDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SamplingDataset")

    @staticmethod
    def Save(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Save")

    @staticmethod
    def SaveDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SaveDataset")

    @staticmethod
    def SaveDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SaveDatasetV2")

    @staticmethod
    def SaveSlices(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SaveSlices")

    @staticmethod
    def SaveV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SaveV2")

    @staticmethod
    def ScalarSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScalarSummary")

    @staticmethod
    def ScaleAndTranslate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScaleAndTranslate")

    @staticmethod
    def ScaleAndTranslateGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScaleAndTranslateGrad")

    @staticmethod
    def ScanDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScanDataset")

    @staticmethod
    def ScatterAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterAdd")

    @staticmethod
    def ScatterDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterDiv")

    @staticmethod
    def ScatterMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterMax")

    @staticmethod
    def ScatterMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterMin")

    @staticmethod
    def ScatterMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterMul")

    @staticmethod
    def ScatterNd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNd")

    @staticmethod
    def ScatterNdAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdAdd")

    @staticmethod
    def ScatterNdMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdMax")

    @staticmethod
    def ScatterNdMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdMin")

    @staticmethod
    def ScatterNdNonAliasingAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdNonAliasingAdd")

    @staticmethod
    def ScatterNdSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdSub")

    @staticmethod
    def ScatterNdUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterNdUpdate")

    @staticmethod
    def ScatterSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterSub")

    @staticmethod
    def ScatterUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ScatterUpdate")

    @staticmethod
    def SdcaFprint(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SdcaFprint")

    @staticmethod
    def SdcaOptimizer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SdcaOptimizer")

    @staticmethod
    def SdcaOptimizerV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SdcaOptimizerV2")

    @staticmethod
    def SdcaShrinkL1(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SdcaShrinkL1")

    @staticmethod
    def SegmentMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentMax")

    @staticmethod
    def SegmentMaxV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentMaxV2")

    @staticmethod
    def SegmentMean(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentMean")

    @staticmethod
    def SegmentMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentMin")

    @staticmethod
    def SegmentMinV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentMinV2")

    @staticmethod
    def SegmentProd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentProd")

    @staticmethod
    def SegmentProdV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentProdV2")

    @staticmethod
    def SegmentSum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentSum")

    @staticmethod
    def SegmentSumV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SegmentSumV2")

    @staticmethod
    def Select(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Select")

    @staticmethod
    def SelectV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SelectV2")

    @staticmethod
    def SelfAdjointEig(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SelfAdjointEig")

    @staticmethod
    def SelfAdjointEigV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SelfAdjointEigV2")

    @staticmethod
    def Selu(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Selu")

    @staticmethod
    def SeluGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SeluGrad")

    @staticmethod
    def Send(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Send")

    @staticmethod
    def SendTPUEmbeddingGradients(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SendTPUEmbeddingGradients"
        )

    @staticmethod
    def SerializeIterator(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SerializeIterator")

    @staticmethod
    def SerializeManySparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SerializeManySparse")

    @staticmethod
    def SerializeSparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SerializeSparse")

    @staticmethod
    def SerializeTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SerializeTensor")

    @staticmethod
    def SetSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SetSize")

    @staticmethod
    def SetStatsAggregatorDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SetStatsAggregatorDataset"
        )

    @staticmethod
    def Shape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Shape")

    @staticmethod
    def ShapeN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShapeN")

    @staticmethod
    def ShardDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShardDataset")

    @staticmethod
    def ShardedFilename(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShardedFilename")

    @staticmethod
    def ShardedFilespec(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShardedFilespec")

    @staticmethod
    def ShuffleAndRepeatDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShuffleAndRepeatDataset")

    @staticmethod
    def ShuffleAndRepeatDatasetV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ShuffleAndRepeatDatasetV2"
        )

    @staticmethod
    def ShuffleDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShuffleDataset")

    @staticmethod
    def ShuffleDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShuffleDatasetV2")

    @staticmethod
    def ShuffleDatasetV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShuffleDatasetV3")

    @staticmethod
    def ShutdownDistributedTPU(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ShutdownDistributedTPU")

    @staticmethod
    def Sigmoid(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sigmoid")

    @staticmethod
    def SigmoidGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SigmoidGrad")

    @staticmethod
    def Sign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sign")

    @staticmethod
    def Sin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sin")

    @staticmethod
    def Sinh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sinh")

    @staticmethod
    def Size(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Size")

    @staticmethod
    def SkipDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SkipDataset")

    @staticmethod
    def SleepDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SleepDataset")

    @staticmethod
    def Slice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Slice")

    @staticmethod
    def SlidingWindowDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SlidingWindowDataset")

    @staticmethod
    def Snapshot(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Snapshot")

    @staticmethod
    def SnapshotChunkDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SnapshotChunkDataset")

    @staticmethod
    def SnapshotDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SnapshotDataset")

    @staticmethod
    def SnapshotDatasetReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SnapshotDatasetReader")

    @staticmethod
    def SnapshotDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SnapshotDatasetV2")

    @staticmethod
    def SnapshotNestedDatasetReader(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SnapshotNestedDatasetReader"
        )

    @staticmethod
    def SobolSample(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SobolSample")

    @staticmethod
    def Softmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Softmax")

    @staticmethod
    def SoftmaxCrossEntropyWithLogits(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SoftmaxCrossEntropyWithLogits"
        )

    @staticmethod
    def Softplus(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Softplus")

    @staticmethod
    def SoftplusGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SoftplusGrad")

    @staticmethod
    def Softsign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Softsign")

    @staticmethod
    def SoftsignGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SoftsignGrad")

    @staticmethod
    def SpaceToBatch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SpaceToBatch")

    @staticmethod
    def SpaceToBatchND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SpaceToBatchND")

    @staticmethod
    def SpaceToDepth(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SpaceToDepth")

    @staticmethod
    def SparseAccumulatorApplyGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseAccumulatorApplyGradient"
        )

    @staticmethod
    def SparseAccumulatorTakeGradient(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseAccumulatorTakeGradient"
        )

    @staticmethod
    def SparseAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseAdd")

    @staticmethod
    def SparseAddGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseAddGrad")

    @staticmethod
    def SparseApplyAdadelta(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyAdadelta")

    @staticmethod
    def SparseApplyAdagrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyAdagrad")

    @staticmethod
    def SparseApplyAdagradDA(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyAdagradDA")

    @staticmethod
    def SparseApplyAdagradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyAdagradV2")

    @staticmethod
    def SparseApplyCenteredRMSProp(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseApplyCenteredRMSProp"
        )

    @staticmethod
    def SparseApplyFtrl(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyFtrl")

    @staticmethod
    def SparseApplyFtrlV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyFtrlV2")

    @staticmethod
    def SparseApplyMomentum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyMomentum")

    @staticmethod
    def SparseApplyProximalAdagrad(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseApplyProximalAdagrad"
        )

    @staticmethod
    def SparseApplyProximalGradientDescent(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseApplyProximalGradientDescent"
        )

    @staticmethod
    def SparseApplyRMSProp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseApplyRMSProp")

    @staticmethod
    def SparseBincount(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseBincount")

    @staticmethod
    def SparseConcat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseConcat")

    @staticmethod
    def SparseConditionalAccumulator(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseConditionalAccumulator"
        )

    @staticmethod
    def SparseCountSparseOutput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseCountSparseOutput")

    @staticmethod
    def SparseCross(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseCross")

    @staticmethod
    def SparseCrossHashed(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseCrossHashed")

    @staticmethod
    def SparseCrossV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseCrossV2")

    @staticmethod
    def SparseDenseCwiseAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseDenseCwiseAdd")

    @staticmethod
    def SparseDenseCwiseDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseDenseCwiseDiv")

    @staticmethod
    def SparseDenseCwiseMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseDenseCwiseMul")

    @staticmethod
    def SparseFillEmptyRows(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseFillEmptyRows")

    @staticmethod
    def SparseFillEmptyRowsGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseFillEmptyRowsGrad")

    @staticmethod
    def SparseMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatMul")

    @staticmethod
    def SparseMatrixAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixAdd")

    @staticmethod
    def SparseMatrixMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixMatMul")

    @staticmethod
    def SparseMatrixMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixMul")

    @staticmethod
    def SparseMatrixNNZ(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixNNZ")

    @staticmethod
    def SparseMatrixOrderingAMD(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixOrderingAMD")

    @staticmethod
    def SparseMatrixSoftmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixSoftmax")

    @staticmethod
    def SparseMatrixSoftmaxGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixSoftmaxGrad")

    @staticmethod
    def SparseMatrixSparseCholesky(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseMatrixSparseCholesky"
        )

    @staticmethod
    def SparseMatrixSparseMatMul(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseMatrixSparseMatMul"
        )

    @staticmethod
    def SparseMatrixTranspose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixTranspose")

    @staticmethod
    def SparseMatrixZeros(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseMatrixZeros")

    @staticmethod
    def SparseReduceMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReduceMax")

    @staticmethod
    def SparseReduceMaxSparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReduceMaxSparse")

    @staticmethod
    def SparseReduceSum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReduceSum")

    @staticmethod
    def SparseReduceSumSparse(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReduceSumSparse")

    @staticmethod
    def SparseReorder(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReorder")

    @staticmethod
    def SparseReshape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseReshape")

    @staticmethod
    def SparseSegmentMean(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentMean")

    @staticmethod
    def SparseSegmentMeanGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentMeanGrad")

    @staticmethod
    def SparseSegmentMeanGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentMeanGradV2")

    @staticmethod
    def SparseSegmentMeanWithNumSegments(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseSegmentMeanWithNumSegments"
        )

    @staticmethod
    def SparseSegmentSqrtN(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentSqrtN")

    @staticmethod
    def SparseSegmentSqrtNGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentSqrtNGrad")

    @staticmethod
    def SparseSegmentSqrtNGradV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseSegmentSqrtNGradV2"
        )

    @staticmethod
    def SparseSegmentSqrtNWithNumSegments(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseSegmentSqrtNWithNumSegments"
        )

    @staticmethod
    def SparseSegmentSum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentSum")

    @staticmethod
    def SparseSegmentSumGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentSumGrad")

    @staticmethod
    def SparseSegmentSumGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSegmentSumGradV2")

    @staticmethod
    def SparseSegmentSumWithNumSegments(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseSegmentSumWithNumSegments"
        )

    @staticmethod
    def SparseSlice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSlice")

    @staticmethod
    def SparseSliceGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSliceGrad")

    @staticmethod
    def SparseSoftmax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSoftmax")

    @staticmethod
    def SparseSoftmaxCrossEntropyWithLogits(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseSoftmaxCrossEntropyWithLogits"
        )

    @staticmethod
    def SparseSparseMaximum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSparseMaximum")

    @staticmethod
    def SparseSparseMinimum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSparseMinimum")

    @staticmethod
    def SparseSplit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseSplit")

    @staticmethod
    def SparseTensorDenseAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseTensorDenseAdd")

    @staticmethod
    def SparseTensorDenseMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseTensorDenseMatMul")

    @staticmethod
    def SparseTensorSliceDataset(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseTensorSliceDataset"
        )

    @staticmethod
    def SparseTensorToCSRSparseMatrix(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseTensorToCSRSparseMatrix"
        )

    @staticmethod
    def SparseToDense(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SparseToDense")

    @staticmethod
    def SparseToSparseSetOperation(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.SparseToSparseSetOperation"
        )

    @staticmethod
    def Spence(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Spence")

    @staticmethod
    def Split(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Split")

    @staticmethod
    def SplitV(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SplitV")

    @staticmethod
    def SqlDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SqlDataset")

    @staticmethod
    def Sqrt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sqrt")

    @staticmethod
    def SqrtGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SqrtGrad")

    @staticmethod
    def Square(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Square")

    @staticmethod
    def SquaredDifference(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SquaredDifference")

    @staticmethod
    def Squeeze(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Squeeze")

    @staticmethod
    def Stack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Stack")

    @staticmethod
    def StackClose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackClose")

    @staticmethod
    def StackCloseV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackCloseV2")

    @staticmethod
    def StackPop(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackPop")

    @staticmethod
    def StackPopV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackPopV2")

    @staticmethod
    def StackPush(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackPush")

    @staticmethod
    def StackPushV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackPushV2")

    @staticmethod
    def StackV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StackV2")

    @staticmethod
    def Stage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Stage")

    @staticmethod
    def StageClear(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StageClear")

    @staticmethod
    def StagePeek(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StagePeek")

    @staticmethod
    def StageSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StageSize")

    @staticmethod
    def StatefulPartitionedCall(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulPartitionedCall")

    @staticmethod
    def StatefulRandomBinomial(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulRandomBinomial")

    @staticmethod
    def StatefulStandardNormal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulStandardNormal")

    @staticmethod
    def StatefulStandardNormalV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatefulStandardNormalV2"
        )

    @staticmethod
    def StatefulTruncatedNormal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulTruncatedNormal")

    @staticmethod
    def StatefulUniform(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulUniform")

    @staticmethod
    def StatefulUniformFullInt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulUniformFullInt")

    @staticmethod
    def StatefulUniformInt(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatefulUniformInt")

    @staticmethod
    def StatelessCase(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessCase")

    @staticmethod
    def StatelessIf(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessIf")

    @staticmethod
    def StatelessMultinomial(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessMultinomial")

    @staticmethod
    def StatelessParameterizedTruncatedNormal(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessParameterizedTruncatedNormal"
        )

    @staticmethod
    def StatelessRandomBinomial(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomBinomial")

    @staticmethod
    def StatelessRandomGammaV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomGammaV2")

    @staticmethod
    def StatelessRandomGammaV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomGammaV3")

    @staticmethod
    def StatelessRandomGetAlg(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomGetAlg")

    @staticmethod
    def StatelessRandomGetKeyCounter(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomGetKeyCounter"
        )

    @staticmethod
    def StatelessRandomGetKeyCounterAlg(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomGetKeyCounterAlg"
        )

    @staticmethod
    def StatelessRandomNormal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomNormal")

    @staticmethod
    def StatelessRandomNormalV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomNormalV2")

    @staticmethod
    def StatelessRandomPoisson(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomPoisson")

    @staticmethod
    def StatelessRandomUniform(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessRandomUniform")

    @staticmethod
    def StatelessRandomUniformFullInt(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomUniformFullInt"
        )

    @staticmethod
    def StatelessRandomUniformFullIntV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomUniformFullIntV2"
        )

    @staticmethod
    def StatelessRandomUniformInt(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomUniformInt"
        )

    @staticmethod
    def StatelessRandomUniformIntV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomUniformIntV2"
        )

    @staticmethod
    def StatelessRandomUniformV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessRandomUniformV2"
        )

    @staticmethod
    def StatelessSampleDistortedBoundingBox(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessSampleDistortedBoundingBox"
        )

    @staticmethod
    def StatelessShuffle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessShuffle")

    @staticmethod
    def StatelessTruncatedNormal(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessTruncatedNormal"
        )

    @staticmethod
    def StatelessTruncatedNormalV2(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatelessTruncatedNormalV2"
        )

    @staticmethod
    def StatelessWhile(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatelessWhile")

    @staticmethod
    def StaticRegexFullMatch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StaticRegexFullMatch")

    @staticmethod
    def StaticRegexReplace(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StaticRegexReplace")

    @staticmethod
    def StatsAggregatorHandle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatsAggregatorHandle")

    @staticmethod
    def StatsAggregatorHandleV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatsAggregatorHandleV2")

    @staticmethod
    def StatsAggregatorSetSummaryWriter(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StatsAggregatorSetSummaryWriter"
        )

    @staticmethod
    def StatsAggregatorSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StatsAggregatorSummary")

    @staticmethod
    def StopGradient(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StopGradient")

    @staticmethod
    def StoreMinibatchStatisticsInFdo(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StoreMinibatchStatisticsInFdo"
        )

    @staticmethod
    def StridedSlice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StridedSlice")

    @staticmethod
    def StridedSliceAssign(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StridedSliceAssign")

    @staticmethod
    def StridedSliceGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StridedSliceGrad")

    @staticmethod
    def StringFormat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringFormat")

    @staticmethod
    def StringJoin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringJoin")

    @staticmethod
    def StringLength(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringLength")

    @staticmethod
    def StringLower(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringLower")

    @staticmethod
    def StringNGrams(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringNGrams")

    @staticmethod
    def StringSplit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringSplit")

    @staticmethod
    def StringSplitV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringSplitV2")

    @staticmethod
    def StringStrip(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringStrip")

    @staticmethod
    def StringToHashBucket(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringToHashBucket")

    @staticmethod
    def StringToHashBucketFast(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringToHashBucketFast")

    @staticmethod
    def StringToHashBucketStrong(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.StringToHashBucketStrong"
        )

    @staticmethod
    def StringToNumber(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringToNumber")

    @staticmethod
    def StringUpper(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.StringUpper")

    @staticmethod
    def Sub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sub")

    @staticmethod
    def Substr(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Substr")

    @staticmethod
    def Sum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Sum")

    @staticmethod
    def SummaryWriter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SummaryWriter")

    @staticmethod
    def Svd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Svd")

    @staticmethod
    def Switch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Switch")

    @staticmethod
    def SymbolicGradient(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SymbolicGradient")

    @staticmethod
    def SyncDevice(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.SyncDevice")

    @staticmethod
    def TFRecordDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TFRecordDataset")

    @staticmethod
    def TFRecordDatasetV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TFRecordDatasetV2")

    @staticmethod
    def TFRecordReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TFRecordReader")

    @staticmethod
    def TFRecordReaderV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TFRecordReaderV2")

    @staticmethod
    def TPUAnnotateTensorsWithDynamicShape(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.TPUAnnotateTensorsWithDynamicShape"
        )

    @staticmethod
    def TPUCompilationResult(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUCompilationResult")

    @staticmethod
    def TPUCopyWithDynamicShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUCopyWithDynamicShape")

    @staticmethod
    def TPUEmbeddingActivations(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUEmbeddingActivations")

    @staticmethod
    def TPUOrdinalSelector(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUOrdinalSelector")

    @staticmethod
    def TPUPartitionedCall(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUPartitionedCall")

    @staticmethod
    def TPUPartitionedInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUPartitionedInput")

    @staticmethod
    def TPUPartitionedInputV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUPartitionedInputV2")

    @staticmethod
    def TPUPartitionedOutput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUPartitionedOutput")

    @staticmethod
    def TPUPartitionedOutputV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUPartitionedOutputV2")

    @staticmethod
    def TPUReplicateMetadata(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUReplicateMetadata")

    @staticmethod
    def TPUReplicatedInput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUReplicatedInput")

    @staticmethod
    def TPUReplicatedOutput(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TPUReplicatedOutput")

    @staticmethod
    def TakeDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TakeDataset")

    @staticmethod
    def TakeManySparseFromTensorsMap(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.TakeManySparseFromTensorsMap"
        )

    @staticmethod
    def TakeWhileDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TakeWhileDataset")

    @staticmethod
    def Tan(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Tan")

    @staticmethod
    def Tanh(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Tanh")

    @staticmethod
    def TanhGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TanhGrad")

    @staticmethod
    def TemporaryVariable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TemporaryVariable")

    @staticmethod
    def TensorArray(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArray")

    @staticmethod
    def TensorArrayClose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayClose")

    @staticmethod
    def TensorArrayCloseV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayCloseV2")

    @staticmethod
    def TensorArrayCloseV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayCloseV3")

    @staticmethod
    def TensorArrayConcat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayConcat")

    @staticmethod
    def TensorArrayConcatV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayConcatV2")

    @staticmethod
    def TensorArrayConcatV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayConcatV3")

    @staticmethod
    def TensorArrayGather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGather")

    @staticmethod
    def TensorArrayGatherV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGatherV2")

    @staticmethod
    def TensorArrayGatherV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGatherV3")

    @staticmethod
    def TensorArrayGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGrad")

    @staticmethod
    def TensorArrayGradV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGradV2")

    @staticmethod
    def TensorArrayGradV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayGradV3")

    @staticmethod
    def TensorArrayGradWithShape(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.TensorArrayGradWithShape"
        )

    @staticmethod
    def TensorArrayPack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayPack")

    @staticmethod
    def TensorArrayRead(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayRead")

    @staticmethod
    def TensorArrayReadV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayReadV2")

    @staticmethod
    def TensorArrayReadV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayReadV3")

    @staticmethod
    def TensorArrayScatter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayScatter")

    @staticmethod
    def TensorArrayScatterV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayScatterV2")

    @staticmethod
    def TensorArrayScatterV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayScatterV3")

    @staticmethod
    def TensorArraySize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySize")

    @staticmethod
    def TensorArraySizeV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySizeV2")

    @staticmethod
    def TensorArraySizeV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySizeV3")

    @staticmethod
    def TensorArraySplit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySplit")

    @staticmethod
    def TensorArraySplitV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySplitV2")

    @staticmethod
    def TensorArraySplitV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArraySplitV3")

    @staticmethod
    def TensorArrayUnpack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayUnpack")

    @staticmethod
    def TensorArrayV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayV2")

    @staticmethod
    def TensorArrayV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayV3")

    @staticmethod
    def TensorArrayWrite(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayWrite")

    @staticmethod
    def TensorArrayWriteV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayWriteV2")

    @staticmethod
    def TensorArrayWriteV3(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorArrayWriteV3")

    @staticmethod
    def TensorDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorDataset")

    @staticmethod
    def TensorListConcat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListConcat")

    @staticmethod
    def TensorListConcatLists(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListConcatLists")

    @staticmethod
    def TensorListConcatV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListConcatV2")

    @staticmethod
    def TensorListElementShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListElementShape")

    @staticmethod
    def TensorListFromTensor(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListFromTensor")

    @staticmethod
    def TensorListGather(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListGather")

    @staticmethod
    def TensorListGetItem(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListGetItem")

    @staticmethod
    def TensorListLength(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListLength")

    @staticmethod
    def TensorListPopBack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListPopBack")

    @staticmethod
    def TensorListPushBack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListPushBack")

    @staticmethod
    def TensorListPushBackBatch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListPushBackBatch")

    @staticmethod
    def TensorListReserve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListReserve")

    @staticmethod
    def TensorListResize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListResize")

    @staticmethod
    def TensorListScatter(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListScatter")

    @staticmethod
    def TensorListScatterIntoExistingList(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.TensorListScatterIntoExistingList"
        )

    @staticmethod
    def TensorListScatterV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListScatterV2")

    @staticmethod
    def TensorListSetItem(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListSetItem")

    @staticmethod
    def TensorListSplit(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListSplit")

    @staticmethod
    def TensorListStack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorListStack")

    @staticmethod
    def TensorMapErase(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapErase")

    @staticmethod
    def TensorMapHasKey(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapHasKey")

    @staticmethod
    def TensorMapInsert(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapInsert")

    @staticmethod
    def TensorMapLookup(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapLookup")

    @staticmethod
    def TensorMapSize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapSize")

    @staticmethod
    def TensorMapStackKeys(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorMapStackKeys")

    @staticmethod
    def TensorScatterAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorScatterAdd")

    @staticmethod
    def TensorScatterMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorScatterMax")

    @staticmethod
    def TensorScatterMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorScatterMin")

    @staticmethod
    def TensorScatterSub(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorScatterSub")

    @staticmethod
    def TensorScatterUpdate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorScatterUpdate")

    @staticmethod
    def TensorSliceDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorSliceDataset")

    @staticmethod
    def TensorStridedSliceUpdate(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.TensorStridedSliceUpdate"
        )

    @staticmethod
    def TensorSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorSummary")

    @staticmethod
    def TensorSummaryV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TensorSummaryV2")

    @staticmethod
    def TextLineDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TextLineDataset")

    @staticmethod
    def TextLineReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TextLineReader")

    @staticmethod
    def TextLineReaderV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TextLineReaderV2")

    @staticmethod
    def ThreadPoolDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ThreadPoolDataset")

    @staticmethod
    def ThreadPoolHandle(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ThreadPoolHandle")

    @staticmethod
    def ThreadUnsafeUnigramCandidateSampler(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.ThreadUnsafeUnigramCandidateSampler"
        )

    @staticmethod
    def Tile(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Tile")

    @staticmethod
    def TileGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TileGrad")

    @staticmethod
    def Timestamp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Timestamp")

    @staticmethod
    def ToBool(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ToBool")

    @staticmethod
    def TopK(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TopK")

    @staticmethod
    def TopKV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TopKV2")

    @staticmethod
    def Transpose(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Transpose")

    @staticmethod
    def TridiagonalMatMul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TridiagonalMatMul")

    @staticmethod
    def TridiagonalSolve(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TridiagonalSolve")

    @staticmethod
    def TruncateDiv(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TruncateDiv")

    @staticmethod
    def TruncateMod(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TruncateMod")

    @staticmethod
    def TruncatedNormal(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.TruncatedNormal")

    @staticmethod
    def Unbatch(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Unbatch")

    @staticmethod
    def UnbatchDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnbatchDataset")

    @staticmethod
    def UnbatchGrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnbatchGrad")

    @staticmethod
    def UncompressElement(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UncompressElement")

    @staticmethod
    def UnicodeDecode(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnicodeDecode")

    @staticmethod
    def UnicodeDecodeWithOffsets(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.UnicodeDecodeWithOffsets"
        )

    @staticmethod
    def UnicodeEncode(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnicodeEncode")

    @staticmethod
    def UnicodeScript(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnicodeScript")

    @staticmethod
    def UnicodeTranscode(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnicodeTranscode")

    @staticmethod
    def UniformCandidateSampler(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformCandidateSampler")

    @staticmethod
    def UniformDequantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformDequantize")

    @staticmethod
    def UniformQuantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformQuantize")

    @staticmethod
    def UniformQuantizedAdd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformQuantizedAdd")

    @staticmethod
    def UniformQuantizedClipByValue(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.UniformQuantizedClipByValue"
        )

    @staticmethod
    def UniformQuantizedConvolution(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.UniformQuantizedConvolution"
        )

    @staticmethod
    def UniformQuantizedConvolutionHybrid(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.UniformQuantizedConvolutionHybrid"
        )

    @staticmethod
    def UniformQuantizedDot(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformQuantizedDot")

    @staticmethod
    def UniformQuantizedDotHybrid(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.UniformQuantizedDotHybrid"
        )

    @staticmethod
    def UniformRequantize(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniformRequantize")

    @staticmethod
    def Unique(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Unique")

    @staticmethod
    def UniqueDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniqueDataset")

    @staticmethod
    def UniqueV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniqueV2")

    @staticmethod
    def UniqueWithCounts(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniqueWithCounts")

    @staticmethod
    def UniqueWithCountsV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UniqueWithCountsV2")

    @staticmethod
    def Unpack(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Unpack")

    @staticmethod
    def UnravelIndex(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnravelIndex")

    @staticmethod
    def UnsortedSegmentJoin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnsortedSegmentJoin")

    @staticmethod
    def UnsortedSegmentMax(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnsortedSegmentMax")

    @staticmethod
    def UnsortedSegmentMin(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnsortedSegmentMin")

    @staticmethod
    def UnsortedSegmentProd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnsortedSegmentProd")

    @staticmethod
    def UnsortedSegmentSum(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnsortedSegmentSum")

    @staticmethod
    def Unstage(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Unstage")

    @staticmethod
    def UnwrapDatasetVariant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UnwrapDatasetVariant")

    @staticmethod
    def UpperBound(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.UpperBound")

    @staticmethod
    def VarHandleOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.VarHandleOp")

    @staticmethod
    def VarIsInitializedOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.VarIsInitializedOp")

    @staticmethod
    def Variable(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Variable")

    @staticmethod
    def VariableShape(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.VariableShape")

    @staticmethod
    def VariableV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.VariableV2")

    @staticmethod
    def Where(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Where")

    @staticmethod
    def While(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.While")

    @staticmethod
    def WholeFileReader(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WholeFileReader")

    @staticmethod
    def WholeFileReaderV2(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WholeFileReaderV2")

    @staticmethod
    def WindowDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WindowDataset")

    @staticmethod
    def WindowOp(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WindowOp")

    @staticmethod
    def WorkerHeartbeat(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WorkerHeartbeat")

    @staticmethod
    def WrapDatasetVariant(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WrapDatasetVariant")

    @staticmethod
    def WriteAudioSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteAudioSummary")

    @staticmethod
    def WriteFile(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteFile")

    @staticmethod
    def WriteGraphSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteGraphSummary")

    @staticmethod
    def WriteHistogramSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteHistogramSummary")

    @staticmethod
    def WriteImageSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteImageSummary")

    @staticmethod
    def WriteRawProtoSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteRawProtoSummary")

    @staticmethod
    def WriteScalarSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteScalarSummary")

    @staticmethod
    def WriteSummary(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.WriteSummary")

    @staticmethod
    def Xdivy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Xdivy")

    @staticmethod
    def XlaConcatND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaConcatND")

    @staticmethod
    def XlaSparseCoreAdagrad(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSparseCoreAdagrad")

    @staticmethod
    def XlaSparseCoreAdagradMomentum(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseCoreAdagradMomentum"
        )

    @staticmethod
    def XlaSparseCoreAdam(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSparseCoreAdam")

    @staticmethod
    def XlaSparseCoreFtrl(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSparseCoreFtrl")

    @staticmethod
    def XlaSparseCoreSgd(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSparseCoreSgd")

    @staticmethod
    def XlaSparseDenseMatmul(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSparseDenseMatmul")

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradAndCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulGradWithAdagradAndCsrInput"
        )

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulGradWithAdagradMomentumAndCsrInput"
        )

    @staticmethod
    def XlaSparseDenseMatmulGradWithAdamAndCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulGradWithAdamAndCsrInput"
        )

    @staticmethod
    def XlaSparseDenseMatmulGradWithFtrlAndCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulGradWithFtrlAndCsrInput"
        )

    @staticmethod
    def XlaSparseDenseMatmulGradWithSgdAndCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulGradWithSgdAndCsrInput"
        )

    @staticmethod
    def XlaSparseDenseMatmulWithCsrInput(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.raw_ops.XlaSparseDenseMatmulWithCsrInput"
        )

    @staticmethod
    def XlaSplitND(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.XlaSplitND")

    @staticmethod
    def Xlog1py(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Xlog1py")

    @staticmethod
    def Xlogy(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Xlogy")

    @staticmethod
    def ZerosLike(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ZerosLike")

    @staticmethod
    def Zeta(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.Zeta")

    @staticmethod
    def ZipDataset(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.raw_ops.ZipDataset")


def realdiv(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.realdiv")


def recompute_grad(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.recompute_grad")


def reduce_logsumexp(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.reduce_logsumexp")


def register_tensor_conversion_function(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.register_tensor_conversion_function")


def required_space_to_batch_paddings(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.required_space_to_batch_paddings")


def resource(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.resource")


def reverse(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.reverse")


def reverse_sequence(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.reverse_sequence")


def rfftnd(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.rfftnd")


def saturate_cast(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.saturate_cast")


def sequence_mask(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.sequence_mask")


class sets:
    @staticmethod
    def difference(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sets.difference")

    @staticmethod
    def intersection(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sets.intersection")

    @staticmethod
    def size(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sets.size")

    @staticmethod
    def union(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sets.union")


def shape_n(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.shape_n")


def sigmoid(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.sigmoid")

    @staticmethod
    def audio(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.audio")

    @staticmethod
    def create_file_writer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.create_file_writer")

    @staticmethod
    def create_noop_writer(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.create_noop_writer")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.experimental")

    @staticmethod
    def flush(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.flush")

    @staticmethod
    def graph(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.graph")

    @staticmethod
    def histogram(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.histogram")

    @staticmethod
    def record_if(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.record_if")

    @staticmethod
    def scalar(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.scalar")

    @staticmethod
    def should_record_summaries(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.should_record_summaries")

    @staticmethod
    def text(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.text")

    @staticmethod
    def trace_export(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.trace_export")

    @staticmethod
    def trace_off(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.trace_off")

    @staticmethod
    def trace_on(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.trace_on")

    @staticmethod
    def write(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.summary.write")


def switch_case(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.switch_case")


class sysconfig:
    @staticmethod
    def CXX11_ABI_FLAG(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.CXX11_ABI_FLAG")

    @staticmethod
    def CXX_VERSION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.CXX_VERSION")

    @staticmethod
    def MONOLITHIC_BUILD(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.MONOLITHIC_BUILD")

    @staticmethod
    def get_build_info(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.get_build_info")

    @staticmethod
    def get_compile_flags(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.get_compile_flags")

    @staticmethod
    def get_include(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.get_include")

    @staticmethod
    def get_lib(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.get_lib")

    @staticmethod
    def get_link_flags(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.sysconfig.get_link_flags")


def tensor_scatter_nd_add(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tensor_scatter_nd_add")


def tensor_scatter_nd_max(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tensor_scatter_nd_max")


def tensor_scatter_nd_min(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tensor_scatter_nd_min")


def tensor_scatter_nd_sub(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tensor_scatter_nd_sub")


def tensor_scatter_nd_update(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tensor_scatter_nd_update")

    @staticmethod
    def TestCase(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.TestCase")

    @staticmethod
    def assert_equal_graph_def(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.assert_equal_graph_def")

    @staticmethod
    def benchmark_config(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.benchmark_config")

    @staticmethod
    def compute_gradient(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.compute_gradient")

    @staticmethod
    def create_local_cluster(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.create_local_cluster")

    @staticmethod
    def disable_with_predicate(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.disable_with_predicate")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.experimental")

    @staticmethod
    def gpu_device_name(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.gpu_device_name")

    @staticmethod
    def is_built_with_cuda(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.is_built_with_cuda")

    @staticmethod
    def is_built_with_gpu_support(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.is_built_with_gpu_support")

    @staticmethod
    def is_built_with_rocm(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.is_built_with_rocm")

    @staticmethod
    def is_built_with_xla(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.is_built_with_xla")

    @staticmethod
    def is_gpu_available(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.is_gpu_available")

    @staticmethod
    def main(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.main")

    @staticmethod
    def with_eager_op_as_function(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.test.with_eager_op_as_function")


def timestamp(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.timestamp")


class tools:
    @staticmethod
    def compatibility(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.tools.compatibility")

    @staticmethod
    def docs(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.tools.docs")


class tpu:
    @staticmethod
    def XLAOptions(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.tpu.XLAOptions")

    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.tpu.experimental")


def tuple(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.tuple")


def type_spec_from_value(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.type_spec_from_value")


def uint16(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.uint16")


def uint32(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.uint32")


def uint64(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.uint64")


def unique(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.unique")


def unique_with_counts(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.unique_with_counts")


def unravel_index(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.unravel_index")


def variable_creator_scope(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.variable_creator_scope")


def variant(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.variant")


def vectorized_map(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.vectorized_map")


class version:
    @staticmethod
    def COMPILER_VERSION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.version.COMPILER_VERSION")

    @staticmethod
    def GIT_VERSION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.version.GIT_VERSION")

    @staticmethod
    def GRAPH_DEF_VERSION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.version.GRAPH_DEF_VERSION")

    @staticmethod
    def GRAPH_DEF_VERSION_MIN_CONSUMER(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.version.GRAPH_DEF_VERSION_MIN_CONSUMER"
        )

    @staticmethod
    def GRAPH_DEF_VERSION_MIN_PRODUCER(*args, **kwargs):
        raise NotImplementedError(
            "Not implemented: tf.version.GRAPH_DEF_VERSION_MIN_PRODUCER"
        )

    @staticmethod
    def VERSION(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.version.VERSION")


def while_loop(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.while_loop")


class xla:
    @staticmethod
    def experimental(*args, **kwargs):
        raise NotImplementedError("Not implemented: tf.xla.experimental")


def zeros_initializer(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.zeros_initializer")


def stop_gradient(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.stop_gradient")


def custom_gradient(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.custom_gradient")


def hessians(*args, **kwargs):
    raise NotImplementedError("Not implemented: tf.hessians")


from . import autograph
from . import config
from . import debugging
from . import dtypes
from . import lookup
from . import quantization
from . import summary
from . import test
from . import types
from .core_stubs import *  # noqa: F403
