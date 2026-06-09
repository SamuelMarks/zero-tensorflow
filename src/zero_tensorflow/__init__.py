"""zero_tensorflow API."""

import functools
from typing import Any, Optional
import numpy as np

from ml_switcheroo_ir import LogicalNode

from . import data
from . import nn
from . import keras

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


class Tensor:
    """Dual-state Tensor Primitive (Eager NumPy + Traced LogicalNode)."""

    def __init__(self, value: Any, _traced_node: Optional[LogicalNode] = None):
        """__init__ docstring."""
        if _traced_node is not None:
            self._traced_node = _traced_node
            self._value = None
            self._shape = _traced_node.shape_metadata
        else:
            if isinstance(value, Tensor):
                value = value.numpy()
            self._value = np.array(value)
            self._traced_node = None
            self._shape = self._value.shape

    @property
    def shape(self):
        """shape docstring."""
        return self._shape

    def numpy(self):
        """numpy docstring."""
        if self._traced_node is not None:
            raise ValueError("Cannot call numpy() on a traced Tensor")
        return self._value

    def __add__(self, other):
        """__add__ docstring."""
        return math.add(self, other)

    def __sub__(self, other):
        """__sub__ docstring."""
        return math.subtract(self, other)

    def __mul__(self, other):
        """__mul__ docstring."""
        return math.multiply(self, other)

    def __truediv__(self, other):
        """__truediv__ docstring."""
        return math.divide(self, other)

    def __radd__(self, other):
        """__radd__ docstring."""
        return math.add(other, self)

    def __rsub__(self, other):
        """__rsub__ docstring."""
        return math.subtract(other, self)

    def __rmul__(self, other):
        """__rmul__ docstring."""
        return math.multiply(other, self)

    def __rtruediv__(self, other):
        """__rtruediv__ docstring."""
        return math.divide(other, self)

    def __eq__(self, other):
        """__eq__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"eq_{len(ctx.nodes)}", op_type="Equal")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value == other)

    def __ne__(self, other):
        """__ne__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"ne_{len(ctx.nodes)}", op_type="NotEqual")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value != other)

    def __lt__(self, other):
        """__lt__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"lt_{len(ctx.nodes)}", op_type="Less")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value < other)

    def __le__(self, other):
        """__le__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"le_{len(ctx.nodes)}", op_type="LessOrEqual")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value <= other)

    def __gt__(self, other):
        """__gt__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"gt_{len(ctx.nodes)}", op_type="Greater")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value > other)

    def __ge__(self, other):
        """__ge__ docstring."""
        if self._traced_node is not None:
            ctx = _TracingContext.get()
            if ctx is not None:
                node = LogicalNode(id=f"ge_{len(ctx.nodes)}", op_type="GreaterOrEqual")
                ctx.nodes.append(node)
                return Tensor(None, _traced_node=node)
        return Tensor(self._value >= other)

    def __bool__(self):
        """__bool__ docstring."""
        if self._traced_node is not None:
            raise TypeError(
                "Using a `Tensor` as a Python `bool` is not allowed in Graph execution. Use `tf.cond` instead."
            )
        return bool(self._value)

    def __nonzero__(self):
        """__nonzero__ docstring."""
        return self.__bool__()


class Variable:
    """Variable docstring."""

    def __init__(self, initial_value, trainable=True):
        """__init__ docstring."""
        self._tensor = (
            Tensor(initial_value)
            if not isinstance(initial_value, Tensor)
            else initial_value
        )
        self.trainable = trainable

    @property
    def value(self):
        """value docstring."""
        return (
            self._tensor.numpy() if self._tensor._traced_node is None else self._tensor
        )

    def assign(self, value):
        """assign docstring."""
        self._tensor = Tensor(value) if not isinstance(value, Tensor) else value
        return self

    def assign_add(self, delta):
        """assign_add docstring."""
        self._tensor = self._tensor + delta
        return self

    def assign_sub(self, delta):
        """assign_sub docstring."""
        self._tensor = self._tensor - delta
        return self


class _TracingContext:
    """_TracingContext docstring."""

    _current_context = None

    def __init__(self):
        """__init__ docstring."""
        self.nodes = []

    @classmethod
    def enter(cls):
        """enter docstring."""
        ctx = cls()
        cls._current_context = ctx
        return ctx

    @classmethod
    def exit(cls):
        """exit docstring."""
        cls._current_context = None

    @classmethod
    def get(cls):
        """get docstring."""
        return cls._current_context


def function(func):
    """function docstring."""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        """wrapper docstring."""
        ctx = _TracingContext.enter()
        try:
            # Emit input nodes
            traced_args = []
            for i, arg in enumerate(args):
                if isinstance(arg, (Tensor, np.ndarray, list, float, int)):
                    node = LogicalNode(id=f"input_{i}", op_type="Input")
                    ctx.nodes.append(node)
                    traced_args.append(Tensor(None, _traced_node=node))
                else:
                    traced_args.append(arg)

            result = func(*traced_args, **kwargs)
            return result
        finally:
            _TracingContext.exit()

    return wrapper


class GradientTape:
    """GradientTape docstring."""

    def __init__(self, persistent=False):
        """__init__ docstring."""
        self.persistent = persistent
        self.watched = []
        self._forward_ops = []

    def __enter__(self):
        """__enter__ docstring."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """__exit__ docstring."""
        pass

    def watch(self, tensor):
        """watch docstring."""
        self.watched.append(tensor)

    def gradient(self, target, sources):
        """gradient docstring."""
        if not isinstance(sources, list):
            sources = [sources]
        return [Tensor(1.0) for _ in sources]


class math:
    """math docstring."""

    @staticmethod
    def add(x, y):
        """add docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"add_{len(ctx.nodes)}", op_type="Add")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)

        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        y_val = y.numpy() if isinstance(y, Tensor) else np.array(y)
        return Tensor(x_val + y_val)

    @staticmethod
    def subtract(x, y):
        """subtract docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"sub_{len(ctx.nodes)}", op_type="Sub")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)

        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        y_val = y.numpy() if isinstance(y, Tensor) else np.array(y)
        return Tensor(x_val - y_val)

    @staticmethod
    def multiply(x, y):
        """multiply docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"mul_{len(ctx.nodes)}", op_type="Mul")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)

        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        y_val = y.numpy() if isinstance(y, Tensor) else np.array(y)
        return Tensor(x_val * y_val)

    @staticmethod
    def divide(x, y):
        """divide docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"div_{len(ctx.nodes)}", op_type="Div")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)

        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        y_val = y.numpy() if isinstance(y, Tensor) else np.array(y)
        return Tensor(x_val / y_val)

    @staticmethod
    def exp(x):
        """exp docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"exp_{len(ctx.nodes)}", op_type="Exp")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.exp(x_val))

    @staticmethod
    def log(x):
        """log docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"log_{len(ctx.nodes)}", op_type="Log")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.log(x_val))

    @staticmethod
    def pow(x, y):
        """pow docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"pow_{len(ctx.nodes)}", op_type="Pow")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        y_val = y.numpy() if isinstance(y, Tensor) else np.array(y)
        return Tensor(np.power(x_val, y_val))

    @staticmethod
    def sqrt(x):
        """sqrt docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"sqrt_{len(ctx.nodes)}", op_type="Sqrt")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.sqrt(x_val))

    @staticmethod
    def reduce_sum(x, axis=None, keepdims=False):
        """reduce_sum docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"reduce_sum_{len(ctx.nodes)}", op_type="ReduceSum")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.sum(x_val, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_mean(x, axis=None, keepdims=False):
        """reduce_mean docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"reduce_mean_{len(ctx.nodes)}", op_type="ReduceMean")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.mean(x_val, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_max(x, axis=None, keepdims=False):
        """reduce_max docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"reduce_max_{len(ctx.nodes)}", op_type="ReduceMax")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.max(x_val, axis=axis, keepdims=keepdims))

    @staticmethod
    def reduce_min(x, axis=None, keepdims=False):
        """reduce_min docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"reduce_min_{len(ctx.nodes)}", op_type="ReduceMin")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        x_val = x.numpy() if isinstance(x, Tensor) else np.array(x)
        return Tensor(np.min(x_val, axis=axis, keepdims=keepdims))

    @staticmethod
    def matmul(a, b):
        """matmul docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"matmul_{len(ctx.nodes)}", op_type="MatMul")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        a_val = a.numpy() if isinstance(a, Tensor) else np.array(a)
        b_val = b.numpy() if isinstance(b, Tensor) else np.array(b)
        return Tensor(np.matmul(a_val, b_val))

    @staticmethod
    def tensordot(a, b, axes):
        """tensordot docstring."""
        ctx = _TracingContext.get()
        if ctx is not None:
            node = LogicalNode(id=f"tensordot_{len(ctx.nodes)}", op_type="TensorDot")
            ctx.nodes.append(node)
            return Tensor(None, _traced_node=node)
        a_val = a.numpy() if isinstance(a, Tensor) else np.array(a)
        b_val = b.numpy() if isinstance(b, Tensor) else np.array(b)
        return Tensor(np.tensordot(a_val, b_val, axes=axes))
