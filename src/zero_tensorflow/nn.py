"""TensorFlow nn module."""

from typing import Optional, Union, Any, TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    import zero_tensorflow

from ml_switcheroo_ir import LogicalNode

__all__ = ["elu", "leaky_relu", "relu", "selu", "sigmoid", "softmax", "tanh"]


def elu(
    features: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes the exponential linear function."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(id=f"elu_{len(ctx.nodes)}", op_type="Elu")
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if features is None:
        raise ValueError("features cannot be None")
    x = features.numpy() if isinstance(features, zero_tensorflow.Tensor) else features
    return zero_tensorflow.Tensor(np.where(x > 0, x, np.exp(x) - 1))  # type: ignore


def leaky_relu(
    features: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    alpha: float = 0.2,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Compute the Leaky ReLU activation function."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(
            id=f"leaky_relu_{len(ctx.nodes)}",
            op_type="LeakyRelu",
            attributes={"alpha": alpha},
        )
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if features is None:
        raise ValueError("features cannot be None")
    x = features.numpy() if isinstance(features, zero_tensorflow.Tensor) else features
    return zero_tensorflow.Tensor(np.where(x > 0, x, x * alpha))  # type: ignore


def relu(
    features: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes rectified linear: max(features, 0)."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(id=f"relu_{len(ctx.nodes)}", op_type="Relu")
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if features is None:
        raise ValueError("features cannot be None")
    x = features.numpy() if isinstance(features, zero_tensorflow.Tensor) else features
    return zero_tensorflow.Tensor(np.maximum(x, 0))


def selu(
    features: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes scaled exponential linear: scale * alpha * (exp(features) - 1)."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(id=f"selu_{len(ctx.nodes)}", op_type="Selu")
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if features is None:
        raise ValueError("features cannot be None")
    x = features.numpy() if isinstance(features, zero_tensorflow.Tensor) else features
    scale = 1.0507009873554804934193349852946
    alpha = 1.6732632423543772848170429916717
    return zero_tensorflow.Tensor(scale * np.where(x > 0, x, alpha * (np.exp(x) - 1)))  # type: ignore


def sigmoid(
    x: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes sigmoid of x element-wise."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(id=f"sigmoid_{len(ctx.nodes)}", op_type="Sigmoid")
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if x is None:
        raise ValueError("x cannot be None")
    x_val = x.numpy() if isinstance(x, zero_tensorflow.Tensor) else x
    return zero_tensorflow.Tensor(1 / (1 + np.exp(-x_val)))


def softmax(
    logits: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    axis: Optional[int] = None,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes softmax activations."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(
            id=f"softmax_{len(ctx.nodes)}",
            op_type="Softmax",
            attributes={"axis": axis if axis is not None else -1},
        )
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if logits is None:
        raise ValueError("logits cannot be None")
    if axis is None:
        axis = -1
    x_val = logits.numpy() if isinstance(logits, zero_tensorflow.Tensor) else logits
    e_x = np.exp(x_val - np.max(x_val, axis=axis, keepdims=True))
    return zero_tensorflow.Tensor(e_x / np.sum(e_x, axis=axis, keepdims=True))


def tanh(
    x: Union["zero_tensorflow.Tensor", NDArray[np.float32], NDArray[np.float64]],
    *args: Any,
    name: Optional[str] = None,
    **kwargs: Any,
) -> "zero_tensorflow.Tensor":
    """Computes hyperbolic tangent of x element-wise."""
    import zero_tensorflow

    ctx = zero_tensorflow._TracingContext.get()
    if ctx is not None:
        node = LogicalNode(id=f"tanh_{len(ctx.nodes)}", op_type="Tanh")
        ctx.nodes.append(node)
        return zero_tensorflow.Tensor(None, _traced_node=node)

    if x is None:
        raise ValueError("x cannot be None")
    x_val = x.numpy() if isinstance(x, zero_tensorflow.Tensor) else x
    return zero_tensorflow.Tensor(np.tanh(x_val))
