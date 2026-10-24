"""TensorFlow `nn` (neural network) operations module."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from zero_keras import activations as k_activations
from zero_keras import ops as k_ops

__all__ = [
    "avg_pool3d",
    "batch_norm_with_global_normalization",
    "conv1d",
    "conv2d_transpose",
    "conv3d_transpose",
    "crelu",
    "ctc_loss",
    "depthwise_conv2d",
    "elu",
    "fractional_max_pool",
    "gelu",
    "leaky_relu",
    "local_response_normalization",
    "max_pool3d",
    "nce_loss",
    "relu",
    "sampled_softmax_loss",
    "selu",
    "separable_conv2d",
    "sigmoid",
    "silu",
    "softmax",
    "swish",
    "tanh",
]


@runtime_checkable
class _DataWrapperProtocol(Protocol):
    """Protocol for wrapper objects exposing a `.data` tensor attribute."""

    @property
    def data(self) -> msc_core.Tensor:
        """Return underlying tensor data.

        Returns:
            msc_core.Tensor: Wrapped compiler tensor.
        """


def _check_none(features: TensorOperand) -> None:
    """Validate that `features` is not `None`.

    Args:
        features: Input tensor or value to check.

    Raises:
        ValueError: If `features` is `None`.
    """
    if features is None:
        raise ValueError("features cannot be None")


def _unwrap_k(res: msc_core.Tensor | _DataWrapperProtocol) -> msc_core.Tensor:
    """Unwrap a `zero_keras` activation or op output into a compiler `Tensor`.

    Args:
        res: Result returned by `zero_keras`.

    Returns:
        msc_core.Tensor: Unwrapped compiler `Tensor`.
    """
    if isinstance(res, msc_core.Tensor):
        return res
    return res.data


def _wrap_single(x: msc_core.Tensor) -> Tensor:
    """Wrap a single compiler `Tensor` into a `zero_tensorflow.Tensor`.

    Args:
        x: Compiler `Tensor` to wrap.

    Returns:
        Tensor: Wrapped `zero_tensorflow.Tensor`.
    """
    return Tensor(x)


def _unwrap_args(args: tuple[TensorOperand, ...]) -> list[msc_core.Tensor | TensorOperand]:
    """Convert `Tensor` arguments in `args` to compiler `Tensor`s.

    Args:
        args: Positional arguments tuple.

    Returns:
        list[msc_core.Tensor | TensorOperand]: Unwrapped positional arguments.
    """
    return [_to_tensor(a) if isinstance(a, Tensor) else a for a in args]


def avg_pool3d(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Perform 3D average pooling on the input tensor.

    Args:
        *args: Positional pooling arguments (`input`, `ksize`, `strides`, `padding`).
        **kwargs: Keyword pooling arguments.

    Returns:
        Tensor: Pooled 3D output tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.average_pool(*_unwrap_args(args), **kwargs)))


def batch_norm_with_global_normalization(
    t: Tensor | TensorOperand = None,
    m: Tensor | TensorOperand = None,
    v: Tensor | TensorOperand = None,
    beta: Tensor | TensorOperand = None,
    gamma: Tensor | TensorOperand = None,
    variance_epsilon: float = 1e-5,
    scale_after_normalization: bool = True,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Perform batch normalization with global normalization.

    Args:
        t: A 4D input Tensor.
        m: A 1D mean Tensor matching the last dimension of `t`.
        v: A 1D variance Tensor matching the last dimension of `t`.
        beta: A 1D offset Tensor matching the last dimension of `t`.
        gamma: A 1D scale Tensor matching the last dimension of `t`.
        variance_epsilon: A small float number to avoid dividing by 0.
        scale_after_normalization: Whether to multiply by `gamma` after normalization.
        name: Optional name for this operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Normalized tensor.

    Raises:
        NotImplementedError: If required normalization operands are omitted.
    """
    del name, args, kwargs
    if t is None or m is None or v is None or beta is None:
        raise NotImplementedError("batch_norm_with_global_normalization requires t, m, v, and beta")
    t_in = _to_tensor(t)
    m_in = _to_tensor(m, dtype=t_in.dtype)
    v_in = _to_tensor(v, dtype=t_in.dtype)
    beta_in = _to_tensor(beta, dtype=t_in.dtype)
    eps = _to_tensor(variance_epsilon, dtype=t_in.dtype)
    inv_std = _ops.sqrt(_ops.add(v_in, eps))
    normed = _ops.divide(_ops.subtract(t_in, m_in), inv_std)
    if scale_after_normalization and gamma is not None:
        normed = _ops.multiply(normed, _to_tensor(gamma, dtype=t_in.dtype))
    return _wrap_single(_ops.add(normed, beta_in))


def conv1d(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute a 1-D convolution given 3-D input and filter tensors.

    Args:
        *args: Positional convolution arguments (`input`, `filters`, `stride`, `padding`).
        **kwargs: Keyword convolution arguments.

    Returns:
        Tensor: 1-D convolved output tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.conv(*_unwrap_args(args), **kwargs)))


def conv2d_transpose(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute the transpose of 2-D convolution.

    Args:
        *args: Positional arguments (`input`, `filters`, `output_shape`, `strides`, `padding`).
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Transposed 2-D convolution output tensor.

    Raises:
        NotImplementedError: If called without required input arguments.
    """
    if not args and "input" not in kwargs and "inputs" not in kwargs:
        raise NotImplementedError("conv2d_transpose requires input and filters arguments")
    return _wrap_single(_unwrap_k(k_ops.conv_transpose(*_unwrap_args(args), **kwargs)))


def conv3d_transpose(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute the transpose of 3-D convolution.

    Args:
        *args: Positional arguments (`input`, `filters`, `output_shape`, `strides`, `padding`).
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Transposed 3-D convolution output tensor.

    Raises:
        NotImplementedError: If called without required input arguments.
    """
    if not args and "input" not in kwargs and "inputs" not in kwargs:
        raise NotImplementedError("conv3d_transpose requires input and filters arguments")
    return _wrap_single(_unwrap_k(k_ops.conv_transpose(*_unwrap_args(args), **kwargs)))


def crelu(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute Concatenated ReLU (`[relu(x), relu(-x)]`).

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Concatenated ReLU output tensor.
    """
    del args, name, kwargs
    _check_none(features)
    t = _to_tensor(features)
    pos = _ops.maximum(t, _to_tensor(0.0, dtype=t.dtype))
    neg = _ops.maximum(_ops.negative(t), _to_tensor(0.0, dtype=t.dtype))
    return _wrap_single(as_tensor_obj(_ops.concatenate([pos, neg], axis=-1)))


def ctc_loss(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute CTC (Connectionist Temporal Classification) loss.

    Args:
        *args: Positional CTC arguments (`labels`, `logits`, `label_length`, `logit_length`).
        **kwargs: Keyword CTC arguments.

    Returns:
        Tensor: CTC loss tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.ctc_loss(*_unwrap_args(args), **kwargs)))


def depthwise_conv2d(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute a 2-D depthwise convolution.

    Args:
        *args: Positional depthwise convolution arguments.
        **kwargs: Keyword depthwise convolution arguments.

    Returns:
        Tensor: Depthwise convolved output tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.depthwise_conv(*_unwrap_args(args), **kwargs)))


def elu(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the exponential linear unit: `exp(features) - 1` if `< 0`, `features` otherwise.

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Exponential linear unit activation tensor.
    """
    del args, name, kwargs
    _check_none(features)
    t = _to_tensor(features)
    alpha_t = _to_tensor(1.0, dtype=t.dtype)

    cond = _ops.greater(t, _to_tensor(0.0, dtype=t.dtype))
    cond_f = _ops.cast(cond, t.dtype)
    not_cond_f = _ops.subtract(_to_tensor(1.0, dtype=t.dtype), cond_f)

    res = _ops.add(
        _ops.multiply(cond_f, t),
        _ops.multiply(
            not_cond_f,
            _ops.multiply(alpha_t, _ops.subtract(_ops.exp(t), _to_tensor(1.0, dtype=t.dtype))),
        ),
    )
    return _wrap_single(res)


def fractional_max_pool(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Perform fractional max pooling on the input.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Pooled output tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented operation.
    """
    del args, kwargs
    raise NotImplementedError(
        "fractional_max_pool is not yet implemented in ml-switcheroo-compiler"
    )


def gelu(
    features: TensorOperand,
    approximate: bool = False,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Gaussian Error Linear Unit (GELU) activation function.

    Args:
        features: Input `Tensor`.
        approximate: Whether to enable tanh approximation.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: GELU activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(
        _unwrap_k(k_activations.gelu(_to_tensor(features), approximate=approximate, **kwargs))
    )


def leaky_relu(
    features: TensorOperand,
    alpha: float = 0.2,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Leaky ReLU activation function.

    Args:
        features: Input `Tensor` representing preactivation values.
        alpha: Slope of the activation function at `x < 0`.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Leaky ReLU activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(
        _unwrap_k(k_activations.leaky_relu(_to_tensor(features), negative_slope=alpha, **kwargs))
    )


def local_response_normalization(
    input: Tensor | TensorOperand = None,
    depth_radius: int = 5,
    bias: float = 1.0,
    alpha: float = 1.0,
    beta: float = 0.5,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Perform Local Response Normalization.

    Args:
        input: 4-D `Tensor` of shape `[batch, height, width, depth]`.
        depth_radius: 0-D `int`. Half-width of the 1-D normalization window.
        bias: An offset float (usually positive to avoid dividing by 0).
        alpha: A scale factor float, usually positive.
        beta: An exponent float.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Normalized tensor.

    Raises:
        NotImplementedError: If `input` is not provided.
    """
    del depth_radius, name, args, kwargs
    if input is None:
        raise NotImplementedError("local_response_normalization requires an input tensor")
    t_in = _to_tensor(input)
    sq = _ops.multiply(t_in, t_in)
    denom_base = _ops.add(
        _to_tensor(bias, dtype=t_in.dtype),
        _ops.multiply(_to_tensor(alpha, dtype=t_in.dtype), sq),
    )
    scale = _ops.power(denom_base, _to_tensor(beta, dtype=t_in.dtype))
    return _wrap_single(_ops.divide(t_in, scale))


def max_pool3d(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Perform 3D max pooling on the input.

    Args:
        *args: Positional pooling arguments.
        **kwargs: Keyword pooling arguments.

    Returns:
        Tensor: 3D max-pooled tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.max_pool(*_unwrap_args(args), **kwargs)))


def nce_loss(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute and return the noise-contrastive estimation training loss.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: NCE loss tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented operation.
    """
    del args, kwargs
    raise NotImplementedError("nce_loss is not yet implemented in ml-switcheroo-compiler")


def relu(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute rectified linear activation: `max(features, 0)`.

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: ReLU activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.relu(_to_tensor(features), *args, **kwargs)))


def sampled_softmax_loss(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute and return the sampled softmax training loss.

    Args:
        *args: Positional arguments.
        **kwargs: Keyword arguments.

    Returns:
        Tensor: Sampled softmax loss tensor.

    Raises:
        NotImplementedError: Always raised for unimplemented operation.
    """
    del args, kwargs
    raise NotImplementedError(
        "sampled_softmax_loss is not yet implemented in ml-switcheroo-compiler"
    )


def selu(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute scaled exponential linear: `scale * alpha * (exp(features) - 1)` if `< 0`.

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: SELU activated tensor.
    """
    del args, name, kwargs
    _check_none(features)
    t = _to_tensor(features)

    scale = _to_tensor(1.0507009873554804934193349852946, dtype=t.dtype)
    alpha = _to_tensor(1.6732632423543772848170429916717, dtype=t.dtype)
    cond = _ops.greater(t, _to_tensor(0.0, dtype=t.dtype))
    cond_f = _ops.cast(cond, t.dtype)
    not_cond_f = _ops.subtract(_to_tensor(1.0, dtype=t.dtype), cond_f)

    res = _ops.add(
        _ops.multiply(cond_f, t),
        _ops.multiply(
            not_cond_f,
            _ops.multiply(alpha, _ops.subtract(_ops.exp(t), _to_tensor(1.0, dtype=t.dtype))),
        ),
    )
    return _wrap_single(_ops.multiply(scale, res))


def separable_conv2d(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
    """Compute a 2-D depthwise separable convolution.

    Args:
        *args: Positional separable convolution arguments.
        **kwargs: Keyword separable convolution arguments.

    Returns:
        Tensor: Separable 2-D convolved tensor.
    """
    return _wrap_single(_unwrap_k(k_ops.separable_conv(*_unwrap_args(args), **kwargs)))


def sigmoid(
    x: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute sigmoid of `x` element-wise (`1 / (1 + exp(-x))`).

    Args:
        x: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Sigmoid activated tensor.
    """
    del name
    _check_none(x)
    return _wrap_single(_unwrap_k(k_activations.sigmoid(_to_tensor(x), *args, **kwargs)))


def silu(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the SiLU (Swish) activation function (`x * sigmoid(x)`).

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: SiLU activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.silu(_to_tensor(features), *args, **kwargs)))


def softmax(
    logits: TensorOperand,
    axis: int | None = None,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute softmax activations (`exp(logits) / sum(exp(logits), axis)`).

    Args:
        logits: Non-empty `Tensor`.
        axis: Dimension along which the softmax will be performed. Defaults to `-1`.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Softmax probability tensor.
    """
    del name
    _check_none(logits)
    resolved_axis = -1 if axis is None else axis
    return _wrap_single(
        _unwrap_k(k_activations.softmax(_to_tensor(logits), axis=resolved_axis, **kwargs))
    )


def swish(
    features: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Swish activation function (`features * sigmoid(features)`).

    Args:
        features: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Swish activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.swish(_to_tensor(features), *args, **kwargs)))


def tanh(
    x: TensorOperand,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute hyperbolic tangent of `x` element-wise.

    Args:
        x: Input `Tensor`.
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Hyperbolic tangent tensor.
    """
    del name
    _check_none(x)
    return _wrap_single(_unwrap_k(k_activations.tanh(_to_tensor(x), *args, **kwargs)))
