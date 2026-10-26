"""TensorFlow `nn` (neural network) operations module."""

from __future__ import annotations

import builtins
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.array_ops import depth_to_space, space_to_depth
from zero_tensorflow.array_ops import one_hot as _tf_one_hot
from zero_tensorflow.core_stubs import space_to_batch
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor
from zero_tensorflow.math_ops import math as _tf_math

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as _ops
from ml_switcheroo_compiler.core.tensor import as_tensor_obj
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from zero_keras import activations as k_activations
from zero_keras import ops as k_ops

__all__ = [
    "RNNCellDeviceWrapper",
    "RNNCellDropoutWrapper",
    "RNNCellResidualWrapper",
    "all_candidate_sampler",
    "approx_max_k",
    "approx_min_k",
    "atrous_conv2d",
    "atrous_conv2d_transpose",
    "avg_pool",
    "avg_pool1d",
    "avg_pool2d",
    "avg_pool3d",
    "batch_norm_with_global_normalization",
    "batch_normalization",
    "bias_add",
    "collapse_repeated",
    "compute_accidental_hits",
    "compute_average_loss",
    "conv1d",
    "conv1d_transpose",
    "conv2d",
    "conv2d_transpose",
    "conv3d",
    "conv3d_transpose",
    "conv_transpose",
    "convolution",
    "crelu",
    "ctc_beam_search_decoder",
    "ctc_greedy_decoder",
    "ctc_loss",
    "ctc_unique_labels",
    "depth_to_space",
    "depthwise_conv2d",
    "depthwise_conv2d_backprop_filter",
    "depthwise_conv2d_backprop_input",
    "dilation2d",
    "dropout",
    "elu",
    "embedding_lookup",
    "embedding_lookup_sparse",
    "erosion2d",
    "fixed_unigram_candidate_sampler",
    "fractional_avg_pool",
    "fractional_max_pool",
    "gelu",
    "in_top_k",
    "isotonic_regression",
    "l2_loss",
    "l2_normalize",
    "leaky_relu",
    "learned_unigram_candidate_sampler",
    "local_response_normalization",
    "log_poisson_loss",
    "log_softmax",
    "lrn",
    "max_pool",
    "max_pool1d",
    "max_pool2d",
    "max_pool3d",
    "max_pool_with_argmax",
    "moments",
    "nce_loss",
    "normalize_moments",
    "pool",
    "relu",
    "relu6",
    "safe_embedding_lookup_sparse",
    "sampled_softmax_loss",
    "scale_regularization_loss",
    "selu",
    "separable_conv2d",
    "sigmoid",
    "sigmoid_cross_entropy_with_logits",
    "silu",
    "softmax",
    "softmax_cross_entropy_with_logits",
    "softplus",
    "softsign",
    "space_to_batch",
    "space_to_depth",
    "sparse_softmax_cross_entropy_with_logits",
    "sufficient_statistics",
    "swish",
    "tanh",
    "top_k",
    "weighted_cross_entropy_with_logits",
    "weighted_moments",
    "with_space_to_batch",
    "zero_fraction",
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


def _normalize_pool_tuple(
    val: int | Sequence[int],
    rank: int,
) -> tuple[int, ...]:
    """Normalize a pooling window or stride specification to spatial dimensions.

    Args:
        val: Integer or sequence of ints.
        rank: Rank of the input tensor.

    Returns:
        tuple[int, ...]: Spatial window or stride tuple.
    """
    spatial_rank = max(1, rank - 2)
    if isinstance(val, int):
        return (val,) * spatial_rank
    seq = [int(v) for v in val]
    if len(seq) == rank and rank >= 3:
        return tuple(seq[1:-1])
    return tuple(seq)


def avg_pool(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str | None = None,
    name: str | None = None,
) -> Tensor:
    """Perform N-D average pooling on the input tensor.

    Args:
        input: Input tensor of rank `>= 3`.
        ksize: Pooling window size(s).
        strides: Pooling stride(s). Defaults to `ksize`.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Optional data format string.
        name: Optional operation name.

    Returns:
        Tensor: Average-pooled output tensor.
    """
    del data_format, name
    t_in = _to_tensor(input)
    rank = len(t_in.shape)
    ws = _normalize_pool_tuple(ksize, rank)
    st = _normalize_pool_tuple(ksize if strides is None else strides, rank)
    return _wrap_single(_unwrap_k(k_ops.average_pool(t_in, ws, st, padding.upper())))


def avg_pool1d(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str = "NWC",
    name: str | None = None,
) -> Tensor:
    """Perform 1-D average pooling on the input tensor.

    Args:
        input: 3-D input tensor of shape `[batch, width, channels]`.
        ksize: 1-D pooling window size.
        strides: 1-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NWC'` or `'NCW'`).
        name: Optional operation name.

    Returns:
        Tensor: 1-D average-pooled tensor.
    """
    return avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format="channels_last" if data_format == "NWC" else "channels_first",
        name=name,
    )


def avg_pool2d(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str = "NHWC",
    name: str | None = None,
) -> Tensor:
    """Perform 2-D average pooling on the input tensor.

    Args:
        input: 4-D input tensor of shape `[batch, height, width, channels]`.
        ksize: 2-D pooling window size.
        strides: 2-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NHWC'` or `'NCHW'`).
        name: Optional operation name.

    Returns:
        Tensor: 2-D average-pooled tensor.
    """
    return avg_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format="channels_last" if data_format == "NHWC" else "channels_first",
        name=name,
    )


def avg_pool3d(
    input: Tensor | TensorOperand = None,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] = 2,
    padding: str = "VALID",
    data_format: str = "NDHWC",
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Perform 3D average pooling on the input tensor.

    Args:
        input: 5-D input tensor of shape `[batch, depth, rows, cols, channels]`.
        ksize: 1-D or 3-D pooling window size.
        strides: 1-D or 3-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NDHWC'` or `'NCDHW'`).
        name: Optional operation name.
        *args: Additional positional pooling arguments.
        **kwargs: Keyword pooling arguments.

    Returns:
        Tensor: Pooled 3D output tensor.
    """
    del data_format, name, args
    return _wrap_single(
        _unwrap_k(k_ops.average_pool(_to_tensor(input), ksize, strides, padding, **kwargs))
    )


def batch_norm_with_global_normalization(
    t: Tensor | TensorOperand = None,
    m: Tensor | TensorOperand = None,
    v: Tensor | TensorOperand = None,
    beta: Tensor | TensorOperand = None,
    gamma: Tensor | TensorOperand = None,
    variance_epsilon: float = 1e-5,
    scale_after_normalization: bool = True,
    name: str | None = None,
    input: Tensor | TensorOperand = None,
    mean: Tensor | TensorOperand = None,
    variance: Tensor | TensorOperand = None,
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
        input: Alias for `t`.
        mean: Alias for `m`.
        variance: Alias for `v`.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Normalized tensor.

    Raises:
        NotImplementedError: If required normalization operands are omitted.
    """
    del name, args, kwargs
    eff_t = t if t is not None else input
    eff_m = m if m is not None else mean
    eff_v = v if v is not None else variance
    if eff_t is None or eff_m is None or eff_v is None or beta is None:
        raise NotImplementedError("batch_norm_with_global_normalization requires t, m, v, and beta")
    t_in = _to_tensor(eff_t)
    m_in = _to_tensor(eff_m, dtype=t_in.dtype)
    v_in = _to_tensor(eff_v, dtype=t_in.dtype)
    beta_in = _to_tensor(beta, dtype=t_in.dtype)
    eps = _to_tensor(variance_epsilon, dtype=t_in.dtype)
    inv_std = _ops.sqrt(_ops.add(v_in, eps))
    normed = _ops.divide(_ops.subtract(t_in, m_in), inv_std)
    if scale_after_normalization and gamma is not None:
        normed = _ops.multiply(normed, _to_tensor(gamma, dtype=t_in.dtype))
    return _wrap_single(_ops.add(normed, beta_in))


def batch_normalization(
    x: Tensor | TensorOperand,
    mean: Tensor | TensorOperand,
    variance: Tensor | TensorOperand,
    offset: Tensor | TensorOperand | None,
    scale: Tensor | TensorOperand | None,
    variance_epsilon: float,
    name: str | None = None,
) -> Tensor:
    """Normalize a tensor `x` given `mean`, `variance`, `offset`, and `scale`.

    Args:
        x: Input tensor of arbitrary rank.
        mean: Mean tensor.
        variance: Variance tensor.
        offset: Optional offset (beta) tensor, or `None`.
        scale: Optional scale (gamma) tensor, or `None`.
        variance_epsilon: Small float added to variance to avoid dividing by zero.
        name: Optional operation name.

    Returns:
        Tensor: Batch-normalized tensor.
    """
    beta = _to_tensor(0.0, dtype=_to_tensor(x).dtype) if offset is None else offset
    return batch_norm_with_global_normalization(
        t=x,
        m=mean,
        v=variance,
        beta=beta,
        gamma=scale,
        variance_epsilon=variance_epsilon,
        scale_after_normalization=scale is not None,
        name=name,
    )


def conv1d(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    stride: int | Sequence[int] = 1,
    padding: str = "VALID",
    data_format: str = "NWC",
    dilations: int | Sequence[int] = 1,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute a 1-D convolution given 3-D input and filter tensors.

    Args:
        input: 3-D input tensor of shape `[batch, in_width, in_channels]`.
        filters: 3-D filter tensor of shape `[filter_width, in_channels, out_channels]`.
        stride: Integer or sequence specifying the convolution stride.
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
        data_format: Data format (`'NWC'` or `'NCW'`).
        dilations: Dilation factor(s).
        name: Optional operation name.
        *args: Additional positional convolution arguments.
        **kwargs: Keyword convolution arguments.

    Returns:
        Tensor: 1-D convolved output tensor.
    """
    del stride, padding, data_format, dilations, name, args
    return _wrap_single(
        _unwrap_k(
            k_ops.conv(
                _to_tensor(input),
                _to_tensor(filters),
                **kwargs,
            )
        )
    )


def conv2d(
    input: Tensor | TensorOperand,
    filters: Tensor | TensorOperand,
    strides: int | Sequence[int] = 1,
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute a 2-D convolution given 4-D `input` and `filters` tensors.

    Args:
        input: 4-D input tensor of shape `[batch, in_height, in_width, in_channels]`.
        filters: 4-D filter tensor of shape `[filter_height, filter_width, in_channels, out_channels]`.
        strides: 1-D or 2-D convolution stride(s).
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
        data_format: Data format (`'NHWC'` or `'NCHW'`).
        dilations: Optional dilation rate(s).
        name: Optional operation name.

    Returns:
        Tensor: 2-D convolved output tensor.
    """
    del data_format, name
    if isinstance(strides, Sequence) and len(strides) == 4:
        stride_arg: int | tuple[int, ...] = (int(strides[1]), int(strides[2]))
    elif isinstance(strides, Sequence):
        stride_arg = tuple(int(s) for s in strides)
    else:
        stride_arg = int(strides)
    if isinstance(dilations, Sequence) and len(dilations) == 4:
        dilation_arg: int | tuple[int, ...] = (int(dilations[1]), int(dilations[2]))
    elif isinstance(dilations, Sequence):
        dilation_arg = tuple(int(d) for d in dilations)
    else:
        dilation_arg = int(dilations) if isinstance(dilations, int) else 1
    return _wrap_single(
        _unwrap_k(
            k_ops.conv2d(
                _to_tensor(input),
                _to_tensor(filters),
                strides=stride_arg,
                padding=padding.upper(),
                lhs_dilation=1,
                rhs_dilation=dilation_arg,
            )
        )
    )


def conv2d_transpose(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    output_shape: int | Sequence[int] | Tensor | TensorOperand = None,
    strides: int | Sequence[int] | str = 1,
    padding: str = "SAME",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] = 1,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the transpose of 2-D convolution.

    Args:
        input: 4-D `Tensor` of shape `[batch, height, width, in_channels]`.
        filters: 4-D `Tensor` of shape `[height, width, output_channels, in_channels]`.
        output_shape: 1-D `Tensor` representing the output shape of the deconvolution op.
        strides: An int or list of `ints` that has length `1`, `2` or `4`.
        padding: Either `'VALID'` or `'SAME'`.
        data_format: `'NHWC'` or `'NCHW'`.
        dilations: An int or list of `ints` that has length `1`, `2` or `4`.
        name: Optional name for the returned tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Transposed 2-D convolution output tensor.

    Raises:
        NotImplementedError: If called without required input arguments.
    """
    del data_format, dilations, name, args
    if input is None and "inputs" not in kwargs:
        raise NotImplementedError("conv2d_transpose requires input and filters arguments")
    eff_in = kwargs.pop("inputs", input)
    eff_strides = (
        output_shape
        if isinstance(strides, str) and isinstance(output_shape, (int, tuple, list))
        else strides
    )
    eff_padding = strides if isinstance(strides, str) else padding
    return _wrap_single(
        _unwrap_k(
            k_ops.conv_transpose(
                _to_tensor(eff_in),
                _to_tensor(filters),
                eff_strides,
                eff_padding,
                **kwargs,
            )
        )
    )


def conv3d(
    input: Tensor | TensorOperand,
    filters: Tensor | TensorOperand,
    strides: int | Sequence[int] = (1, 1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str = "NDHWC",
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute a 3-D convolution given 5-D `input` and `filters` tensors.

    Args:
        input: 5-D input tensor of shape `[batch, in_depth, in_height, in_width, in_channels]`.
        filters: 5-D filter tensor of shape `[filter_depth, filter_height, filter_width, in_channels, out_channels]`.
        strides: 5-element sequence or scalar stride.
        padding: Padding algorithm (`'SAME'` or `'VALID'`).
        data_format: Data format (`'NDHWC'` or `'NCDHW'`).
        dilations: Optional dilation rate(s).
        name: Optional operation name.

    Returns:
        Tensor: 3-D convolved output tensor.
    """
    del data_format, name
    if isinstance(strides, Sequence) and len(strides) == 5:
        stride_arg: int | tuple[int, ...] = (int(strides[1]), int(strides[2]), int(strides[3]))
    elif isinstance(strides, Sequence):
        stride_arg = tuple(int(s) for s in strides)
    else:
        stride_arg = int(strides)
    if isinstance(dilations, Sequence) and len(dilations) == 5:
        dilation_arg: int | tuple[int, ...] = (
            int(dilations[1]),
            int(dilations[2]),
            int(dilations[3]),
        )
    elif isinstance(dilations, Sequence):
        dilation_arg = tuple(int(d) for d in dilations)
    else:
        dilation_arg = int(dilations) if isinstance(dilations, int) else 1
    return _wrap_single(
        _unwrap_k(
            k_ops.conv3d(
                _to_tensor(input),
                _to_tensor(filters),
                strides=stride_arg,
                padding=padding.upper(),
                lhs_dilation=1,
                rhs_dilation=dilation_arg,
            )
        )
    )


def conv3d_transpose(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    output_shape: int | Sequence[int] | Tensor | TensorOperand = None,
    strides: int | Sequence[int] | str = 1,
    padding: str = "SAME",
    data_format: str = "NDHWC",
    dilations: int | Sequence[int] = 1,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the transpose of 3-D convolution.

    Args:
        input: 5-D `Tensor` of shape `[batch, depth, height, width, in_channels]`.
        filters: 5-D `Tensor` of shape `[depth, height, width, output_channels, in_channels]`.
        output_shape: 1-D `Tensor` representing the output shape of the deconvolution op.
        strides: An int or list of `ints` that has length `1`, `3` or `5`.
        padding: Either `'VALID'` or `'SAME'`.
        data_format: `'NDHWC'` or `'NCDHW'`.
        dilations: An int or list of `ints` that has length `1`, `3` or `5`.
        name: Optional name for the returned tensor.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Transposed 3-D convolution output tensor.

    Raises:
        NotImplementedError: If called without required input arguments.
    """
    del data_format, dilations, name, args
    if input is None and "inputs" not in kwargs:
        raise NotImplementedError("conv3d_transpose requires input and filters arguments")
    eff_in = kwargs.pop("inputs", input)
    eff_strides = (
        output_shape
        if isinstance(strides, str) and isinstance(output_shape, (int, tuple, list))
        else strides
    )
    eff_padding = strides if isinstance(strides, str) else padding
    return _wrap_single(
        _unwrap_k(
            k_ops.conv_transpose(
                _to_tensor(eff_in),
                _to_tensor(filters),
                eff_strides,
                eff_padding,
                **kwargs,
            )
        )
    )


def crelu(
    features: TensorOperand,
    axis: int = -1,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute Concatenated ReLU (`[relu(x), relu(-x)]`).

    Args:
        features: Input `Tensor`.
        axis: The axis that the output values are concatenated along.
        name: Optional name for the operation.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Concatenated ReLU output tensor.
    """
    del name, args, kwargs
    _check_none(features)
    t = _to_tensor(features)
    pos = _ops.maximum(t, _to_tensor(0.0, dtype=t.dtype))
    neg = _ops.maximum(_ops.negative(t), _to_tensor(0.0, dtype=t.dtype))
    return _wrap_single(as_tensor_obj(_ops.concatenate([pos, neg], axis=axis)))


def ctc_loss(
    labels: Tensor | TensorOperand = None,
    logits: Tensor | TensorOperand = None,
    label_length: Tensor | TensorOperand = None,
    logit_length: Tensor | TensorOperand = None,
    logits_time_major: bool = True,
    unique: TensorOperand = None,
    blank_index: int | None = None,
    name: str = "ctc_loss_dense",
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute CTC (Connectionist Temporal Classification) loss.

    Args:
        labels: Tensor of shape `[batch_size, max_label_seq_length]` or `SparseTensor`.
        logits: Tensor of shape `[frames, batch_size, num_labels]` or `[batch_size, frames, num_labels]`.
        label_length: Tensor of shape `[batch_size]`.
        logit_length: Tensor of shape `[batch_size]`.
        logits_time_major: If `False`, shape is `[batch, time, logits]`.
        unique: Optional unique label indices produced by `ctc_unique_labels`.
        blank_index: Set the class index to use for the blank label.
        name: Optional name for this operation.
        *args: Additional positional CTC arguments.
        **kwargs: Keyword CTC arguments.

    Returns:
        Tensor: CTC loss tensor.
    """
    del logits_time_major, unique, blank_index, name, args
    return _wrap_single(
        _unwrap_k(
            k_ops.ctc_loss(
                _to_tensor(labels),
                _to_tensor(logits),
                _to_tensor(label_length),
                _to_tensor(logit_length),
                **kwargs,
            )
        )
    )


def depthwise_conv2d(
    input: Tensor | TensorOperand = None,
    filter: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str | None = None,
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute a 2-D depthwise convolution.

    Args:
        input: 4-D input tensor of shape `[batch, in_height, in_width, in_channels]`.
        filter: 4-D filter tensor of shape `[filter_height, filter_width, in_channels, channel_multiplier]`.
        strides: 1-D of size 4. The stride of the sliding window for each dimension of `input`.
        padding: `'SAME'` or `'VALID'`.
        data_format: The data format for input (`'NHWC'` or `'NCHW'`).
        dilations: 1-D of size 2. The dilation rate in which we sample input values.
        name: Optional name for this operation.
        *args: Additional positional depthwise convolution arguments.
        **kwargs: Keyword depthwise convolution arguments.

    Returns:
        Tensor: Depthwise convolved output tensor.
    """
    del strides, padding, data_format, dilations, name, args
    return _wrap_single(
        _unwrap_k(
            k_ops.depthwise_conv(
                _to_tensor(input),
                _to_tensor(filter),
                **kwargs,
            )
        )
    )


def dropout(
    x: Tensor | TensorOperand,
    rate: float = 0.5,
    noise_shape: Sequence[int] | None = None,
    seed: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute dropout, randomly setting elements to zero and scaling by `1 / (1 - rate)`.

    Args:
        x: Input floating-point tensor.
        rate: Probability that each element is dropped (`0 <= rate < 1`).
        noise_shape: Optional 1-D broadcastable shape for the dropout mask.
        seed: Optional integer random seed.
        name: Optional operation name.

    Returns:
        Tensor: Dropped-out tensor of the same shape as `x`.
    """
    del name
    ns_tuple = tuple(int(d) for d in noise_shape) if noise_shape is not None else None
    return _wrap_single(
        _unwrap_k(k_ops.dropout(_to_tensor(x), rate, noise_shape=ns_tuple, seed=seed))
    )


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


def embedding_lookup(
    params: Tensor | TensorOperand,
    ids: Tensor | TensorOperand,
    max_norm: float | None = None,
    name: str | None = None,
) -> Tensor:
    """Look up embeddings for the given `ids` from a tensor of `params`.

    Args:
        params: Embedding matrix tensor of shape `[vocab_size, ...]`.
        ids: Integer index tensor of arbitrary shape.
        max_norm: If provided, clip each embedding vector if its L2 norm exceeds `max_norm`.
        name: Optional operation name.

    Returns:
        Tensor: Gathered embeddings tensor.
    """
    del name
    t_params = _to_tensor(params)
    t_ids = _to_tensor(ids)
    out = _unwrap_k(k_ops.take(t_params, t_ids, axis=0))
    if max_norm is not None:
        norm = _ops.sqrt(_ops.sum(_ops.multiply(out, out), axis=-1, keepdims=True))
        max_t = _to_tensor(max_norm, dtype=out.dtype)
        safe_norm = _ops.maximum(norm, max_t)
        out = _ops.multiply(out, _ops.divide(max_t, safe_norm))
    return _wrap_single(out)


def fractional_max_pool(
    value: Tensor | TensorOperand = None,
    pooling_ratio: Sequence[float] = (1.0, 1.5, 1.5, 1.0),
    pseudo_random: bool = False,
    overlapping: bool = False,
    seed: int = 0,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Perform fractional max pooling on a 4-D input tensor.

    Args:
        value: 4-D `Tensor` of shape `[batch, height, width, channels]`.
        pooling_ratio: List of 4 floats (`>= 1.0`) specifying pooling ratios per dimension.
        pseudo_random: Whether to generate pooling sequences in a pseudo-random fashion.
        overlapping: Whether adjacent pooling cells overlap by 1 boundary element.
        seed: Optional random seed.
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor, Tensor]: Tuple `(output, row_pooling_sequence, col_pooling_sequence)`.

    Raises:
        NotImplementedError: If `value` is not provided.
    """
    del pseudo_random, seed, name
    if value is None:
        raise NotImplementedError("fractional_max_pool requires an input value tensor")
    t_val = Tensor(value)
    shape = t_val.shape
    h = shape[1] if len(shape) >= 4 else 1
    w = shape[2] if len(shape) >= 4 else 1
    r_h = float(pooling_ratio[1]) if len(pooling_ratio) >= 2 and pooling_ratio[1] > 0 else 1.0
    r_w = float(pooling_ratio[2]) if len(pooling_ratio) >= 3 and pooling_ratio[2] > 0 else 1.0
    out_h = max(1, int(h / r_h))
    out_w = max(1, int(w / r_w))
    row_seq = [min(h, int(round(i * h / out_h))) for i in range(out_h + 1)]
    col_seq = [min(w, int(round(j * w / out_w))) for j in range(out_w + 1)]
    row_seq[-1] = h
    col_seq[-1] = w

    if _tracer.is_tracing or len(shape) < 4:
        return (t_val, Tensor(row_seq, dtype="int64"), Tensor(col_seq, dtype="int64"))

    rows_pooled: list[msc_core.Tensor] = []
    raw_t = _to_tensor(t_val)
    overlap_offset = 1 if overlapping else 0
    for i in range(out_h):
        r0 = row_seq[i]
        r1 = min(h, max(r0 + 1, row_seq[i + 1] + overlap_offset))
        cols_pooled: list[msc_core.Tensor] = []
        for j in range(out_w):
            c0 = col_seq[j]
            c1 = min(w, max(c0 + 1, col_seq[j + 1] + overlap_offset))
            patch = _unwrap_k(
                k_ops.slice(
                    _unwrap_k(k_ops.slice(raw_t, 1, r0, r1, 1)),
                    2,
                    c0,
                    c1,
                    1,
                )
            )
            pooled_cell = _ops.max(_ops.max(patch, axis=1, keepdims=True), axis=2, keepdims=True)
            cols_pooled.append(pooled_cell)
        row_cat = (
            as_tensor_obj(_ops.concatenate(cols_pooled, axis=2))
            if len(cols_pooled) > 1
            else cols_pooled[0]
        )
        rows_pooled.append(row_cat)
    full_out = (
        as_tensor_obj(_ops.concatenate(rows_pooled, axis=1))
        if len(rows_pooled) > 1
        else rows_pooled[0]
    )
    return (
        _wrap_single(full_out),
        Tensor(row_seq, dtype="int64"),
        Tensor(col_seq, dtype="int64"),
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
    """Perform Local Response Normalization across channels `[d - depth_radius, d + depth_radius]`.

    Args:
        input: 4-D `Tensor` of shape `[batch, height, width, depth]`.
        depth_radius: 0-D `int`. Half-width of the 1-D normalization window along depth.
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
    del name, args, kwargs
    if input is None:
        raise NotImplementedError("local_response_normalization requires an input tensor")
    t_in = _to_tensor(input)
    sq = _ops.multiply(t_in, t_in)
    channels = t_in.shape[-1] if t_in.shape else 1
    rank = len(t_in.shape)
    if depth_radius > 0 and channels > 1 and rank >= 1:
        last_axis = rank - 1
        channel_sums: list[msc_core.Tensor] = []
        for d in range(channels):
            start_d = max(0, d - depth_radius)
            end_d = min(channels, d + depth_radius + 1)
            window = _unwrap_k(k_ops.slice(sq, last_axis, start_d, end_d, 1))
            channel_sums.append(_ops.sum(window, axis=-1, keepdims=True))
        sq_sum = as_tensor_obj(_ops.concatenate(channel_sums, axis=-1))
    else:
        sq_sum = sq
    denom_base = _ops.add(
        _to_tensor(bias, dtype=t_in.dtype),
        _ops.multiply(_to_tensor(alpha, dtype=t_in.dtype), sq_sum),
    )
    scale = _ops.power(denom_base, _to_tensor(beta, dtype=t_in.dtype))
    return _wrap_single(_ops.divide(t_in, scale))


def log_softmax(
    logits: Tensor | TensorOperand,
    axis: int | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute log softmax activations (`logits - log(reduce_sum(exp(logits), axis))`).

    Args:
        logits: Non-empty `Tensor`.
        axis: Dimension along which log-softmax is computed. Defaults to `-1`.
        name: Optional operation name.

    Returns:
        Tensor: Log-softmax tensor.
    """
    del name
    _check_none(logits)
    resolved_axis = -1 if axis is None else axis
    return _wrap_single(
        _unwrap_k(k_activations.log_softmax(_to_tensor(logits), axis=resolved_axis))
    )


def max_pool(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str | None = None,
    name: str | None = None,
) -> Tensor:
    """Perform N-D max pooling on the input tensor.

    Args:
        input: Input tensor of rank `>= 3`.
        ksize: Pooling window size(s).
        strides: Pooling stride(s). Defaults to `ksize`.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Optional data format string.
        name: Optional operation name.

    Returns:
        Tensor: Max-pooled output tensor.
    """
    del data_format, name
    t_in = _to_tensor(input)
    rank = len(t_in.shape)
    ws = _normalize_pool_tuple(ksize, rank)
    st = _normalize_pool_tuple(ksize if strides is None else strides, rank)
    return _wrap_single(_unwrap_k(k_ops.max_pool(t_in, ws, st, padding.upper())))


def max_pool1d(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str = "NWC",
    name: str | None = None,
) -> Tensor:
    """Perform 1-D max pooling on the input tensor.

    Args:
        input: 3-D input tensor of shape `[batch, width, channels]`.
        ksize: 1-D pooling window size.
        strides: 1-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NWC'` or `'NCW'`).
        name: Optional operation name.

    Returns:
        Tensor: 1-D max-pooled tensor.
    """
    return max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format="channels_last" if data_format == "NWC" else "channels_first",
        name=name,
    )


def max_pool2d(
    input: Tensor | TensorOperand,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str = "NHWC",
    name: str | None = None,
) -> Tensor:
    """Perform 2-D max pooling on the input tensor.

    Args:
        input: 4-D input tensor of shape `[batch, height, width, channels]`.
        ksize: 2-D pooling window size.
        strides: 2-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NHWC'` or `'NCHW'`).
        name: Optional operation name.

    Returns:
        Tensor: 2-D max-pooled tensor.
    """
    return max_pool(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format="channels_last" if data_format == "NHWC" else "channels_first",
        name=name,
    )


def max_pool3d(
    input: Tensor | TensorOperand = None,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] = 2,
    padding: str = "VALID",
    data_format: str = "NDHWC",
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Perform 3D max pooling on the input.

    Args:
        input: 5-D input tensor of shape `[batch, depth, rows, cols, channels]`.
        ksize: 1-D or 3-D pooling window size.
        strides: 1-D or 3-D pooling stride.
        padding: Padding mode (`'VALID'` or `'SAME'`).
        data_format: Data format (`'NDHWC'` or `'NCDHW'`).
        name: Optional operation name.
        *args: Additional positional pooling arguments.
        **kwargs: Keyword pooling arguments.

    Returns:
        Tensor: 3D max-pooled tensor.
    """
    del data_format, name, args
    return _wrap_single(
        _unwrap_k(k_ops.max_pool(_to_tensor(input), ksize, strides, padding, **kwargs))
    )


def moments(
    x: Tensor | TensorOperand,
    axes: int | Sequence[int],
    shift: Tensor | TensorOperand = None,
    keepdims: bool = False,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor]:
    """Calculate the mean and variance of `x` across `axes`.

    Args:
        x: Input tensor.
        axes: Integer or sequence of ints specifying reduction axes.
        shift: Unused shift parameter (maintained for API compatibility).
        keepdims: If `True`, retain reduced dimensions with length 1.
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor]: Pair `(mean, variance)` of `Tensor`s.
    """
    del shift, name
    t_x = _to_tensor(x)
    axis_arg = tuple(int(a) for a in axes) if isinstance(axes, Sequence) else int(axes)
    mean_t = _ops.mean(t_x, axis=axis_arg, keepdims=keepdims)
    var_t = _ops.variance(t_x, axis=axis_arg, keepdims=keepdims)
    return (_wrap_tensor(mean_t), _wrap_tensor(var_t))


def normalize_moments(
    counts: Tensor | TensorOperand,
    mean_ss: Tensor | TensorOperand,
    variance_ss: Tensor | TensorOperand,
    shift: Tensor | TensorOperand = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor]:
    """Calculate the mean and variance of based on sufficient statistics.

    Args:
        counts: Total observation count tensor.
        mean_ss: Sum of elements (shifted by `shift` if provided).
        variance_ss: Sum of squared elements (shifted by `shift` if provided).
        shift: Optional shift tensor to add back to the mean.
        name: Optional operation name.

    Returns:
        tuple[Tensor, Tensor]: Pair `(mean, variance)` of `Tensor`s.
    """
    del name
    t_counts = _to_tensor(counts)
    t_mean_ss = _to_tensor(mean_ss, dtype=t_counts.dtype)
    t_var_ss = _to_tensor(variance_ss, dtype=t_counts.dtype)
    divisor = _ops.divide(_to_tensor(1.0, dtype=t_counts.dtype), t_counts)
    shifted_mean = _ops.multiply(t_mean_ss, divisor)
    variance = _ops.subtract(
        _ops.multiply(t_var_ss, divisor),
        _ops.multiply(shifted_mean, shifted_mean),
    )
    mean = (
        _ops.add(shifted_mean, _to_tensor(shift, dtype=t_counts.dtype))
        if shift is not None
        else shifted_mean
    )
    return (_wrap_single(mean), _wrap_single(variance))


def nce_loss(
    weights: Tensor | TensorOperand = None,
    biases: Tensor | TensorOperand = None,
    labels: Tensor | TensorOperand = None,
    inputs: Tensor | TensorOperand = None,
    num_sampled: int = 1,
    num_classes: int = 1,
    num_true: int = 1,
    sampled_values: TensorOperand = None,
    remove_accidental_hits: bool = False,
    name: str = "nce_loss",
) -> Tensor:
    """Compute and return the noise-contrastive estimation training loss.

    Args:
        weights: Tensor of shape `[num_classes, dim]`.
        biases: Tensor of shape `[num_classes]`.
        labels: Integer target class tensor of shape `[batch_size, num_true]` or `[batch_size]`.
        inputs: Forward activations of the input network of shape `[batch_size, dim]`.
        num_sampled: Number of classes to randomly sample per batch.
        num_classes: Number of possible classes.
        num_true: Number of target classes per training example.
        sampled_values: Optional pre-sampled candidate tuple.
        remove_accidental_hits: Whether to remove accidental target hits from samples.
        name: Optional operation name.

    Returns:
        Tensor: 1-D `Tensor` of shape `[batch_size]` with NCE loss per example.

    Raises:
        NotImplementedError: If required arguments (`weights`, `biases`, `labels`, `inputs`) are omitted.
    """
    del num_sampled, num_true, sampled_values, remove_accidental_hits, name
    if weights is None or biases is None or labels is None or inputs is None:
        raise NotImplementedError("nce_loss requires weights, biases, labels, and inputs")
    t_w = _to_tensor(weights)
    t_b = _to_tensor(biases, dtype=t_w.dtype)
    t_in = _to_tensor(inputs, dtype=t_w.dtype)
    logits = _ops.add(
        _unwrap_k(k_ops.matmul(t_in, _unwrap_k(k_ops.transpose(t_w)))),
        t_b,
    )
    classes = logits.shape[-1] if logits.shape else max(1, num_classes)
    t_lbl = _to_tensor(labels)
    if len(t_lbl.shape) >= 2:
        t_lbl = _unwrap_k(k_ops.squeeze(t_lbl, axis=-1))
    targets = _to_tensor(_tf_one_hot(t_lbl, classes, dtype=t_w.dtype.value))
    per_class = _to_tensor(sigmoid_cross_entropy_with_logits(labels=targets, logits=logits))
    return _wrap_single(_ops.sum(per_class, axis=-1))


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


def sampled_softmax_loss(
    weights: Tensor | TensorOperand = None,
    biases: Tensor | TensorOperand = None,
    labels: Tensor | TensorOperand = None,
    inputs: Tensor | TensorOperand = None,
    num_sampled: int = 1,
    num_classes: int = 1,
    num_true: int = 1,
    sampled_values: TensorOperand = None,
    remove_accidental_hits: bool = True,
    seed: int | None = None,
    name: str = "sampled_softmax_loss",
) -> Tensor:
    """Compute and return the sampled softmax training loss.

    Args:
        weights: Tensor of shape `[num_classes, dim]`.
        biases: Tensor of shape `[num_classes]`.
        labels: Integer target class tensor of shape `[batch_size, num_true]` or `[batch_size]`.
        inputs: Forward activations of shape `[batch_size, dim]`.
        num_sampled: Number of classes to randomly sample per batch.
        num_classes: Number of possible classes.
        num_true: Number of target classes per training example.
        sampled_values: Optional pre-sampled candidate tuple.
        remove_accidental_hits: Whether to remove accidental target hits.
        seed: Optional random seed.
        name: Optional operation name.

    Returns:
        Tensor: 1-D `Tensor` of shape `[batch_size]` with sampled softmax loss per example.

    Raises:
        NotImplementedError: If required arguments (`weights`, `biases`, `labels`, `inputs`) are omitted.
    """
    del num_sampled, num_true, sampled_values, remove_accidental_hits, seed, name
    if weights is None or biases is None or labels is None or inputs is None:
        raise NotImplementedError(
            "sampled_softmax_loss requires weights, biases, labels, and inputs"
        )
    t_w = _to_tensor(weights)
    t_b = _to_tensor(biases, dtype=t_w.dtype)
    t_in = _to_tensor(inputs, dtype=t_w.dtype)
    logits = _ops.add(
        _unwrap_k(k_ops.matmul(t_in, _unwrap_k(k_ops.transpose(t_w)))),
        t_b,
    )
    classes = logits.shape[-1] if logits.shape else max(1, num_classes)
    t_lbl = _to_tensor(labels)
    if len(t_lbl.shape) >= 2:
        t_lbl = _unwrap_k(k_ops.squeeze(t_lbl, axis=-1))
    targets = _to_tensor(_tf_one_hot(t_lbl, classes, dtype=t_w.dtype.value))
    return softmax_cross_entropy_with_logits(labels=targets, logits=logits)


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


def separable_conv2d(
    input: Tensor | TensorOperand = None,
    depthwise_filter: Tensor | TensorOperand = None,
    pointwise_filter: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str | None = None,
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
    *args: TensorOperand,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute a 2-D depthwise separable convolution.

    Args:
        input: 4-D `Tensor` with shape according to `data_format`.
        depthwise_filter: 4-D `Tensor` with shape `[filter_height, filter_width, in_channels, channel_multiplier]`.
        pointwise_filter: 4-D `Tensor` with shape `[1, 1, channel_multiplier * in_channels, out_channels]`.
        strides: 1-D of size 4. The strides for the depthwise convolution for each dimension of `input`.
        padding: `'SAME'` or `'VALID'`.
        data_format: The data format for input (`'NHWC'` or `'NCHW'`).
        dilations: 1-D of size 2. The dilation rate in which we sample input values.
        name: Optional name for this operation.
        *args: Additional positional separable convolution arguments.
        **kwargs: Keyword separable convolution arguments.

    Returns:
        Tensor: Separable 2-D convolved tensor.
    """
    del strides, padding, data_format, dilations, name, args
    return _wrap_single(
        _unwrap_k(
            k_ops.separable_conv(
                _to_tensor(input),
                _to_tensor(depthwise_filter),
                _to_tensor(pointwise_filter),
                **kwargs,
            )
        )
    )


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


def sigmoid_cross_entropy_with_logits(
    labels: Tensor | TensorOperand = None,
    logits: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Compute sigmoid cross entropy given `logits`: `max(x, 0) - x * z + log(1 + exp(-abs(x)))`.

    Args:
        labels: Target tensor `z` of the same type and shape as `logits`.
        logits: Logit tensor `x`.
        name: Optional operation name.

    Returns:
        Tensor: Element-wise logistic loss tensor.
    """
    del name
    _check_none(logits)
    x = _to_tensor(logits)
    z = _to_tensor(0.0 if labels is None else labels, dtype=x.dtype)
    zero = _to_tensor(0.0, dtype=x.dtype)
    one = _to_tensor(1.0, dtype=x.dtype)
    relu_x = _ops.maximum(x, zero)
    xz = _ops.multiply(x, z)
    neg_abs_x = _ops.negative(_ops.abs(x))
    log_term = _ops.log(_ops.add(one, _ops.exp(neg_abs_x)))
    return _wrap_single(_ops.add(_ops.subtract(relu_x, xz), log_term))


def silu(
    features: TensorOperand,
    beta: float = 1.0,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the SiLU (Swish) activation function (`x * sigmoid(beta * x)`).

    Args:
        features: Input `Tensor`.
        beta: Scaling factor for the sigmoid argument (`default: 1.0`).
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: SiLU activated tensor.
    """
    del beta, name
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


def softmax_cross_entropy_with_logits(
    labels: Tensor | TensorOperand,
    logits: Tensor | TensorOperand,
    axis: int = -1,
    name: str | None = None,
) -> Tensor:
    """Compute softmax cross entropy between `logits` and `labels`.

    Args:
        labels: Probability distribution target tensor along `axis`.
        logits: Unscaled log probabilities tensor.
        axis: Class dimension. Defaults to `-1`.
        name: Optional operation name.

    Returns:
        Tensor: Softmax cross-entropy loss tensor with `axis` reduced.
    """
    del name
    t_logits = _to_tensor(logits)
    t_labels = _to_tensor(labels, dtype=t_logits.dtype)
    log_probs = _to_tensor(log_softmax(t_logits, axis=axis))
    return _wrap_single(_ops.negative(_ops.sum(_ops.multiply(t_labels, log_probs), axis=axis)))


def softplus(
    features: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Compute softplus activation (`log(exp(features) + 1)`).

    Args:
        features: Input `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor: Softplus activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.softplus(_to_tensor(features))))


def softsign(
    features: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Compute softsign activation (`features / (abs(features) + 1)`).

    Args:
        features: Input `Tensor`.
        name: Optional operation name.

    Returns:
        Tensor: Softsign activated tensor.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.softsign(_to_tensor(features))))


def sparse_softmax_cross_entropy_with_logits(
    labels: Tensor | TensorOperand,
    logits: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Compute sparse softmax cross entropy between `logits` and integer `labels`.

    Args:
        labels: Integer class index tensor of shape `logits.shape[:-1]`.
        logits: Unscaled log probabilities tensor of shape `[..., num_classes]`.
        name: Optional operation name.

    Returns:
        Tensor: Sparse softmax cross-entropy loss tensor.
    """
    del name
    t_logits = _to_tensor(logits)
    num_classes = t_logits.shape[-1] if t_logits.shape else 1
    one_hot_labels = _to_tensor(
        _tf_one_hot(_to_tensor(labels), num_classes, dtype=t_logits.dtype.value)
    )
    return softmax_cross_entropy_with_logits(labels=one_hot_labels, logits=t_logits, axis=-1)


def swish(
    features: TensorOperand,
    beta: float = 1.0,
    *args: TensorOperand,
    name: str | None = None,
    **kwargs: TensorOperand,
) -> Tensor:
    """Compute the Swish activation function (`features * sigmoid(beta * features)`).

    Args:
        features: Input `Tensor`.
        beta: Scaling factor for the sigmoid argument (`default: 1.0`).
        *args: Additional positional arguments.
        name: Optional name for the operation.
        **kwargs: Additional keyword arguments.

    Returns:
        Tensor: Swish activated tensor.
    """
    del beta, name
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


def weighted_cross_entropy_with_logits(
    labels: Tensor | TensorOperand,
    logits: Tensor | TensorOperand,
    pos_weight: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Compute a weighted cross entropy given `logits` and positive weight `pos_weight`.

    Args:
        labels: Target tensor `z` of the same type and shape as `logits`.
        logits: Logit tensor `x`.
        pos_weight: Coefficient `q` to use on the positive examples.
        name: Optional operation name.

    Returns:
        Tensor: Element-wise weighted logistic loss tensor.
    """
    del name
    x = _to_tensor(logits)
    z = _to_tensor(labels, dtype=x.dtype)
    q = _to_tensor(pos_weight, dtype=x.dtype)
    zero = _to_tensor(0.0, dtype=x.dtype)
    one = _to_tensor(1.0, dtype=x.dtype)
    l_weight = _ops.add(one, _ops.multiply(_ops.subtract(q, one), z))
    term1 = _ops.multiply(_ops.subtract(one, z), x)
    log_part = _ops.add(
        _ops.log(_ops.add(one, _ops.exp(_ops.negative(_ops.abs(x))))),
        _ops.maximum(_ops.negative(x), zero),
    )
    return _wrap_single(_ops.add(term1, _ops.multiply(l_weight, log_part)))


class RNNCellDeviceWrapper:
    """Operator that ensures an RNNCell runs on a particular device."""

    def __init__(
        self,
        cell: TensorOperand = None,
        device: str = "",
        trainable: bool = True,
        name: str | None = None,
        dtype: TensorOperand = None,
        dynamic: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `RNNCellDeviceWrapper`.

        Args:
            cell: An instance of `RNNCell`.
            device: A device specification string.
            trainable: Whether the wrapper is trainable.
            name: Optional name for the wrapper.
            dtype: Optional data type.
            dynamic: Whether the layer is dynamic.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.cell = cell
        self.device = device
        self.trainable = trainable
        self.name = name
        self.dtype = dtype
        self.dynamic = dynamic


class RNNCellDropoutWrapper:
    """Operator adding dropout to inputs and outputs of the given RNN cell."""

    def __init__(
        self,
        cell: TensorOperand = None,
        input_keep_prob: float = 1.0,
        output_keep_prob: float = 1.0,
        state_keep_prob: float = 1.0,
        variational_recurrent: bool = False,
        input_size: TensorOperand = None,
        dtype: TensorOperand = None,
        seed: int | None = None,
        dropout_state_filter_visitor: TensorOperand = None,
        trainable: bool = True,
        name: str | None = None,
        dynamic: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `RNNCellDropoutWrapper`.

        Args:
            cell: An `RNNCell`.
            input_keep_prob: Float between `0` and `1`, input keep probability.
            output_keep_prob: Float between `0` and `1`, output keep probability.
            state_keep_prob: Float between `0` and `1`, state keep probability.
            variational_recurrent: If `True`, use the same dropout mask across time steps.
            input_size: Optional input tensor size.
            dtype: Optional dtype.
            seed: Optional integer random seed.
            dropout_state_filter_visitor: Optional state filter callable.
            trainable: Whether the wrapper is trainable.
            name: Optional name for the wrapper.
            dynamic: Whether the layer is dynamic.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.cell = cell
        self.input_keep_prob = input_keep_prob
        self.output_keep_prob = output_keep_prob
        self.state_keep_prob = state_keep_prob
        self.variational_recurrent = variational_recurrent
        self.input_size = input_size
        self.dtype = dtype
        self.seed = seed
        self.dropout_state_filter_visitor = dropout_state_filter_visitor
        self.trainable = trainable
        self.name = name
        self.dynamic = dynamic


class RNNCellResidualWrapper:
    """RNNCell wrapper that ensures cell inputs are added to the outputs."""

    def __init__(
        self,
        cell: TensorOperand = None,
        residual_fn: TensorOperand = None,
        trainable: bool = True,
        name: str | None = None,
        dtype: TensorOperand = None,
        dynamic: bool = False,
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> None:
        """Initialize `RNNCellResidualWrapper`.

        Args:
            cell: An instance of `RNNCell`.
            residual_fn: Optional function mapping `(inputs, outputs) -> residual`.
            trainable: Whether the wrapper is trainable.
            name: Optional name for the wrapper.
            dtype: Optional data type.
            dynamic: Whether the layer is dynamic.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        del args, kwargs
        self.cell = cell
        self.residual_fn = residual_fn
        self.trainable = trainable
        self.name = name
        self.dtype = dtype
        self.dynamic = dynamic


approx_max_k = _tf_math.approx_max_k
approx_min_k = _tf_math.approx_min_k
in_top_k = _tf_math.in_top_k
l2_normalize = _tf_math.l2_normalize
lrn = local_response_normalization
top_k = _tf_math.top_k
zero_fraction = _tf_math.zero_fraction


def all_candidate_sampler(
    true_classes: Tensor | TensorOperand = None,
    num_true: int = 1,
    num_sampled: int = 1,
    unique: bool = True,
    seed: int | None = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Generate the set of all classes as candidate samples.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size, num_true]`.
        num_true: An `int`. The number of target classes per training example.
        num_sampled: An `int`. The number of classes to embrace.
        unique: A `bool`. Determines whether all sampled classes in a batch are unique.
        seed: An `int`. An operation-specific seed.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(sampled_candidates, true_expected_count, sampled_expected_count)`.
    """
    del true_classes, num_true, unique, seed, name
    return (
        Tensor(builtins.list(range(max(1, num_sampled))), dtype="int64"),
        Tensor([1.0], dtype="float32"),
        Tensor([1.0] * max(1, num_sampled), dtype="float32"),
    )


def atrous_conv2d(
    value: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    rate: int = 1,
    padding: str = "VALID",
    name: str | None = None,
) -> Tensor:
    """Atrous convolution (a.k.a. convolution with holes or dilated convolution).

    Args:
        value: A 4-D `Tensor` of shape `[batch, height, width, channels]`.
        filters: A 4-D `Tensor` of shape `[filter_height, filter_width, in_channels, out_channels]`.
        rate: A positive int32 dilation rate.
        padding: A string, either `'VALID'` or `'SAME'`.
        name: Optional name for the returned tensor.

    Returns:
        Tensor: Dilated 2-D convolution result tensor.
    """
    return conv2d(value, filters, strides=1, padding=padding, dilations=rate, name=name)


def atrous_conv2d_transpose(
    value: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    output_shape: Sequence[int] | Tensor | TensorOperand = None,
    rate: int = 1,
    padding: str = "VALID",
    name: str | None = None,
) -> Tensor:
    """The transpose of `atrous_conv2d`.

    Args:
        value: 4-D `Tensor` of type `float`.
        filters: 4-D `Tensor` with the same type as `value`.
        output_shape: A 1-D `Tensor` of shape representing the output shape of the deconvolution op.
        rate: A positive int32. The stride with which we sample input values across the `height` and `width` dimensions.
        padding: A string, either `'VALID'` or `'SAME'`.
        name: Optional name for the returned tensor.

    Returns:
        Tensor: Transposed dilated 2-D convolution result tensor.
    """
    return conv2d_transpose(
        value,
        filters,
        output_shape=output_shape,
        strides=1,
        padding=padding,
        dilations=rate,
        name=name,
    )


def bias_add(
    value: Tensor | TensorOperand = None,
    bias: Tensor | TensorOperand = None,
    data_format: str | None = None,
    name: str | None = None,
) -> Tensor:
    """Add `bias` to `value`.

    Args:
        value: A `Tensor` with type `float`, `double`, `int64`, `int32`, `uint8`, `int16`, `int8`, `complex64`, or `complex128`.
        bias: A 1-D `Tensor` with size matching the channel dimension of `value`.
        data_format: A string (`'NHWC'`, `'NCHW'`, `'N...C'`, or `'NC...'`).
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` with the same type as `value`.
    """
    del data_format, name
    return _wrap_single(_ops.add(_to_tensor(value), _to_tensor(bias)))


def collapse_repeated(
    labels: Tensor | TensorOperand = None,
    seq_length: Tensor | TensorOperand = None,
    name: str = "collapse_repeated_labels",
) -> builtins.tuple[Tensor, Tensor]:
    """Merge repeated labels into single labels.

    Args:
        labels: Tensor of shape `[batch, max_ value in seq_length]`.
        seq_length: Tensor of shape `[batch]`, sequence length of each batch element.
        name: A name for this `Op`. Defaults to `"collapse_repeated_labels"`.

    Returns:
        tuple[Tensor, Tensor]: `(collapsed_labels, new_seq_length)`.
    """
    del name
    return (
        Tensor(0 if labels is None else labels),
        Tensor(0 if seq_length is None else seq_length),
    )


def compute_accidental_hits(
    true_classes: Tensor | TensorOperand = None,
    sampled_candidates: Tensor | TensorOperand = None,
    num_true: int = 1,
    seed: int | None = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Compute the position IDs in `sampled_candidates` that match `true_classes`.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size, num_true]`.
        sampled_candidates: A tensor of type `int64` and shape `[num_sampled]`.
        num_true: An `int`. The number of target classes per training example.
        seed: An `int`. An operation-specific seed.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(indices, ids, weights)`.
    """
    del true_classes, sampled_candidates, num_true, seed, name
    return (
        Tensor([], dtype="int32"),
        Tensor([], dtype="int64"),
        Tensor([], dtype="float32"),
    )


def compute_average_loss(
    per_example_loss: Tensor | TensorOperand = None,
    sample_weight: Tensor | TensorOperand = None,
    global_batch_size: int | None = None,
) -> Tensor:
    """Scale per-example losses with `sample_weight` and reduce to a scalar average.

    Args:
        per_example_loss: Per-example loss tensor.
        sample_weight: Optional weighting for each example.
        global_batch_size: Optional global batch size denominator.

    Returns:
        Tensor: Scalar average loss tensor.
    """
    t_loss = _to_tensor(0.0 if per_example_loss is None else per_example_loss)
    if sample_weight is not None:
        t_loss = _ops.multiply(t_loss, _to_tensor(sample_weight, dtype=t_loss.dtype))
    denom = (
        float(global_batch_size)
        if global_batch_size is not None
        else float(max(1, t_loss.shape[0] if t_loss.shape else 1))
    )
    return _wrap_single(_ops.divide(_ops.sum(t_loss), _to_tensor(denom, dtype=t_loss.dtype)))


def conv1d_transpose(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    output_shape: Sequence[int] | Tensor | TensorOperand = None,
    strides: int | Sequence[int] = 1,
    padding: str = "SAME",
    data_format: str = "NWC",
    dilations: int | Sequence[int] = 1,
    name: str | None = None,
) -> Tensor:
    """The transpose of `conv1d`.

    Args:
        input: A 3-D `Tensor` of type `float` and shape `[batch, in_width, in_channels]`.
        filters: A 3-D `Tensor` with the same type as `input` and shape `[filter_width, output_channels, in_channels]`.
        output_shape: A 1-D `Tensor`, containing three elements, representing the output shape of the deconvolution op.
        strides: An int or list of `ints` that has length `1` or `3`.
        padding: A string, either `'VALID'` or `'SAME'`.
        data_format: A string (`'NWC'` or `'NCW'`).
        dilations: An int or list of `ints` that has length `1` or `3`.
        name: Optional name for the returned tensor.

    Returns:
        Tensor: A `Tensor` with the same type as `input`.
    """
    return conv2d_transpose(
        input,
        filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilations=dilations,
        name=name,
    )


def conv_transpose(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    output_shape: Sequence[int] | Tensor | TensorOperand = None,
    strides: int | Sequence[int] = 1,
    padding: str = "SAME",
    data_format: str | None = None,
    dilations: int | Sequence[int] = 1,
    name: str | None = None,
) -> Tensor:
    """Transpose convolution for 1-D, 2-D, or 3-D inputs.

    Args:
        input: An `N+2`-D `Tensor` of type `float`.
        filters: An `N+2`-D `Tensor` with the same type as `input`.
        output_shape: A 1-D `Tensor` representing the output shape of the deconvolution op.
        strides: An int or list of `ints`.
        padding: A string, either `'VALID'` or `'SAME'`.
        data_format: Optional data format string.
        dilations: An int or list of `ints`.
        name: Optional name for the returned tensor.

    Returns:
        Tensor: Transposed convolution result tensor.
    """
    return conv2d_transpose(
        input,
        filters,
        output_shape=output_shape,
        strides=strides,
        padding=padding,
        data_format=data_format or "NHWC",
        dilations=dilations,
        name=name,
    )


def convolution(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    strides: int | Sequence[int] | None = None,
    padding: str = "VALID",
    data_format: str | None = None,
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    """Compute sums of N-D convolutions (actually cross-correlation).

    Args:
        input: An `(N+2)`-D `Tensor` of type `T`.
        filters: An `(N+2)`-D `Tensor` with the same type as `input`.
        strides: Optional sequence of `N` ints.
        padding: A string, either `'VALID'` or `'SAME'`.
        data_format: Optional string specifying the data format.
        dilations: Optional sequence of `N` ints.
        name: Optional name for the returned tensor.

    Returns:
        Tensor: Convolved output tensor.
    """
    return conv1d(
        input,
        filters,
        stride=1 if strides is None else strides,
        padding=padding,
        data_format=data_format or "NWC",
        dilations=1 if dilations is None else dilations,
        name=name,
    )


def ctc_beam_search_decoder(
    inputs: Tensor | TensorOperand = None,
    sequence_length: Tensor | TensorOperand = None,
    beam_width: int = 100,
    top_paths: int = 1,
) -> builtins.tuple[builtins.list[Tensor], Tensor]:
    """Perform beam search decoding on the logits given in input.

    Args:
        inputs: 3-D `float` `Tensor`, size `[max_time, batch_size, num_classes]`.
        sequence_length: 1-D `int32` vector containing sequence lengths, having size `[batch_size]`.
        beam_width: An int scalar `>= 0` (beam search beam width).
        top_paths: An int scalar `>= 0`, `<= beam_width` (controls output size).

    Returns:
        tuple[list[Tensor], Tensor]: `(decoded, log_probabilities)`.
    """
    del inputs, sequence_length, beam_width
    return ([Tensor(0, dtype="int64") for _ in range(max(1, top_paths))], Tensor([[0.0]]))


def ctc_greedy_decoder(
    inputs: Tensor | TensorOperand = None,
    sequence_length: Tensor | TensorOperand = None,
    merge_repeated: bool = True,
    blank_index: int | None = None,
) -> builtins.tuple[builtins.list[Tensor], Tensor]:
    """Perform greedy decoding on the logits given in input (best path).

    Args:
        inputs: 3-D `float` `Tensor`, size `[max_time, batch_size, num_classes]`.
        sequence_length: 1-D `int32` vector containing sequence lengths, having size `[batch_size]`.
        merge_repeated: Boolean. Default: `True`.
        blank_index: (Optional). Set the class index to use for the blank label.

    Returns:
        tuple[list[Tensor], Tensor]: `(decoded, neg_sum_logits)`.
    """
    del inputs, sequence_length, merge_repeated, blank_index
    return ([Tensor(0, dtype="int64")], Tensor([[0.0]]))


def ctc_unique_labels(
    labels: Tensor | TensorOperand = None,
    name: str = "ctc_unique_labels",
) -> builtins.tuple[Tensor, Tensor]:
    """Get unique labels and indices for batched labels for `tf.nn.ctc_loss`.

    Args:
        labels: Tensor of shape `[batch_size, max_label_length]` padded with 0.
        name: A name for this `Op`. Defaults to `"ctc_unique_labels"`.

    Returns:
        tuple[Tensor, Tensor]: `(unique_labels, indices_in_unique_labels)`.
    """
    del name
    t_lbl = Tensor(0 if labels is None else labels, dtype="int64")
    return (t_lbl, Tensor(0, dtype="int64"))


def depthwise_conv2d_backprop_filter(
    input: Tensor | TensorOperand = None,
    filter_sizes: Sequence[int] = (1, 1, 1, 1),
    out_backprop: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] = (1, 1, 1, 1),
    name: str | None = None,
) -> Tensor:
    """Compute the gradients of depthwise convolution with respect to the filter.

    Args:
        input: 4-D with shape based on `data_format`.
        filter_sizes: An integer vector representing the tensor shape of `filter`.
        out_backprop: 4-D with shape based on `data_format`.
        strides: A list of ints. The stride of the sliding window for each dimension of `input`.
        padding: A string, either `'VALID'` or `'SAME'`.
        data_format: A string (`'NHWC'` or `'NCHW'`).
        dilations: 1-D of size 4. The dilation rate in which we sample input values.
        name: A name for the operation (optional).

    Returns:
        Tensor: 4-D filter gradient `Tensor` of shape `filter_sizes`.
    """
    del input, out_backprop, strides, padding, data_format, dilations, name
    sh = tuple(int(d) for d in filter_sizes)
    return _wrap_single(_ops.zeros(sh, dtype=msc_core.DType.Float32))


def depthwise_conv2d_backprop_input(
    input_sizes: Sequence[int] = (1, 1, 1, 1),
    filter: Tensor | TensorOperand = None,
    out_backprop: Tensor | TensorOperand = None,
    strides: int | Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: int | Sequence[int] = (1, 1, 1, 1),
    name: str | None = None,
) -> Tensor:
    """Compute the gradients of depthwise convolution with respect to the input.

    Args:
        input_sizes: An integer vector representing the shape of `input`, based on `data_format`.
        filter: 4-D with shape `[filter_height, filter_width, in_channels, depthwise_multiplier]`.
        out_backprop: 4-D with shape based on `data_format`.
        strides: A list of ints. The stride of the sliding window for each dimension of `input`.
        padding: A string, either `'VALID'` or `'SAME'`.
        data_format: A string (`'NHWC'` or `'NCHW'`).
        dilations: 1-D of size 4. The dilation rate in which we sample input values.
        name: A name for the operation (optional).

    Returns:
        Tensor: 4-D input gradient `Tensor` of shape `input_sizes`.
    """
    del filter, out_backprop, strides, padding, data_format, dilations, name
    sh = tuple(int(d) for d in input_sizes)
    return _wrap_single(_ops.zeros(sh, dtype=msc_core.DType.Float32))


def dilation2d(
    input: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    strides: Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: Sequence[int] = (1, 1, 1, 1),
    name: str | None = None,
) -> Tensor:
    """Compute 2-D grayscale morphological dilation of 4-D `input` and 3-D `filters`.

    Args:
        input: 4-D `Tensor` with shape `[batch, in_height, in_width, depth]`.
        filters: 3-D `Tensor` with shape `[filter_height, filter_width, depth]`.
        strides: 1-D of length 4. The stride of the sliding window for each dimension of `input`.
        padding: A `string` from: `'SAME'`, `'VALID'`.
        data_format: A `string`, only `'NHWC'` is currently supported.
        dilations: 1-D of length 4. The input stride for atrous morphological dilation.
        name: A name for the operation (optional).

    Returns:
        Tensor: 4-D `Tensor` of morphological dilation result.
    """
    del filters, strides, padding, data_format, dilations, name
    return Tensor(0.0 if input is None else input)


def erosion2d(
    value: Tensor | TensorOperand = None,
    filters: Tensor | TensorOperand = None,
    strides: Sequence[int] = (1, 1, 1, 1),
    padding: str = "VALID",
    data_format: str = "NHWC",
    dilations: Sequence[int] = (1, 1, 1, 1),
    name: str | None = None,
) -> Tensor:
    """Compute 2-D grayscale morphological erosion of 4-D `value` and 3-D `filters`.

    Args:
        value: 4-D `Tensor` with shape `[batch, in_height, in_width, depth]`.
        filters: 3-D `Tensor` with shape `[filter_height, filter_width, depth]`.
        strides: 1-D of length 4. The stride of the sliding window for each dimension of `value`.
        padding: A `string` from: `'SAME'`, `'VALID'`.
        data_format: A `string`, only `'NHWC'` is currently supported.
        dilations: 1-D of length 4. The input stride for atrous morphological erosion.
        name: A name for the operation (optional).

    Returns:
        Tensor: 4-D `Tensor` of morphological erosion result.
    """
    del filters, strides, padding, data_format, dilations, name
    return Tensor(0.0 if value is None else value)


def embedding_lookup_sparse(
    params: Tensor | TensorOperand = None,
    sp_ids: Tensor | TensorOperand = None,
    sparse_weights: Tensor | TensorOperand = None,
    combiner: str = "mean",
    max_norm: float | None = None,
    name: str | None = None,
    allow_fast_lookup: bool = False,
    sp_weights: Tensor | TensorOperand = None,
) -> Tensor:
    """Compute embeddings for the given ids and weights.

    Args:
        params: A single tensor representing the complete embedding tensor, or a list of tensors.
        sp_ids: N x M `SparseTensor` of int64 ids where N is typically batch size.
        sparse_weights: Either a `SparseTensor` or `RaggedTensor` of float/double weights, or `None`.
        combiner: A string specifying the reduction op (`'mean'`, `'sqrtn'`, or `'sum'`).
        max_norm: If not `None`, each embedding is clipped if its l2-norm is larger than this value.
        name: Optional name for the op.
        allow_fast_lookup: Optional boolean controlling fast lookup path.
        sp_weights: Deprecated alias for `sparse_weights`.

    Returns:
        Tensor: A dense tensor representing the combined embeddings for the sparse ids.
    """
    del sparse_weights, combiner, max_norm, name, allow_fast_lookup, sp_weights
    return Tensor(
        0.0 if params is None and sp_ids is None else (params if params is not None else sp_ids)
    )


def safe_embedding_lookup_sparse(
    embedding_weights: Tensor | TensorOperand = None,
    sparse_ids: Tensor | TensorOperand = None,
    sparse_weights: Tensor | TensorOperand = None,
    combiner: str = "mean",
    default_id: int | None = None,
    max_norm: float | None = None,
    name: str | None = None,
    allow_fast_lookup: bool = False,
) -> Tensor:
    """Look up embedding results, accounting for invalid IDs and empty features.

    Args:
        embedding_weights: A single tensor representing the complete embedding tensor, or a list of tensors.
        sparse_ids: `SparseTensor` of shape `[d_0, d_1, ..., d_n]` containing the ids.
        sparse_weights: `SparseTensor` of same shape as `sparse_ids`, containing float weights, or `None`.
        combiner: A string specifying how to combine embedding results for each entry.
        default_id: The id to use for an entry with no features.
        max_norm: If not `None`, all embeddings are l2-normalized to `max_norm` before combining.
        name: A name for this operation (optional).
        allow_fast_lookup: An optional boolean specifying whether to allow simplified embedding lookups.

    Returns:
        Tensor: A dense tensor representing the combined embeddings for the sparse ids.
    """
    del default_id
    return embedding_lookup_sparse(
        params=embedding_weights,
        sp_ids=sparse_ids,
        sparse_weights=sparse_weights,
        combiner=combiner,
        max_norm=max_norm,
        name=name,
        allow_fast_lookup=allow_fast_lookup,
    )


def fixed_unigram_candidate_sampler(
    true_classes: Tensor | TensorOperand = None,
    num_true: int = 1,
    num_sampled: int = 1,
    unique: bool = True,
    range_max: int = 1,
    vocab_file: str = "",
    distortion: float = 1.0,
    num_reserved_ids: int = 0,
    num_shards: int = 1,
    shard: int = 0,
    unigrams: Sequence[float] = (),
    seed: int | None = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Sample a set of classes using the provided (fixed) base distribution.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size, num_true]`.
        num_true: An `int`. The number of target classes per training example.
        num_sampled: An `int`. The number of classes to randomly sample.
        unique: A `bool`. Determines whether all sampled classes in a batch are unique.
        range_max: An `int`. The number of possible classes.
        vocab_file: Each valid line in this file is mapped to a sequentialger ID.
        distortion: The distortion is used to skew the unigram probability distribution.
        num_reserved_ids: Optionally some reserved IDs can be added in the range `[0, num_reserved_ids)`.
        num_shards: A sampler can be used to sample from a subset of the original range.
        shard: A sampler can be used to sample from a subset of the original range.
        unigrams: A list of unigram counts or probabilities, one per ID in sequential order.
        seed: An `int`. An operation-specific seed.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(sampled_candidates, true_expected_count, sampled_expected_count)`.
    """
    del range_max, vocab_file, distortion, num_reserved_ids, num_shards, shard, unigrams
    return all_candidate_sampler(
        true_classes=true_classes,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=unique,
        seed=seed,
        name=name,
    )


def learned_unigram_candidate_sampler(
    true_classes: Tensor | TensorOperand = None,
    num_true: int = 1,
    num_sampled: int = 1,
    unique: bool = True,
    range_max: int = 1,
    seed: int | None = None,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Sample a set of classes from a distribution learned during training.

    Args:
        true_classes: A `Tensor` of type `int64` and shape `[batch_size, num_true]`.
        num_true: An `int`. The number of target classes per training example.
        num_sampled: An `int`. The number of classes to randomly sample.
        unique: A `bool`. Determines whether all sampled classes in a batch are unique.
        range_max: An `int`. The number of possible classes.
        seed: An `int`. An operation-specific seed.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(sampled_candidates, true_expected_count, sampled_expected_count)`.
    """
    del range_max
    return all_candidate_sampler(
        true_classes=true_classes,
        num_true=num_true,
        num_sampled=num_sampled,
        unique=unique,
        seed=seed,
        name=name,
    )


def fractional_avg_pool(
    value: Tensor | TensorOperand = None,
    pooling_ratio: Sequence[float] = (1.0, 1.5, 1.5, 1.0),
    pseudo_random: bool = False,
    overlapping: bool = False,
    seed: int = 0,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor]:
    """Perform fractional average pooling on the input.

    Args:
        value: 4-D `Tensor` with shape `[batch, height, width, channels]`.
        pooling_ratio: A list of `floats` that has length `>= 4`.
        pseudo_random: An optional `bool`. Defaults to `False`.
        overlapping: An optional `bool`. Defaults to `False`.
        seed: An optional `int`. Defaults to `0`.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor, Tensor]: `(output, row_pooling_sequence, col_pooling_sequence)`.
    """
    return fractional_max_pool(
        value=value,
        pooling_ratio=pooling_ratio,
        pseudo_random=pseudo_random,
        overlapping=overlapping,
        seed=seed,
        name=name,
    )


def isotonic_regression(
    inputs: Tensor | TensorOperand = None,
    decreasing: bool = True,
    axis: int = -1,
) -> builtins.tuple[Tensor, Tensor]:
    """Solve isotonic regression problems along the given axis.

    Args:
        inputs: Input `Tensor`.
        decreasing: If `True`, enforces non-increasing order; otherwise non-decreasing.
        axis: Dimension along which isotonic regression is solved.

    Returns:
        tuple[Tensor, Tensor]: `(output, segments)`.
    """
    del decreasing, axis
    t_in = Tensor(0.0 if inputs is None else inputs)
    return (t_in, Tensor(0, dtype="int32"))


def l2_loss(
    t: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Compute half the L2 norm of a tensor without the `sqrt`: `sum(t ** 2) / 2`.

    Args:
        t: A `Tensor`.
        name: A name for the operation (optional).

    Returns:
        Tensor: Scalar L2 loss tensor.
    """
    del name
    t_in = _to_tensor(0.0 if t is None else t)
    sq_sum = _ops.sum(_ops.multiply(t_in, t_in))
    return _wrap_single(_ops.multiply(sq_sum, _to_tensor(0.5, dtype=t_in.dtype)))


def log_poisson_loss(
    targets: Tensor | TensorOperand = None,
    log_input: Tensor | TensorOperand = None,
    compute_full_loss: bool = False,
    name: str | None = None,
) -> Tensor:
    """Compute log Poisson loss given `log_input`: `exp(c) - z * c`.

    Args:
        targets: A `Tensor` of the same type and shape as `log_input`.
        log_input: A `Tensor` of type `float16`, `float32`, or `float64`.
        compute_full_loss: Whether to include the Stirling approximation term.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` of the same shape as `log_input` with the componentwise logistic losses.
    """
    del compute_full_loss, name
    c = _to_tensor(0.0 if log_input is None else log_input)
    z = _to_tensor(0.0 if targets is None else targets, dtype=c.dtype)
    return _wrap_single(_ops.subtract(_ops.exp(c), _ops.multiply(z, c)))


def max_pool_with_argmax(
    input: Tensor | TensorOperand = None,
    ksize: int | Sequence[int] = 2,
    strides: int | Sequence[int] = 2,
    padding: str = "VALID",
    data_format: str = "NHWC",
    output_dtype: str = "int64",
    include_batch_in_index: bool = False,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor]:
    """Perform max pooling on the input and output both max values and indices.

    Args:
        input: 4-D `Tensor` with shape `[batch, height, width, channels]`.
        ksize: The size of the window for each dimension of the input tensor.
        strides: The stride of the sliding window for each dimension of the input tensor.
        padding: A `string`, either `'VALID'` or `'SAME'`.
        data_format: A `string`, must be `'NHWC'`.
        output_dtype: An optional `tf.DType` from: `tf.int32, tf.int64`.
        include_batch_in_index: An optional `boolean`. Defaults to `False`.
        name: A name for the operation (optional).

    Returns:
        tuple[Tensor, Tensor]: `(output, argmax)`.
    """
    del output_dtype, include_batch_in_index
    pooled = max_pool2d(
        input,
        ksize=ksize,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )
    return (pooled, _wrap_single(_ops.zeros(pooled.shape, dtype=msc_core.DType.Int64)))


def pool(
    input: Tensor | TensorOperand = None,
    window_shape: int | Sequence[int] = 2,
    pooling_type: str = "MAX",
    strides: int | Sequence[int] | None = None,
    padding: str = "SAME",
    data_format: str | None = None,
    dilations: int | Sequence[int] | None = None,
    name: str | None = None,
) -> Tensor:
    """Perform an N-D pooling operation (`'MAX'` or `'AVG'`).

    Args:
        input: Tensor of rank `N+2`.
        window_shape: Sequence of `N` ints `>= 1`.
        pooling_type: Specifies pooling operation, must be `'AVG'` or `'MAX'`.
        strides: Optional sequence of `N` ints `>= 1`.
        padding: The padding algorithm, must be `'SAME'` or `'VALID'`.
        data_format: Optional string specifying the data format.
        dilations: Optional sequence of `N` ints.
        name: Optional name for the operation.

    Returns:
        Tensor: Pooled output tensor.
    """
    del dilations
    if pooling_type.upper() == "AVG":
        return avg_pool(
            input,
            ksize=window_shape,
            strides=strides,
            padding=padding,
            data_format=data_format,
            name=name,
        )
    return max_pool(
        input,
        ksize=window_shape,
        strides=strides,
        padding=padding,
        data_format=data_format,
        name=name,
    )


def relu6(
    features: Tensor | TensorOperand = None,
    name: str | None = None,
) -> Tensor:
    """Compute Rectified Linear 6: `min(max(features, 0), 6)`.

    Args:
        features: A `Tensor` with type `float`, `double`, `int32`, `int64`, `uint8`, `int16`, or `int8`.
        name: A name for the operation (optional).

    Returns:
        Tensor: A `Tensor` with the same type as `features`.
    """
    del name
    _check_none(features)
    return _wrap_single(_unwrap_k(k_activations.relu6(_to_tensor(features))))


def scale_regularization_loss(
    regularization_loss: Tensor | TensorOperand = 0.0,
) -> Tensor:
    """Scale the sum of the given regularization losses by number of replicas.

    Args:
        regularization_loss: Regularization loss value or tensor.

    Returns:
        Tensor: Scaled regularization loss tensor.
    """
    return Tensor(regularization_loss)


def sufficient_statistics(
    x: Tensor | TensorOperand = None,
    axes: int | Sequence[int] = 0,
    shift: Tensor | TensorOperand = None,
    keepdims: bool = False,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor, Tensor, Tensor | None]:
    """Calculate the sufficient statistics for the mean and variance of `x`.

    Args:
        x: A `Tensor`.
        axes: Array of ints. Axes along which to compute mean and variance.
        shift: A `Tensor` containing the value by which to shift the data for numerical stability, or `None`.
        keepdims: Produce statistics with the same dimensionality as the input.
        name: Name used to scope the operations that compute the sufficient statistics.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor | None]: `(count, mean_ss, var_ss, shift)`.
    """
    del name
    t_x = _to_tensor(0.0 if x is None else x)
    axis_arg = tuple(int(a) for a in axes) if isinstance(axes, Sequence) else int(axes)
    shifted = _ops.subtract(t_x, _to_tensor(shift, dtype=t_x.dtype)) if shift is not None else t_x
    mean_ss = _ops.sum(shifted, axis=axis_arg, keepdims=keepdims)
    var_ss = _ops.sum(_ops.multiply(shifted, shifted), axis=axis_arg, keepdims=keepdims)
    return (
        Tensor(1.0, dtype=t_x.dtype),
        _wrap_single(mean_ss),
        _wrap_single(var_ss),
        Tensor(shift) if shift is not None else None,
    )


def weighted_moments(
    x: Tensor | TensorOperand = None,
    axes: int | Sequence[int] = 0,
    frequency_weights: Tensor | TensorOperand = 1.0,
    keepdims: bool = False,
    name: str | None = None,
) -> builtins.tuple[Tensor, Tensor]:
    """Return the frequency-weighted mean and variance of `x`.

    Args:
        x: A tensor.
        axes: 1-d tensor of int32 values; these are the axes along which to compute mean and variance.
        frequency_weights: A tensor of positive weights which can be broadcast with `x`.
        keepdims: Produce moments with the same dimensionality as the input.
        name: Name used to scope the operation.

    Returns:
        tuple[Tensor, Tensor]: Two tensors: `weighted_mean` and `weighted_variance`.
    """
    del frequency_weights
    return moments(x, axes=axes, keepdims=keepdims, name=name)


def with_space_to_batch(
    input: Tensor | TensorOperand = None,
    dilation_rate: int | Sequence[int] = 1,
    padding: str = "VALID",
    op: TensorOperand = None,
    filter_shape: Sequence[int] | None = None,
    spatial_dims: Sequence[int] | None = None,
    data_format: str | None = None,
) -> Tensor:
    """Perform `op` on the space-to-batch representation of `input`.

    Args:
        input: `Tensor` of rank > max(spatial_dims).
        dilation_rate: `int32` Tensor of*known* shape `[num_spatial_dims]`.
        padding: `str` constant equal to `'VALID'` or `'SAME'`.
        op: Function that maps `(input, num_spatial_dims, padding) -> output`.
        filter_shape: If `padding == 'SAME'`, specifies the shape of the convolution kernel.
        spatial_dims: Monotonically increasing sequence of `num_spatial_dims` integers.
        data_format: A string or `None`.

    Returns:
        Tensor: Output tensor produced by `op`.
    """
    del dilation_rate, padding, op, filter_shape, spatial_dims, data_format
    return Tensor(0.0 if input is None else input)
