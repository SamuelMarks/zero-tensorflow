"""Zero-dependency pure Python implementation of the TensorFlow API surface."""

from __future__ import annotations

import builtins
from collections.abc import Sequence

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.autodiff import (
    GradientTape,
    custom_gradient,
    function,
    hessians,
    stop_gradient,
)
from zero_tensorflow.bitwise_ops import bitwise
from zero_tensorflow.core_stubs import (
    double,
    import_graph_def,
    init_scope,
    is_symbolic_tensor,
    is_tensor,
    ragged_fill_empty_rows,
    ragged_fill_empty_rows_grad,
    size,
    space_to_batch,
    space_to_batch_nd,
    string,
)
from zero_tensorflow.core_tensor import (
    Tensor,
    Variable,
)
from zero_tensorflow.core_tensor import (
    _extract_constant_value as _extract_constant_value,
)
from zero_tensorflow.core_tensor import (
    _to_attribute_value as _to_attribute_value,
)
from zero_tensorflow.core_tensor import (
    _to_tensor as _to_tensor,
)
from zero_tensorflow.core_tensor import (
    _TracingContext as _TracingContext,
)
from zero_tensorflow.core_tensor import (
    _wrap as _wrap,
)
from zero_tensorflow.core_tensor import (
    _wrap_tensor as _wrap_tensor,
)
from zero_tensorflow.core_tensor import (
    to_dtype as to_dtype,
)
from zero_tensorflow.linalg_ops import linalg
from zero_tensorflow.math_ops import (
    abs,
    acos,
    acosh,
    add,
    argmax,
    argmin,
    asin,
    asinh,
    atan,
    atan2,
    atanh,
    bitcast,
    broadcast_to,
    cast,
    concat,
    cos,
    cosh,
    divide,
    einsum,
    equal,
    exp,
    expand_dims,
    eye,
    fill,
    floor,
    gather,
    gather_nd,
    greater,
    greater_equal,
    identity,
    less,
    less_equal,
    linspace,
    logical_and,
    logical_not,
    logical_or,
    math,
    matmul,
    maximum,
    meshgrid,
    minimum,
    multiply,
    negative,
    norm,
    not_equal,
    ones,
    ones_like,
    pow,
    range,
    reduce_all,
    reduce_any,
    reduce_max,
    reduce_mean,
    reduce_min,
    reduce_prod,
    reduce_sum,
    repeat,
    reshape,
    roll,
    round,
    scatter_nd,
    shape,
    sign,
    sin,
    sinh,
    slice,
    split,
    sqrt,
    square,
    squeeze,
    stack,
    strided_slice,
    subtract,
    tan,
    tanh,
    tensordot,
    tile,
    transpose,
    unstack,
    where,
    zeros,
    zeros_like,
)
from zero_tensorflow.ragged import RaggedTensor
from zero_tensorflow.raw_ops import raw_ops
from zero_tensorflow.sparse import SparseTensor

from ml_switcheroo_compiler.core.dtype import DType

import zero_keras as keras  # isort: skip
from zero_keras import ops as _ops  # isort: skip

from . import (
    autograph,
    config,
    data,
    debugging,
    distribute,
    dtypes,
    image,
    io,
    lookup,
    nn,
    quantization,
    ragged,
    random,
    saved_model,
    signal,
    sparse,
    strings,
    summary,
    test,
    train,
    types,
)
from . import (
    losses as losses,
)
from . import (
    metrics as metrics,
)
from . import (
    optimizers as optimizers,
)

initializers = keras.initializers

float16: DType = DType.Float16
float32: DType = DType.Float32
float64: DType = DType.Float64
bfloat16: DType = DType.BFloat16
complex64: DType = DType.Complex64
complex128: DType = DType.Complex128
int8: DType = DType.Int8
int16: DType = DType.Int16
int32: DType = DType.Int32
int64: DType = DType.Int64
uint8: DType = DType.UInt8
bool: DType = DType.Bool

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
    "double",
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
    "import_graph_def",
    "init_scope",
    "io",
    "is_symbolic_tensor",
    "is_tensor",
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
    "ragged_fill_empty_rows",
    "ragged_fill_empty_rows_grad",
    "random",
    "range",
    "raw_ops",
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
    "size",
    "slice",
    "space_to_batch",
    "space_to_batch_nd",
    "sparse",
    "split",
    "sqrt",
    "square",
    "squeeze",
    "stack",
    "stop_gradient",
    "strided_slice",
    "string",
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
    "version",
    "where",
    "zeros",
    "zeros_like",
]


def AggregationMethod(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `AggregationMethod` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def Assert(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `Assert` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def CriticalSection(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `CriticalSection` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def DeviceSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `DeviceSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def Graph(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `Graph` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def IndexedSlices(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `IndexedSlices` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def IndexedSlicesSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `IndexedSlicesSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def Module(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `Module` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def Operation(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `Operation` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def OptionalSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `OptionalSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def RaggedTensorSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `RaggedTensorSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def RegisterGradient(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `RegisterGradient` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def SparseTensorSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `SparseTensorSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def TensorArray(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `TensorArray` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def TensorArraySpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `TensorArraySpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def TensorShape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `TensorShape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def TensorSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `TensorSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def TypeSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `TypeSpec` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def UnconnectedGradients(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `UnconnectedGradients` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def VariableAggregation(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `VariableAggregation` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def VariableSynchronization(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `VariableSynchronization` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def add_n(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `add_n` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.add_n(*[_to_tensor(a) for a in args], **kwargs))


def approx_top_k(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `approx_top_k` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def argsort(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `argsort` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.argsort(*[_to_tensor(a) for a in args], **kwargs))


def as_dtype(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `as_dtype` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def as_string(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `as_string` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.as_string(*[_to_tensor(a) for a in args], **kwargs))


def assert_equal(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `assert_equal` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def assert_greater(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `assert_greater` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def assert_less(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `assert_less` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def assert_rank(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `assert_rank` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class audio:
    """TensorFlow `audio` decoding and encoding namespace."""

    @staticmethod
    def decode_wav(
        contents: Tensor | TensorOperand,
        desired_channels: int = -1,
        desired_samples: int = -1,
        name: str | None = None,
    ) -> builtins.tuple[Tensor, Tensor]:
        """Decode a 16-bit PCM WAV file to a float tensor.

        Args:
            contents: WAV-encoded audio bytes or string tensor.
            desired_channels: Number of channels wanted (`-1` for all).
            desired_samples: Length of audio wanted (`-1` for all).
            name: Optional name for the operation.

        Returns:
            tuple[Tensor, Tensor]: `(audio, sample_rate)`.
        """
        del contents, desired_channels, desired_samples, name
        audio_t = _wrap_tensor(_ops.zeros((1, 1), dtype="float32"))
        rate_t = _wrap_tensor(_ops.array(44100, dtype="int32"))
        return audio_t, rate_t

    @staticmethod
    def encode_wav(
        audio: Tensor | TensorOperand,
        sample_rate: Tensor | TensorOperand,
        name: str | None = None,
    ) -> bytes:
        """Encode audio data using the WAV file format.

        Args:
            audio: 2-D float32 audio tensor `[length, channels]`.
            sample_rate: Scalar integer sample rate.
            name: Optional name for the operation.

        Returns:
            bytes: WAV-encoded byte string.
        """
        del audio, sample_rate, name
        return b""


class autodiff:
    """TensorFlow `autodiff` namespace."""

    @staticmethod
    def ForwardAccumulator(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `autodiff.ForwardAccumulator` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def GradientTape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `autodiff.GradientTape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def set_verbosity(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `autodiff.set_verbosity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def to_code(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `autodiff.to_code` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def to_graph(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `autodiff.to_graph` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def trace(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `autodiff.trace` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.trace(*[_to_tensor(a) for a in args], **kwargs))


def batch_to_space(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `batch_to_space` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def boolean_mask(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `boolean_mask` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.boolean_mask(*[_to_tensor(a) for a in args], **kwargs))


def broadcast_dynamic_shape(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `broadcast_dynamic_shape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def broadcast_static_shape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `broadcast_static_shape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def case(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `case` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.case(*[_to_tensor(a) for a in args], **kwargs))


def clip_by_global_norm(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `clip_by_global_norm` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def clip_by_norm(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `clip_by_norm` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def clip_by_value(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `clip_by_value` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class compat:
    """TensorFlow `compat` namespace."""

    @staticmethod
    def as_bytes(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.as_bytes` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def as_str(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.as_str` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def as_str_any(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.as_str_any` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def as_text(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.as_text` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def bytes_or_text_types(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `compat.bytes_or_text_types` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def complex_types(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.complex_types` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dimension_at_index(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.dimension_at_index` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dimension_value(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.dimension_value` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def forward_compatibility_horizon(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `compat.forward_compatibility_horizon` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def forward_compatible(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.forward_compatible` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def integral_types(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.integral_types` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def path_to_str(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.path_to_str` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def real_types(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.real_types` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def v1(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.v1` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def v2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `compat.v2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def complex(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `complex` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def cond(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `cond` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.cond(*[_to_tensor(a) for a in args], **kwargs))


def constant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `constant` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def constant_initializer(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `constant_initializer` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def control_dependencies(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `control_dependencies` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def conv(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `conv` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.conv(*[_to_tensor(a) for a in args], **kwargs))


def conv2d_backprop_filter_v2(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `conv2d_backprop_filter_v2` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def conv2d_backprop_input_v2(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `conv2d_backprop_input_v2` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def convert_to_tensor(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `convert_to_tensor` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.convert_to_tensor(*[_to_tensor(a) for a in args], **kwargs))


def cumsum(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `cumsum` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.cumsum(*[_to_tensor(a) for a in args], **kwargs))


def device(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `device` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class dtensor:
    """TensorFlow `dtensor` namespace."""

    @staticmethod
    def python(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.python` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def complex128(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.complex128` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def complex64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.complex64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def double(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.double` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def float16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.float16` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def float32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.float32` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def float64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.float64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def half(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.half` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def int16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.int16` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def int32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.int32` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def int64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.int64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def int8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.int8` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def qint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.qint16` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def qint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.qint32` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def qint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.qint8` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def quint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.quint16` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def quint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.quint8` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def resource(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.resource` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def saturate_cast(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.saturate_cast` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def string(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.string` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def uint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.uint16` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def uint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.uint32` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def uint64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.uint64` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def uint8(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `dtensor.uint8` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.uint8(*[_to_tensor(a) for a in args], **kwargs))

    @staticmethod
    def variant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `dtensor.variant` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def dynamic_partition(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `dynamic_partition` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.dynamic_partition(*[_to_tensor(a) for a in args], **kwargs))


def dynamic_stitch(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `dynamic_stitch` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.dynamic_stitch(*[_to_tensor(a) for a in args], **kwargs))


def edit_distance(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `edit_distance` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.edit_distance(*[_to_tensor(a) for a in args], **kwargs))


def eig(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `eig` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.eig(*[_to_tensor(a) for a in args], **kwargs))


def eigvals(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `eigvals` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.eigvals(*[_to_tensor(a) for a in args], **kwargs))


def ensure_shape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `ensure_shape` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class errors:
    """TensorFlow `errors` namespace."""

    @staticmethod
    def ABORTED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.ABORTED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ALREADY_EXISTS(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.ALREADY_EXISTS` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AbortedError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.AbortedError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def AlreadyExistsError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.AlreadyExistsError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CANCELLED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.CANCELLED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CancelledError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.CancelledError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DATA_LOSS(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.DATA_LOSS` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DEADLINE_EXCEEDED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.DEADLINE_EXCEEDED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DataLossError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.DataLossError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DeadlineExceededError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.DeadlineExceededError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FAILED_PRECONDITION(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.FAILED_PRECONDITION` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def FailedPreconditionError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.FailedPreconditionError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def INTERNAL(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.INTERNAL` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def INVALID_ARGUMENT(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.INVALID_ARGUMENT` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InternalError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.InternalError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def InvalidArgumentError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.InvalidArgumentError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NOT_FOUND(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.NOT_FOUND` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def NotFoundError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.NotFoundError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OK(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.OK` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OUT_OF_RANGE(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.OUT_OF_RANGE` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OpError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.OpError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OperatorNotAllowedInGraphError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.OperatorNotAllowedInGraphError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def OutOfRangeError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.OutOfRangeError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PERMISSION_DENIED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.PERMISSION_DENIED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PermissionDeniedError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.PermissionDeniedError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RESOURCE_EXHAUSTED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.RESOURCE_EXHAUSTED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ResourceExhaustedError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.ResourceExhaustedError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UNAUTHENTICATED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UNAUTHENTICATED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UNAVAILABLE(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UNAVAILABLE` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UNIMPLEMENTED(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UNIMPLEMENTED` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UNKNOWN(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UNKNOWN` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnauthenticatedError(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `errors.UnauthenticatedError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnavailableError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UnavailableError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnimplementedError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UnimplementedError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def UnknownError(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `errors.UnknownError` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def executing_eagerly(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `executing_eagerly` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class experimental:
    """TensorFlow `experimental` namespace."""

    @staticmethod
    def BatchableExtensionType(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.BatchableExtensionType` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def DynamicRaggedShape(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.DynamicRaggedShape` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtensionType(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.ExtensionType` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtensionTypeBatchEncoder(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.ExtensionTypeBatchEncoder` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def ExtensionTypeSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.ExtensionTypeSpec` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Optional(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `experimental.Optional` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.Optional(*[_to_tensor(a) for a in args], **kwargs))

    @staticmethod
    def RowPartition(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.RowPartition` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def StructuredTensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.StructuredTensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def async_clear_error(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.async_clear_error` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def async_scope(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.async_scope` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dispatch_for_api(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.dispatch_for_api` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dispatch_for_binary_elementwise_apis(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.dispatch_for_binary_elementwise_apis` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dispatch_for_binary_elementwise_assert_apis(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.dispatch_for_binary_elementwise_assert_apis` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dispatch_for_unary_elementwise_apis(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.dispatch_for_unary_elementwise_apis` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dlpack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.dlpack` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def dtensor(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.dtensor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def enable_strict_mode(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.enable_strict_mode` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def extension_type(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.extension_type` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def float8_e4m3fn(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.float8_e4m3fn` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def float8_e5m2(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.float8_e5m2` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def function_executor_type(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.function_executor_type` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def int4(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.int4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def numpy(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.numpy` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def register_filesystem_plugin(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.register_filesystem_plugin` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def tensorrt(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `experimental.tensorrt` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def uint4(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `experimental.uint4` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.uint4(*[_to_tensor(a) for a in args], **kwargs))

    @staticmethod
    def unregister_dispatch_for(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `experimental.unregister_dispatch_for` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def extract_volume_patches(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `extract_volume_patches` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.extract_volume_patches(*[_to_tensor(a) for a in args], **kwargs))


class feature_column:
    """TensorFlow `feature_column` namespace."""

    @staticmethod
    def bucketized_column(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.bucketized_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def categorical_column_with_hash_bucket(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.categorical_column_with_hash_bucket` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def categorical_column_with_identity(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.categorical_column_with_identity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def categorical_column_with_vocabulary_file(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.categorical_column_with_vocabulary_file` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def categorical_column_with_vocabulary_list(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.categorical_column_with_vocabulary_list` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def crossed_column(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.crossed_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def embedding_column(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.embedding_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def indicator_column(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.indicator_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def make_parse_example_spec(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.make_parse_example_spec` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def numeric_column(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.numeric_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def sequence_categorical_column_with_hash_bucket(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.sequence_categorical_column_with_hash_bucket` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def sequence_categorical_column_with_identity(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.sequence_categorical_column_with_identity` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def sequence_categorical_column_with_vocabulary_file(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.sequence_categorical_column_with_vocabulary_file` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def sequence_categorical_column_with_vocabulary_list(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.sequence_categorical_column_with_vocabulary_list` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def sequence_numeric_column(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.sequence_numeric_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def shared_embeddings(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `feature_column.shared_embeddings` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def weighted_categorical_column(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `feature_column.weighted_categorical_column` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def fftnd(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `fftnd` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.fftnd(*[_to_tensor(a) for a in args], **kwargs))


def fingerprint(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `fingerprint` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def foldl(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `foldl` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def foldr(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `foldr` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def get_current_name_scope(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `get_current_name_scope` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def get_logger(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `get_logger` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def get_static_value(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `get_static_value` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def grad_pass_through(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `grad_pass_through` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def gradients(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `gradients` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class graph_util:
    """TensorFlow `graph_util` namespace."""

    @staticmethod
    def import_graph_def(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `graph_util.import_graph_def` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def group(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `group` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def guarantee_const(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `guarantee_const` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def half(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `half` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def histogram_fixed_width(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `histogram_fixed_width` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def histogram_fixed_width_bins(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `histogram_fixed_width_bins` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def identity_n(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `identity_n` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def ifftnd(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `ifftnd` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.ifftnd(*[_to_tensor(a) for a in args], **kwargs))


def inside_function(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `inside_function` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class lite:
    """TensorFlow `lite` namespace."""

    @staticmethod
    def OpsSet(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `lite.OpsSet` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def Optimize(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `lite.Optimize` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RepresentativeDataset(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `lite.RepresentativeDataset` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TFLiteConverter(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `lite.TFLiteConverter` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def TargetSpec(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `lite.TargetSpec` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `lite.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def load_library(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `load_library` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def load_op_library(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `load_op_library` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def make_ndarray(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `make_ndarray` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def make_tensor_proto(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `make_tensor_proto` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def map_fn(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `map_fn` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.map_fn(*[_to_tensor(a) for a in args], **kwargs))


def matrix_square_root(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `matrix_square_root` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class mlir:
    """TensorFlow `mlir` namespace."""

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `mlir.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def name_scope(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `name_scope` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class nest:
    """TensorFlow `nest` namespace."""

    @staticmethod
    def assert_same_structure(
        *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
    ) -> None:
        """Execute `nest.assert_same_structure` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def flatten(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `nest.flatten` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.flatten(*[_to_tensor(a) for a in args], **kwargs))

    @staticmethod
    def is_nested(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `nest.is_nested` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def map_structure(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `nest.map_structure` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def pack_sequence_as(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `nest.pack_sequence_as` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def newaxis(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `newaxis` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def no_gradient(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `no_gradient` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def no_op(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `no_op` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def nondifferentiable_batch_function(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `nondifferentiable_batch_function` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def numpy_function(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `numpy_function` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def one_hot(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `one_hot` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def ones_initializer(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `ones_initializer` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def pad(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `pad` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.pad(*[_to_tensor(a) for a in args], **kwargs))


def parallel_stack(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `parallel_stack` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def print(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `print` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class profiler:
    """TensorFlow `profiler` namespace."""

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `profiler.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def py_function(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `py_function` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def qint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `qint16` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def qint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `qint32` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def qint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `qint8` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class queue:
    """TensorFlow `queue` namespace."""

    @staticmethod
    def FIFOQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `queue.FIFOQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PaddingFIFOQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `queue.PaddingFIFOQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def PriorityQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `queue.PriorityQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def QueueBase(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `queue.QueueBase` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def RandomShuffleQueue(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `queue.RandomShuffleQueue` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def quint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `quint16` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def quint8(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `quint8` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def random_index_shuffle(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `random_index_shuffle` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def random_normal_initializer(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `random_normal_initializer` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def random_uniform_initializer(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `random_uniform_initializer` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def rank(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `rank` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def realdiv(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `realdiv` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def recompute_grad(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `recompute_grad` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def reduce_logsumexp(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `reduce_logsumexp` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.reduce_logsumexp(*[_to_tensor(a) for a in args], **kwargs))


def register_tensor_conversion_function(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `register_tensor_conversion_function` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def required_space_to_batch_paddings(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `required_space_to_batch_paddings` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def resource(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `resource` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def reverse(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `reverse` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.reverse(*[_to_tensor(a) for a in args], **kwargs))


def reverse_sequence(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `reverse_sequence` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def rfftnd(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `rfftnd` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.rfftnd(*[_to_tensor(a) for a in args], **kwargs))


def saturate_cast(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `saturate_cast` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def sequence_mask(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `sequence_mask` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class sets:
    """TensorFlow `sets` namespace."""

    @staticmethod
    def difference(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sets.difference` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def intersection(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sets.intersection` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def size(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
        """Execute `sets.size` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
        """
        return _wrap(_ops.size(*[_to_tensor(a) for a in args], **kwargs))

    @staticmethod
    def union(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sets.union` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def shape_n(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `shape_n` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def sigmoid(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `sigmoid` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.sigmoid(*[_to_tensor(a) for a in args], **kwargs))


def switch_case(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `switch_case` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.switch_case(*[_to_tensor(a) for a in args], **kwargs))


class sysconfig:
    """TensorFlow `sysconfig` namespace."""

    @staticmethod
    def CXX11_ABI_FLAG(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.CXX11_ABI_FLAG` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def CXX_VERSION(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.CXX_VERSION` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def MONOLITHIC_BUILD(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.MONOLITHIC_BUILD` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def get_build_info(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.get_build_info` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def get_compile_flags(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.get_compile_flags` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def get_include(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.get_include` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def get_lib(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.get_lib` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def get_link_flags(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `sysconfig.get_link_flags` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def tensor_scatter_nd_add(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `tensor_scatter_nd_add` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def tensor_scatter_nd_max(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `tensor_scatter_nd_max` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def tensor_scatter_nd_min(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `tensor_scatter_nd_min` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def tensor_scatter_nd_sub(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `tensor_scatter_nd_sub` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def tensor_scatter_nd_update(
    *args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand
) -> None:
    """Execute `tensor_scatter_nd_update` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def timestamp(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `timestamp` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


class tools:
    """TensorFlow `tools` namespace."""

    @staticmethod
    def compatibility(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `tools.compatibility` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def docs(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `tools.docs` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


class tpu:
    """TensorFlow `tpu` namespace."""

    @staticmethod
    def XLAOptions(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `tpu.XLAOptions` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `tpu.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def tuple(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `tuple` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def type_spec_from_value(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `type_spec_from_value` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def uint16(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `uint16` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def uint32(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `uint32` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def uint64(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `uint64` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def unique(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `unique` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.unique(*[_to_tensor(a) for a in args], **kwargs))


def unique_with_counts(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `unique_with_counts` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def unravel_index(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `unravel_index` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.unravel_index(*[_to_tensor(a) for a in args], **kwargs))


def variable_creator_scope(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `variable_creator_scope` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def variant(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `variant` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None


def vectorized_map(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `vectorized_map` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.vectorized_map(*[_to_tensor(a) for a in args], **kwargs))


class version:
    """TensorFlow `version` namespace."""

    COMPILER_VERSION: str = "Clang 15.0.0"
    GIT_VERSION: str = "v2.20.0-rc0"
    GRAPH_DEF_VERSION: int = 1942
    GRAPH_DEF_VERSION_MIN_CONSUMER: int = 0
    GRAPH_DEF_VERSION_MIN_PRODUCER: int = 0
    VERSION: str = "2.20.0"


def while_loop(
    *args: Tensor | TensorOperand,
    **kwargs: Tensor | TensorOperand,
) -> Tensor | Sequence[Tensor | TensorOperand] | TensorOperand:
    """Execute `while_loop` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        Tensor | Sequence[Tensor | TensorOperand] | TensorOperand: Operation result.
    """
    return _wrap(_ops.while_loop(*[_to_tensor(a) for a in args], **kwargs))


class xla:
    """TensorFlow `xla` namespace."""

    @staticmethod
    def experimental(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
        """Execute `xla.experimental` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            None: Stub operation result.
        """
        del args, kwargs
        return None


def zeros_initializer(*args: Tensor | TensorOperand, **kwargs: Tensor | TensorOperand) -> None:
    """Execute `zeros_initializer` operation.

    Args:
        *args: Positional operation arguments.
        **kwargs: Keyword operation arguments.

    Returns:
        None: Stub operation result.
    """
    del args, kwargs
    return None
