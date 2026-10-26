"""TensorFlow data types (`dtypes`) module."""

from __future__ import annotations

from zero_tensorflow._typing import DTypeLike, TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor, to_dtype

import ml_switcheroo_compiler.core.dtype as dt

from zero_keras import ops as _ops  # isort: skip

DType = dt.DType
as_dtype = to_dtype

bfloat16: dt.DType = dt.DType.BFloat16
bool: dt.DType = dt.DType.Bool
complex64: dt.DType = dt.DType.Complex64
complex128: dt.DType = dt.DType.Complex128
double: dt.DType = dt.DType.Float64
float16: dt.DType = dt.DType.Float16
float32: dt.DType = dt.DType.Float32
float64: dt.DType = dt.DType.Float64
half: dt.DType = dt.DType.Float16
int8: dt.DType = dt.DType.Int8
int16: dt.DType = dt.DType.Int16
int32: dt.DType = dt.DType.Int32
int64: dt.DType = dt.DType.Int64
qint8: dt.DType = dt.DType.Int8
qint16: dt.DType = dt.DType.Int16
qint32: dt.DType = dt.DType.Int32
quint8: dt.DType = dt.DType.UInt8
quint16: dt.DType = dt.DType.UInt16
resource: dt.DType = dt.DType.Int64
string: dt.DType = dt.DType.String
uint8: dt.DType = dt.DType.UInt8
uint16: dt.DType = dt.DType.UInt16
uint32: dt.DType = dt.DType.UInt32
uint64: dt.DType = dt.DType.UInt64
variant: dt.DType = dt.DType.Int64

QUANTIZED_DTYPES: frozenset[dt.DType] = frozenset({qint8, qint16, qint32, quint8, quint16})


def cast(
    x: Tensor | TensorOperand,
    dtype: DTypeLike | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Cast a tensor `x` to a new `dtype`.

    Args:
        x: Input tensor or scalar.
        dtype: Target data type.
        name: Optional operation name.

    Returns:
        Tensor: Cast tensor.
    """
    del name
    target_dt = to_dtype(dtype)
    return _wrap_tensor(_ops.cast(_to_tensor(x), target_dt.value))


def saturate_cast(
    value: Tensor | TensorOperand,
    dtype: DTypeLike | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Perform a saturating cast of `value` to `dtype`.

    Args:
        value: Input tensor or scalar.
        dtype: Target data type.
        name: Optional operation name.

    Returns:
        Tensor: Cast tensor.
    """
    del name
    target_dt = to_dtype(dtype)
    return _wrap_tensor(_ops.cast(_to_tensor(value), target_dt.value))


def complex(
    real: Tensor | TensorOperand,
    imag: Tensor | TensorOperand,
    name: str | None = None,
) -> Tensor:
    """Convert two real numbers/tensors to a complex tensor.

    Args:
        real: Real component tensor or scalar.
        imag: Imaginary component tensor or scalar.
        name: Optional operation name.

    Returns:
        Tensor: Complex tensor.
    """
    del name
    return _wrap_tensor(_ops.add(_to_tensor(real), _ops.multiply(_to_tensor(imag), 1j)))


class experimental:
    """Experimental low-bitwidth data types."""

    float8_e4m3fn: dt.DType = dt.DType.Float8E4M3FN
    float8_e5m2: dt.DType = dt.DType.Float8E5M2
    int4: dt.DType = dt.DType.Int4
    uint4: dt.DType = dt.DType.UInt4


__all__ = [
    "DType",
    "QUANTIZED_DTYPES",
    "as_dtype",
    "bfloat16",
    "bool",
    "cast",
    "complex",
    "complex128",
    "complex64",
    "double",
    "experimental",
    "float16",
    "float32",
    "float64",
    "half",
    "int16",
    "int32",
    "int64",
    "int8",
    "qint16",
    "qint32",
    "qint8",
    "quint16",
    "quint8",
    "resource",
    "saturate_cast",
    "string",
    "uint16",
    "uint32",
    "uint64",
    "uint8",
    "variant",
]
