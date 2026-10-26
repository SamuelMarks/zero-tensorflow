"""TensorFlow data types (`dtypes`) module."""

from __future__ import annotations

import builtins

import zero_tensorflow.core_tensor as _ct
from zero_tensorflow._typing import DTypeLike, TensorCompatible, TensorOperand

import ml_switcheroo_compiler.core.dtype as dt

from zero_keras import ops as _ops  # isort: skip

_FLOATING_NAMES: frozenset[str] = frozenset(
    {
        "float16",
        "float32",
        "float64",
        "bfloat16",
        "float8_e4m3fn",
        "float8_e5m2",
        "float8_e4m3fnuz",
        "float8_e4m3b11fnuz",
        "float8_e5m2fnuz",
    }
)

_INTEGER_NAMES: frozenset[str] = frozenset(
    {
        "int4",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint4",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }
)

_UNSIGNED_NAMES: frozenset[str] = frozenset(
    {
        "uint4",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "quint8",
        "quint16",
    }
)

_COMPLEX_NAMES: frozenset[str] = frozenset({"complex64", "complex128"})

_DTYPE_VALUE_LIMITS: dict[str, tuple[int | float, int | float]] = {
    "bool": (False, True),
    "int4": (-8, 7),
    "uint4": (0, 15),
    "int8": (-128, 127),
    "qint8": (-128, 127),
    "uint8": (0, 255),
    "quint8": (0, 255),
    "int16": (-32768, 32767),
    "qint16": (-32768, 32767),
    "uint16": (0, 65535),
    "quint16": (0, 65535),
    "int32": (-2147483648, 2147483647),
    "qint32": (-2147483648, 2147483647),
    "uint32": (0, 4294967295),
    "int64": (-9223372036854775808, 9223372036854775807),
    "uint64": (0, 18446744073709551615),
    "float16": (-65504.0, 65504.0),
    "bfloat16": (-3.3895313892515355e38, 3.3895313892515355e38),
    "float32": (-3.4028234663852886e38, 3.4028234663852886e38),
    "float64": (-1.7976931348623157e308, 1.7976931348623157e308),
    "float8_e4m3fn": (-448.0, 448.0),
    "float8_e5m2": (-57344.0, 57344.0),
    "float8_e4m3fnuz": (-240.0, 240.0),
    "float8_e4m3b11fnuz": (-30.0, 30.0),
    "float8_e5m2fnuz": (-57344.0, 57344.0),
}

_DTYPE_SIZES: dict[str, int] = {
    "bool": 1,
    "int4": 1,
    "uint4": 1,
    "int8": 1,
    "uint8": 1,
    "qint8": 1,
    "quint8": 1,
    "float8_e4m3fn": 1,
    "float8_e5m2": 1,
    "float8_e4m3fnuz": 1,
    "float8_e4m3b11fnuz": 1,
    "float8_e5m2fnuz": 1,
    "float16": 2,
    "bfloat16": 2,
    "int16": 2,
    "uint16": 2,
    "qint16": 2,
    "quint16": 2,
    "float32": 4,
    "int32": 4,
    "uint32": 4,
    "qint32": 4,
    "float64": 8,
    "int64": 8,
    "uint64": 8,
    "complex64": 8,
    "resource": 8,
    "variant": 8,
    "complex128": 16,
    "string": 0,
}


class DType(str):
    """TensorFlow data type descriptor supporting compiler and TF type algebra."""

    _tf_name: str
    _msc_dtype: dt.DType
    _is_quantized_flag: builtins.bool
    _enum_id: int

    Float16: DType
    Float32: DType
    Float64: DType
    BFloat16: DType
    Float8E4M3FN: DType
    Float8E5M2: DType
    Float8E4M3FNUZ: DType
    Float8E4M3B11FNUZ: DType
    Float8E5M2FNUZ: DType
    Complex64: DType
    Complex128: DType
    Int4: DType
    Int8: DType
    Int16: DType
    Int32: DType
    Int64: DType
    UInt4: DType
    UInt8: DType
    UInt16: DType
    UInt32: DType
    UInt64: DType
    Bool: DType
    String: DType

    def __new__(
        cls,
        value: str | dt.DType,
        _tf_name: str | None = None,
        _is_quantized: builtins.bool = False,
        _enum_id: int = 0,
    ) -> DType:
        """Create a `DType` descriptor.

        Args:
            value: Underlying compiler dtype string or `dt.DType`.
            _tf_name: Canonical TensorFlow dtype name if distinct from `value`.
            _is_quantized: Whether this dtype represents a quantized integer type.
            _enum_id: TensorFlow `DataType` protobuf enum identifier.

        Returns:
            DType: Initialized `DType` instance.
        """
        raw_str = value.value if isinstance(value, dt.DType) else str(value)
        obj = super().__new__(cls, raw_str)
        obj._tf_name = _tf_name if _tf_name is not None else raw_str
        obj._msc_dtype = dt.DType(raw_str)
        obj._is_quantized_flag = _is_quantized
        obj._enum_id = _enum_id
        return obj

    def __eq__(self, other: str | dt.DType | DType | TensorOperand) -> builtins.bool:
        """Compare equality with another `DType`, compiler `DType`, or string name.

        Args:
            other: Candidate dtype or string to compare against.

        Returns:
            bool: `True` if `other` represents the same underlying dtype string.
        """
        return isinstance(other, str) and str.__eq__(self, other)

    def __hash__(self) -> int:
        """Return the string hash of this `DType`.

        Returns:
            int: Hash of the underlying dtype string.
        """
        return str.__hash__(self)

    @property
    def name(self) -> str:
        """Return the canonical TensorFlow dtype name.

        Returns:
            str: Canonical dtype name (e.g., `'float32'`, `'qint8'`, `'resource'`).
        """
        return self._tf_name

    @property
    def value(self) -> str:
        """Return the underlying compiler dtype string value.

        Returns:
            str: Underlying dtype string.
        """
        return str(self)

    @property
    def as_compiler_dtype(self) -> dt.DType:
        """Return the backing `ml_switcheroo_compiler.core.dtype.DType` enum member.

        Returns:
            dt.DType: Backing compiler `DType`.
        """
        return self._msc_dtype

    @property
    def is_floating(self) -> builtins.bool:
        """Return whether this `DType` is a non-quantized floating-point type.

        Returns:
            bool: `True` if floating-point.
        """
        return not self._is_quantized_flag and self._tf_name in _FLOATING_NAMES

    @property
    def is_integer(self) -> builtins.bool:
        """Return whether this `DType` is a non-quantized integer type.

        Returns:
            bool: `True` if integer.
        """
        return not self._is_quantized_flag and self._tf_name in _INTEGER_NAMES

    @property
    def is_unsigned(self) -> builtins.bool:
        """Return whether this `DType` is unsigned.

        Returns:
            bool: `True` if unsigned integer or quantized unsigned integer.
        """
        return self._tf_name in _UNSIGNED_NAMES

    @property
    def is_complex(self) -> builtins.bool:
        """Return whether this `DType` is a complex floating-point type.

        Returns:
            bool: `True` if complex (`complex64` or `complex128`).
        """
        return self._tf_name in _COMPLEX_NAMES

    @property
    def is_bool(self) -> builtins.bool:
        """Return whether this `DType` is a boolean type.

        Returns:
            bool: `True` if `bool`.
        """
        return self._tf_name == "bool"

    @property
    def is_quantized(self) -> builtins.bool:
        """Return whether this `DType` is a quantized data type.

        Returns:
            bool: `True` if quantized (`qint8`, `qint16`, `qint32`, `quint8`, `quint16`).
        """
        return self._is_quantized_flag

    @property
    def is_numpy_compatible(self) -> builtins.bool:
        """Return whether this `DType` has a compatible NumPy representation.

        Returns:
            bool: `True` unless `resource` or `variant`.
        """
        return self._tf_name not in ("resource", "variant")

    @property
    def limits(self) -> tuple[int | float, int | float]:
        """Return the `(min, max)` representable value bounds for this `DType`.

        Returns:
            tuple[int | float, int | float]: Minimum and maximum representable values.

        Raises:
            ValueError: If this `DType` does not have ordered numeric bounds.
        """
        if self._tf_name not in _DTYPE_VALUE_LIMITS:
            raise ValueError(f"{self._tf_name} does not have defined limits.")
        return _DTYPE_VALUE_LIMITS[self._tf_name]

    @property
    def min(self) -> int | float:
        """Return the minimum representable value for this `DType`.

        Returns:
            int | float: Minimum representable value.
        """
        return self.limits[0]

    @property
    def max(self) -> int | float:
        """Return the maximum representable value for this `DType`.

        Returns:
            int | float: Maximum representable value.
        """
        return self.limits[1]

    @property
    def real_dtype(self) -> DType:
        """Return the real component `DType` corresponding to this `DType`.

        Returns:
            DType: `float32` for `complex64`, `float64` for `complex128`, else `self`.
        """
        if self._tf_name == "complex64":
            return float32
        if self._tf_name == "complex128":
            return float64
        return self

    @property
    def size(self) -> int:
        """Return the storage size in bytes per element (`0` for variable-width `string`).

        Returns:
            int: Element byte size.
        """
        return _DTYPE_SIZES[self._tf_name]

    @property
    def as_datatype_enum(self) -> int:
        """Return the TensorFlow `DataType` protobuf enum integer value.

        Returns:
            int: Protobuf `DataType` enum value.
        """
        return self._enum_id


bfloat16: DType = DType(dt.DType.BFloat16, _tf_name="bfloat16", _enum_id=14)
bool: DType = DType(dt.DType.Bool, _tf_name="bool", _enum_id=10)
complex64: DType = DType(dt.DType.Complex64, _tf_name="complex64", _enum_id=8)
complex128: DType = DType(dt.DType.Complex128, _tf_name="complex128", _enum_id=18)
float16: DType = DType(dt.DType.Float16, _tf_name="float16", _enum_id=19)
half: DType = float16
float32: DType = DType(dt.DType.Float32, _tf_name="float32", _enum_id=1)
float64: DType = DType(dt.DType.Float64, _tf_name="float64", _enum_id=2)
double: DType = float64
int8: DType = DType(dt.DType.Int8, _tf_name="int8", _enum_id=6)
int16: DType = DType(dt.DType.Int16, _tf_name="int16", _enum_id=5)
int32: DType = DType(dt.DType.Int32, _tf_name="int32", _enum_id=3)
int64: DType = DType(dt.DType.Int64, _tf_name="int64", _enum_id=9)
qint8: DType = DType(dt.DType.Int8, _tf_name="qint8", _is_quantized=True, _enum_id=11)
qint16: DType = DType(dt.DType.Int16, _tf_name="qint16", _is_quantized=True, _enum_id=15)
qint32: DType = DType(dt.DType.Int32, _tf_name="qint32", _is_quantized=True, _enum_id=13)
quint8: DType = DType(dt.DType.UInt8, _tf_name="quint8", _is_quantized=True, _enum_id=12)
quint16: DType = DType(dt.DType.UInt16, _tf_name="quint16", _is_quantized=True, _enum_id=16)
resource: DType = DType(dt.DType.Int64, _tf_name="resource", _enum_id=20)
string: DType = DType(dt.DType.String, _tf_name="string", _enum_id=7)
uint8: DType = DType(dt.DType.UInt8, _tf_name="uint8", _enum_id=4)
uint16: DType = DType(dt.DType.UInt16, _tf_name="uint16", _enum_id=17)
uint32: DType = DType(dt.DType.UInt32, _tf_name="uint32", _enum_id=22)
uint64: DType = DType(dt.DType.UInt64, _tf_name="uint64", _enum_id=23)
variant: DType = DType(dt.DType.Int64, _tf_name="variant", _enum_id=21)

_float8_e4m3fn: DType = DType(dt.DType.Float8E4M3FN, _tf_name="float8_e4m3fn", _enum_id=25)
_float8_e5m2: DType = DType(dt.DType.Float8E5M2, _tf_name="float8_e5m2", _enum_id=24)
_float8_e4m3fnuz: DType = DType(dt.DType.Float8E4M3FNUZ, _tf_name="float8_e4m3fnuz", _enum_id=26)
_float8_e4m3b11fnuz: DType = DType(
    dt.DType.Float8E4M3B11FNUZ,
    _tf_name="float8_e4m3b11fnuz",
    _enum_id=27,
)
_float8_e5m2fnuz: DType = DType(dt.DType.Float8E5M2FNUZ, _tf_name="float8_e5m2fnuz", _enum_id=28)
_int4: DType = DType(dt.DType.Int4, _tf_name="int4", _enum_id=29)
_uint4: DType = DType(dt.DType.UInt4, _tf_name="uint4", _enum_id=30)

DType.Float16 = float16
DType.Float32 = float32
DType.Float64 = float64
DType.BFloat16 = bfloat16
DType.Float8E4M3FN = _float8_e4m3fn
DType.Float8E5M2 = _float8_e5m2
DType.Float8E4M3FNUZ = _float8_e4m3fnuz
DType.Float8E4M3B11FNUZ = _float8_e4m3b11fnuz
DType.Float8E5M2FNUZ = _float8_e5m2fnuz
DType.Complex64 = complex64
DType.Complex128 = complex128
DType.Int4 = _int4
DType.Int8 = int8
DType.Int16 = int16
DType.Int32 = int32
DType.Int64 = int64
DType.UInt4 = _uint4
DType.UInt8 = uint8
DType.UInt16 = uint16
DType.UInt32 = uint32
DType.UInt64 = uint64
DType.Bool = bool
DType.String = string

QUANTIZED_DTYPES: frozenset[DType] = frozenset({qint8, qint16, qint32, quint8, quint16})

_DTYPE_LOOKUP: dict[dt.DType | str | type, DType] = {
    builtins.bool: bool,
    int: int32,
    float: float32,
    builtins.complex: complex64,
    str: string,
    bytes: string,
    "float": float32,
    "float16": float16,
    "half": float16,
    "float32": float32,
    "float64": float64,
    "double": float64,
    "bfloat16": bfloat16,
    "float8_e4m3fn": _float8_e4m3fn,
    "float8_e5m2": _float8_e5m2,
    "float8_e4m3fnuz": _float8_e4m3fnuz,
    "fp8_e4m3fnuz": _float8_e4m3fnuz,
    "float8_e4m3b11fnuz": _float8_e4m3b11fnuz,
    "float8_e5m2fnuz": _float8_e5m2fnuz,
    "fp8_e5m2fnuz": _float8_e5m2fnuz,
    "complex": complex64,
    "complex64": complex64,
    "complex128": complex128,
    "int": int32,
    "int4": _int4,
    "int8": int8,
    "int16": int16,
    "int32": int32,
    "int64": int64,
    "uint4": _uint4,
    "uint8": uint8,
    "uint16": uint16,
    "uint32": uint32,
    "uint64": uint64,
    "qint8": qint8,
    "quint8": quint8,
    "qint16": qint16,
    "quint16": quint16,
    "qint32": qint32,
    "bool": bool,
    "bool_": bool,
    "string": string,
    "str": string,
    "bytes": string,
    "resource": resource,
    "variant": variant,
    "object": variant,
    dt.DType.Float16: float16,
    dt.DType.Float32: float32,
    dt.DType.Float64: float64,
    dt.DType.BFloat16: bfloat16,
    dt.DType.Float8E4M3FN: _float8_e4m3fn,
    dt.DType.Float8E5M2: _float8_e5m2,
    dt.DType.Float8E4M3FNUZ: _float8_e4m3fnuz,
    dt.DType.Float8E4M3B11FNUZ: _float8_e4m3b11fnuz,
    dt.DType.Float8E5M2FNUZ: _float8_e5m2fnuz,
    dt.DType.Complex64: complex64,
    dt.DType.Complex128: complex128,
    dt.DType.Int4: _int4,
    dt.DType.Int8: int8,
    dt.DType.Int16: int16,
    dt.DType.Int32: int32,
    dt.DType.Int64: int64,
    dt.DType.UInt4: _uint4,
    dt.DType.UInt8: uint8,
    dt.DType.UInt16: uint16,
    dt.DType.UInt32: uint32,
    dt.DType.UInt64: uint64,
    dt.DType.Bool: bool,
    dt.DType.String: string,
}


def as_dtype(type_value: DTypeLike | type | TensorOperand = None) -> DType:
    """Convert `type_value` into a canonical `zero_tensorflow.dtypes.DType`.

    Args:
        type_value: Input dtype representation (`DType`, `dt.DType`, string, Python type, or `None`).

    Returns:
        DType: Resolved `DType` descriptor.
    """
    return _ct.to_dtype(type_value)


def cast(
    x: _ct.Tensor | TensorCompatible | TensorOperand,
    dtype: DTypeLike | TensorOperand,
    name: str | None = None,
) -> _ct.Tensor:
    """Cast a tensor `x` to a new `dtype`.

    Args:
        x: Input tensor or scalar.
        dtype: Target data type.
        name: Optional operation name.

    Returns:
        _ct.Tensor: Cast tensor.
    """
    del name
    target_dt = _ct.to_dtype(dtype)
    return _ct._wrap_tensor(_ops.cast(_ct._to_tensor(x), target_dt.value))


def saturate_cast(
    value: _ct.Tensor | TensorCompatible | TensorOperand,
    dtype: DTypeLike | TensorOperand,
    name: str | None = None,
) -> _ct.Tensor:
    """Perform a saturating cast of `value` to `dtype`.

    Args:
        value: Input tensor or scalar.
        dtype: Target data type.
        name: Optional operation name.

    Returns:
        _ct.Tensor: Cast tensor clamped to `[target_dtype.min, target_dtype.max]`.
    """
    del name
    target_dt = _ct.to_dtype(dtype)
    t_val = _ct._to_tensor(value)
    if target_dt.is_integer or target_dt.is_quantized:
        t_val = _ops.clip(t_val, float(target_dt.min), float(target_dt.max))
    return _ct._wrap_tensor(_ops.cast(t_val, target_dt.value))


def complex(
    real: _ct.Tensor | TensorCompatible | TensorOperand,
    imag: _ct.Tensor | TensorCompatible | TensorOperand,
    name: str | None = None,
) -> _ct.Tensor:
    """Convert two real numbers/tensors to a complex tensor.

    Args:
        real: Real component tensor or scalar.
        imag: Imaginary component tensor or scalar.
        name: Optional operation name.

    Returns:
        _ct.Tensor: Complex tensor.
    """
    del name
    return _ct._wrap_tensor(_ops.add(_ct._to_tensor(real), _ops.multiply(_ct._to_tensor(imag), 1j)))


class experimental:
    """Experimental low-bitwidth data types."""

    float8_e4m3fn: DType = _float8_e4m3fn
    float8_e5m2: DType = _float8_e5m2
    int4: DType = _int4
    uint4: DType = _uint4


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
