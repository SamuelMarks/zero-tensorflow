"""Unit tests for `zero_tensorflow.dtypes`."""

from __future__ import annotations

import zero_tensorflow.dtypes as tf_dtypes
from zero_tensorflow import debugging
from zero_tensorflow.dtypes import complex, experimental, saturate_cast

import ml_switcheroo_compiler.core.dtype as dt
import pytest


def test_dtypes() -> None:
    """Verify `tf.dtypes` constants, `DType` properties, `as_dtype`, `cast`, `saturate_cast`, and `complex`."""
    assert experimental.float8_e4m3fn == dt.DType.Float8E4M3FN
    assert experimental.float8_e5m2 == dt.DType.Float8E5M2
    assert experimental.int4 == dt.DType.Int4
    assert experimental.uint4 == dt.DType.UInt4

    assert tf_dtypes.as_dtype("bfloat16") == dt.DType.BFloat16
    assert tf_dtypes.as_dtype("uint16") == dt.DType.UInt16
    assert tf_dtypes.as_dtype("qint8") == dt.DType.Int8
    assert tf_dtypes.qint8 in tf_dtypes.QUANTIZED_DTYPES

    # Verify full TensorFlow DType properties
    assert tf_dtypes.float32.name == "float32"
    assert tf_dtypes.float32.value == "float32"
    assert tf_dtypes.float32.as_compiler_dtype == dt.DType.Float32
    assert tf_dtypes.float32.is_floating is True
    assert tf_dtypes.float32.is_integer is False
    assert tf_dtypes.float32.is_unsigned is False
    assert tf_dtypes.float32.is_complex is False
    assert tf_dtypes.float32.is_bool is False
    assert tf_dtypes.float32.is_quantized is False
    assert tf_dtypes.float32.is_numpy_compatible is True
    assert tf_dtypes.float32.size == 4
    assert tf_dtypes.float32.as_datatype_enum == 1
    assert tf_dtypes.float32.real_dtype == tf_dtypes.float32
    assert tf_dtypes.float32.min < 0.0 < tf_dtypes.float32.max

    assert tf_dtypes.uint8.is_integer is True
    assert tf_dtypes.uint8.is_unsigned is True
    assert tf_dtypes.uint8.limits == (0, 255)
    assert tf_dtypes.uint8.min == 0
    assert tf_dtypes.uint8.max == 255
    assert tf_dtypes.uint8.size == 1

    assert tf_dtypes.qint8.name == "qint8"
    assert tf_dtypes.qint8.is_quantized is True
    assert tf_dtypes.qint8.is_integer is False
    assert tf_dtypes.qint8.limits == (-128, 127)

    assert tf_dtypes.complex64.is_complex is True
    assert tf_dtypes.complex64.real_dtype == tf_dtypes.float32
    assert tf_dtypes.complex128.real_dtype == tf_dtypes.float64
    assert tf_dtypes.bool.is_bool is True
    assert tf_dtypes.resource.is_numpy_compatible is False
    assert tf_dtypes.variant.is_numpy_compatible is False
    assert tf_dtypes.string.size == 0

    with pytest.raises(ValueError, match="does not have defined limits"):
        _ = tf_dtypes.string.limits

    c_t = tf_dtypes.cast([1.5, 2.5], "int32")
    debugging.assert_near(c_t, [1, 2])
    sc_t = saturate_cast([1.0, 2.0], "float32")
    debugging.assert_near(sc_t, [1.0, 2.0])
    sc_u8 = saturate_cast([-10.0, 100.0, 300.0], "uint8")
    debugging.assert_equal(sc_u8, [0, 100, 255])
    sc_i8 = saturate_cast([-200.0, 0.0, 200.0], "int8")
    debugging.assert_equal(sc_i8, [-128, 0, 127])
    cplx = complex(1.0, 2.0)
    assert cplx.numpy() == (1.0 + 2.0j)
