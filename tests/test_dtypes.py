"""Unit tests for `zero_tensorflow.dtypes`."""

from __future__ import annotations

import zero_tensorflow.dtypes as tf_dtypes
from zero_tensorflow import debugging
from zero_tensorflow.dtypes import complex, experimental, saturate_cast

import ml_switcheroo_compiler.core.dtype as dt


def test_dtypes() -> None:
    """Verify `tf.dtypes` constants, `as_dtype`, `cast`, `saturate_cast`, and `complex`."""
    assert experimental.float8_e4m3fn == dt.DType.Float8E4M3FN
    assert experimental.float8_e5m2 == dt.DType.Float8E5M2
    assert experimental.int4 == dt.DType.Int4
    assert experimental.uint4 == dt.DType.UInt4

    assert tf_dtypes.as_dtype("bfloat16") == dt.DType.BFloat16
    assert tf_dtypes.as_dtype("uint16") == dt.DType.UInt16
    assert tf_dtypes.as_dtype("qint8") == dt.DType.Int8
    assert tf_dtypes.qint8 in tf_dtypes.QUANTIZED_DTYPES

    c_t = tf_dtypes.cast([1.5, 2.5], "int32")
    debugging.assert_near(c_t, [1, 2])
    sc_t = saturate_cast([1.0, 2.0], "float32")
    debugging.assert_near(sc_t, [1.0, 2.0])
    cplx = complex(1.0, 2.0)
    assert cplx.numpy() == (1.0 + 2.0j)
