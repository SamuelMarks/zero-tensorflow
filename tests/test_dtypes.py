import ml_switcheroo_compiler.core.dtype as dt

from zero_tensorflow.dtypes import experimental


def test_dtypes():
    assert experimental.float8_e4m3fn == dt.DType.Float8E4M3FN
    assert experimental.float8_e5m2 == dt.DType.Float8E5M2
    assert experimental.int4 == dt.DType.Int4
    assert hasattr(experimental, "uint4")
