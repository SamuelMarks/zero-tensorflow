"""Stub module for dtypes."""

import ml_switcheroo_compiler.core.dtype as dt


class experimental:
    """Stub for experimental module."""

    float8_e4m3fn = dt.DType.Float8E4M3FN
    float8_e5m2 = dt.DType.Float8E5M2
    int4 = dt.DType.Int4
    uint4 = getattr(dt.DType, "UInt4", dt.DType.Int4)
