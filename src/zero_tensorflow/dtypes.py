"""TensorFlow data types (`dtypes`) module."""

from __future__ import annotations

import ml_switcheroo_compiler.core.dtype as dt

__all__ = ["experimental"]


class experimental:
    """Experimental low-bitwidth data types."""

    float8_e4m3fn: dt.DType = dt.DType.Float8E4M3FN
    float8_e5m2: dt.DType = dt.DType.Float8E5M2
    int4: dt.DType = dt.DType.Int4
    uint4: dt.DType = dt.DType.UInt4
