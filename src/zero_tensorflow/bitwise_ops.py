"""TensorFlow `bitwise` operations module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorCompatible, TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor, to_dtype

import ml_switcheroo_compiler.ops as _ops
from zero_keras import ops as _k_ops

__all__ = ["bitwise"]


class bitwise:
    """TensorFlow `bitwise` namespace."""

    @staticmethod
    def bitwise_and(
        x: TensorCompatible | TensorOperand = 0,
        y: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise AND of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise AND tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        t_y = (
            _to_tensor(y)
            if isinstance(y, Tensor)
            else _to_tensor(y, dtype=target_dt.as_compiler_dtype)
        )
        res = _ops.bitwise_and(t_x, t_y)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))

    @staticmethod
    def invert(
        x: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Invert (flip) each bit of `x` element-wise using `x`'s signed/unsigned bit-width.

        Args:
            x: Integer tensor (`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`).
            name: Optional operation name.

        Returns:
            Tensor: Bitwise NOT tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        res = _ops.bitwise_not(t_x)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))

    @staticmethod
    def bitwise_or(
        x: TensorCompatible | TensorOperand = 0,
        y: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise OR of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise OR tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        t_y = (
            _to_tensor(y)
            if isinstance(y, Tensor)
            else _to_tensor(y, dtype=target_dt.as_compiler_dtype)
        )
        res = _ops.bitwise_or(t_x, t_y)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))

    @staticmethod
    def bitwise_xor(
        x: TensorCompatible | TensorOperand = 0,
        y: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise XOR of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise XOR tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        t_y = (
            _to_tensor(y)
            if isinstance(y, Tensor)
            else _to_tensor(y, dtype=target_dt.as_compiler_dtype)
        )
        res = _ops.bitwise_xor(t_x, t_y)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))

    @staticmethod
    def left_shift(
        x: TensorCompatible | TensorOperand = 0,
        y: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise left-shift of `x` by `y` bits element-wise.

        Args:
            x: Value integer tensor.
            y: Shift amount integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Left-shifted tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        t_y = (
            _to_tensor(y)
            if isinstance(y, Tensor)
            else _to_tensor(y, dtype=target_dt.as_compiler_dtype)
        )
        res = _ops.left_shift(t_x, t_y)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))

    @staticmethod
    def right_shift(
        x: TensorCompatible | TensorOperand = 0,
        y: TensorCompatible | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise right-shift of `x` by `y` bits element-wise.

        Args:
            x: Value integer tensor.
            y: Shift amount integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Right-shifted tensor with `x`'s integer `DType`.
        """
        del name
        t_x = _to_tensor(x)
        target_dt = to_dtype(t_x.dtype)
        t_y = (
            _to_tensor(y)
            if isinstance(y, Tensor)
            else _to_tensor(y, dtype=target_dt.as_compiler_dtype)
        )
        res = _ops.right_shift(t_x, t_y)
        return _wrap_tensor(_k_ops.cast(res, target_dt.as_compiler_dtype))
