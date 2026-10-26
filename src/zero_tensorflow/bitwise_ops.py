"""TensorFlow `bitwise` operations module."""

from __future__ import annotations

from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.core_tensor import Tensor, _to_tensor, _wrap_tensor

import ml_switcheroo_compiler.ops as _ops

__all__ = ["bitwise"]


class bitwise:
    """TensorFlow `bitwise` namespace."""

    @staticmethod
    def bitwise_and(
        x: Tensor | TensorOperand = 0,
        y: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise AND of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise AND tensor.
        """
        del name
        res = _ops.bitwise_and(_to_tensor(x), _to_tensor(y))
        return _wrap_tensor(res)

    @staticmethod
    def invert(
        x: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Invert (flip) each bit of `x` element-wise.

        Args:
            x: Integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise NOT tensor.
        """
        del name
        res = _ops.bitwise_not(_to_tensor(x))
        return _wrap_tensor(res)

    @staticmethod
    def bitwise_or(
        x: Tensor | TensorOperand = 0,
        y: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise OR of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise OR tensor.
        """
        del name
        res = _ops.bitwise_or(_to_tensor(x), _to_tensor(y))
        return _wrap_tensor(res)

    @staticmethod
    def bitwise_xor(
        x: Tensor | TensorOperand = 0,
        y: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise XOR of `x` and `y` element-wise.

        Args:
            x: First integer tensor.
            y: Second integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Bitwise XOR tensor.
        """
        del name
        res = _ops.bitwise_xor(_to_tensor(x), _to_tensor(y))
        return _wrap_tensor(res)

    @staticmethod
    def left_shift(
        x: Tensor | TensorOperand = 0,
        y: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise left-shift of `x` by `y` bits element-wise.

        Args:
            x: Value integer tensor.
            y: Shift amount integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Left-shifted tensor.
        """
        del name
        res = _ops.left_shift(_to_tensor(x), _to_tensor(y))
        return _wrap_tensor(res)

    @staticmethod
    def right_shift(
        x: Tensor | TensorOperand = 0,
        y: Tensor | TensorOperand = 0,
        name: str | None = None,
    ) -> Tensor:
        """Compute the bitwise right-shift of `x` by `y` bits element-wise.

        Args:
            x: Value integer tensor.
            y: Shift amount integer tensor.
            name: Optional operation name.

        Returns:
            Tensor: Right-shifted tensor.
        """
        del name
        res = _ops.right_shift(_to_tensor(x), _to_tensor(y))
        return _wrap_tensor(res)
