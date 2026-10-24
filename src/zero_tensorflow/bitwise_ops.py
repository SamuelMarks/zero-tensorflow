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
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.bitwise_and` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.bitwise_and(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)

    @staticmethod
    def invert(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.invert` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.bitwise_not(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)

    @staticmethod
    def bitwise_or(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.bitwise_or` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.bitwise_or(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)

    @staticmethod
    def bitwise_xor(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.bitwise_xor` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.bitwise_xor(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)

    @staticmethod
    def left_shift(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.left_shift` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.left_shift(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)

    @staticmethod
    def right_shift(
        *args: Tensor | TensorOperand,
        **kwargs: Tensor | TensorOperand,
    ) -> Tensor:
        """Execute `bitwise.right_shift` operation.

        Args:
            *args: Positional operation arguments.
            **kwargs: Keyword operation arguments.

        Returns:
            Tensor: Operation result.
        """
        del kwargs
        res = _ops.right_shift(*[_to_tensor(a) for a in args])
        return _wrap_tensor(res)
