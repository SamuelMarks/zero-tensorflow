"""Unit tests for `test_autodiff_extensions`."""

from __future__ import annotations

from zero_tensorflow import GradientTape, Tensor, Variable, custom_gradient, hessians, stop_gradient

import numpy as np


def test_jacobian() -> None:
    """Verify `GradientTape.jacobian` computes numerical derivatives and handles non-Tensor targets."""
    x = Variable([2.0, 3.0])
    with GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * x
    jac = tape.jacobian(y, x)
    assert isinstance(jac, Tensor)
    np.testing.assert_allclose(jac.numpy(), [4.0, 6.0])
    assert tape.jacobian(None, x) is None


def test_batch_jacobian() -> None:
    """Verify `GradientTape.batch_jacobian` computes batched Jacobian tensors."""
    x = Variable([[1.0, 2.0], [3.0, 4.0]])
    with GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * 3.0
    bjac = tape.batch_jacobian(y, x)
    assert isinstance(bjac, Tensor)
    assert bjac.shape == (2, 1, 2)
    np.testing.assert_allclose(bjac.numpy(), [[[3.0, 3.0]], [[3.0, 3.0]]])

    bjac_list = tape.batch_jacobian(y, [x, Tensor(1.0)])
    assert isinstance(bjac_list, list) and len(bjac_list) == 2
    assert isinstance(bjac_list[0], Tensor)
    assert bjac_list[1] is None

    s = Variable(2.0)
    with GradientTape() as stape:
        stape.watch(s)
        sy = s * 5.0
    s_bjac = stape.batch_jacobian(sy, s)
    assert isinstance(s_bjac, Tensor) and s_bjac.numpy() == 5.0
    assert stape.batch_jacobian(None, s) is None


def test_custom_gradient() -> None:
    """Verify `custom_gradient` decorator with both `(y, grad_fn)` and direct tensor returns."""

    @custom_gradient
    def f_with_grad(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> tuple[Tensor, CallableOperandProtocol]:
        """Compute `x * 2.0` and return a custom backward function.

        Args:
            *args: Positional input operand (`x`).
            **kwargs: Unused keyword arguments.

        Returns:
            tuple[Tensor, CallableOperandProtocol]: Forward result and gradient callable.
        """
        del kwargs
        x_val = args[0] if args else 0.0

        def grad_fn(*g_args: TensorOperand, **g_kwargs: TensorOperand) -> TensorOperand:
            """Compute custom gradient `2.0 * dy`.

            Args:
                *g_args: Upstream gradient arguments.
                **g_kwargs: Unused keyword arguments.

            Returns:
                TensorOperand: Scaled upstream gradient.
            """
            del g_kwargs
            return g_args[0] if g_args else 2.0

        return (Tensor(x_val) * 2.0, grad_fn)

    res1 = f_with_grad(Tensor(3.0)._tensor)
    assert isinstance(res1, Tensor)
    assert res1.numpy() == 6.0

    assert isinstance(custom_gradient(None)(), Tensor)

    @custom_gradient
    def f_direct(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
        """Compute `x + 1.0` directly.

        Args:
            *args: Input operand (`x`).
            **kwargs: Unused keyword arguments.

        Returns:
            Tensor: Forward result.
        """
        del kwargs
        return Tensor(args[0] if args else 0.0) + 1.0

    res2 = f_direct(Tensor(4.0)._tensor)
    assert isinstance(res2, Tensor)
    assert res2.numpy() == 5.0


def test_stop_gradient() -> None:
    """Verify `stop_gradient` detaches a tensor while preserving its numerical value."""
    x = Variable(3.0)
    with GradientTape() as tape:
        tape.watch(x)
        stopped = stop_gradient(x)
        y = stopped * 2.0 + x * 4.0
    assert isinstance(stopped, Tensor)
    assert stopped.numpy() == 3.0
    grad = tape.gradient(y, x)
    assert isinstance(grad, Tensor)
    assert grad.numpy() == 4.0


def test_hessians() -> None:
    """Verify `hessians` computes second derivatives inside a traced graph and zeros when untraced."""
    x = Variable(3.0)
    with GradientTape() as tape:
        tape.watch(x)
        y = x * x
        h_traced = hessians(y, [x])
    assert len(h_traced) == 1
    assert isinstance(h_traced[0], Tensor)
    assert h_traced[0].numpy() == 2.0

    h_untraced = hessians(Tensor(1.0), Tensor(2.0))
    assert len(h_untraced) == 1
    assert h_untraced[0].numpy() == 0.0


from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand  # noqa: E402
