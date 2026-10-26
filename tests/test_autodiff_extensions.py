"""Unit tests for `test_autodiff_extensions`."""

from __future__ import annotations

from zero_tensorflow import (
    GradientTape,
    Tensor,
    Variable,
    custom_gradient,
    debugging,
    function,
    hessians,
    stop_gradient,
)
from zero_tensorflow._typing import CallableOperandProtocol, TensorOperand
from zero_tensorflow.dtypes import DType, float32

import pytest


class _DummyTensorSpec:
    """Test helper implementing `TensorSpecProtocol` for `@tf.function(input_signature=...)`."""

    def __init__(self, shape: tuple[int | None, ...], dtype: DType) -> None:
        """Initialize `_DummyTensorSpec`.

        Args:
            shape: Expected tensor shape tuple.
            dtype: Expected tensor `DType`.
        """
        self._shape = shape
        self._dtype = dtype
        self._name: str | None = "spec"

    @property
    def shape(self) -> tuple[int | None, ...]:
        """Return spec shape.

        Returns:
            tuple[int | None, ...]: Shape tuple.
        """
        return self._shape

    @property
    def dtype(self) -> DType:
        """Return spec dtype.

        Returns:
            DType: DType value.
        """
        return self._dtype

    @property
    def name(self) -> str | None:
        """Return spec name.

        Returns:
            str | None: Spec name.
        """
        return self._name

    def is_compatible_with(self, spec_or_tensor: TensorOperand) -> bool:
        """Check compatibility with `spec_or_tensor`.

        Args:
            spec_or_tensor: Candidate tensor or spec.

        Returns:
            bool: Always `True` for test stub.
        """
        del spec_or_tensor
        return True


def test_jacobian() -> None:
    """Verify `GradientTape.jacobian` computes full `target.shape + source.shape` Jacobians."""
    x = Variable([2.0, 3.0])
    with GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * x
    jac = tape.jacobian(y, x)
    assert isinstance(jac, Tensor)
    assert jac.shape == (2, 2)
    debugging.assert_near(jac, [[4.0, 0.0], [0.0, 6.0]])

    jac_no_pfor = tape.jacobian(y, [x, "non_tensor", Tensor(1.0)], experimental_use_pfor=False)
    assert isinstance(jac_no_pfor, list) and len(jac_no_pfor) == 3
    assert isinstance(jac_no_pfor[0], Tensor)
    assert jac_no_pfor[1] is None
    assert jac_no_pfor[2] is None
    assert tape.jacobian(None, x) is None

    # Non-persistent scalar and vector jacobians
    s = Variable(4.0)
    with GradientTape() as stape:
        stape.watch(s)
        sy = s * s
    s_jac = stape.jacobian(sy, s)
    assert isinstance(s_jac, Tensor) and s_jac.numpy() == 8.0

    v = Variable([1.0, 2.0])
    with GradientTape() as vtape:
        vtape.watch(v)
        vy = v * 3.0
    v_jac = vtape.jacobian(vy, v)
    assert isinstance(v_jac, Tensor) and v_jac.shape == (2, 2)


def test_batch_jacobian() -> None:
    """Verify `GradientTape.batch_jacobian` computes batched Jacobian tensors of shape `(B, *Y, *X)`."""
    x = Variable([[1.0, 2.0], [3.0, 4.0]])
    with GradientTape(persistent=True) as tape:
        tape.watch(x)
        y = x * 3.0
    bjac = tape.batch_jacobian(y, x)
    assert isinstance(bjac, Tensor)
    assert bjac.shape == (2, 2, 2)
    debugging.assert_near(
        bjac,
        [[[3.0, 0.0], [0.0, 3.0]], [[3.0, 0.0], [0.0, 3.0]]],
    )

    bjac_list = tape.batch_jacobian(
        y,
        [x, "non_tensor", Tensor([[1.0, 1.0], [1.0, 1.0]])],
        experimental_use_pfor=False,
    )
    assert isinstance(bjac_list, list) and len(bjac_list) == 3
    assert isinstance(bjac_list[0], Tensor)
    assert bjac_list[1] is None
    assert bjac_list[2] is None

    s = Variable(2.0)
    with GradientTape() as stape:
        stape.watch(s)
        sy = s * 5.0
    s_bjac = stape.batch_jacobian(sy, s)
    assert isinstance(s_bjac, Tensor) and s_bjac.numpy() == 5.0
    assert stape.batch_jacobian(None, s) is None


def test_nested_tapes_and_custom_gradient() -> None:
    """Verify nested `GradientTape` frame stack and `@custom_gradient` backward traversal."""
    x = Variable(2.0)
    with GradientTape() as outer:
        outer.watch(x)
        with GradientTape() as inner:
            inner.watch(x)
            y = x * x * x
        dy_dx = inner.gradient(y, x)
    assert isinstance(dy_dx, Tensor) and dy_dx.numpy() == 12.0
    d2y_dx2 = outer.gradient(dy_dx, x)
    assert isinstance(d2y_dx2, Tensor) and d2y_dx2.numpy() == 12.0

    @custom_gradient
    def f_with_grad(
        *args: TensorOperand,
        **kwargs: TensorOperand,
    ) -> tuple[Tensor, CallableOperandProtocol]:
        """Compute `x * 2.0` and return a custom backward function `dy * 10.0`.

        Args:
            *args: Positional input operand (`x`).
            **kwargs: Unused keyword arguments.

        Returns:
            tuple[Tensor, CallableOperandProtocol]: Forward result and gradient callable.
        """
        del kwargs
        x_val = args[0] if args else 0.0

        def grad_fn(*g_args: TensorOperand, **g_kwargs: TensorOperand) -> TensorOperand:
            """Compute custom gradient `10.0 * dy`.

            Args:
                *g_args: Upstream gradient arguments.
                **g_kwargs: Unused keyword arguments.

            Returns:
                TensorOperand: Scaled upstream gradient.
            """
            del g_kwargs
            dy = g_args[0] if g_args else Tensor(1.0)
            return dy * 10.0 if isinstance(dy, Tensor) else 10.0

        return (Tensor(x_val) * 2.0, grad_fn)

    res1 = f_with_grad(Tensor(3.0)._tensor)
    assert isinstance(res1, Tensor)
    assert res1.numpy() == 6.0

    # Direct custom_gradient differentiation inside nested tapes
    w = Variable(3.0)
    with GradientTape(persistent=True) as ctape:
        ctape.watch(w)
        with GradientTape() as inner_ctape:
            inner_ctape.watch(w)
            direct_cy = f_with_grad(w)
            u = w * 2.0
            cy = f_with_grad(u)
            cz = cy * 3.0
    g_w = ctape.gradient(direct_cy, w)
    assert isinstance(g_w, Tensor) and g_w.numpy() == 10.0
    g_direct = ctape.gradient(cy, w)
    assert isinstance(g_direct, Tensor) and g_direct.numpy() == 20.0
    g_cot = ctape.gradient(cy, w, output_gradients=Tensor(2.0))
    assert isinstance(g_cot, Tensor) and g_cot.numpy() == 40.0
    g_chain = ctape.gradient(cz, w)
    assert isinstance(g_chain, Tensor) and g_chain.numpy() == 60.0

    assert isinstance(custom_gradient(None)(), Tensor)

    @custom_gradient
    def f_direct(*args: TensorOperand, **kwargs: TensorOperand) -> Tensor:
        """Compute `x + 1.0` directly without a `grad_fn` tuple.

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


def test_stop_gradient_and_hessians() -> None:
    """Verify `stop_gradient` and scalar/vector `hessians`."""
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

    with GradientTape() as htape:
        htape.watch(x)
        hy = x * x
        h_traced = hessians(hy, [x])
    assert len(h_traced) == 1
    assert isinstance(h_traced[0], Tensor)
    assert h_traced[0].numpy() == 2.0

    vx = Variable([2.0, 3.0])
    with GradientTape() as vhtape:
        vhtape.watch(vx)
        vy = vx * vx
        vh_traced = hessians(vy, [vx])
    assert len(vh_traced) == 1
    assert vh_traced[0].shape == (2, 2)
    debugging.assert_near(vh_traced[0], [[2.0, 0.0], [0.0, 2.0]])

    h_untraced = hessians(Tensor(1.0), Tensor(2.0))
    assert len(h_untraced) == 1
    assert h_untraced[0].numpy() == 0.0


def test_tf_function_signatures_and_caching() -> None:
    """Verify `tf.function(input_signature=..., jit_compile=..., reduce_retracing=...)`."""
    spec = _DummyTensorSpec((2,), float32)

    def raw_scaled(*t_args: Tensor) -> Tensor:
        """Double first input tensor `t_args[0]`.

        Args:
            *t_args: 1-D input tensor(s) of length 2.

        Returns:
            Tensor: `t_args[0] * 2.0`.
        """
        return t_args[0] * 2.0

    scaled = function(
        raw_scaled,
        input_signature=[spec],
        jit_compile=True,
        reduce_retracing=True,
    )

    assert scaled.jit_compile is True
    assert scaled.reduce_retracing is True
    cf1 = scaled.get_concrete_function()
    cf2 = scaled.get_concrete_function(Tensor([1.0, 2.0], dtype=float32))
    assert cf1 is cf2
    out = scaled(Tensor([3.0, 4.0], dtype=float32))
    debugging.assert_near(out, [6.0, 8.0])

    with pytest.raises(ValueError, match="Expected 1 arguments"):
        scaled(Tensor([1.0, 2.0]), Tensor([3.0, 4.0]))
    with pytest.raises(ValueError, match="Incompatible argument shape"):
        scaled(Tensor([1.0, 2.0, 3.0]))
