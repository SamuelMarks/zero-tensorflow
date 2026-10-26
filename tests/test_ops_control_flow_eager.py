"""Unit tests for `test_ops_control_flow_eager`."""

from __future__ import annotations

from zero_tensorflow import Tensor as ZTensor

from ml_switcheroo_compiler.core import config
from ml_switcheroo_compiler.core.tensor import Tensor, as_tensor_obj
from ml_switcheroo_compiler.ops import control_flow


def test_control_flow_eager() -> None:
    """Verify eager execution of `cond`, `while_loop`, `scan`, `vmap`, `pmap`, and `stop_gradient`."""
    config.eager_mode = True
    x = as_tensor_obj(ZTensor([[1.0, 2.0], [3.0, 4.0]])._tensor)

    def true_fn() -> Tensor:
        """Return `x` for the true branch.

        Returns:
            Tensor: Tensor `x`.
        """
        return x

    def false_fn() -> Tensor:
        """Return `x` for the false branch.

        Returns:
            Tensor: Tensor `x`.
        """
        return x

    res_true = control_flow.cond(as_tensor_obj(ZTensor(True)._tensor), true_fn, false_fn)
    res_false = control_flow.cond(as_tensor_obj(ZTensor(False)._tensor), true_fn, false_fn)
    assert res_true is x
    assert res_false is x
    ctr = [0]

    def cond_fn(val: Tensor) -> Tensor:
        """Increment loop counter and check `< 2`.

        Args:
            val: Loop state tensor.

        Returns:
            Tensor: Boolean condition tensor.
        """
        del val
        ctr[0] += 1
        return as_tensor_obj(ZTensor(ctr[0] < 2)._tensor)

    def body_fn(val: Tensor) -> Tensor:
        """Return loop state `val`.

        Args:
            val: Loop state tensor.

        Returns:
            Tensor: Unmodified loop state tensor.
        """
        return val

    loop_res = control_flow.while_loop(cond_fn, body_fn, x)
    assert loop_res is not None

    def scan_fn_empty(c: Tensor, xs: Tensor) -> tuple[Tensor, float]:
        """Return `(c, 1.0)` for scan step.

        Args:
            c: Carry state tensor.
            xs: Current element tensor.

        Returns:
            tuple[Tensor, float]: Updated carry and scalar output.
        """
        del xs
        return (c, 1.0)

    empty_x = as_tensor_obj(ZTensor([])._tensor)
    control_flow.scan(scan_fn_empty, x, empty_x)

    def vmap_fn(val: Tensor) -> Tensor:
        """Return `val` unchanged.

        Args:
            val: Input slice tensor.

        Returns:
            Tensor: Output slice tensor.
        """
        return val

    vmapped = control_flow.vmap(vmap_fn)
    assert callable(vmapped)
    assert vmapped(x) is not None
    pmapped = control_flow.pmap(vmap_fn)
    assert callable(pmapped)
    assert pmapped(x) is not None
    stopped = control_flow.stop_gradient(x)
    assert stopped is not None
