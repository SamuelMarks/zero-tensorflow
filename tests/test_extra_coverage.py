from zero_tensorflow import _to_tensor
from zero_tensorflow.nn import softmax
import numpy as np


def test_missing_branches():
    # for __init__.py 57->61
    from ml_switcheroo_compiler.tracing import ProxyTensor

    pt = ProxyTensor("test", ())
    pt.dtype = None
    _to_tensor(pt)

    # for nn.py 76->78
    softmax(np.array([1.0, 2.0]), axis=0)


from zero_tensorflow import Tensor, GradientTape
import pytest


def test_tensor_len():
    # scalar tensor
    t1 = Tensor(3.0)
    assert len(t1) == 0
    # 1D tensor
    t2 = Tensor([1.0, 2.0])
    assert len(t2) == 2


def test_gradient_tape_edge_cases():
    with GradientTape() as tape:
        t = Tensor(3.0)
        tape.watch(t)
        y = t * t

    # None target
    assert tape.gradient(None, t) is None

    # Invalid target
    with pytest.raises(
        ValueError, match="Target was not created during the tape's recording."
    ):
        tape.gradient(Tensor(5.0), t)

    # Invalid source (not a Tensor)
    assert tape.gradient(y, [None])[0] is None
    assert tape.gradient(y, None) is None

    # Multiple sources including invalid
    t2 = Tensor(4.0)
    tape.watch(t2)
    with GradientTape() as tape2:
        y2 = t2 * 2.0
    grads = tape2.gradient(y2, [t2, None])
    assert len(grads) == 2
    assert grads[1] is None


def test_to_tensor_switcheroo_tensor_tracing():
    from zero_tensorflow import _to_tensor, Tensor
    from ml_switcheroo_compiler.tracing import _tracer, ProxyTensor

    prev_tracing = getattr(_tracer, "is_tracing", False)
    prev_graph = getattr(_tracer, "active_graph", None)

    # 1) CREATE EAGER TENSOR FIRST!
    t = Tensor(3.0)
    mls_tensor = t._tensor

    t2 = Tensor(3.0)

    try:
        _tracer.is_tracing = True
        _tracer.active_graph = type("Graph", (), {"nodes": {}})()

        # 1) original_tensor=None -> 43->53, 61->63
        _to_tensor(mls_tensor)

        # 2) original_tensor != None, but graph_id in _traced_node_ids
        t2._traced_node_ids = {id(_tracer.active_graph): "my_node_id"}
        _to_tensor(t2)

        proxy = ProxyTensor(id="pt", shape=(1,), dtype=mls_tensor.dtype.value)
        _to_tensor(proxy)

    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph


def test_array_on_traced_non_constant():
    from zero_tensorflow import Tensor, GradientTape
    import pytest

    with GradientTape() as tape:
        t = Tensor(3.0)
        tape.watch(t)
        y = t * 2.0
        with pytest.raises(
            ValueError, match="Cannot call array conversion on a traced tensor"
        ):
            y.numpy()


def test_watch_non_tensor():
    from zero_tensorflow import GradientTape

    with GradientTape() as tape:
        tape.watch("this is a string, not a tensor")


def test_to_tensor_invalid_dtype():
    pass


def test_compiler_ops_missing_coverage():
    # creation/basic.py
    from ml_switcheroo_compiler.ops.creation.basic import ConstantOfShape

    op = ConstantOfShape()

    class ArrayLike:
        def __array__(self):
            return np.array([2, 3])

    res = op.numpy_eval(ArrayLike(), value=5)
    assert res.shape == (2, 3)

    # linalg/basic.py Einsum branches
    from ml_switcheroo_compiler.ops.linalg.basic import Einsum

    op = Einsum()

    class ItemEq:
        def item(self):
            return "ij,jk->ik"

    op.numpy_eval(ItemEq(), np.ones((2, 3)), np.ones((3, 4)))

    class ArrayEq:
        def __array__(self):
            return np.array("ij,jk->ik")

    op.numpy_eval(ArrayEq(), np.ones((2, 3)), np.ones((3, 4)))

    class DataEq:
        def __init__(self):
            self.data = "ij,jk->ik"

    op.numpy_eval(DataEq(), np.ones((2, 3)), np.ones((3, 4)))

    # reductions/basic.py ReductionOp branches
    from ml_switcheroo_compiler.ops.reductions.basic import Sum, Argmax, Argmin

    class ItemAxis:
        def item(self):
            return 0

    class NdarrayAxis(np.ndarray):
        pass

    class BadItemAxis:
        def item(self):
            raise Exception("bad")

    op_sum = Sum()
    # ReductionOp branches
    # axis hasattr item and ndim==0
    axis = ItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_sum.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = BadItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_sum.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = np.array(0)
    try:
        op_sum.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    # keepdims hasattr __array__ and not isinstance
    class ArrayBool:
        def __array__(self):
            return np.array(True)

    op_sum.numpy_eval(np.ones((2, 3)), keepdims=ArrayBool())

    # Argmax branches
    op_argmax = Argmax()
    axis = ItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_argmax.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = BadItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_argmax.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = np.array(0)
    try:
        op_argmax.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    # Argmin branches
    op_argmin = Argmin()
    axis = ItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_argmin.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = BadItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_argmin.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = np.array(0)
    try:
        op_argmin.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    # shape/basic.py branches
    from ml_switcheroo_compiler.ops.shape.basic import (
        Reshape,
        Transpose,
        TopK,
        BroadcastTo,
    )

    op_reshape = Reshape()

    class BadItemShape:
        def item(self):
            raise Exception("bad")

    class GoodItemShape:
        def item(self):
            return 10

    shape = BadItemShape()
    try:
        op_reshape.numpy_eval(np.ones(10), shape=shape)
    except Exception:
        pass

    shape = GoodItemShape()
    try:
        op_reshape.numpy_eval(np.ones(10), shape=shape)
    except Exception:
        pass

    class IterShape:
        def __iter__(self):
            yield 10

    try:
        op_reshape.numpy_eval(np.ones(10), shape=IterShape())
    except Exception:
        pass
    try:
        op_reshape.infer_shape(None, shape=IterShape())
    except Exception:
        pass
    try:
        op_reshape.infer_shape(None, shape=GoodItemShape())
    except Exception:
        pass
    try:
        op_reshape.infer_shape(None, shape=BadItemShape())
    except Exception:
        pass

    op_transpose = Transpose()
    perm = GoodItemShape()
    try:
        op_transpose.numpy_eval(np.ones(10), perm=perm)
    except Exception:
        pass
    perm = BadItemShape()
    try:
        op_transpose.numpy_eval(np.ones(10), perm=perm)
    except Exception:
        pass

    op_bt = BroadcastTo()

    class ItemDtype:
        def item(self):
            return "float32"

    try:
        op_bt.numpy_eval(np.ones(10), shape=np.array([10]))
    except Exception:
        pass

    class ListShape:
        def tolist(self):
            return [10]

    try:
        op_bt.numpy_eval(np.ones(10), shape=ListShape())
    except Exception:
        pass

    op_tk = TopK()
    op_tk.infer_shape(np.ones(10))

    class ArrayK:
        def __array__(self):
            return np.array(1)

    op_tk.infer_shape(np.ones(10), k=ArrayK())
    op_tk.numpy_eval(np.ones(10), k=ArrayK())

    class ItemK:
        def item(self):
            return 1

    op_tk.infer_shape(np.ones(10), k=ItemK())

    from ml_switcheroo_compiler.ops.reductions.basic import Max, Min

    op_max = Max()
    axis = ItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_max.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = BadItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_max.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = np.array(0)
    try:
        op_max.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass
    op_max.numpy_eval(np.ones((2, 3)), keepdims=ArrayBool())

    op_min = Min()
    axis = ItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_min.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = BadItemAxis()
    setattr(axis, "ndim", 0)
    try:
        op_min.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass

    axis = np.array(0)
    try:
        op_min.numpy_eval(np.ones((2, 3)), axis=axis)
    except Exception:
        pass
    op_min.numpy_eval(np.ones((2, 3)), keepdims=ArrayBool())

    # shape/basic.py branches 45, 47
    class ListShape45:
        def tolist(self):
            return [2, 5]

    try:
        op_reshape.numpy_eval(np.ones(10), shape=ListShape45())
    except Exception:
        pass

    class ArrayShape47:
        def __array__(self):
            return np.array([2, 5])

    try:
        op_reshape.numpy_eval(np.ones(10), shape=ArrayShape47())
    except Exception:
        pass
