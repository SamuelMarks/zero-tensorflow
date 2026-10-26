"""Unit tests for `test_extra_coverage`."""

from __future__ import annotations

import zero_tensorflow as tf
import zero_tensorflow._typing as zt_typing
from zero_tensorflow import GradientTape, Tensor, _to_tensor, debugging, to_dtype
from zero_tensorflow.nn import softmax

import pytest
from ml_switcheroo_compiler.core.dtype import DType
from ml_switcheroo_ir import LogicalNode


def test_tensor_numpy_and_comparisons() -> None:
    """Verify `Tensor` arithmetic, comparisons, boolean conversion, and `len()`."""
    t1 = Tensor([1.0, 2.0])
    t2 = Tensor([1.0, 3.0])

    class _NonOperand:
        """Unrelated sentinel class not convertible to a `TensorOperand`."""

    eq = tf.math.equal(t1, t2)
    assert isinstance(t1 == t2, (Tensor, bool))
    debugging.assert_equal(eq, [True, False])
    eq_non = t1 == _NonOperand()
    assert eq_non is False

    ne = tf.math.not_equal(t1, t2)
    assert isinstance(t1 != t2, (Tensor, bool))
    debugging.assert_equal(ne, [False, True])
    ne_non = t1 != _NonOperand()
    assert ne_non is True

    assert isinstance(t1 < t2, Tensor)
    assert isinstance(t1 <= t2, Tensor)
    assert isinstance(t1 > t2, Tensor)
    assert isinstance(t1 >= t2, Tensor)

    assert len(t1) == 2
    with pytest.raises(TypeError, match=r"Scalar tensor has no `len\(\)`"):
        len(Tensor(1.0))

    assert bool(Tensor(1.0)) is True
    assert bool(Tensor(0.0)) is False
    assert Tensor(1.0).__nonzero__() is True
    with pytest.raises(ValueError, match="ambiguous"):
        bool(t1)


def test_tensor_extended_methods_and_operators() -> None:
    """Verify `ndim`, `size`, `device`, `ref`, slicing, iteration, matmul, div/mod, and bitwise ops."""
    t_empty = Tensor()
    assert t_empty.ndim == 0
    assert t_empty.size == 0
    assert "CPU:0" in t_empty.device
    assert t_empty.backing_device == t_empty.device
    assert t_empty.consumers() == []

    t_mat = Tensor([[1.0, 2.0], [3.0, 4.0]])
    assert t_mat.ndim == 2
    assert t_mat.size == 4
    ref1 = t_mat.ref()
    ref2 = t_mat.ref()
    assert ref1 == ref2
    assert hash(ref1) == hash(ref2)
    assert ref1.deref() is t_mat

    # Indexing and slicing
    debugging.assert_near(t_mat[0], [1.0, 2.0])
    debugging.assert_near(t_mat[0:1, 1], [2.0])
    debugging.assert_near(t_mat[..., 1], [2.0, 4.0])
    debugging.assert_near(t_mat[Tensor(1)], [3.0, 4.0])
    debugging.assert_near(t_mat[Tensor([0, 1])], [1.0, 2.0])
    debugging.assert_near(t_mat[:, ::-1], [[2.0, 1.0], [4.0, 3.0]])
    assert t_mat[None, 0].shape == (1, 2)
    assert t_mat.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    t_scalar = Tensor(5.0)
    assert t_scalar.tolist() == 5.0
    assert t_scalar[0].numpy() == 5.0

    # Boolean mask indexing
    mask_1d = Tensor([True, False], dtype="bool")
    debugging.assert_near(Tensor([10.0, 20.0])[mask_1d], [10.0])
    mask_1d_empty = Tensor([False, False], dtype="bool")
    assert Tensor([10.0, 20.0])[mask_1d_empty].shape == (0,)
    mask_2d = Tensor([[True, False], [False, True]], dtype="bool")
    debugging.assert_near(t_mat[mask_2d], [1.0, 4.0])
    assert t_scalar[Tensor(True, dtype="bool")].numpy() == [5.0]
    assert t_scalar[Tensor(False, dtype="bool")].shape == (0,)

    # Iteration
    rows = list(t_mat)
    assert len(rows) == 2
    debugging.assert_near(rows[0], [1.0, 2.0])
    t_sym = Tensor(1.0, _traced_node=LogicalNode(id="sym1", op_type="Custom"))
    with pytest.raises(TypeError, match="Iterating over a symbolic"):
        list(t_sym)

    # Consumers in active graph
    with GradientTape() as tape:
        x_c = Tensor(2.0)
        tape.watch(x_c)
        _ = x_c + 1.0
        assert len(x_c.consumers()) == 1
        assert t_sym.consumers() == []
        untracked = Tensor(9.0)
        assert untracked.consumers() == []

    # Matmul, pow, floordiv, mod
    debugging.assert_near(t_mat @ t_mat, [[7.0, 10.0], [15.0, 22.0]])
    debugging.assert_near([[1.0, 0.0], [0.0, 1.0]] @ t_mat, [[1.0, 2.0], [3.0, 4.0]])
    debugging.assert_near(2.0 ** Tensor([2.0, 3.0]), [4.0, 8.0])
    debugging.assert_near(Tensor([7.0, 8.0]) // 3.0, [2.0, 2.0])
    debugging.assert_near(7.0 // Tensor([2.0, 3.0]), [3.0, 2.0])
    debugging.assert_near(Tensor([7.0, 8.0]) % 3.0, [1.0, 2.0])
    debugging.assert_near(7.0 % Tensor([2.0, 4.0]), [1.0, 3.0])

    # Bitwise and logical operators on int and bool tensors
    ti = Tensor([6, 3], dtype="int32")
    debugging.assert_equal(ti & 2, [2, 2])
    debugging.assert_equal(2 & ti, [2, 2])
    debugging.assert_equal(ti | 1, [7, 3])
    debugging.assert_equal(1 | ti, [7, 3])
    debugging.assert_equal(ti ^ 1, [7, 2])
    debugging.assert_equal(1 ^ ti, [7, 2])
    debugging.assert_equal(~ti, [-7, -4])
    debugging.assert_equal(ti << 1, [12, 6])
    debugging.assert_equal(1 << ti, [64, 8])
    debugging.assert_equal(ti >> 1, [3, 1])
    debugging.assert_equal(16 >> ti, [0, 2])

    tb = Tensor([True, False], dtype="bool")
    debugging.assert_equal(tb & True, [True, False])
    debugging.assert_equal(True & tb, [True, False])
    debugging.assert_equal(tb | False, [True, False])
    debugging.assert_equal(False | tb, [True, False])
    debugging.assert_equal(tb ^ True, [False, True])
    debugging.assert_equal(True ^ tb, [False, True])
    debugging.assert_equal(~tb, [False, True])


def test_gradient_tape_unwatched_and_errors() -> None:
    """Verify `GradientTape` error handling and unwatched source behavior."""
    x = Tensor(2.0)
    unwatched = Tensor(5.0)
    with GradientTape() as tape:
        tape.watch(x)
        y = x * 3.0

    grad_unwatched = tape.gradient(y, unwatched)
    assert grad_unwatched is None

    with pytest.raises(RuntimeError, match="non-persistent"):
        tape.gradient(y, x)

    with GradientTape(persistent=True) as ptape:
        ptape.watch(x)
        y2 = x * 4.0

    g1 = ptape.gradient(y2, x)
    g2 = ptape.gradient(y2, x, unconnected_gradients="zero")
    g_unconn = ptape.gradient(y2, unwatched, unconnected_gradients="zero")
    assert isinstance(g1, Tensor) and g1.numpy() == 4.0
    assert isinstance(g2, Tensor) and g2.numpy() == 4.0
    assert isinstance(g_unconn, Tensor) and g_unconn.numpy() == 0.0

    with pytest.raises(ValueError, match="Unknown value for unconnected_gradients"):
        ptape.gradient(y2, x, unconnected_gradients="invalid")

    with pytest.raises(ValueError, match="not created during the tape"):
        ptape.gradient(unwatched, x)

    assert ptape.gradient(None, x) is None


def test_softmax_axis() -> None:
    """Verify `nn.softmax` numerical accuracy along default and explicit axes."""
    logits = Tensor([[1.0, 1.0], [0.0, 0.0]])
    probs = softmax(logits, axis=-1)
    debugging.assert_near(probs, [[0.5, 0.5], [0.5, 0.5]])


def test_to_tensor_invalid_dtype() -> None:
    """Verify `to_dtype` mappings for built-in types and `TypeError` on invalid dtypes."""

    class int16:  # noqa: N801
        """Custom class whose lowercase name matches a DType alias."""

    class UnknownDTypeClass:
        """Custom class whose name does not match any DType alias."""

    assert to_dtype(None) == DType.Float32
    assert to_dtype(tf.float32) == DType.Float32
    assert to_dtype(DType.Int64) == DType.Int64
    assert to_dtype("float64") == DType.Float64
    assert to_dtype("INT32") == DType.Int32
    assert to_dtype(bool) == DType.Bool
    assert to_dtype(int) == DType.Int32
    assert to_dtype(float) == DType.Float32
    assert to_dtype(complex) == DType.Complex64
    assert to_dtype(str) == DType.String
    assert to_dtype(bytes) == DType.String
    assert to_dtype(int16) == DType.Int16

    with pytest.raises(TypeError, match="Cannot convert"):
        to_dtype("not_a_valid_dtype")

    with pytest.raises(TypeError, match="Cannot convert"):
        to_dtype(UnknownDTypeClass)

    with pytest.raises(TypeError, match="Cannot convert"):
        to_dtype([1, 2, 3])


def test_compiler_ops_missing_coverage() -> None:
    """Verify `_to_tensor`, `Variable` scatter/read operations, and `_typing` protocols."""
    v = tf.Variable(
        [1.0, 2.0, 3.0],
        name="weights",
        dtype="float32",
        synchronization="ON_WRITE",
        aggregation="SUM",
        constraint=lambda t: t,
        shape=(3,),
    )
    assert v.name == "weights"
    assert v.dtype == DType.Float32
    assert v.trainable is True
    v.trainable = False
    assert v.trainable is False
    assert v.synchronization == "ON_WRITE"
    assert v.aggregation == "SUM"
    assert v.constraint is not None

    snap = v.read_value()
    assert isinstance(snap, Tensor)
    debugging.assert_near(snap, [1.0, 2.0, 3.0])
    assert _to_tensor(v) is not None

    snap_assign = v.assign([4.0, 5.0, 6.0], read_value=False)
    assert not isinstance(snap_assign, tf.Variable)
    snap_add = v.assign_add([1.0, 1.0, 1.0], read_value=False)
    debugging.assert_near(snap_add, [5.0, 6.0, 7.0])
    snap_sub = v.assign_sub([1.0, 1.0, 1.0], read_value=False)
    debugging.assert_near(snap_sub, [4.0, 5.0, 6.0])

    v.scatter_nd_update([[0], [2]], [10.0, 30.0])
    debugging.assert_near(v, [10.0, 5.0, 30.0])
    v.scatter_nd_add([1], [5.0])
    debugging.assert_near(v, [10.0, 10.0, 30.0])
    v.scatter_nd_sub(0, 2.0)
    debugging.assert_near(v, [8.0, 10.0, 30.0])
    v.batch_scatter_update([1.0, 2.0, 3.0])
    debugging.assert_near(v, [1.0, 2.0, 3.0])
    gathered = v.sparse_read([0, 2])
    debugging.assert_near(gathered, [1.0, 3.0])

    @tf.function
    def traced_mutate() -> Tensor:
        """Record variable mutations inside a traced `@tf.function` graph.

        Returns:
            Tensor: Updated variable snapshot.
        """
        v.assign([2.0, 3.0, 4.0])
        v.assign_add([1.0, 1.0, 1.0])
        v.assign_sub([1.0, 1.0, 1.0])
        return v.read_value()

    debugging.assert_near(traced_mutate(), [2.0, 3.0, 4.0])

    # Exercise runtime-checkable protocols in _typing.py
    assert isinstance(v, zt_typing.TrackableProtocol)
    assert isinstance(tf.ragged.RaggedTensor([[1]]), zt_typing.RaggedTensorProtocol)
    assert isinstance(tf.sparse.from_dense([[1.0]]), zt_typing.SparseTensorProtocol)
    assert zt_typing.ensure_tensor_compatible(42) == 42
    assert zt_typing.ensure_tensor_compatible([1, 2]) == [1, 2]
    with pytest.raises(TypeError, match="not TensorCompatible"):
        zt_typing.ensure_tensor_compatible({"a": 1})
    assert zt_typing.ensure_scalar("abc") == "abc"
    with pytest.raises(TypeError, match="Expected scalar value"):
        zt_typing.ensure_scalar([1])
    assert zt_typing.ensure_shape_tuple(None) == ()
    assert zt_typing.ensure_shape_tuple(4) == (4,)
    assert zt_typing.ensure_shape_tuple([2, None]) == (2, None)
    assert zt_typing.ensure_shape_tuple("invalid") == ()
    tf.core_tensor._ModuleTrackerProtocol()._register_variable(v)
