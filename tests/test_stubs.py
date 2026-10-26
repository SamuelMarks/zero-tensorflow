"""Behavioral unit tests for `zero_tensorflow.spec_types`, `raw_ops`, and namespace classes."""

from __future__ import annotations

import inspect

import zero_tensorflow as tf
from zero_tensorflow import RaggedTensor, SparseTensor, Tensor, Variable
from zero_tensorflow._typing import TensorOperand
from zero_tensorflow.spec_types import _normalize_dims


def test_tensor_shape_and_specs() -> None:
    """Verify `TensorShape`, `TypeSpec`, `TensorSpec`, `RaggedTensorSpec`, `SparseTensorSpec`, and `OptionalSpec`."""
    assert _normalize_dims(None) == ((), True)
    assert _normalize_dims(tf.TensorShape([2, 3])) == ((2, 3), False)
    assert _normalize_dims(3) == ((3,), False)
    assert _normalize_dims(Tensor([2, 4])) == ((2, 4), False)
    assert _normalize_dims("invalid") == ((), False)

    unk = tf.TensorShape(None)
    assert unk.dims is None
    assert unk.ndims is None
    assert unk.rank is None
    try:
        unk.as_list()
        raise AssertionError("Expected ValueError on unknown rank")
    except ValueError:
        pass
    assert unk.is_compatible_with([2, 3]) is True
    assert unk.merge_with([2, 3]) == tf.TensorShape([2, 3])
    assert tf.TensorShape([2, 3]).merge_with(unk) == tf.TensorShape([2, 3])
    assert unk.concatenate([4]).dims is None
    assert tf.TensorShape([2]).concatenate(unk).dims is None

    s1 = tf.TensorShape([2, None])
    s2 = tf.TensorShape([2, 4])
    assert s1.dims == [2, None]
    assert s1.ndims == 2
    assert s1.rank == 2
    assert s1.as_list() == [2, None]
    assert s1.is_compatible_with(s2) is True
    assert s1.is_compatible_with([2, 4, 5]) is False
    assert s1.is_compatible_with([3, 4]) is False
    assert s1.merge_with(s2) == tf.TensorShape([2, 4])
    assert s2.merge_with(s1) == tf.TensorShape([2, 4])
    try:
        s1.merge_with([3, 4])
        raise AssertionError("Expected ValueError on rank/dim mismatch")
    except ValueError:
        pass
    try:
        s1.merge_with([2, 4, 5])
        raise AssertionError("Expected ValueError on rank mismatch")
    except ValueError:
        pass
    assert s2.concatenate([5]) == tf.TensorShape([2, 4, 5])

    base_spec = tf.TypeSpec()
    assert base_spec.value_type is Tensor
    assert base_spec.is_compatible_with(tf.TypeSpec()) is True
    assert base_spec.most_specific_compatible_type(tf.TypeSpec()) is base_spec
    try:
        tf.RaggedTensorSpec().most_specific_compatible_type(tf.SparseTensorSpec())
        raise AssertionError("Expected ValueError on incompatible TypeSpec")
    except ValueError:
        pass

    ts = tf.TensorSpec([2, 2], dtype=tf.float32, name="x")
    assert ts.value_type is Tensor
    assert ts.name == "x"
    t = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    assert ts.is_compatible_with(t) is True
    assert ts.is_compatible_with(tf.TensorSpec([2, None], dtype=tf.float32)) is True
    assert ts.is_compatible_with("not_a_tensor") is False
    ts_from = tf.TensorSpec.from_tensor(t, name="from_t")
    assert ts_from.shape == (2, 2)
    assert ts_from.name == "from_t"

    rt = RaggedTensor.from_row_splits([1, 2, 3], [0, 2, 3])
    rts = tf.RaggedTensorSpec.from_value(rt)
    assert rts.value_type is RaggedTensor
    assert rts.shape == rt.shape
    assert tf.RaggedTensorSpec().shape == tf.TensorShape(None)

    sp = SparseTensor(indices=[[0, 0]], values=[1.0], dense_shape=[2, 2])
    sps = tf.SparseTensorSpec.from_value(sp)
    assert sps.value_type is SparseTensor
    assert sps.shape == (2, 2)
    assert tf.SparseTensorSpec().shape == tf.TensorShape(None)

    opt_s = tf.OptionalSpec.from_value(t)
    assert isinstance(opt_s.element_spec, tf.TensorSpec)

    assert isinstance(tf.type_spec_from_value(t), tf.TensorSpec)
    assert isinstance(tf.type_spec_from_value(rt), tf.RaggedTensorSpec)
    assert isinstance(tf.type_spec_from_value(sp), tf.SparseTensorSpec)


def test_indexed_slices_and_tensor_array() -> None:
    """Verify `IndexedSlices`, `IndexedSlicesSpec`, `TensorArray`, and `TensorArraySpec`."""
    isl = tf.IndexedSlices(
        values=Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32),
        indices=Tensor([0, 2], dtype=tf.int32),
        dense_shape=[4, 2],
        name="slices",
    )
    assert isl.shape == (4, 2)
    assert isl.dtype == tf.float32
    assert "CPU" in isl.device

    isl_no_dense = tf.IndexedSlices(values=[1.0, 2.0], indices=[0, 1], dense_shape=Tensor(4))
    assert isl_no_dense.shape == (2,)
    isl_none_dense = tf.IndexedSlices(values=[1.0, 2.0], indices=[0, 1], dense_shape=None)
    assert isl_none_dense.shape == (2,)
    isl_null_dense = tf.IndexedSlices(values=[1.0, 2.0], indices=[0, 1], dense_shape=Tensor(None))
    assert isl_null_dense.shape == (2,)
    assert isinstance(tf.type_spec_from_value(isl), tf.IndexedSlicesSpec)
    isl_spec = tf.IndexedSlicesSpec(shape=[4, 2], dtype=tf.float32)
    assert isl_spec.value_type is tf.IndexedSlices
    assert tf.IndexedSlicesSpec().shape == tf.TensorShape(None)

    ta = tf.TensorArray(dtype=tf.float32, size=2, dynamic_size=False, clear_after_read=True)
    assert ta.concat().shape == (0,)
    ta_scalar = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=False, infer_shape=False)
    assert ta_scalar.stack().shape == (1,)
    assert isinstance(tf.type_spec_from_value(ta), tf.TensorArraySpec)
    assert tf.TensorArraySpec().value_type is tf.TensorArray
    assert ta.size().numpy() == 2
    ta.write(0, [1.0, 2.0])
    ta.write(1, [3.0, 4.0])
    try:
        ta.write(5, [5.0, 6.0])
        raise AssertionError("Expected IndexError on fixed TensorArray out-of-bounds write")
    except IndexError:
        pass
    try:
        ta.read(5)
        raise AssertionError("Expected IndexError on out-of-bounds read")
    except IndexError:
        pass

    r0 = ta.read(0)
    assert r0.shape == (2,)
    # Slot 0 was cleared after read, so reading again returns 0.0
    assert ta.read(0).numpy() == 0.0

    ta_dyn = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True, clear_after_read=False)
    assert ta_dyn.stack().shape == (0,)
    assert ta_dyn.concat().shape == (0,)
    assert ta_dyn.gather([]).shape == (0,)
    assert ta_dyn.gather(Tensor(0)).shape == (0,)
    ta_dyn.scatter(Tensor(0), [[1.0]])
    ta_dyn.write(2, [10.0, 20.0])
    assert ta_dyn.size().numpy() == 3
    assert ta_dyn.stack().shape == (3, 2)
    assert ta_dyn.concat().shape == (2,)
    assert ta_dyn.gather([2]).shape == (1, 2)
    assert ta_dyn.gather(Tensor([2])).shape == (1, 2)

    ta_dyn.unstack([[1.0, 2.0], [3.0, 4.0]])
    assert ta_dyn.size().numpy() == 2
    ta_dyn.scatter(Tensor([0, 1]), [[7.0, 8.0], [9.0, 10.0]])
    assert ta_dyn.read(1).shape == (2,)
    assert ta_dyn.identity() is ta_dyn
    ta_dyn.close()
    assert ta_dyn.size().numpy() == 0

    assert isinstance(tf.type_spec_from_value([1.0, 2.0]), tf.TensorSpec)


def test_graph_operation_device_and_module() -> None:
    """Verify `Operation`, `Graph`, `DeviceSpec`, `CriticalSection`, `RegisterGradient`, and `Module`."""
    from ml_switcheroo_compiler.transforms.autodiff_rules.vjp_registry import _VJP_REGISTRY
    from ml_switcheroo_ir import LogicalGraph, LogicalNode

    op = tf.Operation("my_add", op_type="AddV2", inputs=[Tensor(1.0)], outputs=[Tensor(2.0)])
    assert op.name == "my_add"
    assert op.type == "AddV2"
    assert len(op.inputs) == 1
    assert len(op.outputs) == 1
    assert "CPU" in op.device
    assert op.node_def.op_type == "AddV2"

    g = tf.Graph()
    with g.as_default() as g_active:
        assert g_active is g
        tf.import_graph_def([LogicalNode(id="op_in_g", op_type="AddV2")])
    assert g.as_graph_def() is g._logical_graph
    ops = g.get_operations()
    assert len(ops) == 1
    first_op_name = ops[0].name
    assert g.get_operation_by_name(first_op_name).name == first_op_name
    assert isinstance(g.get_tensor_by_name(f"{first_op_name}:0"), Tensor)
    try:
        g.get_operation_by_name("nonexistent_op")
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass
    try:
        g.get_tensor_by_name("nonexistent_op:0")
        raise AssertionError("Expected KeyError")
    except KeyError:
        pass

    with g.name_scope("outer"), g.name_scope("inner") as scope:
        assert scope == "outer/inner/"
    with g.name_scope("") as empty_scope:
        assert empty_scope == ""
    with g.control_dependencies([op]):
        pass
    with g.device("/device:CPU:0"):
        pass

    ds = tf.DeviceSpec.from_string("/job:worker/replica:1/task:2/device:GPU:3")
    assert ds.job == "worker"
    assert ds.replica == 1
    assert ds.task == 2
    assert ds.device_type == "GPU"
    assert ds.device_index == 3
    assert ds.to_string() == "/job:worker/replica:1/task:2/device:GPU:3"

    ds_short = tf.DeviceSpec().parse_from_string("/device:CPU:*")
    assert ds_short.device_type == "CPU"
    assert ds_short.device_index is None
    assert ds_short.to_string() == "/device:CPU:*"
    assert tf.DeviceSpec().parse_from_string("GPU:0").device_type == "GPU"
    assert tf.DeviceSpec().parse_from_string("GPU:*").device_index is None
    assert tf.DeviceSpec().parse_from_string("/unknown:foo").device_type is None
    assert tf.DeviceSpec().parse_from_string("").to_string() == ""

    merged = ds.make_merged_spec(tf.DeviceSpec(job="ps", device_type="TPU", device_index=0))
    assert merged.job == "ps"
    assert merged.device_type == "TPU"
    assert merged.device_index == 0

    cs = tf.CriticalSection(name="mutex")
    assert cs.execute(lambda: 42) == 42

    assert tf.VariableAggregation.SUM.value == "sum"
    assert tf.VariableSynchronization.ON_WRITE.value == "on_write"
    assert tf.UnconnectedGradients.ZERO.value == "zero"
    assert tf.AggregationMethod.ADD_N == 0

    called_grad: list[str] = []

    @tf.RegisterGradient("CustomTestOp")
    def _custom_grad(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Record custom gradient invocation.

        Args:
            *args: Positional arguments (`op, grad`).
            **kwargs: Unused keyword arguments.

        Returns:
            TensorOperand: Unmodified incoming gradient.
        """
        del kwargs
        grad_arg = args[1] if len(args) > 1 else None
        called_grad.append(str(grad_arg))
        return grad_arg

    tf.RegisterGradient("")(_custom_grad)
    vjp_fn = _VJP_REGISTRY["CustomTestOp"]
    res_cot = vjp_fn(
        LogicalGraph(),
        LogicalNode(id="n1", op_type="CustomTestOp", inputs=["x"]),
        "cot_0",
    )
    assert res_cot == ["cot_0"]
    assert called_grad == ["cot_0"]

    class _SubMod(tf.Module):
        """Child module for testing `tf.Module` hierarchy."""

        def __init__(self) -> None:
            """Initialize `_SubMod` with one trainable and one non-trainable variable."""
            super().__init__(name="sub")
            self.w = Variable([1.0, 2.0], trainable=True)
            self.b = Variable([0.0], trainable=False)

    class _RootMod(tf.Module):
        """Root module owning `_SubMod`."""

        def __init__(self) -> None:
            """Initialize `_RootMod`."""
            super().__init__(name="root")
            self.v = Variable([3.0], trainable=True)
            self.child = _SubMod()

    root = _RootMod()
    assert len(root.variables) == 3
    assert len(root.trainable_variables) == 2
    assert root.submodules == (root.child,)


def test_raw_ops_and_namespace_stubs() -> None:
    """Verify all `tf.raw_ops` and namespace stub methods execute deterministically."""
    for name, member in inspect.getmembers(tf.raw_ops):
        if not name.startswith("_") and callable(member):
            member()

    for ns_cls in (
        tf.dtensor,
        tf.experimental,
        tf.feature_column,
        tf.graph_util,
        tf.lite,
        tf.mlir.experimental,
        tf.profiler,
        tf.tools,
        tf.tpu,
        tf.xla,
    ):
        for name, member in inspect.getmembers(ns_cls):
            if not name.startswith("_") and callable(member):
                member()
