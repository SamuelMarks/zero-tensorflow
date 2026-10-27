"""Behavioral unit tests for `zero_tensorflow.control_flow_ops` and `zero_tensorflow.array_ops`."""

from __future__ import annotations

import math

import zero_tensorflow as tf
from zero_tensorflow import SparseTensor, Tensor, debugging
from zero_tensorflow._typing import ArrayLikeProtocol, TensorOperand
from zero_tensorflow.array_ops import _to_float, _to_int, _to_py_data
from zero_tensorflow.control_flow_ops import _to_bool, scan
from zero_tensorflow.spec_types import _GRADIENT_REGISTRY

import pytest
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_ir import LogicalGraph


def _add_pair(a: Tensor, b: Tensor) -> Tensor:
    """Return `a + b`.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Tensor: Sum tensor.
    """
    return a + b


def _sub_pair(a: Tensor, b: Tensor) -> Tensor:
    """Return `a - b`.

    Args:
        a: First tensor.
        b: Second tensor.

    Returns:
        Tensor: Difference tensor.
    """
    return a - b


def _double_tensor(x: Tensor) -> Tensor:
    """Return `x * 2.0`.

    Args:
        x: Input tensor.

    Returns:
        Tensor: Doubled tensor.
    """
    return x * 2.0


def _triple_tensor(x: Tensor) -> Tensor:
    """Return `x * 3.0`.

    Args:
        x: Input tensor.

    Returns:
        Tensor: Tripled tensor.
    """
    return x * 3.0


def _inc_tensor(x: Tensor) -> Tensor:
    """Return `x + 1.0`.

    Args:
        x: Input tensor.

    Returns:
        Tensor: Incremented tensor.
    """
    return x + 1.0


def _while_cond_1d(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Check whether `args[0] < 5`.

    Args:
        *args: Loop state arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Boolean result.
    """
    del kwargs
    x = args[0] if args else 0
    if isinstance(x, Tensor):
        return _to_bool(x < 5)
    return False


def _while_body_add2(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Increment `args[0]` by `2`.

    Args:
        *args: Loop state arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Updated state.
    """
    del kwargs
    x = args[0] if args else Tensor(0)
    return (x + 2) if isinstance(x, Tensor) else Tensor(2)


def _while_body_add1(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Increment `args[0]` by `1`.

    Args:
        *args: Loop state arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Updated state.
    """
    del kwargs
    x = args[0] if args else Tensor(0)
    return (x + 1) if isinstance(x, Tensor) else Tensor(1)


def _while_cond_tuple(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Check whether `args[0] < 3`.

    Args:
        *args: Loop state tuple elements.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Boolean result.
    """
    del kwargs
    i = args[0] if args else 0
    return _to_bool(i < 3) if isinstance(i, Tensor) else False


def _while_body_tuple(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Advance `(i, s)` to `(i + 1, s + 10)`.

    Args:
        *args: Loop state tuple elements.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Updated `(i, s)` tuple.
    """
    del kwargs
    i = args[0] if len(args) > 0 and isinstance(args[0], Tensor) else Tensor(0)
    s = args[1] if len(args) > 1 and isinstance(args[1], Tensor) else Tensor(0)
    return (i + 1, s + 10)


def _identity_op(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Return `args[0]` unchanged.

    Args:
        *args: Positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: First positional argument or `None`.
    """
    del kwargs
    return args[0] if args else None


def _np_add5(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Add `5.0` to the first positional argument.

    Args:
        *args: Positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Shifted value.
    """
    del kwargs
    x = args[0] if args else 0.0
    val = x.tolist() if isinstance(x, ArrayLikeProtocol) else x
    return val + 5.0 if isinstance(val, (int, float)) else 5.0


def _py_mul4(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Multiply the first `Tensor` positional argument by `4.0`.

    Args:
        *args: Positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Scaled tensor.
    """
    del kwargs
    x = args[0] if args else Tensor(0.0)
    return (x * 4.0) if isinstance(x, Tensor) else Tensor(0.0)


def _nest_double(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Double an integer leaf value.

    Args:
        *args: Leaf positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Doubled integer.
    """
    del kwargs
    x = args[0] if args else 0
    return x * 2 if isinstance(x, int) else 0


def _nest_inc(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
    """Increment an integer leaf value by `1`.

    Args:
        *args: Leaf positional arguments.
        **kwargs: Unused keyword arguments.

    Returns:
        TensorOperand: Incremented integer.
    """
    del kwargs
    x = args[0] if args else 0
    return x + 1 if isinstance(x, int) else 0


def test_control_flow_ops_behavioral() -> None:
    """Verify control flow, fold/scan/map, assertions, gradients, scopes, and runtime helpers."""
    assert _to_bool(Tensor(True)) is True
    assert _to_bool(Tensor(0)) is False
    assert _to_bool(Tensor([1, 2])) is True
    assert _to_bool(Tensor([])) is False
    assert _to_bool([1]) is True
    assert _to_bool(1) is True

    assert tf.Assert(None, ["msg"]) is None
    op_assert = tf.Assert(True, ["ok"])
    assert isinstance(op_assert, tf.Operation)
    with pytest.raises(ValueError, match="Assertion failed"):
        tf.Assert(False, ["failed_check"])

    tf.assert_equal(Tensor(2), Tensor(2))
    tf.assert_equal(None, None)
    tf.assert_greater(Tensor(5), Tensor(2))
    tf.assert_greater(None, None)
    tf.assert_less(Tensor(1), Tensor(3))
    tf.assert_less(None, None)
    tf.assert_rank(Tensor([1, 2]), 1)
    tf.assert_rank(Tensor([1, 2]), Tensor(1))
    tf.assert_rank(None, None)

    assert tf.cond(None, lambda: 1, lambda: 2) is None
    c_true = tf.cond(Tensor(True), lambda: Tensor(10.0), lambda: Tensor(20.0))
    assert isinstance(c_true, Tensor) and c_true.numpy() == 10.0
    c_false = tf.cond(False, Tensor(10.0), Tensor(20.0))
    assert isinstance(c_false, Tensor) and c_false.numpy() == 20.0

    case_seq = tf.case(
        [(False, lambda: Tensor(1.0)), (True, lambda: Tensor(2.0))],
        default=lambda: Tensor(3.0),
    )
    assert isinstance(case_seq, Tensor) and case_seq.numpy() == 2.0
    case_map = tf.case(
        {False: lambda: Tensor(4.0), True: lambda: Tensor(5.0)},
        default=lambda: Tensor(6.0),
    )
    assert isinstance(case_map, Tensor) and case_map.numpy() == 5.0
    case_map_def = tf.case({False: lambda: Tensor(4.0)}, default=lambda: Tensor(6.0))
    assert isinstance(case_map_def, Tensor) and case_map_def.numpy() == 6.0
    case_def = tf.case([(False, lambda: Tensor(1.0))], default=lambda: Tensor(9.0))
    assert isinstance(case_def, Tensor) and case_def.numpy() == 9.0
    assert tf.case("invalid", default=None) is None

    assert tf.switch_case(None, [lambda: 1]) is None
    sw_seq = tf.switch_case(Tensor(1), [lambda: Tensor(10.0), lambda: Tensor(20.0)])
    assert isinstance(sw_seq, Tensor) and sw_seq.numpy() == 20.0
    sw_map = tf.switch_case(0, {1: lambda: Tensor(6.0), 0: lambda: Tensor(7.0)})
    assert isinstance(sw_map, Tensor) and sw_map.numpy() == 7.0
    sw_map_def = tf.switch_case(2, {0: lambda: Tensor(7.0)}, default=lambda: Tensor(8.0))
    assert isinstance(sw_map_def, Tensor) and sw_map_def.numpy() == 8.0
    sw_def = tf.switch_case(5, [lambda: Tensor(1.0)], default=lambda: Tensor(99.0))
    assert isinstance(sw_def, Tensor) and sw_def.numpy() == 99.0
    assert tf.switch_case(5, "invalid", default=None) is None

    assert tf.while_loop(None, None, None) is None
    assert isinstance(tf.while_loop(None, None, Tensor(3.0)), Tensor)
    wl_single = tf.while_loop(_while_cond_1d, _while_body_add2, Tensor(1))
    assert isinstance(wl_single, Tensor) and wl_single.numpy() == 5
    wl_max = tf.while_loop(_while_cond_1d, _while_body_add1, Tensor(0), maximum_iterations=2)
    assert isinstance(wl_max, Tensor) and wl_max.numpy() == 2
    wl_tuple = tf.while_loop(
        _while_cond_tuple,
        _while_body_tuple,
        (Tensor(0), Tensor(0)),
    )
    assert isinstance(wl_tuple, tuple) and isinstance(wl_tuple[1], Tensor)
    assert wl_tuple[1].numpy() == 30

    assert scan(None, Tensor([1.0, 2.0])) is None
    assert scan(_add_pair, Tensor([])) is not None
    sc_fwd = scan(_add_pair, Tensor([1.0, 2.0, 3.0]))
    assert isinstance(sc_fwd, Tensor)
    debugging.assert_near(sc_fwd, [1.0, 3.0, 6.0])
    sc_rev = scan(_add_pair, [Tensor(1.0), Tensor(2.0)], initializer=Tensor(10.0), reverse=True)
    assert isinstance(sc_rev, Tensor)
    debugging.assert_near(sc_rev, [13.0, 12.0])

    assert tf.foldl(None, Tensor([1.0])) is None
    fl_empty = tf.foldl(_add_pair, Tensor([]))
    assert fl_empty is not None and fl_empty.numpy() == 0.0
    fl_empty_init = tf.foldl(_add_pair, Tensor([]), initializer=Tensor(5.0))
    assert fl_empty_init is not None and fl_empty_init.numpy() == 5.0
    fl_res = tf.foldl(_add_pair, Tensor([1.0, 2.0, 3.0]))
    assert fl_res is not None and fl_res.numpy() == 6.0
    fl_init = tf.foldl(_add_pair, [Tensor(2.0), Tensor(3.0)], initializer=Tensor(10.0))
    assert fl_init is not None and fl_init.numpy() == 15.0

    assert tf.foldr(None, Tensor([1.0])) is None
    fr_empty = tf.foldr(_add_pair, Tensor([]))
    assert fr_empty is not None and fr_empty.numpy() == 0.0
    fr_empty_init = tf.foldr(_add_pair, Tensor([]), initializer=Tensor(4.0))
    assert fr_empty_init is not None and fr_empty_init.numpy() == 4.0
    fr_res = tf.foldr(_sub_pair, Tensor([1.0, 2.0, 3.0]))
    assert isinstance(fr_res, Tensor)
    fr_init = tf.foldr(_add_pair, [Tensor(1.0), Tensor(2.0)], initializer=Tensor(5.0))
    assert fr_init is not None and fr_init.numpy() == 8.0

    assert tf.map_fn(None, Tensor([1.0])) is None
    mf_empty = tf.map_fn(_double_tensor, Tensor([]))
    assert isinstance(mf_empty, Tensor) and mf_empty.shape == (0,)
    mapped = tf.map_fn(_double_tensor, Tensor([1.0, 2.0, 3.0]))
    assert isinstance(mapped, Tensor)
    debugging.assert_near(mapped, [2.0, 4.0, 6.0])
    mapped_seq = tf.map_fn(_inc_tensor, [Tensor(3.0), Tensor(4.0)])
    assert isinstance(mapped_seq, Tensor)
    debugging.assert_near(mapped_seq, [4.0, 5.0])

    vmap = tf.vectorized_map(_triple_tensor, Tensor([1.0, 2.0]))
    assert isinstance(vmap, Tensor)
    debugging.assert_near(vmap, [3.0, 6.0])

    with tf.name_scope("scope_a"), tf.name_scope("scope_b"):
        assert tf.get_current_name_scope() == "scope_a/scope_b"
    with tf.name_scope(""):
        assert tf.get_current_name_scope() == ""
    assert tf.get_current_name_scope() == ""
    with tf.control_dependencies([op_assert]):
        pass
    with tf.device("/cpu:0"):
        pass
    with tf.variable_creator_scope("creator"):
        pass

    assert tf.executing_eagerly() is True
    assert tf.inside_function() is False
    assert tf.get_logger().name == "tensorflow"
    assert tf.get_static_value(Tensor(12.0)) == 12.0
    assert tf.get_static_value(tf.Variable(8.0)) == 8.0
    sym_t = Tensor()
    assert tf.get_static_value(sym_t) is None
    assert tf.get_static_value(42) == 42

    gc = tf.guarantee_const(Tensor(5.0))
    assert gc is not None and gc.numpy() == 5.0
    assert tf.guarantee_const(None) is None
    gpt_fn = tf.grad_pass_through(_identity_op)
    assert gpt_fn is not None and gpt_fn(Tensor(3.0)) is not None
    assert tf.grad_pass_through(None) is None
    rg_fn = tf.recompute_grad(_identity_op)
    assert rg_fn is not None and rg_fn(Tensor(4.0)) is not None
    assert tf.recompute_grad(None) is None

    assert tf.gradients(None, None) is None
    x_var = tf.Variable(3.0)
    grads = tf.gradients(x_var, [x_var])
    assert isinstance(grads, list) and grads[0] is not None and grads[0].numpy() == 1.0
    grads_single = tf.gradients(x_var, x_var)
    assert (
        isinstance(grads_single, list)
        and grads_single[0] is not None
        and grads_single[0].numpy() == 1.0
    )

    assert isinstance(tf.group(op_assert), tf.Operation)
    id_n = tf.identity_n([Tensor(1.0), Tensor(2.0)])
    assert id_n is not None and id_n[0].numpy() == 1.0
    assert tf.identity_n(None) is None

    assert tf.load_library("libfoo.so") == {"library": "libfoo.so"}
    assert tf.load_library(None) is None
    assert tf.load_op_library("libop.so") == {"library": "libop.so"}
    assert tf.no_op().type == "NoOp"
    tf.no_gradient("CustomNonDiffOp")
    tf.no_gradient("")
    assert _GRADIENT_REGISTRY["CustomNonDiffOp"]() is None
    nd_wrapped = tf.nondifferentiable_batch_function(1, 2)(_identity_op)
    assert nd_wrapped(Tensor(2.0)) is not None

    np_fn_res = tf.numpy_function(_np_add5, [Tensor(2.0)], tf.float32)
    assert isinstance(np_fn_res, Tensor) and np_fn_res.numpy() == 7.0
    assert tf.numpy_function(None, None) is None

    py_fn_res = tf.py_function(_py_mul4, [Tensor(3.0)], tf.float32)
    assert isinstance(py_fn_res, Tensor) and py_fn_res.numpy() == 12.0
    assert tf.py_function(None, None) is None

    assert isinstance(tf.print("hello", Tensor(1)), tf.Operation)
    tf.register_tensor_conversion_function(int, _identity_op)
    tf.register_tensor_conversion_function(int, None)
    assert tf.timestamp().shape == ()
    tup_res = tf.tuple([Tensor(1.0), 42])
    assert tup_res is not None and tup_res[1].numpy() == 42
    assert tf.tuple(None) is None

    acc = tf.autodiff.ForwardAccumulator(primals=Tensor(2.0), tangents=Tensor(1.0))
    jvp_res = acc.jvp(Tensor(4.0))
    assert jvp_res is not None and jvp_res.numpy() == 1.0
    with acc as active_acc:
        assert active_acc is acc

    assert tf.compat.as_bytes("abc") == b"abc"
    assert tf.compat.as_bytes(b"xyz") == b"xyz"
    assert tf.compat.as_bytes(123) == b""
    assert tf.compat.as_text(b"abc") == "abc"
    assert tf.compat.as_text("xyz") == "xyz"
    assert tf.compat.as_text(123) == ""
    assert tf.compat.as_str("xyz") == "xyz"
    assert tf.compat.as_str_any(b"abc") == "abc"
    assert tf.compat.as_str_any(123) == "123"
    assert tf.compat.bytes_or_text_types() == (bytes, str)
    assert tf.compat.complex_types() == (complex,)
    assert tf.compat.integral_types() == (int,)
    assert tf.compat.real_types() == (int, float)
    assert tf.compat.dimension_at_index(tf.TensorShape([4, 5]), 1) == 5
    assert tf.compat.dimension_at_index(tf.TensorShape(None), 0) is None
    assert tf.compat.dimension_value(7) == 7
    assert tf.compat.forward_compatible(2099, 1, 1) is False
    with tf.compat.forward_compatibility_horizon(2026, 1, 1):
        pass
    assert tf.compat.path_to_str("/tmp/b") == "/tmp/b"
    assert tf.compat.path_to_str(123) == "123"
    assert tf.compat.v1() is None
    assert tf.compat.v2() is None

    err = tf.errors.OpError(None, None, "custom_err", tf.errors.INVALID_ARGUMENT)
    assert err.message == "custom_err"
    assert err.error_code == tf.errors.INVALID_ARGUMENT

    assert "tf.pipeline" in tf.mlir.experimental.convert_function(None)
    assert "tf.pipeline" in tf.mlir.experimental.convert_graph_def(None)

    tf.nest.assert_same_structure([1, (2, 3)], ["a", ("b", "c")])
    tf.nest.assert_same_structure({"a": 1}, {"a": 2})
    with pytest.raises(ValueError, match="do not have the same leaf count"):
        tf.nest.assert_same_structure([1, 2], [1])
    with pytest.raises(ValueError, match="same nesting depth"):
        tf.nest.assert_same_structure([1], 1)
    with pytest.raises(ValueError, match="Dictionary keys"):
        tf.nest.assert_same_structure({"a": 1}, {"b": 1})
    with pytest.raises(TypeError, match="different container types"):
        tf.nest.assert_same_structure([1, 2], (1, 2), check_types=True)
    with pytest.raises(ValueError, match="Sequence lengths"):
        tf.nest.assert_same_structure([[1, 2]], [[1], [2]])

    sp_c = tf.SparseTensor([[0, 0]], [1.0], [2, 2])
    rg_c = tf.RaggedTensor([[1, 2], [3]])
    is_c = tf.IndexedSlices(Tensor([1.0]), Tensor([0]), dense_shape=Tensor([2]))
    assert len(tf.nest.flatten(sp_c, expand_composites=True)) == 3
    assert len(tf.nest.flatten(rg_c, expand_composites=True)) == 2
    assert len(tf.nest.flatten(is_c, expand_composites=True)) == 3
    assert (
        len(tf.nest.flatten(tf.IndexedSlices(Tensor([1.0]), Tensor([0])), expand_composites=True))
        == 2
    )
    assert isinstance(
        tf.nest.map_structure(_identity_op, sp_c, expand_composites=True),
        tf.SparseTensor,
    )

    with pytest.raises(ValueError, match="exclusive case"):
        tf.case([(True, lambda: Tensor(1.0)), (True, lambda: Tensor(2.0))], exclusive=True)
    with pytest.raises(ValueError, match="strict case"):
        tf.case([(False, lambda: Tensor(1.0))], strict=True, default=None)
    sw_clamped = tf.switch_case(99, [lambda: Tensor(10.0), lambda: Tensor(20.0)], default=None)
    assert isinstance(sw_clamped, Tensor) and sw_clamped.numpy() == 20.0

    with pytest.raises(ValueError, match="not compatible with shape_invariant"):
        tf.while_loop(
            _while_cond_1d,
            _while_body_add1,
            Tensor([1, 2]),
            shape_invariants=tf.TensorShape([3]),
        )
    wl_inv = tf.while_loop(
        _while_cond_1d,
        _while_body_add2,
        Tensor(1),
        shape_invariants=[tf.TensorShape([]), None],
    )
    assert isinstance(wl_inv, Tensor) and wl_inv.numpy() == 5

    @tf.function
    def _traced_cf(a: Tensor) -> Tensor:
        """Trace `cond` and `while_loop` inside `@tf.function`.

        Args:
            a: Input tensor.

        Returns:
            Tensor: Conditional loop result.
        """
        c_res = tf.cond(Tensor(True), lambda: a + 1.0, lambda: a + 2.0)
        _ = tf.cond(Tensor(True), lambda: (a, a), lambda: (a, a))
        w_res = tf.while_loop(_while_cond_1d, _while_body_add1, c_res)
        return Tensor(w_res)

    _ = _traced_cf(Tensor(1.0))

    def _map_dict_fn(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Double dict entry `'a'`.

        Args:
            *args: Input dict slice.
            **kwargs: Unused keyword arguments.

        Returns:
            TensorOperand: Mapped dict.
        """
        del kwargs
        d = args[0]
        assert isinstance(d, dict)
        return {"out": Tensor(d["a"]) * 2.0}

    def _map_pair_fn(*args: TensorOperand, **kwargs: TensorOperand) -> TensorOperand:
        """Sum tuple pair.

        Args:
            *args: Input pair slice.
            **kwargs: Unused keyword arguments.

        Returns:
            TensorOperand: Single-element list.
        """
        del kwargs
        p = args[0]
        assert isinstance(p, tuple)
        return [Tensor(p[0]) + Tensor(p[1])]

    def _map_tup_fn(x: Tensor) -> tuple[Tensor, Tensor]:
        """Return `(x, x * 2.0)`.

        Args:
            x: Input tensor.

        Returns:
            tuple[Tensor, Tensor]: Pair of tensors.
        """
        return (x, x * 2.0)

    assert tf.map_fn(_map_dict_fn, {"a": Tensor([1.0, 2.0])}) is not None
    assert tf.map_fn(_map_pair_fn, (Tensor([1.0, 2.0]), Tensor([3.0, 4.0]))) is not None
    assert tf.map_fn(_map_tup_fn, Tensor([1.0, 2.0])) is not None

    # Feature columns verification
    num_col = tf.feature_column.numeric_column("age", shape=1, normalizer_fn=lambda t: t * 2.0)
    assert num_col.name == "age"
    debugging.assert_near(num_col.transform_feature({"age": [10.0, 20.0]}), [20.0, 40.0])
    buck_col = tf.feature_column.bucketized_column(num_col, boundaries=[15.0, 30.0])
    debugging.assert_equal(buck_col.transform_feature({"age": [5.0, 10.0, 20.0]}), [0, 1, 2])
    assert tf.feature_column.bucketized_column(None, boundaries="bad").transform_feature(
        {"bucketized": [1.0]}
    ).shape == (1,)
    id_col = tf.feature_column.categorical_column_with_identity("id_feat", num_buckets=4)
    vlist_col = tf.feature_column.categorical_column_with_vocabulary_list(
        "color", ["red", "blue"], num_oov_buckets=2
    )
    assert vlist_col.transform_feature({"color": ["red", "green"]}).shape == (2,)
    vlist_def = tf.feature_column.categorical_column_with_vocabulary_list(
        "num_id", [10, 20], num_oov_buckets=0, default_value=-1
    )
    debugging.assert_equal(vlist_def.transform_feature({"num_id": [10, 99]}), [0, -1])
    assert (
        tf.feature_column.categorical_column_with_vocabulary_list("empty", "bad").num_buckets == 1
    )
    hb_col = tf.feature_column.categorical_column_with_hash_bucket("tag", hash_bucket_size=8)
    assert hb_col.transform_feature({"tag": ["a", "b"]}).shape == (2,)
    ind_col = tf.feature_column.indicator_column(id_col)
    debugging.assert_near(
        ind_col.transform_feature({"id_feat": [1, 3]}), [[0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    assert tf.feature_column.indicator_column(None).transform_feature({}).shape == (1, 1)
    emb_col = tf.feature_column.embedding_column(id_col, dimension=2)
    assert emb_col.transform_feature({"id_feat": [0, 1]}).shape == (2, 2)
    assert tf.feature_column.embedding_column(None, dimension=2).transform_feature({}).shape == (
        1,
        2,
    )
    cross_col = tf.feature_column.crossed_column([num_col, "tag"], hash_bucket_size=4)
    assert cross_col.key == "age_X_tag"
    parse_spec = tf.feature_column.make_parse_example_spec(
        [num_col, id_col, ind_col, buck_col, "ignored"]
    )
    assert "age" in parse_spec and "id_feat" in parse_spec
    assert len(tf.feature_column.shared_embeddings([id_col], dimension=2)) == 1
    assert tf.feature_column.weighted_categorical_column(id_col, "w").is_sparse is True
    vf_col = tf.feature_column.categorical_column_with_vocabulary_file(
        "vf", "dummy_file", vocabulary_size=2
    )
    assert vf_col["type"] == "vocabulary_file"

    # TFLite converter & interpreter verification
    @tf.function
    def _tflite_double(x: Tensor) -> Tensor:
        """Double input tensor `x`.

        Args:
            x: Input tensor.

        Returns:
            Tensor: `x * 2.0`.
        """
        return x * 2.0

    cf_tflite = _tflite_double.get_concrete_function(Tensor([3.0, 4.0]))
    conv = tf.lite.TFLiteConverter.from_concrete_functions([cf_tflite])
    tflite_bytes = conv.convert()
    assert tflite_bytes.startswith(b"TFL3")
    interp = tf.lite.Interpreter(model_content=tflite_bytes)
    interp.allocate_tensors()
    in_det = interp.get_input_details()
    out_det = interp.get_output_details()
    in_idx = in_det[0]["index"]
    out_idx = out_det[0]["index"]
    assert isinstance(in_idx, int) and isinstance(out_idx, int)
    interp.set_tensor(in_idx, Tensor([5.0, 6.0]))
    interp.invoke()
    debugging.assert_near(Tensor(interp.get_tensor(out_idx)), [10.0, 12.0])

    conv_k = tf.lite.TFLiteConverter.from_keras_model(lambda x: (x + 1.0,))
    interp_k = tf.lite.Interpreter(model_content=conv_k.convert())
    interp_k.allocate_tensors()
    interp_k.set_tensor(0, Tensor([2.0]))
    interp_k.invoke()
    debugging.assert_near(Tensor(interp_k.get_tensor(1)), [3.0])

    conv_s = tf.lite.TFLiteConverter.from_saved_model("/tmp/nonexistent_sm")
    interp_s = tf.lite.Interpreter(model_content=conv_s.convert())
    interp_s.allocate_tensors()
    interp_s.set_tensor(0, Tensor([9.0]))
    interp_s.invoke()
    debugging.assert_near(Tensor(interp_s.get_tensor(1)), [9.0])

    # XLA compile & TPU XLAOptions verification
    xla_out = tf.xla.experimental.compile(lambda a: a * 3.0, [Tensor(4.0)])
    assert len(xla_out) == 1 and xla_out[0].numpy() == 12.0
    xla_tup = tf.xla.experimental.compile(lambda a: (a, a + 1.0), [Tensor(2.0)])
    assert len(xla_tup) == 2
    assert tf.xla.experimental.compile(None) == []
    assert isinstance(tf.tpu.XLAOptions(enable_xla=True), dict)
    assert tf.nest.flatten(None) == []
    assert tf.nest.flatten([1, [2, (3, 4)]]) == [1, 2, 3, 4]
    assert tf.nest.flatten({"b": 2, "a": 1}) == [1, 2]
    assert tf.nest.is_nested([1, 2]) is True
    assert tf.nest.is_nested(1) is False
    assert tf.nest.map_structure(_nest_double, [1, (2, 3)]) == [2, (4, 6)]
    assert tf.nest.map_structure(_nest_inc, {"k": 5}) == {"k": 6}
    assert tf.nest.map_structure(None, [1]) is None
    assert tf.nest.pack_sequence_as([1, (2, 3)], [10, 20, 30]) == [10, (20, 30)]
    assert tf.nest.pack_sequence_as(1, "not_a_seq") is None

    for q_cls in (
        tf.queue.FIFOQueue,
        tf.queue.PaddingFIFOQueue,
        tf.queue.PriorityQueue,
        tf.queue.RandomShuffleQueue,
    ):
        q = q_cls(capacity=4)
        q.enqueue(Tensor(10.0))
        q.enqueue(Tensor(20.0))
        assert q.size().numpy() == 2
        deq = q.dequeue()
        assert isinstance(deq, Tensor) and deq.numpy() == 10.0
        empty_q = q_cls()
        assert empty_q.dequeue() is None

    assert tf.sets.intersection(None, None) is None
    s_inter = tf.sets.intersection([1, 2, 3], [2, 3, 4])
    assert s_inter is not None
    debugging.assert_equal(s_inter, [2, 3])
    assert tf.sets.difference(None, None) is None
    s_diff = tf.sets.difference([1, 2, 3], [2])
    assert s_diff is not None
    debugging.assert_equal(s_diff, [1, 3])
    assert tf.sets.union(None, None) is None
    s_union = tf.sets.union([1, 2], [2, 3])
    assert s_union is not None
    debugging.assert_equal(s_union, [1, 2, 3])
    assert tf.sets.size(None) is None
    s_size = tf.sets.size([1, 2, 2, 3])
    assert s_size is not None and s_size.numpy() == 3

    assert isinstance(tf.sysconfig.get_build_info(), dict)
    assert isinstance(tf.sysconfig.get_compile_flags(), list)
    assert isinstance(tf.sysconfig.get_include(), str)
    assert isinstance(tf.sysconfig.get_lib(), str)
    assert isinstance(tf.sysconfig.get_link_flags(), list)
    assert tf.version.VERSION.startswith("2.")


def test_array_ops_behavioral() -> None:
    """Verify `zero_tensorflow.array_ops` operations in eager and traced modes."""
    assert _to_py_data(None) is None
    assert _to_float("bad") == 0.0
    assert _to_int("bad") == 0

    assert tf.add_n(None) is None
    add_empty = tf.add_n([])
    assert add_empty is not None and add_empty.numpy() == 0.0
    add_single = tf.add_n(Tensor(5.0))
    assert add_single is not None and add_single.numpy() == 5.0
    add_two = tf.add_n([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
    assert add_two is not None
    debugging.assert_near(add_two, [4.0, 6.0])

    assert tf.approx_top_k(None) is None
    topk_res = tf.approx_top_k(Tensor([1.0, 5.0, 3.0, 2.0]), k=2)
    assert topk_res is not None
    vals, idxs = topk_res
    debugging.assert_near(vals, [5.0, 3.0])
    debugging.assert_equal(idxs, [1, 2])

    assert tf.argsort(None) is None
    as_asc = tf.argsort(Tensor([3.0, 1.0, 2.0]))
    assert as_asc is not None
    debugging.assert_equal(as_asc, [1, 2, 0])
    as_desc = tf.argsort(Tensor([3.0, 1.0, 2.0]), direction="DESCENDING")
    assert as_desc is not None
    debugging.assert_equal(as_desc, [0, 2, 1])

    assert tf.sort(None) is None
    s_asc = tf.sort(Tensor([3.0, 1.0, 2.0]))
    assert s_asc is not None
    debugging.assert_near(s_asc, [1.0, 2.0, 3.0])
    s_desc = tf.sort(Tensor([3.0, 1.0, 2.0]), direction="DESCENDING")
    assert s_desc is not None
    debugging.assert_near(s_desc, [3.0, 2.0, 1.0])

    assert tf.searchsorted(None, None) is None
    ss_left = tf.searchsorted(Tensor([1.0, 3.0, 5.0]), Tensor([2.0, 3.0, 6.0]), side="left")
    assert ss_left is not None
    debugging.assert_equal(ss_left, [1, 1, 3])
    ss_right = tf.searchsorted(Tensor([1.0, 3.0]), Tensor(3.0), side="right")
    assert ss_right is not None and ss_right.numpy() == 2

    assert tf.as_dtype(None) is None
    assert tf.as_dtype("float32") == tf.float32
    assert tf.as_string(None) is None
    assert isinstance(tf.as_string(Tensor([1, 2])), Tensor)
    assert isinstance(tf.as_string(Tensor(42)), Tensor)

    assert tf.batch_to_space(None) is None
    bts_in = tf.ones((4, 1, 1, 1), dtype=tf.float32)
    bts_out = tf.batch_to_space(bts_in, crops=[[0, 0]], block_shape=[2, 2])
    assert bts_out is not None and bts_out.shape == (1, 2, 2, 1)
    bts_crop = tf.batch_to_space(bts_in, crops=[[0, 1], [0, 1]], block_shape=2)
    assert bts_crop is not None and bts_crop.shape == (1, 1, 1, 1)
    bts_1d = tf.batch_to_space(Tensor([1.0, 2.0]), block_shape=4)
    assert bts_1d is not None and bts_1d.shape == (2,)

    assert tf.nn.space_to_depth(None) is None
    s2d_1d = tf.nn.space_to_depth(Tensor([1.0, 2.0]), block_size=2)
    assert s2d_1d is not None and s2d_1d.shape == (2,)
    s2d_in = tf.ones((1, 4, 4, 1), dtype=tf.float32)
    s2d_out = tf.nn.space_to_depth(s2d_in, block_size=2)
    assert s2d_out is not None and s2d_out.shape == (1, 2, 2, 4)

    assert tf.nn.depth_to_space(None) is None
    d2s_1d = tf.nn.depth_to_space(Tensor([1.0, 2.0]), block_size=2)
    assert d2s_1d is not None and d2s_1d.shape == (2,)
    d2s_out = tf.nn.depth_to_space(s2d_out, block_size=2)
    assert d2s_out is not None and d2s_out.shape == (1, 4, 4, 1)

    assert tf.boolean_mask(None, None) is None
    bm_nomask = tf.boolean_mask(Tensor([1.0, 2.0]), None)
    assert bm_nomask is not None and bm_nomask.shape == (2,)
    bm_res = tf.boolean_mask(Tensor([10.0, 20.0, 30.0]), Tensor([True, False, True]))
    assert bm_res is not None
    debugging.assert_near(bm_res, [10.0, 30.0])

    assert tf.broadcast_dynamic_shape(None, None) is None
    b_dyn = tf.broadcast_dynamic_shape([1, 3], [2, 1])
    assert b_dyn is not None
    debugging.assert_equal(b_dyn, [2, 3])
    with pytest.raises(ValueError, match="Incompatible shapes"):
        tf.broadcast_dynamic_shape([2, 3], [4, 3])

    assert tf.broadcast_static_shape(None, None) is None
    assert tf.broadcast_static_shape([1, 3], [2, 1]) == tf.TensorShape([2, 3])
    assert tf.broadcast_static_shape([None, 3], [2, 1]) == tf.TensorShape([None, 3])
    with pytest.raises(ValueError, match="Incompatible static shapes"):
        tf.broadcast_static_shape([2, 3], [4, 3])

    assert tf.clip_by_value(None) is None
    cbv = tf.clip_by_value(Tensor([-1.0, 0.5, 2.0]), 0.0, 1.0)
    assert cbv is not None
    debugging.assert_near(cbv, [0.0, 0.5, 1.0])

    assert tf.clip_by_norm(None) is None
    cbn1 = tf.clip_by_norm(Tensor([3.0, 4.0]), 2.5)
    assert cbn1 is not None
    debugging.assert_near(cbn1, [1.5, 2.0])
    cbn2 = tf.clip_by_norm(Tensor([1.0, 1.0]), 5.0)
    assert cbn2 is not None
    debugging.assert_near(cbn2, [1.0, 1.0])

    assert tf.clip_by_global_norm(None) is None
    cgn1 = tf.clip_by_global_norm([Tensor([3.0]), Tensor([4.0])], 2.5)
    assert cgn1 is not None
    c_list, g_norm = cgn1
    assert g_norm.numpy() == 5.0
    debugging.assert_near(c_list[0], [1.5])
    cgn2 = tf.clip_by_global_norm([Tensor([3.0])], 10.0, use_norm=Tensor(3.0))
    assert cgn2 is not None and cgn2[1].numpy() == 3.0

    assert tf.complex(None, None) is None
    cplx0 = tf.complex(Tensor(1.0), Tensor(2.0))
    assert cplx0 is not None and cplx0.numpy() == complex(1.0, 2.0)
    cplx1 = tf.complex(Tensor([1.0, 3.0]), Tensor([2.0, 4.0]))
    assert cplx1 is not None and cplx1.shape == (2,)

    assert tf.constant(None) is None
    c_arr = tf.constant([1.0, 2.0])
    assert c_arr is not None and c_arr.shape == (2,)

    class _FakeConstArray:
        """Pure-Python 1-D `ArrayLikeProtocol` test helper for `tf.constant`."""

        @property
        def shape(self) -> tuple[int, ...]:
            """Return 1-D shape."""
            return (2,)

        @property
        def size(self) -> int:
            """Return element count."""
            return 2

        def tolist(self) -> list[float]:
            """Return Python list."""
            return [1.0, 2.0]

        def item(self) -> float:
            """Return first element."""
            return 1.0

    c_proto = tf.constant(_FakeConstArray())
    assert c_proto is not None and c_proto.shape == (2,)
    c_fill = tf.constant(5.0, shape=(2, 2))
    assert c_fill is not None and c_fill.shape == (2, 2)
    c_resh = tf.constant([1.0, 2.0, 3.0, 4.0], shape=(2, 2))
    assert c_resh is not None and c_resh.shape == (2, 2)

    assert tf.convert_to_tensor(None) is None
    ctt = tf.convert_to_tensor([1.0, 2.0], dtype_hint=tf.float32)
    assert ctt is not None and ctt.dtype == tf.float32

    assert tf.constant_initializer(3.0)((2,)).shape == (2,)
    assert tf.ones_initializer()((2,)).shape == (2,)
    assert tf.zeros_initializer()((2,)).shape == (2,)
    assert isinstance(
        tf.random_normal_initializer(0.0, 1.0, seed=1),
        tf.keras.initializers.RandomNormal,
    )
    assert isinstance(
        tf.random_uniform_initializer(-1.0, 1.0, seed=1),
        tf.keras.initializers.RandomUniform,
    )

    conv_in = tf.ones((1, 4, 4, 1), dtype=tf.float32)
    conv_k = tf.ones((2, 2, 1, 1), dtype=tf.float32)
    conv_res = tf.conv(conv_in, conv_k)
    assert isinstance(conv_res, Tensor) and conv_res.shape == (1, 3, 3, 1)
    assert tf.conv2d_backprop_filter_v2(None, None) is None
    cbf = tf.conv2d_backprop_filter_v2(conv_in, [2, 2, 1, 1], conv_in)
    assert cbf is not None and cbf.shape == (2, 2, 1, 1)
    assert tf.conv2d_backprop_input_v2(None, None) is None
    cbi = tf.conv2d_backprop_input_v2([1, 4, 4, 1], conv_k, conv_in)
    assert cbi is not None and cbi.shape == (1, 4, 4, 1)

    cs_res = tf.cumsum(Tensor([1.0, 2.0, 3.0]))
    assert isinstance(cs_res, Tensor)
    debugging.assert_near(cs_res, [1.0, 3.0, 6.0])

    assert tf.dynamic_partition(None, None) is None
    parts = tf.dynamic_partition(Tensor([10.0, 20.0, 30.0]), Tensor([0, 99, 0]), num_partitions=2)
    assert isinstance(parts, list) and len(parts) == 2
    debugging.assert_near(parts[0], [10.0, 30.0])

    assert tf.dynamic_stitch(None, None) is None
    ds_empty = tf.dynamic_stitch([], [])
    assert ds_empty is not None and ds_empty.shape == (0,)
    stitched = tf.dynamic_stitch(
        [Tensor([0, 2, -1]), Tensor([1])], [Tensor([10.0, 30.0, 99.0]), Tensor([20.0])]
    )
    assert stitched is not None
    debugging.assert_near(stitched, [10.0, 20.0, 30.0])

    assert tf.edit_distance(None, None) is None
    sp_a = SparseTensor(indices=[[0, 0], [0, 1]], values=[1, 2], dense_shape=[1, 2])
    sp_b = SparseTensor(indices=[[0, 0], [0, 1]], values=[1, 3], dense_shape=[1, 2])
    ed = tf.edit_distance(sp_a, sp_b, normalize=True)
    assert ed is not None
    debugging.assert_near(ed, [0.5])
    ed_unnorm = tf.edit_distance(sp_a, sp_b, normalize=False)
    assert ed_unnorm is not None
    debugging.assert_near(ed_unnorm, [1.0])

    mat = Tensor([[2.0, 0.0], [0.0, 3.0]], dtype=tf.float32)
    assert tf.eig(mat) is not None
    assert tf.eigvals(mat) is not None
    assert tf.linalg.eigh(mat) is not None
    assert tf.linalg.pinv(mat) is not None
    assert tf.linalg.slogdet(mat) is not None

    assert tf.ensure_shape(None, None) is None
    es1 = tf.ensure_shape(Tensor([1.0, 2.0]), None)
    assert es1 is not None and es1.shape == (2,)
    es2 = tf.ensure_shape(Tensor([1.0, 2.0]), [2])
    assert es2 is not None and es2.shape == (2,)
    with pytest.raises(ValueError, match="not compatible"):
        tf.ensure_shape(Tensor([1.0, 2.0]), [3])

    assert tf.extract_volume_patches(None) is None
    vol = tf.ones((1, 2, 2, 2, 1), dtype=tf.float32)
    assert tf.extract_volume_patches(vol, [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], "VALID") is not None
    assert tf.extract_volume_patches(vol, [1, 2, 2, 2, 1], [1, 1, 1, 1, 1], "SAME") is not None
    assert tf.extract_volume_patches(vol, [1, 5, 5, 5, 1], [1, 1, 1, 1, 1], "VALID") is not None
    assert tf.extract_volume_patches(Tensor([1.0, 2.0])) is not None
    assert tf.fftnd(None) is None
    assert tf.fftnd(Tensor([1.0, 0.0])) is not None
    assert tf.ifftnd(None) is None
    assert tf.ifftnd(Tensor([1.0, 0.0])) is not None
    assert tf.rfftnd(None) is None
    assert tf.rfftnd(Tensor([1.0, 0.0])) is not None

    assert tf.fingerprint(None) is None
    fp1 = tf.fingerprint(Tensor([[1, 2], [3, 4]]))
    assert fp1 is not None and fp1.shape == (2, 8)
    fp2 = tf.fingerprint(Tensor(5))
    assert fp2 is not None and fp2.shape == (1, 8)

    assert tf.histogram_fixed_width_bins(None, None) is None
    bins = tf.histogram_fixed_width_bins(Tensor([-1.0, 0.2, 0.8, 2.0]), [0.0, 1.0], nbins=2)
    assert bins is not None
    debugging.assert_equal(bins, [0, 0, 1, 1])
    bins_scalar = tf.histogram_fixed_width_bins(Tensor(0.5), [0.0, 1.0], nbins=2)
    assert bins_scalar is not None and bins_scalar.shape == ()

    assert tf.histogram_fixed_width(None, None) is None
    hist = tf.histogram_fixed_width(Tensor([-1.0, 0.2, 0.8, 2.0]), [0.0, 1.0], nbins=2)
    assert hist is not None
    debugging.assert_equal(hist, [2, 2])

    assert tf.make_tensor_proto(None) is None
    proto = tf.make_tensor_proto([1.0, 2.0], dtype=tf.float32)
    assert proto is not None and proto["shape"] == [2]
    assert tf.make_ndarray(None) is None
    nd_proto = tf.make_ndarray(proto)
    assert nd_proto is not None
    debugging.assert_near(nd_proto, [1.0, 2.0])
    nd_t = tf.make_ndarray(Tensor([3.0, 4.0]))
    assert nd_t is not None
    debugging.assert_near(nd_t, [3.0, 4.0])

    assert tf.matrix_square_root(None) is None
    msqrt = tf.matrix_square_root(Tensor([[4.0, 0.0], [0.0, 9.0]]))
    assert msqrt is not None
    debugging.assert_near(msqrt, [[2.0, 0.0], [0.0, 3.0]])
    tf.newaxis()

    assert tf.one_hot(None, 3) is None
    oh = tf.one_hot(Tensor([0, 2]), depth=3)
    assert oh is not None
    debugging.assert_near(oh, [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    oh_ax0 = tf.one_hot(Tensor([0, -1]), depth=2, axis=0)
    assert oh_ax0 is not None and oh_ax0.shape == (2, 2)
    debugging.assert_near(oh_ax0, [[1.0, 0.0], [0.0, 0.0]])

    padded = tf.pad(Tensor([[1.0]]), [[0, 1], [1, 0]])
    assert isinstance(padded, Tensor)
    debugging.assert_near(padded, [[0.0, 1.0], [0.0, 0.0]])

    assert tf.parallel_stack(None) is None
    ps1 = tf.parallel_stack([Tensor([1.0, 2.0]), Tensor([3.0, 4.0])])
    assert ps1 is not None and ps1.shape == (2, 2)
    ps2 = tf.parallel_stack(Tensor([1.0, 2.0]))
    assert ps2 is not None and ps2.shape == (2,)

    for dt_alias in (
        tf.half,
        tf.qint8,
        tf.qint16,
        tf.qint32,
        tf.quint8,
        tf.quint16,
        tf.uint16,
        tf.uint32,
        tf.uint64,
        tf.resource,
        tf.variant,
    ):
        assert isinstance(dt_alias, tf.DType)

    assert tf.random_index_shuffle(None, [1, 2], 10) is None
    ris0 = tf.random_index_shuffle(Tensor(0), [1, 2], 0)
    assert ris0 is not None and ris0.numpy() == 0
    shuf = tf.random_index_shuffle(Tensor([0, 1, 2]), [11, 22], 2)
    assert shuf is not None and shuf.shape == (3,)

    assert tf.rank(None) is None
    rk = tf.rank(Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert rk is not None and rk.numpy() == 2

    assert tf.realdiv(None, None) is None
    rdiv = tf.realdiv(Tensor([6.0, 9.0]), Tensor([2.0, 3.0]))
    assert rdiv is not None
    debugging.assert_near(rdiv, [3.0, 3.0])

    assert tf.truncatediv(None, None) is None
    tdiv0 = tf.truncatediv(Tensor(7), Tensor(2))
    assert tdiv0 is not None and tdiv0.numpy() == 3
    tdiv1 = tf.truncatediv(Tensor([-7, 7]), Tensor([2, 2]))
    assert tdiv1 is not None
    debugging.assert_equal(tdiv1, [-3, 3])

    assert tf.truncatemod(None, None) is None
    tmod0 = tf.truncatemod(Tensor(-7), Tensor(2))
    assert tmod0 is not None and tmod0.numpy() == -1
    tmod1 = tf.truncatemod(Tensor([-7, 7]), Tensor([2, 2]))
    assert tmod1 is not None
    debugging.assert_equal(tmod1, [-1, 1])

    r_lse = tf.reduce_logsumexp(Tensor([0.0, 0.0]))
    assert isinstance(r_lse, Tensor)
    debugging.assert_near(r_lse, math.log(2.0), rtol=1e-5)

    assert tf.required_space_to_batch_paddings(None, None) is None
    req_sb = tf.required_space_to_batch_paddings([5, 6], [2, 3], base_paddings=[[0, 0]])
    assert req_sb is not None
    pads, crops = req_sb
    debugging.assert_equal(pads, [[0, 1], [0, 0]])
    debugging.assert_equal(crops, [[0, 1], [0, 0]])

    rev_res = tf.reverse(Tensor([1.0, 2.0, 3.0]), [0])
    assert isinstance(rev_res, Tensor)
    debugging.assert_near(rev_res, [3.0, 2.0, 1.0])

    assert tf.reverse_sequence(None, None) is None
    rs_none = tf.reverse_sequence(Tensor([1.0, 2.0]), None)
    assert rs_none is not None and rs_none.shape == (2,)
    rev_seq = tf.reverse_sequence(
        Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
        seq_lengths=[2, 1],
        seq_axis=1,
    )
    assert rev_seq is not None
    debugging.assert_near(rev_seq, [[2.0, 1.0, 3.0], [4.0, 5.0, 6.0]])
    rs_ax0 = tf.reverse_sequence(Tensor([[1.0, 2.0]]), seq_lengths=[1], seq_axis=0)
    assert rs_ax0 is not None and rs_ax0.shape == (1, 2)

    assert tf.saturate_cast(None, None) is None
    sc_none = tf.saturate_cast(Tensor([1.0]), None)
    assert sc_none is not None and sc_none.shape == (1,)
    sc_u8 = tf.saturate_cast(Tensor([-10.0, 300.0]), tf.uint8)
    assert sc_u8 is not None
    debugging.assert_equal(sc_u8, [0, 255])
    sc_f32 = tf.saturate_cast(Tensor([1.5]), tf.float32)
    assert sc_f32 is not None and sc_f32.dtype == tf.float32

    assert tf.sequence_mask(None) is None
    sm1 = tf.sequence_mask(2, maxlen=4)
    assert sm1 is not None
    debugging.assert_equal(sm1, [True, True, False, False])
    sm2 = tf.sequence_mask([1, 2], maxlen=2, dtype=tf.int32)
    assert sm2 is not None
    debugging.assert_equal(sm2, [[1, 0], [1, 1]])

    assert tf.shape_n(None) is None
    shapes = tf.shape_n([Tensor([1.0, 2.0]), Tensor([[1.0]])])
    assert shapes is not None and len(shapes) == 2 and shapes[0].shape == (1,)

    sig_res = tf.sigmoid(Tensor([0.0]))
    assert isinstance(sig_res, Tensor)
    debugging.assert_near(sig_res, [0.5])

    assert tf.tensor_scatter_nd_update(None, None, None) is None
    tsu_none = tf.tensor_scatter_nd_update(Tensor([1.0, 2.0]), None, None)
    assert tsu_none is not None and tsu_none.shape == (2,)
    tsu_0d = tf.tensor_scatter_nd_update(Tensor(1.0), [0], [5.0])
    assert tsu_0d is not None and tsu_0d.numpy() == 5.0
    tsu = tf.tensor_scatter_nd_update(Tensor([1.0, 2.0, 3.0]), [[1], [-1]], [9.0, 8.0])
    assert tsu is not None
    debugging.assert_near(tsu, [1.0, 9.0, 3.0])
    tsa = tf.tensor_scatter_nd_add(Tensor([1.0, 2.0, 3.0]), [[1]], [5.0])
    assert tsa is not None
    debugging.assert_near(tsa, [1.0, 7.0, 3.0])
    tss = tf.tensor_scatter_nd_sub(Tensor([1.0, 5.0, 3.0]), [[1]], [2.0])
    assert tss is not None
    debugging.assert_near(tss, [1.0, 3.0, 3.0])
    tsmax = tf.tensor_scatter_nd_max(Tensor([1.0, 5.0, 3.0]), [[1]], [8.0])
    assert tsmax is not None
    debugging.assert_near(tsmax, [1.0, 8.0, 3.0])
    tsmin = tf.tensor_scatter_nd_min(Tensor([1.0, 5.0, 3.0]), [[1]], [2.0])
    assert tsmin is not None
    debugging.assert_near(tsmin, [1.0, 2.0, 3.0])

    assert tf.unique(None) is None
    u_res = tf.unique(Tensor([10.0, 20.0, 10.0, 30.0]))
    assert isinstance(u_res, tuple)
    u_y, u_idx = u_res
    debugging.assert_near(u_y, [10.0, 20.0, 30.0])
    debugging.assert_equal(u_idx, [0, 1, 0, 2])

    assert tf.unique_with_counts(None) is None
    uc_res = tf.unique_with_counts(Tensor([10.0, 20.0, 10.0]))
    assert uc_res is not None
    uc_y, uc_idx, uc_cnt = uc_res
    debugging.assert_near(uc_y, [10.0, 20.0])
    debugging.assert_equal(uc_idx, [0, 1, 0])
    debugging.assert_equal(uc_cnt, [2, 1])

    assert tf.unravel_index(None, None) is None
    ui0 = tf.unravel_index(Tensor(5), [2, 3])
    assert ui0 is not None
    debugging.assert_equal(ui0, [1, 2])
    ui1 = tf.unravel_index(Tensor([2, 5]), [2, 3])
    assert ui1 is not None
    debugging.assert_equal(ui1, [[0, 1], [2, 2]])

    # Traced mode branch for boolean_mask and pad
    prev_tr = _tracer.is_tracing
    prev_gr = _tracer.active_graph
    _tracer.is_tracing = True
    _tracer.active_graph = LogicalGraph(name="traced_array_ops")
    try:
        t_sym = Tensor([1.0, 2.0])
        assert isinstance(tf.boolean_mask(t_sym, Tensor([True, False])), Tensor)
        assert isinstance(tf.pad(t_sym, [[1, 1]], mode="REFLECT"), Tensor)
    finally:
        _tracer.is_tracing = prev_tr
        _tracer.active_graph = prev_gr


def test_math_ops_behavioral() -> None:
    """Verify `tf.math` unary, binary, reduction, and special mathematical functions."""
    t_half = Tensor([0.5, 0.25], dtype=tf.float32)
    t_gt1 = Tensor([1.5, 2.0], dtype=tf.float32)
    t_pos = Tensor([2.0, 4.0], dtype=tf.float32)
    t_bool = Tensor([True, False])
    t_bool2 = Tensor([True, True])

    for unary_half in (
        tf.math.acos,
        tf.math.asin,
        tf.math.atan,
        tf.math.atanh,
        tf.math.erf,
        tf.math.erfc,
        tf.math.erfinv,
    ):
        assert isinstance(unary_half(t_half), Tensor)

    for unary_gt1 in (
        tf.math.acosh,
        tf.math.asinh,
        tf.math.ceil,
        tf.math.conj,
        tf.math.cosh,
        tf.math.count_nonzero,
        tf.math.digamma,
        tf.math.expm1,
        tf.math.floor,
        tf.math.imag,
        tf.math.log1p,
        tf.math.negative,
        tf.math.real,
        tf.math.reciprocal,
        tf.math.round,
        tf.math.rsqrt,
        tf.math.sign,
        tf.math.sinh,
        tf.math.reduce_std,
        tf.math.tan,
        tf.math.reduce_variance,
        tf.math.angle,
        tf.math.bessel_i0,
        tf.math.bessel_i0e,
        tf.math.bessel_i1,
        tf.math.bessel_i1e,
        tf.math.cumprod,
        tf.math.cumsum,
        tf.math.cumulative_logsumexp,
        tf.math.is_finite,
        tf.math.is_inf,
        tf.math.is_nan,
        tf.math.is_non_decreasing,
        tf.math.is_strictly_increasing,
        tf.math.l2_normalize,
        tf.math.log_sigmoid,
        tf.math.log_softmax,
        tf.math.reciprocal_no_nan,
        tf.math.reduce_euclidean_norm,
        tf.math.reduce_logsumexp,
        tf.math.rint,
        tf.math.softmax,
        tf.math.softplus,
        tf.math.softsign,
        tf.math.zero_fraction,
    ):
        assert isinstance(unary_gt1(t_gt1), Tensor)

    assert isinstance(tf.math.reduce_all(t_bool), Tensor)
    assert isinstance(tf.math.reduce_any(t_bool), Tensor)
    assert isinstance(tf.math.logical_xor(t_bool, t_bool2), Tensor)
    assert isinstance(tf.math.argmax(t_pos), Tensor)
    assert isinstance(tf.math.argmin(t_pos), Tensor)

    for bin_fn in (
        tf.math.atan2,
        tf.math.greater_equal,
        tf.math.less_equal,
        tf.math.mod,
        tf.math.nextafter,
        tf.math.divide_no_nan,
        tf.math.floordiv,
        tf.math.floormod,
        tf.math.igamma,
        tf.math.igammac,
        tf.math.multiply_no_nan,
        tf.math.polygamma,
        tf.math.squared_difference,
        tf.math.truediv,
        tf.math.xdivy,
        tf.math.xlog1py,
        tf.math.xlogy,
        tf.math.zeta,
    ):
        assert isinstance(bin_fn(t_gt1, t_pos), Tensor)

    assert isinstance(tf.math.accumulate_n([t_pos, t_gt1]), Tensor)
    assert isinstance(tf.math.add_n([t_pos, t_gt1]), Tensor)
    assert tf.math.approx_max_k(t_pos, 1) is not None
    assert tf.math.approx_min_k(t_pos, 1) is not None
    assert isinstance(
        tf.math.confusion_matrix(Tensor([0, 1], dtype=tf.int32), Tensor([0, 1], dtype=tf.int32)),
        Tensor,
    )
    assert isinstance(
        tf.math.in_top_k(Tensor([0], dtype=tf.int32), Tensor([[0.1, 0.9]]), 1),
        Tensor,
    )
    assert isinstance(tf.math.invert_permutation(Tensor([1, 0], dtype=tf.int32)), Tensor)
    assert isinstance(tf.math.polyval(Tensor(1.0), Tensor(3.0)), Tensor)
    debugging.assert_near(tf.math.polyval([2.0, 3.0, 1.0], Tensor([1.0, 2.0])), [6.0, 15.0])
    debugging.assert_near(tf.math.polyval(Tensor([2.0, 3.0, 1.0]), Tensor([1.0, 2.0])), [6.0, 15.0])
    debugging.assert_near(tf.math.polyval([], Tensor([1.0, 2.0])), [0.0, 0.0])
    debugging.assert_near(tf.math.polyval(None, Tensor([1.0, 2.0])), [0.0, 0.0])
    x_pv = tf.Variable([2.0, 3.0])
    with tf.GradientTape() as tape_pv:
        y_pv = tf.math.polyval([2.0, 3.0, 1.0], x_pv)
    g_pv = tape_pv.gradient(y_pv, x_pv)
    assert isinstance(g_pv, Tensor)
    debugging.assert_near(g_pv, [11.0, 15.0])

    # Bincount & Confusion Matrix behavioral verification
    debugging.assert_equal(
        tf.math.bincount(Tensor([1, 1, 2, 3], dtype=tf.int32), minlength=5, maxlength=3),
        [0, 2, 1],
    )
    debugging.assert_equal(
        tf.math.bincount(Tensor([1, 1, 2], dtype=tf.int32), binary_output=True),
        [0, 1, 1],
    )
    debugging.assert_near(
        tf.math.bincount(Tensor([0, 1, 1], dtype=tf.int32), weights=Tensor([0.5, 1.5, 2.0])),
        [0.5, 3.5],
    )
    debugging.assert_equal(
        tf.math.bincount(Tensor([[0, 1, 3], [2, 2, 0]], dtype=tf.int32), maxlength=2, axis=-1),
        [[1, 1], [1, 0]],
    )
    debugging.assert_equal(
        tf.math.bincount(
            Tensor([[0, 1, 1], [2, 2, 0]], dtype=tf.int32), axis=-1, binary_output=True
        ),
        [[1, 1, 0], [1, 0, 1]],
    )
    debugging.assert_near(
        tf.math.bincount(
            Tensor([[0, 1], [1, 0]], dtype=tf.int32),
            weights=Tensor([[1.0, 2.0], [3.0, 4.0]]),
            axis=-1,
        ),
        [[1.0, 2.0], [4.0, 3.0]],
    )
    debugging.assert_equal(
        tf.math.confusion_matrix([0, 1, 2, 1], [0, 2, 2, 1], num_classes=Tensor(3, dtype=tf.int32)),
        [[1, 0, 0], [0, 1, 1], [0, 0, 1]],
    )
    debugging.assert_equal(
        tf.math.confusion_matrix([0, 2], [0, 2], num_classes=1),
        [[1]],
    )
    debugging.assert_near(
        tf.math.confusion_matrix(
            [0, 1, 2], [0, 1, 2], num_classes=2, weights=[1.5, 2.5, 9.0], dtype=tf.float32
        ),
        [[1.5, 0.0], [0.0, 2.5]],
    )

    # Cumulative ops with exclusive and reverse
    mat_cum = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    debugging.assert_near(
        tf.math.cumsum(mat_cum, axis=-1, exclusive=True, reverse=True),
        [[5.0, 3.0, 0.0], [11.0, 6.0, 0.0]],
    )
    debugging.assert_near(
        tf.math.cumprod(mat_cum, axis=-1, exclusive=True, reverse=True),
        [[6.0, 3.0, 1.0], [30.0, 6.0, 1.0]],
    )
    debugging.assert_near(
        tf.math.cumulative_logsumexp(Tensor([0.0, 0.0]), axis=[0], exclusive=True, reverse=True),
        [0.0, float("-inf")],
    )

    # N-D Segment & Unsorted Segment reductions
    seg_data_2d = Tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    seg_ids = Tensor([0, 0, 2], dtype=tf.int32)
    debugging.assert_near(
        tf.math.segment_sum(seg_data_2d, seg_ids), [[4.0, 6.0], [0.0, 0.0], [5.0, 6.0]]
    )
    debugging.assert_near(
        tf.math.segment_mean(seg_data_2d, seg_ids), [[2.0, 3.0], [0.0, 0.0], [5.0, 6.0]]
    )
    debugging.assert_near(
        tf.math.segment_prod(seg_data_2d, seg_ids), [[3.0, 8.0], [1.0, 1.0], [5.0, 6.0]]
    )
    debugging.assert_near(
        tf.math.segment_min(seg_data_2d, seg_ids), [[1.0, 2.0], [0.0, 0.0], [5.0, 6.0]]
    )
    debugging.assert_near(
        tf.math.segment_max(seg_data_2d, seg_ids), [[3.0, 4.0], [0.0, 0.0], [5.0, 6.0]]
    )
    useg_ids = Tensor([1, 0, 1], dtype=tf.int32)
    num_s_t = Tensor(3, dtype=tf.int32)
    debugging.assert_near(
        tf.math.unsorted_segment_sum(seg_data_2d, Tensor([1, -1, 1], dtype=tf.int32), num_s_t),
        [[0.0, 0.0], [6.0, 8.0], [0.0, 0.0]],
    )
    debugging.assert_near(
        tf.math.unsorted_segment_mean(seg_data_2d, useg_ids, 2),
        [[3.0, 4.0], [3.0, 4.0]],
    )
    debugging.assert_near(
        tf.math.unsorted_segment_prod(seg_data_2d, useg_ids, 2),
        [[3.0, 4.0], [5.0, 12.0]],
    )
    debugging.assert_near(
        tf.math.unsorted_segment_min(seg_data_2d, useg_ids, 2),
        [[3.0, 4.0], [1.0, 2.0]],
    )
    debugging.assert_near(
        tf.math.unsorted_segment_max(seg_data_2d, useg_ids, 2),
        [[3.0, 4.0], [5.0, 6.0]],
    )
    assert tf.math.unsorted_segment_sqrt_n(seg_data_2d, useg_ids, 2).shape == (2, 2)

    # Special functions (dawsn, fresnel_cos, Fresnel_cos, fresnel_sin, spence, erfcinv, lbeta, lgamma, ndtri)
    debugging.assert_near(
        tf.math.dawsn(Tensor([0.0, 1.0, -1.0, 7.0])),
        [0.0, 0.5380795, -0.5380795, 0.0721899],
        atol=1e-4,
    )
    debugging.assert_near(tf.math.dawsn(Tensor(1, dtype=tf.int32)), 0.5380795, atol=1e-4)
    debugging.assert_near(
        tf.math.fresnel_cos(Tensor([0.0, 1.0, -1.0, 3.0])),
        [0.0, 0.7798934, -0.7798934, 0.6057207],
        atol=1e-4,
    )
    debugging.assert_near(tf.math.Fresnel_cos(Tensor([1.0])), [0.7798934], atol=1e-4)
    debugging.assert_near(
        tf.math.fresnel_sin(Tensor([0.0, 1.0, -1.0, 3.0])),
        [0.0, 0.4382591, -0.4382591, 0.4963131],
        atol=1e-4,
    )
    sp_out = tf.math.spence(Tensor([0.0, 0.25, 1.0, 1.25, 1.75, 4.0]))
    debugging.assert_near(
        sp_out,
        [1.6449341, 0.9784694, 0.0, -0.2359003, -0.6427613, -1.9393754],
        atol=1e-4,
    )
    sp_neg = tf.math.spence(Tensor(-1.0)).tolist()
    assert isinstance(sp_neg, float) and sp_neg != sp_neg
    debugging.assert_near(tf.math.erfcinv(Tensor([1.0, 0.5])), [0.0, 0.47693628], atol=1e-5)
    debugging.assert_near(tf.math.lgamma(Tensor([1.0, 2.0, 3.0])), [0.0, 0.0, 0.6931472], atol=1e-5)
    debugging.assert_near(tf.math.lbeta(Tensor(1.0)), 0.0)
    debugging.assert_near(
        tf.math.lbeta(Tensor([[2.0, 3.0], [1.0, 1.0]])), [-2.4849066, 0.0], atol=1e-5
    )
    debugging.assert_near(
        tf.math.betainc(Tensor([2.0]), Tensor([1.5]), Tensor([0.5])),
        [0.38128157],
        atol=1e-4,
    )
    debugging.assert_near(tf.math.ndtri(Tensor([0.5, 0.84134475])), [0.0, 1.0], atol=1e-5)
    debugging.assert_equal(tf.range(4), [0, 1, 2, 3])

    assert isinstance(tf.math.scalar_mul(2.0, t_pos), Tensor)

    mat2 = Tensor([[1.0, 2.0], [3.0, 4.0]], dtype=tf.float32)
    assert isinstance(tf.matmul(mat2, mat2), Tensor)
    debugging.assert_near(
        tf.matmul(mat2, mat2, transpose_a=True, transpose_b=True),
        [[7.0, 15.0], [10.0, 22.0]],
    )
    debugging.assert_near(
        tf.matmul(mat2, mat2, adjoint_a=True, adjoint_b=True),
        [[7.0, 15.0], [10.0, 22.0]],
    )
    assert isinstance(tf.tensordot(mat2, mat2, axes=1), Tensor)
    assert tf.meshgrid() == []
    mg1 = tf.meshgrid(t_pos)
    assert len(mg1) == 1 and mg1[0].shape == (2,)
    mg_xy = tf.meshgrid(Tensor([1.0, 2.0]), Tensor([3.0, 4.0, 5.0]), indexing="xy")
    assert len(mg_xy) == 2 and mg_xy[0].shape == (3, 2) and mg_xy[1].shape == (3, 2)
    mg_ij = tf.meshgrid(
        Tensor([1.0, 2.0]),
        Tensor([3.0, 4.0, 5.0]),
        Tensor([6.0, 7.0, 8.0, 9.0]),
        indexing="ij",
    )
    assert len(mg_ij) == 3 and mg_ij[0].shape == (2, 3, 4)
    with pytest.raises(ValueError, match="indexing"):
        tf.meshgrid(t_pos, t_gt1, indexing="invalid")

    for ord_v in ("euclidean", "fro", 1, 2, float("inf"), float("-inf"), 3):
        n_res = tf.norm(mat2, ord=ord_v, axis=[0, 1], keepdims=True)
        assert n_res.shape == (1, 1)
        ln_res = tf.linalg.norm(mat2, ord=ord_v, axis=(0, 1), keepdims=True)
        assert ln_res.shape == (1, 1)
    with pytest.raises((ValueError, TypeError)):
        tf.norm(mat2, ord="invalid_ord")
    normed, norm_v = tf.linalg.normalize(Tensor([[3.0, 4.0]]), ord=2, axis=1)
    debugging.assert_near(normed, [[0.6, 0.8]])
    debugging.assert_near(norm_v, [[5.0]])

    bm_ax1 = tf.boolean_mask(
        Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), Tensor([True, False, True]), axis=1
    )
    assert bm_ax1 is not None and bm_ax1.shape == (2, 2)
    debugging.assert_near(bm_ax1, [[1.0, 3.0], [4.0, 6.0]])
    bm_empty = tf.boolean_mask(Tensor([[1.0, 2.0], [3.0, 4.0]]), Tensor([False, False]), axis=-1)
    assert bm_empty is not None and bm_empty.shape == (2, 0)

    cgn_empty = tf.clip_by_global_norm([], 1.0)
    assert cgn_empty is not None and cgn_empty[0] == [] and cgn_empty[1].numpy() == 0.0

    with tf.GradientTape() as tape:
        x_clip = Tensor([3.0, 4.0])
        tape.watch(x_clip)
        y_clip = tf.clip_by_norm(x_clip, 2.5)
        assert y_clip is not None
        loss_clip = tf.reduce_sum(y_clip)
    g_clip = tape.gradient(loss_clip, x_clip)
    assert isinstance(g_clip, Tensor) and g_clip.shape == (2,)

    ds_no_items = tf.dynamic_stitch([Tensor([], dtype=tf.int32)], [Tensor([], dtype=tf.float32)])
    assert ds_no_items is not None and ds_no_items.shape == (0,)
    ds_2d = tf.dynamic_stitch(
        [Tensor([0, 2]), Tensor([1])],
        [Tensor([[1.0, 2.0], [5.0, 6.0]]), Tensor([[3.0, 4.0]])],
    )
    assert ds_2d is not None and ds_2d.shape == (3, 2)
    debugging.assert_near(ds_2d, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    ed_dense = tf.edit_distance(Tensor([[1, 2], [3, 4]]), Tensor([[1, 3], [3, 4]]), normalize=True)
    assert ed_dense is not None and ed_dense.shape == (2,)
    debugging.assert_near(ed_dense, [0.5, 0.0])

    es_t = tf.ensure_shape(Tensor([1.0, 2.0]), Tensor([2], dtype=tf.int32))
    assert es_t is not None and es_t.shape == (2,)
    es_ts = tf.ensure_shape(Tensor([1.0, 2.0]), tf.TensorShape([2]))
    assert es_ts is not None and es_ts.shape == (2,)

    sp_parts = tf.split(Tensor([[1, 2, 3, 4, 5, 6]]), [1, 2, 3], axis=1)
    assert [p.shape for p in sp_parts] == [(1, 1), (1, 2), (1, 3)]
    sp_infer = tf.split(Tensor([[1, 2, 3, 4, 5, 6]]), Tensor([2, -1, 1], dtype=tf.int32), axis=-1)
    assert [p.shape for p in sp_infer] == [(1, 2), (1, 3), (1, 1)]

    assert tf.squeeze(Tensor([[1.0, 2.0]]), axis=0).shape == (2,)
    with pytest.raises(ValueError, match="squeeze"):
        tf.squeeze(Tensor([[1.0, 2.0]]), axis=1)
    with pytest.raises(ValueError, match="squeeze"):
        tf.squeeze(Tensor([[1.0]]), axis=[0, 5])

    t_cplx = Tensor([[complex(1.0, 2.0), complex(3.0, 4.0)]])
    t_conj_T = tf.transpose(t_cplx, conjugate=True)
    assert t_conj_T.shape == (2, 1)
    assert t_conj_T.tolist() == [[complex(1.0, -2.0)], [complex(3.0, -4.0)]]

    w_0d_t = tf.where(Tensor(True))
    assert w_0d_t.shape == (1, 0)
    w_0d_f = tf.where(Tensor(False))
    assert w_0d_f.shape == (0, 0)
    w_2d = tf.where(Tensor([[True, False], [False, True]]))
    assert w_2d.shape == (2, 2) and w_2d.tolist() == [[0, 0], [1, 1]]
    w_2d_empty = tf.where(Tensor([[False, False], [False, False]]))
    assert w_2d_empty.shape == (0, 2)

    bc_0d = tf.bitcast(Tensor(1.0, dtype=tf.float32), tf.int32)
    assert bc_0d.shape == () and bc_0d.numpy() == 1065353216
    bc_inc = tf.bitcast(Tensor(0x01020304, dtype=tf.int32), tf.uint8)
    assert bc_inc.shape == (4,) and bc_inc.tolist() == [4, 3, 2, 1]
    bc_dec = tf.bitcast(Tensor([[4, 3, 2, 1]], dtype=tf.uint8), tf.int32)
    assert bc_dec.shape == (1,) and bc_dec.tolist() == [0x01020304]
    with pytest.raises(ValueError, match="Cannot bitcast"):
        tf.bitcast(Tensor([1, 2, 3], dtype=tf.uint8), tf.int32)

    g_batched = tf.gather(
        Tensor([[10, 20, 30], [40, 50, 60]]),
        Tensor([[2, 0], [1, 2]], dtype=tf.int32),
        batch_dims=1,
    )
    assert g_batched.shape == (2, 2) and g_batched.tolist() == [[30, 10], [50, 60]]
    g_neg_b = tf.gather(
        Tensor([[10, 20], [30, 40]]),
        Tensor([[1], [0]], dtype=tf.int32),
        axis=-1,
        batch_dims=-1,
    )
    assert g_neg_b.shape == (2, 1) and g_neg_b.tolist() == [[20], [30]]

    gnd_2d = tf.gather_nd(Tensor([[10, 20], [30, 40]]), Tensor([[0, 1], [1, 0]], dtype=tf.int32))
    assert gnd_2d.shape == (2,) and gnd_2d.tolist() == [20, 30]
    gnd_b1 = tf.gather_nd(
        Tensor([[10, 20, 30], [40, 50, 60]]),
        Tensor([[[2]], [[1]]], dtype=tf.int32),
        batch_dims=-1,
    )
    assert gnd_b1.shape == (2, 1) and gnd_b1.tolist() == [[30], [50]]
    gnd_0d = tf.gather_nd(Tensor(42), Tensor(0, dtype=tf.int32))
    assert isinstance(gnd_0d, Tensor)

    snd_2d = tf.scatter_nd(
        Tensor([[0, 1], [1, 0]], dtype=tf.int32),
        Tensor([5.0, 9.0]),
        [2, 2],
    )
    assert snd_2d.shape == (2, 2) and snd_2d.tolist() == [[0.0, 5.0], [9.0, 0.0]]

    p_ref = tf.pad(Tensor([[1, 2], [3, 4]]), [[1, 0], [0, 1]], mode="REFLECT")
    assert isinstance(p_ref, Tensor) and p_ref.shape == (3, 3)
    p_sym = tf.pad(Tensor([[1, 2], [3, 4]]), [[1, 0], [0, 1]], mode="SYMMETRIC")
    assert isinstance(p_sym, Tensor) and p_sym.shape == (3, 3)
    p_cst = tf.pad(Tensor([1, 2]), [[1, 1]], mode="CONSTANT", constant_values=9)
    assert isinstance(p_cst, Tensor) and p_cst.tolist() == [9, 1, 2, 9]

    fl_2d = tf.foldl(lambda a, b: a + b, Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert fl_2d is not None and fl_2d.shape == (2,) and fl_2d.tolist() == [4.0, 6.0]
    fr_2d = tf.foldr(lambda a, b: a + b, Tensor([[1.0, 2.0], [3.0, 4.0]]))
    assert fr_2d is not None and fr_2d.shape == (2,) and fr_2d.tolist() == [4.0, 6.0]

    assert isinstance(tf.ones_like(t_pos), Tensor)
    assert isinstance(tf.zeros_like(t_pos), Tensor)
