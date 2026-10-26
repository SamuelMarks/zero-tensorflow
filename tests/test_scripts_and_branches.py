"""Unit tests for `scripts/*.py` and remaining edge branches in `src/zero_tensorflow/`."""

from __future__ import annotations

import types
from collections.abc import Sequence
from pathlib import Path

import scripts.check_api_parity as check_api_parity
import scripts.check_imports as check_imports
import scripts.generate_api_snapshot as generate_api_snapshot
import scripts.rebuild_api as rebuild_api
import scripts.rebuild_init as rebuild_init
import scripts.update_badges as update_badges
import zero_tensorflow as tf
import zero_tensorflow._typing as _typing
from zero_tensorflow import data, image, io, nn, random, sparse, strings, train

import ml_switcheroo_compiler.core as msc_core
import ml_switcheroo_compiler.ops as msc_ops
import pytest
from ml_switcheroo_compiler.tracing.state import global_tracing_state as _tracer
from ml_switcheroo_compiler.tracing.tracer import ProxyTensor
from ml_switcheroo_ir import LogicalGraph


class _DummyDataWrapper:
    """Wrapper implementing `_DataWrapperProtocol` for testing `nn._unwrap_k`."""

    def __init__(self, tensor: msc_core.Tensor) -> None:
        """Initialize `_DummyDataWrapper`.

        Args:
            tensor: Compiler tensor to wrap.
        """
        self._data = tensor

    @property
    def data(self) -> msc_core.Tensor:
        """Return wrapped compiler tensor.

        Returns:
            msc_core.Tensor: Wrapped compiler tensor.
        """
        return self._data


class _DummyContainer:
    """Container implementing `TrackableContainerProtocol` for checkpoint tests."""

    def __init__(self, variables: Sequence[tf.Variable]) -> None:
        """Initialize `_DummyContainer`.

        Args:
            variables: Sequence of tracked variables.
        """
        self._vars = list(variables)

    @property
    def variables(self) -> Sequence[tf.Variable]:
        """Return tracked variables.

        Returns:
            Sequence[tf.Variable]: Tracked variables.
        """
        return self._vars


class _NonNumpyAssignable:
    """Assignable object that does not implement `NumpyConvertibleProtocol`."""

    def assign(self, value: _typing.TensorOperand) -> _typing.TensorOperand:
        """Assign value.

        Args:
            value: Value to assign.

        Returns:
            _typing.TensorOperand: Assigned value.
        """
        return value


class _NonNumpyContainer:
    """Container holding a non-numpy assignable variable."""

    @property
    def variables(self) -> Sequence[_typing.AssignableProtocol]:
        """Return non-numpy assignable variables.

        Returns:
            Sequence[_typing.AssignableProtocol]: Sequence of assignables.
        """
        return [_NonNumpyAssignable()]


def test_remaining_src_branches(tmp_path: Path) -> None:
    """Exercise remaining lines and branches across `src/zero_tensorflow/`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert _typing.as_operand(10) == 10
    assert _typing.as_operand([1, 2]) == [1, 2]
    with pytest.raises(TypeError, match="Unsupported tensor operand type"):
        _typing.as_operand({"bad": 1})

    assert tf._to_attribute_value(1 + 2j) == "(1+2j)"
    assert tf._extract_constant_value(5) == 5
    fallback_t = tf._to_tensor(5.0)
    assert fallback_t is not None

    # Core tensor branches: custom class dtype, bad ProxyTensor dtype, reflected ops
    class int8:  # noqa: N801
        """Custom class whose lowercase name matches a DType alias."""

    assert tf.to_dtype(int8) == tf.int8
    bad_pt = ProxyTensor(id="bad_dt", shape=(), dtype="unknown_dtype_xyz")
    assert tf._to_tensor(bad_pt).dtype == tf.float32
    empty_pt = ProxyTensor(id="empty_dt", shape=(), dtype="")
    assert tf._to_tensor(empty_pt).dtype == tf.float32

    t_val = tf.Tensor(4.0)
    assert (-t_val).numpy() == -4.0
    assert (t_val**2.0).numpy() == 16.0
    assert (2.0 + t_val).numpy() == 6.0
    assert (10.0 - t_val).numpy() == 6.0
    assert (3.0 * t_val).numpy() == 12.0
    assert (8.0 / t_val).numpy() == 2.0
    assert bool(tf.Tensor(1.5)) is True

    # Autodiff branches: kwargs in @tf.function, nested tracing, unconnected zero
    @tf.function
    def fn_with_kw(a: int, b: float = 2.0) -> str:
        """Trace function with keyword argument and string return.

        Args:
            a: Integer input.
            b: Float keyword input.

        Returns:
            str: Constant status string.
        """
        _ = tf.Tensor(a) + tf.Tensor(b)
        return "done"

    assert fn_with_kw(1, b=3.0) == "done"

    @tf.function
    def fn_nested() -> tuple[tf.Tensor, str]:
        """Trace nested function returning tuple with non-tensor item.

        Returns:
            tuple[tf.Tensor, str]: Sum tensor and inner status string.
        """
        inner_res = fn_with_kw(2)
        return (tf.Tensor(1.0) + tf.Tensor(2.0), inner_res)

    nested_out = fn_nested()
    assert nested_out[1] == "done"

    @tf.function
    def fn_tensor_out(x_in: tf.Tensor, y_in: tf.Tensor) -> tf.Tensor:
        """Trace function returning a single traced `Tensor` node.

        Args:
            x_in: First input tensor.
            y_in: Second input tensor.

        Returns:
            tf.Tensor: Elementwise sum tensor.
        """
        return x_in + y_in

    tensor_out = fn_tensor_out(tf.Tensor(1.0), tf.Tensor(2.0))
    assert tensor_out.numpy() == 3.0

    empty_t = tf.Tensor()
    x_var = tf.Variable(2.0)
    with tf.GradientTape(persistent=True) as tape:
        with tf.GradientTape() as inner_tape:
            inner_tape.watch(x_var)
            inner_tape.watch(x_var)
            inner_tape.watch("non_tensor")
        tape.watch(x_var)
        assert tape.watched_variables() == (x_var,)
        y_out = x_var * 2.0
        const_y = tf.Tensor(_traced_node=tape._graph.nodes[next(iter(tape._graph.nodes))])
        tape.reset()
        tape.watch(x_var)
        y_out = x_var * 2.0
        const_id = next(iter(tape._graph.nodes))
        const_y = tf.Tensor(_traced_node=tape._graph.nodes[const_id])

    grads = tape.gradient(
        y_out,
        [x_var, empty_t, "non_tensor"],
        output_gradients=empty_t,
    )
    assert isinstance(grads, list) and grads[1] is None and grads[2] is None
    grads_cot = tape.gradient(
        y_out,
        [x_var],
        output_gradients=tf.Tensor(3.0),
    )
    assert isinstance(grads_cot, list) and grads_cot[0] is not None
    assert len(tf.hessians([], [x_var])) == 1
    disconnected_var = tf.Variable(9.0)
    with tf.GradientTape(persistent=True) as tape_disc:
        tape_disc.watch(x_var)
        tape_disc.watch(disconnected_var)
        y_disc = x_var * 2.0
        h_untracked = tf.hessians(y_disc, [empty_t])
        assert len(h_untracked) == 1
        assert len(tf.hessians([], [x_var])) == 1
    assert tape_disc.gradient(y_disc, disconnected_var, unconnected_gradients="none") is None
    zero_g = tape_disc.gradient(y_disc, disconnected_var, unconnected_gradients="zero")
    assert isinstance(zero_g, tf.Tensor) and zero_g.numpy() == 0.0
    assert const_y is not None

    tape.reset()
    assert tape.watched_variables() == ()

    with tf.GradientTape() as tape_sub:
        tape_sub.watch(x_var)
        y_sub = x_var - 1.0
    assert tape_sub.gradient(y_sub, x_var) is not None

    # Bitwise, Linalg, and Math ops remaining branches
    assert isinstance(tf.bitwise.invert(tf.Tensor([1, 2], dtype=tf.int32)), tf.Tensor)
    sym_mat = tf.Tensor([[2.0, 0.0], [0.0, 3.0]])
    assert isinstance(tf.linalg.eigvalsh(sym_mat), tf.Tensor)
    assert isinstance(tf.linalg.einsum("ij,jk->ik", sym_mat, sym_mat), tf.Tensor)
    assert isinstance(tf.linalg.eye(2), tf.Tensor)
    assert isinstance(tf.linalg.matrix_transpose(sym_mat), tf.Tensor)
    assert isinstance(tf.linalg.solve(sym_mat, sym_mat), tf.Tensor)

    t_pos = tf.Tensor([1.5, 2.5])
    t_unit = tf.Tensor([0.2, 0.5])
    t_int = tf.Tensor([0, 1], dtype=tf.int32)
    prev_tr = _tracer.is_tracing
    prev_gr = _tracer.active_graph
    _tracer.is_tracing = True
    _tracer.active_graph = LogicalGraph(name="math_trace")
    try:
        assert isinstance(tf.linalg.lstsq(sym_mat, sym_mat), tf.Tensor)
        assert tf.math.lgamma(t_pos) is not None
        assert tf.math.betainc(t_pos, t_pos, t_unit) is not None
        assert tf.math.bincount(t_int) is not None
        assert tf.math.erfcinv(t_unit) is not None
        assert tf.math.lbeta(tf.Tensor([[1.5, 2.5]])) is not None
        assert tf.math.ndtri(t_unit) is not None
        assert tf.math.segment_max(t_pos, t_int) is not None
        assert tf.math.segment_mean(t_pos, t_int) is not None
        assert tf.math.segment_min(t_pos, t_int) is not None
        assert tf.math.segment_prod(t_pos, t_int) is not None
        assert tf.math.sobol_sample(2, 2) is not None
        assert tf.math.top_k(t_pos, k=1) is not None
        assert tf.math.unsorted_segment_max(t_pos, t_int, 2) is not None
        assert tf.math.unsorted_segment_mean(t_pos, t_int, 2) is not None
        assert tf.math.unsorted_segment_min(t_pos, t_int, 2) is not None
        assert tf.math.unsorted_segment_prod(t_pos, t_int, 2) is not None
        assert tf.math.unsorted_segment_sqrt_n(t_pos, t_int, 2) is not None
        assert tf.math.unsorted_segment_sum(t_pos, t_int, 2) is not None
        assert tf.bitcast(t_pos, tf.float32) is not None
        assert tf.cast(t_pos, tf.float64) is not None
        assert tf.einsum("i,i->", t_pos, t_pos) is not None
        assert tf.gather(t_pos, t_int) is not None
        assert tf.gather_nd(t_pos, [[0], [1]]) is not None
        assert tf.identity(t_pos) is not None
        assert tf.repeat(t_pos, 2) is not None
        assert tf.roll(t_pos, shift=1, axis=0) is not None
        assert tf.scatter_nd([[0], [1]], t_pos, [2]) is not None
    finally:
        _tracer.is_tracing = prev_tr
        _tracer.active_graph = prev_gr

    assert tf.slice(t_pos, [0], [-1]) is not None
    assert tf.slice(t_pos, 0, 0, 1, 1) is not None
    assert tf.strided_slice(t_pos) is not None
    assert tf.concat(t_pos, axis=0) is not None
    assert tf.stack(t_pos, axis=0) is not None

    # Data remaining branches
    assert list(data.NumpyIterator(data.Dataset.from_tensor_slices([1, 2]))) == [1, 2]
    ds_of_ds = data.TensorSliceDataset([data.Dataset.from_tensor_slices([1])])
    assert list(ds_of_ds.batch(1)) == []
    assert list(data.Dataset.from_tensor_slices([1, 2]).unbatch()) == [1, 2]
    assert list(ds_of_ds.enumerate()) == []
    assert list(data.Dataset.from_tensor_slices([1, 2]).take(5)) == [1, 2]
    assert data._iter_file_paths(123) == []
    short_bin = tmp_path / "short.bin"
    short_bin.write_bytes(b"123")
    assert list(data.FixedLengthRecordDataset(str(short_bin), record_bytes=2)) == [b"12"]
    assert data.experimental.at(data.Dataset.from_tensor_slices([1]), 5) is None
    tw_all = data.experimental.take_while(lambda x: True)
    assert tw_all is not None
    assert list(data.Dataset.from_tensor_slices([1, 2]).apply(tw_all)) == [1, 2]
    tf.distribute.experimental_set_strategy("not_a_strategy")

    # Random remaining branches
    assert random._extract_seed(None) is None
    assert random._extract_seed([]) == 0
    assert random.all_candidate_sampler(num_sampled=2, seed=5).shape == (2,)
    assert random.fold_in([], 5) == [0, 5]
    assert random.split([], num=1) == [[0, 0]]

    # Sparse remaining branches
    empty_sp = sparse.SparseTensor(None, None, None)
    assert sparse._extract_values(empty_sp).shape == (1,)
    assert sparse._extract_values(tf.Tensor([1.0, 2.0])).shape == (2,)

    # Strings remaining branches
    assert strings._is_int_dtype(int) is True
    assert strings._is_int_dtype("int32") is True
    assert strings.ngrams(["a", "b"], 0) == []
    assert strings.ngrams(["a"], [2, 1], preserve_short_sequences=False) == ["a"]
    assert strings.ngrams(["a", "b"], 2, pad_values="PAD") == ["PAD a", "a b", "b PAD"]
    assert strings.ngrams(["a"], 3, preserve_short_sequences=False) == []
    assert strings._decode_text_scalar([b"a", "b"], "utf-8", "replace") == "ab"
    assert strings._decode_text_scalar(b"abc", "utf-8", "replace") == "abc"
    assert strings._decode_text_scalar(None, "utf-8", "replace") == ""
    assert strings.unicode_encode([[97, 98], "skip"]) == ["ab"]
    assert strings.unicode_encode(97) == ["a"]
    assert strings.unicode_encode("invalid") == []
    assert strings.unicode_script(97) == []
    assert strings.unicode_transcode("", "utf-8", "utf-8") == []
    assert strings.unsorted_segment_join(["a"], [0], 0) == []
    assert strings.unsorted_segment_join(["a", "b"], [0, 5], 2) == ["a", ""]
    assert strings.unsorted_segment_join("not_a_list", 0, 1) == [""]

    # IO remaining branches
    valid_png_hdr = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x02\x00\x00\x00\x02\x08\x02\x00\x00\x00"
    )
    assert io.decode_png(valid_png_hdr) is not None
    assert io.decode_image(valid_png_hdr) is not None
    gz_file = tmp_path / "wb_str.txt"
    with io.gfile.GFile(str(gz_file), "wb") as gf_str_wb:
        gf_str_wb.write("str_in_binary_mode")
    assert gz_file.read_bytes() == b"str_in_binary_mode"
    assert io.gfile.join("only_one") == "only_one"
    assert io.gfile.listdir("nonexistent_dir_xyz") == []
    io.gfile.mkdir("")
    io.gfile.remove("nonexistent_file_xyz")
    io.gfile.rename("", "")
    io.gfile.rmtree("nonexistent_dir_xyz")
    assert list(io.gfile.walk("nonexistent_dir_xyz")) == []

    # Train remaining branches
    mgr_no_ckpt = train.CheckpointManager(None, "", max_to_keep=None)
    assert mgr_no_ckpt.save("custom_prefix") == "custom_prefix"
    assert mgr_no_ckpt.save("custom_prefix") == "custom_prefix"
    ema_empty = train.ExponentialMovingAverage()
    ema_empty.apply(None)
    ema_empty.apply(["non_numeric"])
    ema_empty.apply(["non_numeric"])
    train._CHECKPOINT_STORE["standalone_dir/ckpt-1"] = {
        "arr": tf.Tensor([1.0, 2.0]).numpy(),
        "seq": [1, 2, 3],
        "scal": 5.0,
    }
    assert train.latest_checkpoint("standalone_dir") == "standalone_dir/ckpt-1"
    assert train.latest_checkpoint("missing_dir_xyz") is None
    assert train.load_checkpoint("missing_dir_xyz") is None
    vars_info = train.list_variables("standalone_dir/ckpt-1")
    assert ("arr", [2]) in vars_info and ("seq", [3]) in vars_info and ("scal", []) in vars_info
    assert train.load_variable("standalone_dir/ckpt-1", "missing_var") is None

    raw_t = msc_ops.zeros((2, 2, 3))
    assert image._ensure_tensor(raw_t) is raw_t
    assert image._ensure_tensor(tf.Tensor(raw_t)) is raw_t

    file_p = tmp_path / "tensor_out.txt"
    io.write_file(str(file_p), raw_t)
    assert io.decode_base64(msc_ops.zeros((1,))) is not None
    assert io.decode_bmp(b"") is not None
    assert io.decode_csv(msc_ops.zeros((1,)), [msc_ops.zeros((1,))]) is not None
    assert io.decode_gif(b"") is not None
    assert io.encode_base64(msc_ops.zeros((1,))) is not None

    src_file = tmp_path / "src.txt"
    dst_file = tmp_path / "dst.txt"
    src_file.write_bytes(b"abc")
    io.gfile.copy(str(src_file), str(dst_file), overwrite=True)
    assert io.gfile.exists(str(dst_file))
    assert isinstance(io.gfile.glob(str(tmp_path / "*.txt")), list)
    sub_dir = tmp_path / "nested" / "dir"
    io.gfile.makedirs(str(sub_dir))
    assert isinstance(io.gfile.stat(str(src_file)), dict)

    wrapped_dummy = _DummyDataWrapper(raw_t)
    assert nn._unwrap_k(wrapped_dummy) is raw_t
    assert nn._unwrap_k(raw_t) is raw_t

    assert isinstance(strings._to_python(lambda x: x), str)
    assert strings._as_str_tensor("hello") is not None
    joined_list = strings.join([["a", "b"], "x"], separator="-")
    assert joined_list == ["a-x", "b-x"]

    prev_tracing = _tracer.is_tracing
    prev_graph = _tracer.active_graph
    _tracer.is_tracing = True
    _tracer.active_graph = LogicalGraph(name="strings_trace")
    try:
        pt = ProxyTensor(id="str_in", shape=(1,), dtype="string")
        traced_str = tf._to_tensor(pt)
        strings.join([traced_str], separator=",")
        strings.split(traced_str)
        strings.length(traced_str)
        strings.regex_replace(traced_str, "a", "b")
        strings.regex_full_match(traced_str, "a")
        strings.to_number(traced_str)
        strings.bytes_split(traced_str)
        strings.lower(traced_str)
        strings.reduce_join(traced_str)
        strings.strip(traced_str)
        strings.substr(traced_str, 0, 1)
        strings.to_hash_bucket(traced_str, 5)
        strings.to_hash_bucket_strong(traced_str, 5, [1, 2])
        strings.upper(traced_str)
    finally:
        _tracer.is_tracing = prev_tracing
        _tracer.active_graph = prev_graph

    v1 = tf.Variable(1.0)
    v2 = tf.Variable(2.0)
    container = _DummyContainer([v1])
    non_np_container = _NonNumpyContainer()
    ckpt = train.Checkpoint(
        cont=container,
        non_np=non_np_container,
        var=v2,
        other=_NonNumpyAssignable(),
        plain="scalar",
    )
    saved_prefix = ckpt.save("test_ckpt_prefix")
    assert saved_prefix == "test_ckpt_prefix"

    v1.assign(10.0)
    v2.assign(20.0)
    status = ckpt.restore(saved_prefix)
    assert status is not None
    assert status.assert_consumed() is status
    assert status.assert_existing_objects_matched() is status
    status.run_restore_ops()
    assert ckpt.restore(None) is None
    assert ckpt.restore("nonexistent_prefix") is None

    train._CHECKPOINT_STORE["partial_ckpt"] = {}
    assert ckpt.restore("partial_ckpt") is not None


def test_scripts_check_imports(tmp_path: Path) -> None:
    """Test `scripts/check_imports.py` across valid, invalid, and syntax-error files.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert check_imports.is_stdlib("sys")
    assert check_imports.is_stdlib("json")
    assert check_imports.is_stdlib("fractions")
    assert not check_imports.is_stdlib("pytest")
    assert not check_imports.is_stdlib("nonexistent_third_party_pkg_xyz")
    assert check_imports.check_module("zero_tensorflow")
    assert check_imports.check_module("ml_switcheroo_ir")
    assert check_imports.check_module("pytest", is_test_or_script=True)
    assert not check_imports.check_module("pytest", is_test_or_script=False)
    assert not check_imports.check_module("numpy", is_test_or_script=True)
    assert not check_imports.check_module("nonexistent_third_party_pkg_xyz")

    nl = chr(10)
    bad_syntax = tmp_path / "bad_syntax.py"
    bad_syntax.write_text(f"def broken(:{nl}", encoding="utf-8")
    assert check_imports.check_file(str(bad_syntax)) == [(0, "SyntaxError")]

    disallowed_file = tmp_path / "disallowed.py"
    disallowed_file.write_text(
        nl.join(
            [
                "import some_unallowed_pkg",
                "from another_unallowed_pkg import foo",
                "from . import relative_mod",
                "import sys",
                "foo('arg')",
                "obj.some_method('arg')",
                "__import__('third_unallowed')",
                "__import__(some_var)",
                "importlib.import_module('fourth_unallowed')",
                "sys.modules.get('fifth_unallowed')",
                "x = sys.modules['sixth_unallowed']",
                "y = sys.modules[__name__]",
                "z = sys.modules[some_var]",
                "d = obj.__dict__",
                "bad_t = Any",
                "def __getattr__(name: str) -> int: return 1",
            ]
        ),
        encoding="utf-8",
    )
    errs = check_imports.check_file(str(disallowed_file))
    assert len(errs) >= 9
    assert check_imports.main(str(disallowed_file)) == 1
    assert check_imports.main("src/zero_tensorflow/dtypes.py") == 0


def test_scripts_check_api_parity(tmp_path: Path) -> None:
    """Test `scripts/check_api_parity.py` parity checking and ecosystem snapshot loading.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert check_api_parity.main("snapshots/tf_api.json") == 0
    assert check_api_parity.main(argv=["--check", "--snapshot", "snapshots/tf_api.json"]) == 0
    assert check_api_parity.main(str(tmp_path / "missing.json")) == 1

    class _FakeMath:
        """Synthetic math namespace for parity error testing."""

        val = 1

        @staticmethod
        def pow() -> None:
            """Return None."""
            return None

        @staticmethod
        def nonexistent_op() -> None:
            """Return None."""
            return None

    class _FakeModule:
        """Synthetic root module for parity error testing."""

        ext_mod = types
        nn = _FakeMath
        missing_top = 123

    errs = check_api_parity.check_parity(
        _FakeModule,
        {"nn": {"type": "module", "contents": {}}},
    )
    assert any("missing_top" in e for e in errs)
    assert any("nonexistent_op" in e for e in errs)

    bad_snap = tmp_path / "bad_snap.json"
    bad_snap.write_text("{}", encoding="utf-8")
    assert check_api_parity.main(str(bad_snap)) == 1

    assert check_api_parity.load_ecosystem_snapshot_paths(str(tmp_path / "nope.json")) == set()
    fake_snap = tmp_path / "eco_snap.json"
    fake_snap.write_text(
        '{"categories": {"array": [{"api_path": "tf.abs", "kind": "function"}, {"kind": "other"}]}}',
        encoding="utf-8",
    )
    paths = check_api_parity.load_ecosystem_snapshot_paths(str(fake_snap))
    assert "tf.abs" in paths
    assert check_api_parity._resolve_v005_snapshot_file(None) != ""

    assert check_api_parity.parse_todo_unchecked(str(tmp_path / "no_todo.md")) == set()
    assert check_api_parity.resolve_tf_symbol(tf, "other.abs") == (False, None)
    assert check_api_parity.resolve_tf_symbol(tf, "tf.nonexistent_xyz") == (False, None)

    from ml_switcheroo_ir.schema.ghost import GhostParam, GhostRef, ParameterKind

    ref_missing = GhostRef(name="nope", api_path="tf.nope_xyz", kind="function")
    assert check_api_parity.verify_v005_ref(tf, ref_missing) == [
        "Missing minimum v0.0.5 API: tf.nope_xyz"
    ]

    ref_not_cls = GhostRef(name="abs", api_path="tf.abs", kind="class")
    assert "expected class" in check_api_parity.verify_v005_ref(tf, ref_not_cls)[0]

    ref_not_fn = GhostRef(name="version", api_path="tf.version.VERSION", kind="function")
    assert "expected callable" in check_api_parity.verify_v005_ref(tf, ref_not_fn)[0]

    class _FakeTargetMod:
        """Fake module for v0.0.5 verification edge cases."""

        keras = "not_zero_keras"

        class DoclessCls:
            """Fake class whose docstring and annotation are cleared at runtime."""

            FIELD_A = 1

            def __init__(self, a: int, b: int) -> None:
                """Initialize `DoclessCls`.

                Args:
                    a: First arg.
                    b: Second arg.
                """
                self.a = a
                self.b = b

    _FakeTargetMod.DoclessCls.__doc__ = ""
    _FakeTargetMod.DoclessCls.__init__.__annotations__.pop("a", None)
    ref_edge = GhostRef(
        name="DoclessCls",
        api_path="tf.DoclessCls",
        kind="class",
        aliases=["tf.keras.Ignored", "tf.MissingAlias"],
        params=[
            GhostParam(name="self", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="a", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="b", kind=ParameterKind.POSITIONAL_OR_KEYWORD, default="1"),
            GhostParam(name="FIELD_A", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
            GhostParam(name="missing_p", kind=ParameterKind.POSITIONAL_OR_KEYWORD),
        ],
    )
    edge_errs = check_api_parity.verify_v005_ref(_FakeTargetMod, ref_edge)
    assert any("missing docstring" in e for e in edge_errs)
    assert any("missing alias tf.MissingAlias" in e for e in edge_errs)
    assert any("missing type annotation" in e for e in edge_errs)
    assert any("missing default value" in e for e in edge_errs)
    assert any("missing parameter/field 'missing_p'" in e for e in edge_errs)

    assert check_api_parity.check_v005_minimum_compliance(
        _FakeTargetMod, ecosystem_snapshot_path=str(tmp_path / "missing_v005.json")
    ) == (0, 0, [])

    v005_test_snap = tmp_path / "v005_test.json"
    v005_test_snap.write_text(
        '{"target": "tensorflow", "version": "2.21.0", "categories": {"array": ['
        '{"name": "DoclessCls", "api_path": "tf.DoclessCls", "kind": "class", "params": []},'
        '{"name": "ExemptItem", "api_path": "tf.ExemptItem", "kind": "class", "params": []},'
        '{"name": "Dense", "api_path": "tf.keras.layers.Dense", "kind": "class", "params": []}'
        "]}}",
        encoding="utf-8",
    )
    todo_plan_tmp = tmp_path / "todo_tmp.md"
    todo_plan_tmp.write_text(
        "- [ ] `tf.ExemptItem` (`class`)\n- [ ] no_backticks\n", encoding="utf-8"
    )
    _, total_c, viols = check_api_parity.check_v005_minimum_compliance(
        _FakeTargetMod,
        ecosystem_snapshot_path=str(v005_test_snap),
        todo_plan_path=str(todo_plan_tmp),
    )
    assert total_c == 2
    assert any("tf.keras must delegate directly to zero_keras" in v for v in viols)
    assert check_api_parity.parse_todo_unchecked(str(tmp_path / "no_todo.md")) == set()


def test_scripts_generate_api_snapshot(tmp_path: Path) -> None:
    """Test `scripts/generate_api_snapshot.py` live and ecosystem snapshot builders.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert generate_api_snapshot.safe_name(tf) == "zero_tensorflow"
    assert generate_api_snapshot.safe_name(123) == ""
    assert generate_api_snapshot.get_public_api(tf, depth=3) == {}

    class _FakeSubMod(types.ModuleType):
        """Typed fake submodule."""

        val: int = 1

    class _FakeRootTF(types.ModuleType):
        """Typed fake root zero_tensorflow module."""

        def __init__(self) -> None:
            """Initialize `_FakeRootTF`."""
            super().__init__("zero_tensorflow")
            self.sub = _FakeSubMod("zero_tensorflow.sub")
            self.ext = types.ModuleType("other_pkg.sub")
            self.Cls = _DummyContainer
            self.fn = lambda: None
            self.const = 42

    fake_tf = _FakeRootTF()
    live_api = generate_api_snapshot.get_public_api(fake_tf)
    assert live_api["sub"]["type"] == "module"
    assert "ext" not in live_api
    assert live_api["Cls"]["type"] == "class"
    assert live_api["fn"]["type"] == "function"
    assert live_api["const"]["type"] == "value"

    eco_file = tmp_path / "tf_eco.json"
    eco_file.write_text(
        '{"categories": {"ops": ['
        '{"api_path": "tf.math.add", "kind": "function"},'
        '{"api_path": "tf.Tensor", "kind": "class"},'
        '{"api_path": "other.ignore", "kind": "function"}'
        "]}}",
        encoding="utf-8",
    )
    out_file = tmp_path / "out_api.json"
    tree = generate_api_snapshot.generate_snapshot(
        output_path=str(out_file),
        ecosystem_snapshot_path=str(eco_file),
    )
    assert "Tensor" in tree
    assert "math" in tree
    assert out_file.exists()

    out_file2 = tmp_path / "out_api2.json"
    missing_eco = tmp_path / "missing_eco.json"
    generate_api_snapshot.generate_snapshot(
        output_path=str(out_file2),
        ecosystem_snapshot_path=str(missing_eco),
    )
    assert out_file2.exists()

    bad_eco = tmp_path / "bad_eco.json"
    bad_eco.write_text('{"categories": "invalid"}', encoding="utf-8")
    assert generate_api_snapshot.build_api_from_ecosystem_snapshot(str(bad_eco)) == {}


def test_scripts_rebuild_api_and_init(tmp_path: Path) -> None:
    """Test `scripts/rebuild_api.py` and `scripts/rebuild_init.py`.

    Args:
        tmp_path: Temporary directory fixture.
    """
    counts = rebuild_api.main()
    assert counts["math"] > 0
    assert counts["top_level"] > 0

    nl = chr(10)
    empty_py = tmp_path / "empty.py"
    empty_py.write_text(f"x = 1{nl}", encoding="utf-8")
    assert rebuild_api.extract_math_methods(str(empty_py)) == {}
    assert rebuild_api._rename_func_source(f"x = 1{nl}", "y") == f"x = 1{nl}"
    assert rebuild_init.rebuild_init_source(f"x = 1{nl}", {}) == f"x = 1{nl}"

    mini_init = tmp_path / "mini_init.py"
    mini_init.write_text(
        nl.join(
            [
                "class math:",
                '    """Doc."""',
                "    @staticmethod",
                "    def abs(*args, **kwargs):",
                "        return 1",
                "",
                "    @staticmethod",
                "    def all(*args, **kwargs):",
                "        return 2",
                "",
                "    @staticmethod",
                "    def binary(*args, **kwargs):",
                "        return 3",
                "",
                "    @staticmethod",
                "    def inv(*args, **kwargs):",
                "        return 4",
                "",
                "    @staticmethod",
                "    def zeta(*args, **kwargs):",
                "        return 5",
                "# tail",
            ]
        ),
        encoding="utf-8",
    )
    mini_snap = tmp_path / "mini_snap.json"
    mini_snap.write_text(
        '{"abs": {"type": "function"}, "math": {"type": "module", "contents": {"abs": {"type": "function"}, "reduce_all": {"type": "function"}, "zeta": {"type": "function"}}}, "reduce_all": {"type": "function"}}',
        encoding="utf-8",
    )
    rebuild_api.build_api_mappings(str(mini_snap), str(mini_init))
    rebuilt_no_write = rebuild_init.main(
        snapshot_path=str(mini_snap),
        init_path=str(mini_init),
        write=False,
    )
    assert "def zeta(" in rebuilt_no_write
    rebuilt = rebuild_init.main(
        snapshot_path=str(mini_snap),
        init_path=str(mini_init),
        write=True,
    )
    assert "def reduce_all(" in rebuilt
    assert "def abs(" in rebuilt


def test_scripts_update_badges(tmp_path: Path) -> None:
    """Test `scripts/update_badges.py` color mapping, formatting, and badge updates.

    Args:
        tmp_path: Temporary directory fixture.
    """
    assert update_badges.get_color(100.0) == "brightgreen"
    assert update_badges.get_color(95.0) == "green"
    assert update_badges.get_color(85.0) == "yellowgreen"
    assert update_badges.get_color(75.0) == "yellow"
    assert update_badges.get_color(65.0) == "orange"
    assert update_badges.get_color(50.0) == "red"

    assert update_badges.format_cov(100.0) == "100"
    assert update_badges.format_cov(99.5) == "99.5"
    assert update_badges.get_doc_coverage() == 100.0
    assert update_badges.get_doc_coverage(()) == 100.0

    nl = chr(10)
    undoc_file = tmp_path / "undoc.py"
    undoc_file.write_text(f"def f():{nl}    return 1{nl}", encoding="utf-8")
    assert update_badges.get_doc_coverage((str(undoc_file),)) < 100.0

    syntax_err_file = tmp_path / "bad_doc.py"
    syntax_err_file.write_text(f"def broken(:{nl}", encoding="utf-8")
    assert update_badges.get_doc_coverage((str(syntax_err_file), str(tmp_path / "missing"))) == 0.0
    empty_init = tmp_path / "__init__.py"
    empty_init.write_text("", encoding="utf-8")
    assert update_badges.get_doc_coverage((str(empty_init),)) == 100.0

    assert update_badges.get_test_coverage(str(tmp_path / "nonexistent_cov.json")) == 100.0
    cov_file = tmp_path / "cov.json"
    cov_file.write_text('{"totals": {"percent_covered": 100.0}}', encoding="utf-8")
    assert update_badges.get_test_coverage(str(cov_file)) == 100.0

    bad_cov = tmp_path / "bad_cov.json"
    bad_cov.write_text("not json", encoding="utf-8")
    assert update_badges.get_test_coverage(str(bad_cov)) == 0.0

    missing_totals_cov = tmp_path / "no_totals.json"
    missing_totals_cov.write_text("{}", encoding="utf-8")
    assert update_badges.get_test_coverage(str(missing_totals_cov)) == 0.0

    missing_pct_cov = tmp_path / "no_pct.json"
    missing_pct_cov.write_text('{"totals": {}}', encoding="utf-8")
    assert update_badges.get_test_coverage(str(missing_pct_cov)) == 0.0

    rep_file = tmp_path / "rep.json"
    assert update_badges.get_official_test_pass_rate(str(rep_file)) == 100.0
    rep_file.write_text('{"summary": {"total": 10, "passed": 10}}', encoding="utf-8")
    assert update_badges.get_official_test_pass_rate(str(rep_file)) == 100.0
    rep_file.write_text('{"summary": {"total": 0, "passed": 0}}', encoding="utf-8")
    assert update_badges.get_official_test_pass_rate(str(rep_file)) == 100.0
    rep_file.write_text("invalid json", encoding="utf-8")
    assert update_badges.get_official_test_pass_rate(str(rep_file)) == 0.0

    rendered_no_doc = update_badges.render_readme_badges("# Plain", 100.0, 100.0, 100.0)
    assert "# Plain" in rendered_no_doc
    readme = tmp_path / "README.md"
    assert update_badges.update_readme(str(readme)) is True
    readme.write_text(
        nl.join(
            [
                "# Title",
                "[![Test Coverage](https://img.shields.io/badge/test_coverage-90%25-green.svg)](#)",
                "[![Doc Coverage](https://img.shields.io/badge/doc_coverage-90%25-green.svg)](#)",
            ]
        ),
        encoding="utf-8",
    )
    assert (
        update_badges.update_readme(str(readme), str(cov_file), str(tmp_path / "no_rep.json"))
        is False
    )
    assert (
        update_badges.update_readme(str(readme), str(cov_file), str(tmp_path / "no_rep.json"))
        is True
    )
    assert (
        update_badges.main(["--readme", str(readme), "--cov-json", str(cov_file), "--enforce"]) == 0
    )
    low_cov = tmp_path / "low_cov.json"
    low_cov.write_text('{"totals": {"percent_covered": 50.0}}', encoding="utf-8")
    assert (
        update_badges.main(["--readme", str(readme), "--cov-json", str(low_cov), "--enforce"]) == 1
    )


def test_ci_workflow_configuration() -> None:
    """Verify CI workflow, pre-commit hooks, and dev setup scripts enforce ecosystem invariants."""
    repo_root = Path(__file__).resolve().parent.parent
    ci_yml = (repo_root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "UV_DYNAMIC_VERSIONING_BYPASS" in ci_yml
    assert "PDM_BUILD_SCM_VERSION" in ci_yml
    assert "path: ../" not in ci_yml
    assert "--cov-report=json" in ci_yml
    assert ci_yml.index("ml-ecosystem-snapshots") < ci_yml.index(
        "pip install -e ../ml-switcheroo-ir"
    )

    pre_commit = (repo_root / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    assert "--cov-report=json" in pre_commit
    assert 'pytest -m "reference"' in pre_commit

    setup_sh = (repo_root / "setup_dev_env.sh").read_text(encoding="utf-8")
    assert "UV_DYNAMIC_VERSIONING_BYPASS" in setup_sh
    assert "PDM_BUILD_SCM_VERSION" in setup_sh

    pyproject = (repo_root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'requires-python = ">=3.10"' in pyproject
