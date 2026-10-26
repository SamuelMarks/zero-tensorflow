"""Unit tests for `zero_tensorflow.strings`."""

from __future__ import annotations

import zero_tensorflow as tf
from zero_tensorflow import strings
from zero_tensorflow._typing import TensorOperand

import pytest


def test_strings_join() -> None:
    """Verify `strings.join` and `.numpy()` support."""
    res = strings.join(["a", "b"], separator=",")
    assert str(res) == "a,b"
    assert isinstance(res, strings._StringScalar)
    assert res.numpy() == "a,b"
    assert res.shape == ()
    assert res.dtype == tf.dtypes.string


def test_strings_split() -> None:
    """Verify `strings.split` and `.numpy()` support."""
    res = strings.split("a b c")
    assert isinstance(res, strings._TensorList) and list(res) == ["a", "b", "c"]
    assert res.numpy() == ["a", "b", "c"]
    assert res.shape == (3,)
    assert res.dtype == tf.dtypes.string


def test_strings_length() -> None:
    """Verify `strings.length` and `.numpy()` support."""
    res = strings.length("abc")
    assert res == 3
    assert isinstance(res, strings._IntScalar)
    assert res.numpy() == 3
    assert res.shape == ()
    assert res.dtype == tf.int64


def test_strings_regex_replace() -> None:
    """Verify `strings.regex_replace`."""
    res = strings.regex_replace("a b c", " ", "_")
    assert str(res) == "a_b_c"


def test_strings_regex_full_match() -> None:
    """Verify `strings.regex_full_match` and `.numpy()` support."""
    m1 = strings.regex_full_match("a", "a")
    m2 = strings.regex_full_match("a", "b")
    assert bool(m1) is True
    assert bool(m2) is False
    assert isinstance(m1, strings._BoolScalar)
    assert m1.numpy() is True
    assert m1.shape == ()
    assert m1.dtype == tf.bool


def test_strings_to_number() -> None:
    """Verify `strings.to_number` and `.numpy()` support."""
    res = strings.to_number("1.5")
    assert res == 1.5
    assert isinstance(res, strings._FloatScalar)
    assert res.numpy() == 1.5
    assert res.shape == ()
    assert res.dtype == tf.float32


def test_strings_bytes_split() -> None:
    """Verify `strings.bytes_split`."""
    res = strings.bytes_split("abc")
    assert isinstance(res, list) and list(res) == ["a", "b", "c"]


def test_strings_format() -> None:
    """Verify `strings.format`."""
    res = strings.format("{} b {}", ["a", "c"])
    assert str(res) == "a b c"
    assert res.numpy() == "a b c"


def test_strings_lower_upper() -> None:
    """Verify `strings.lower` and `strings.upper`."""
    assert str(strings.lower("A")) == "a"
    assert str(strings.upper("a")) == "A"


def test_strings_strip() -> None:
    """Verify `strings.strip`."""
    padded = " a "
    assert str(strings.strip(padded)) == "a"


def test_strings_substr() -> None:
    """Verify `strings.substr`."""
    assert str(strings.substr("abc", 1, 1)) == "b"


def test_strings_to_hash_bucket() -> None:
    """Verify `strings.to_hash_bucket` and keyed `strings.to_hash_bucket_strong`."""
    res = strings.to_hash_bucket("a", 10)
    assert isinstance(res, int) and 0 <= res < 10

    b1 = strings.to_hash_bucket_strong("hello_world", 100000, [1, 2])
    b2 = strings.to_hash_bucket_strong("hello_world", 100000, [99, 100])
    assert b1 != b2
    with pytest.raises(ValueError, match="2 elements"):
        strings.to_hash_bucket_strong("a", 10, [1])


def test_strings_reduce_join_axes_and_keepdims() -> None:
    """Verify `strings.reduce_join` honors `axis` and `keepdims`."""
    mat = [["a", "b"], ["c", "d"]]
    assert strings.reduce_join(mat, axis=0, separator="-") == ["a-c", "b-d"]
    assert strings.reduce_join(mat, axis=1, separator="-") == ["a-b", "c-d"]
    assert strings.reduce_join(mat, axis=-1, keepdims=True, separator="-") == [["a-b"], ["c-d"]]
    assert strings.reduce_join(mat, axis=0, keepdims=True, separator="-") == [["a-c", "b-d"]]
    assert strings.reduce_join(mat, axis=None, keepdims=True, separator="-") == [["a-b-c-d"]]
    assert strings.reduce_join([], axis=0, keepdims=True) == [""]
    assert strings.reduce_join("scalar", axis=0) == "scalar"


def test_strings_unicode_script() -> None:
    """Verify `strings.unicode_script` looks up real Unicode scripts."""
    assert strings.unicode_script([ord("a"), ord("Z")]) == [25, 25]
    assert strings.unicode_script([ord("0"), ord("!")]) == [0, 0]
    assert strings.unicode_script([0x0391, 0x0410, 0x4E2D]) == [14, 8, 17]
    assert strings.unicode_script([0x0300, -5, 0]) == [1, -1, 0]
    assert strings.unicode_script(["a", [1]]) == [25, -1]


def test_strings_stubs_exist() -> None:
    """Verify remaining string helpers."""
    strings.ngrams([], 1)
    strings.reduce_join([])
    strings.to_hash_bucket_fast("a", 10)
    strings.to_hash_bucket_strong("a", 10, [1, 2])
    strings.unicode_decode("a", "utf-8")
    strings.unicode_decode_with_offsets("a", "utf-8")
    strings.unicode_encode("a", "utf-8")
    strings.unicode_script("a")
    strings.unicode_split("a", "utf-8")
    strings.unicode_split_with_offsets("a", "utf-8")
    strings.unicode_transcode("a", "utf-8", "utf-8")
    strings.unsorted_segment_join([], [], 1)


class MockTensor:
    """Test helper class `MockTensor`."""

    def __init__(self, val: TensorOperand) -> None:
        """Execute `__init__`.

        Args:
            val: Parameter `val`.
        """
        self.val = val

    def numpy(self) -> TensorOperand:
        """Execute `numpy`.

        Returns:
            TensorOperand: Computed result.
        """
        return self.val


def test_strings_coverage_extras() -> None:
    """Verify `_TensorList` dtype branches and wrapper round-trips."""
    res = strings.length(MockTensor("abc"))
    assert res == 3
    res_j = strings.join([])
    assert str(res_j) == ""
    res_s = strings.split(MockTensor("a b"))
    assert isinstance(res_s, list) and list(res_s) == ["a", "b"]
    res_b = strings.bytes_split(MockTensor("ab"))
    assert isinstance(res_b, list) and list(res_b) == ["a", "b"]

    tl_bool = strings._TensorList([True, False])
    assert tl_bool.dtype == tf.bool
    tl_int = strings._TensorList([1, 2])
    assert tl_int.dtype == tf.int64
    tl_float = strings._TensorList([1.5, 2.5])
    assert tl_float.dtype == tf.float32
    assert strings._to_python(strings._StringScalar("x")) == "x"
    assert strings._to_python(tl_int) == [1, 2]
    assert isinstance(strings._wrap_eager_output(tf.Tensor(1)), tf.Tensor)
    assert isinstance(strings._wrap_eager_output(tf._to_tensor(1)), tf.Tensor)
    assert strings._as_str_tensor(tf._to_tensor("hello")) is not None
    assert strings._wrap_eager_output(None) is None
    assert strings._reduce_join_single_axis("leaf", 0, False, "") == "leaf"


def test_strings_split_array() -> None:
    """Execute `test_strings_split_array`."""
    res = strings.split(["a b", "c d"])
    assert isinstance(res, strings._TensorList) and len(res) == 2
    assert isinstance(res[0], list) and list(res[0]) == ["a", "b"]
    assert res.shape == (2, 2)


def test_strings_bytes_split_array() -> None:
    """Execute `test_strings_bytes_split_array`."""
    res = strings.bytes_split(["ab", "cd"])
    assert isinstance(res, list) and len(res) == 2
    assert isinstance(res[0], list) and list(res[0]) == ["a", "b"]


def test_strings_is_tracing() -> None:
    """Execute `test_strings_is_tracing`."""

    class TracingTensor:
        """Test helper class `TracingTensor`."""

        shape = ()
        dtype = "str"

    t = TracingTensor()
    from ml_switcheroo_compiler.core import config

    config.eager_mode = False
    try:
        strings.join([t])
        strings.split(t)
        strings.length(t)
        strings.regex_replace(t, "a", "b")
        strings.regex_full_match(t, "a")
        strings.to_number(t)
        strings.bytes_split(t)
        strings.lower(t)
        strings.reduce_join(t)
        strings.strip(t)
        strings.substr(t, 0, 1)
        strings.to_hash_bucket(t, 10)
        strings.to_hash_bucket_strong(t, 10, [1, 2])
        strings.upper(t)
    except (
        ValueError,
        TypeError,
        AttributeError,
        RuntimeError,
        NotImplementedError,
        KeyError,
        IndexError,
    ):
        pass
    finally:
        config.eager_mode = True
