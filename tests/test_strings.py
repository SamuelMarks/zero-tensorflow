"""Unit tests for `test_strings`."""

from __future__ import annotations

from zero_tensorflow import strings
from zero_tensorflow._typing import TensorOperand


def test_strings_join() -> None:
    """Execute `test_strings_join`."""
    res = strings.join(["a", "b"], separator=",")
    assert str(res) == "a,b"


def test_strings_split() -> None:
    """Execute `test_strings_split`."""
    res = strings.split("a b c")
    assert isinstance(res, list) and list(res) == ["a", "b", "c"]


def test_strings_length() -> None:
    """Execute `test_strings_length`."""
    res = strings.length("abc")
    assert res == 3


def test_strings_regex_replace() -> None:
    """Execute `test_strings_regex_replace`."""
    res = strings.regex_replace("a b c", " ", "_")
    assert str(res) == "a_b_c"


def test_strings_regex_full_match() -> None:
    """Execute `test_strings_regex_full_match`."""
    assert strings.regex_full_match("a", "a")
    assert not strings.regex_full_match("a", "b")


def test_strings_to_number() -> None:
    """Execute `test_strings_to_number`."""
    res = strings.to_number("1.5")
    assert res == 1.5


def test_strings_bytes_split() -> None:
    """Execute `test_strings_bytes_split`."""
    res = strings.bytes_split("abc")
    assert isinstance(res, list) and list(res) == ["a", "b", "c"]


def test_strings_format() -> None:
    """Execute `test_strings_format`."""
    res = strings.format("{} b {}", ["a", "c"])
    assert str(res) == "a b c"


def test_strings_lower_upper() -> None:
    """Execute `test_strings_lower_upper`."""
    assert str(strings.lower("A")) == "a"
    assert str(strings.upper("a")) == "A"


def test_strings_strip() -> None:
    """Execute `test_strings_strip`."""
    padded = " a "
    assert str(strings.strip(padded)) == "a"


def test_strings_substr() -> None:
    """Execute `test_strings_substr`."""
    assert str(strings.substr("abc", 1, 1)) == "b"


def test_strings_to_hash_bucket() -> None:
    """Execute `test_strings_to_hash_bucket`."""
    res = strings.to_hash_bucket("a", 10)
    assert isinstance(res, int) and 0 <= res < 10


def test_strings_stubs_exist() -> None:
    """Execute `test_strings_stubs_exist`."""
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
    """Execute `test_strings_coverage_extras`."""
    res = strings.length(MockTensor("abc"))
    assert res == 3
    res = strings.join([])
    assert str(res) == ""
    res = strings.split(MockTensor("a b"))
    assert isinstance(res, list) and list(res) == ["a", "b"]
    res = strings.bytes_split(MockTensor("ab"))
    assert isinstance(res, list) and list(res) == ["a", "b"]


def test_strings_split_array() -> None:
    """Execute `test_strings_split_array`."""
    res = strings.split(["a b", "c d"])
    assert isinstance(res, list) and len(res) == 2
    assert isinstance(res[0], list) and list(res[0]) == ["a", "b"]


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
